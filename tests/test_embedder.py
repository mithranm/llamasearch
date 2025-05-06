# tests/test_embedder.py
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
import tempfile
import threading

from sentence_transformers import SentenceTransformer
from transformers import PretrainedConfig

from llamasearch.core.embedder import EnhancedEmbedder, EmbedderConfig, DEFAULT_MODEL_NAME, DEFAULT_CPU_BATCH_SIZE
from llamasearch.exceptions import ModelNotFoundError

# Mock data_manager for EnhancedEmbedder
MOCK_EMBEDDER_DATAMANAGER_PATH = "llamasearch.core.embedder.data_manager"

class TestEmbedderConfig(unittest.TestCase):
    def test_config_defaults(self):
        config = EmbedderConfig()
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.batch_size, DEFAULT_CPU_BATCH_SIZE)
        self.assertIsNone(config.truncate_dim)

    def test_config_validation_truncate_dim(self):
        with self.assertRaises(ValueError):
            EmbedderConfig(truncate_dim=0)
        with self.assertRaises(ValueError):
            EmbedderConfig(truncate_dim=-100)
        config = EmbedderConfig(truncate_dim=256)
        self.assertEqual(config.truncate_dim, 256)

class TestEnhancedEmbedder(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_embedder_models_")
        self.models_dir = Path(self.temp_dir_obj.name)

        self.data_manager_patcher = patch(MOCK_EMBEDDER_DATAMANAGER_PATH)
        self.mock_data_manager = self.data_manager_patcher.start()
        self.mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}

        self.mock_snapshot_download = patch("llamasearch.core.embedder.snapshot_download").start()
        self.mock_sentence_transformer_cls = patch("llamasearch.core.embedder.SentenceTransformer").start()
        
        self.mock_st_model_instance = MagicMock(spec=SentenceTransformer)
        self.mock_st_model_instance.max_seq_length = 512 # Default
        # Mock methods needed by EnhancedEmbedder
        self.mock_st_model_instance.encode = MagicMock()
        self.mock_st_model_instance.get_sentence_embedding_dimension = MagicMock(return_value=384)
        
        # Make config accessible and mock its attributes
        mock_model_config = MagicMock(spec=PretrainedConfig)
        mock_model_config.hidden_size = 384 # Example dimension
        self.mock_st_model_instance.config = mock_model_config
        
        self.mock_sentence_transformer_cls.return_value = self.mock_st_model_instance
        
        self.mock_torch_set_num_threads = patch("torch.set_num_threads").start()
        self.mock_torch_get_num_threads = patch("torch.get_num_threads", return_value=4).start()
        self.mock_os_cpu_count = patch("os.cpu_count", return_value=8).start()

        # Simulate model files existing locally for _load_model to succeed by default
        self.mock_snapshot_download.return_value = None # Default behavior for local check

    def tearDown(self):
        patch.stopall()
        self.temp_dir_obj.cleanup()

    def test_init_default_config(self):
        embedder = EnhancedEmbedder()
        self.assertEqual(embedder.config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(embedder.config.batch_size, DEFAULT_CPU_BATCH_SIZE)
        self.mock_sentence_transformer_cls.assert_called_once_with(
            DEFAULT_MODEL_NAME, device="cpu", cache_folder=str(self.models_dir), trust_remote_code=True
        )
        self.assertEqual(self.mock_st_model_instance.max_seq_length, 512)

    def test_init_custom_params(self):
        embedder = EnhancedEmbedder(
            model_name="custom/model", max_length=256, batch_size=8, truncate_dim=128
        )
        self.assertEqual(embedder.config.model_name, "custom/model")
        self.assertEqual(embedder.config.max_length, 256)
        self.assertEqual(embedder.config.batch_size, 8)
        self.assertEqual(embedder.config.truncate_dim, 128)
        self.mock_sentence_transformer_cls.assert_called_once_with(
            "custom/model", device="cpu", cache_folder=str(self.models_dir), trust_remote_code=True, truncate_dim=128
        )
        self.assertEqual(self.mock_st_model_instance.max_seq_length, 256)

    def test_load_model_not_found_locally_raises_error(self):
        self.mock_snapshot_download.side_effect = FileNotFoundError("Model not in cache")
        with self.assertRaises(ModelNotFoundError):
            EnhancedEmbedder()
        self.mock_snapshot_download.assert_called_once_with(
            repo_id=DEFAULT_MODEL_NAME,
            cache_dir=self.models_dir,
            local_files_only=True,
            local_dir_use_symlinks=False
        )

    @patch("llamasearch.core.embedder.logger")
    def test_init_invalid_truncate_dim_warning(self, mock_logger):
        embedder = EnhancedEmbedder(truncate_dim=-5)
        self.assertIsNone(embedder.config.truncate_dim)
        mock_logger.warning.assert_any_call(
            "Ignoring invalid truncate_dim value: -5. Using model default."
        )
        # Check that truncate_dim was not passed to ST constructor
        self.mock_sentence_transformer_cls.assert_called_once_with(
            DEFAULT_MODEL_NAME, device="cpu", cache_folder=str(self.models_dir), trust_remote_code=True
        ) # No truncate_dim here

    def test_embed_strings_success(self):
        embedder = EnhancedEmbedder()
        docs = ["doc1", "doc2"]
        expected_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        self.mock_st_model_instance.encode.return_value = expected_embeddings
        
        embeddings = embedder.embed_strings(docs)
        
        self.mock_st_model_instance.encode.assert_called_once_with(
            docs, batch_size=embedder.config.batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        np.testing.assert_array_equal(embeddings, expected_embeddings)

    def test_embed_strings_empty_input(self):
        embedder = EnhancedEmbedder()
        embeddings = embedder.embed_strings([])
        self.assertEqual(embeddings.shape, (0,)) # Based on current SUT, could be (0, D)
        self.mock_st_model_instance.encode.assert_not_called()

    def test_embed_strings_model_not_loaded(self):
        embedder = EnhancedEmbedder()
        embedder.model = None # Simulate model load failure
        with self.assertRaisesRegex(RuntimeError, "Cannot embed strings, model is not loaded."):
            embedder.embed_strings(["doc1"])

    def test_embed_strings_shutdown_event(self):
        embedder = EnhancedEmbedder()
        shutdown_event = threading.Event()
        embedder.set_shutdown_event(shutdown_event)
        
        # Simulate shutdown during embedding
        original_encode = self.mock_st_model_instance.encode
        def encode_with_shutdown(*args, **kwargs):
            shutdown_event.set() # Trigger shutdown
            return original_encode(*args, **kwargs)
        self.mock_st_model_instance.encode.side_effect = encode_with_shutdown
        self.mock_st_model_instance.encode.return_value = np.array([[0.1,0.2]])
        
        embedder.config.batch_size = 1
        self.mock_st_model_instance.encode.side_effect = encode_with_shutdown # Re-assign side_effect
        shutdown_event.clear() # Reset for this run

        embeddings_interrupted = embedder.embed_strings(["doc1", "doc2", "doc3"], show_progress=False)
        
        # Expected: Processes "doc1", then shutdown is set, "doc2" and "doc3" are skipped.
        self.assertEqual(embeddings_interrupted.shape[0], 1) # Only one embedding expected
        self.mock_st_model_instance.encode.assert_any_call(
             ["doc1"], batch_size=1, show_progress_bar=False, convert_to_numpy=True
        )

    @patch("llamasearch.core.embedder.logger")
    def test_embed_strings_truncation_warning(self, mock_logger):
        embedder = EnhancedEmbedder(max_length=5) # Small max_length to trigger truncation
        long_doc = "This is a very long document that will surely be_truncated" * 10
        docs = [long_doc, "short doc"]
        self.mock_st_model_instance.encode.return_value = np.array([[0.1,0.2],[0.3,0.4]])
        
        embedder.embed_strings(docs)
        mock_logger.warning.assert_any_call(
            f"Truncated 1 input strings > ~{5*6} chars for embedding."
        )
        
        # Check that the truncated string was passed to encode for the first doc
        truncated_first_doc = long_doc[:5*6]
        self.mock_st_model_instance.encode.assert_called_once_with(
            [truncated_first_doc, "short doc"], 
            batch_size=embedder.config.batch_size, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )

    def test_embed_string_success(self):
        embedder = EnhancedEmbedder()
        text = "single doc"
        expected_embedding = np.array([0.5, 0.6], dtype=np.float32)
        self.mock_st_model_instance.encode.return_value = expected_embedding.reshape(1, -1) # encode returns 2D
        
        embedding = embedder.embed_string(text)
        
        self.mock_st_model_instance.encode.assert_called_once_with(
            text, show_progress_bar=False, convert_to_numpy=True
        )
        np.testing.assert_array_equal(embedding, expected_embedding)

    def test_get_embedding_dimension_with_truncation(self):
        embedder = EnhancedEmbedder(truncate_dim=128)
        self.assertEqual(embedder.get_embedding_dimension(), 128)

    def test_get_embedding_dimension_from_model_method(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = 768
        self.assertEqual(embedder.get_embedding_dimension(), 768)

    def test_get_embedding_dimension_from_model_config_hidden_size(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension = None # Simulate no method
        self.mock_st_model_instance.config.hidden_size = 512
        self.mock_st_model_instance.config.d_model = None # Ensure hidden_size is picked
        self.assertEqual(embedder.get_embedding_dimension(), 512)

    def test_get_embedding_dimension_from_model_config_d_model(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension = None
        self.mock_st_model_instance.config.hidden_size = None 
        self.mock_st_model_instance.config.d_model = 256
        self.assertEqual(embedder.get_embedding_dimension(), 256)

    def test_get_embedding_dimension_from_direct_attribute(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension = None
        self.mock_st_model_instance.config = None # No config object
        self.mock_st_model_instance._embedding_size = 192 # Add direct attribute
        self.assertEqual(embedder.get_embedding_dimension(), 192)
        del self.mock_st_model_instance._embedding_size # cleanup

        self.mock_st_model_instance.embedding_dim = 193
        self.assertEqual(embedder.get_embedding_dimension(), 193)


    def test_get_embedding_dimension_not_found(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension = None
        self.mock_st_model_instance.config = None
        # Ensure no direct attributes are found
        for attr_name in ['_embedding_size', 'embedding_dim', 'dim', 'output_embedding_dimension']:
            if hasattr(self.mock_st_model_instance, attr_name):
                delattr(self.mock_st_model_instance, attr_name)
        self.assertIsNone(embedder.get_embedding_dimension())

    def test_close_method(self):
        embedder = EnhancedEmbedder()
        shutdown_event = threading.Event()
        embedder.set_shutdown_event(shutdown_event)
        
        with patch.object(embedder, '_try_gc') as mock_gc:
            embedder.close()
        
        self.assertIsNone(embedder.model)
        self.assertTrue(shutdown_event.is_set())
        mock_gc.assert_called_once()

if __name__ == '__main__':
    unittest.main()