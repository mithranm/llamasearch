# tests/test_embedder.py

import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.configuration_utils import PretrainedConfig

from llamasearch.core.embedder import (DEFAULT_CPU_BATCH_SIZE,
                                       DEFAULT_MODEL_NAME, EmbedderConfig,
                                       EnhancedEmbedder)
from llamasearch.exceptions import ModelNotFoundError

MOCK_EMBEDDER_DATAMANAGER_PATH = "llamasearch.core.embedder.data_manager"
MOCK_EMBEDDER_HF_HUB_DOWNLOAD = "llamasearch.core.embedder.snapshot_download"
MOCK_EMBEDDER_SENTENCE_TRANSFORMER_CLS = "llamasearch.core.embedder.SentenceTransformer"


class TestEmbedderConfig(unittest.TestCase):
    def test_config_defaults(self):
        config = EmbedderConfig()
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.batch_size, DEFAULT_CPU_BATCH_SIZE)
        self.assertIsNone(config.truncate_dim)

    def test_config_custom_values(self):
        config = EmbedderConfig(
            model_name="custom/model", max_length=256, batch_size=8, truncate_dim=128
        )
        self.assertEqual(config.model_name, "custom/model")
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.truncate_dim, 128)

    def test_config_validation_truncate_dim(self):
        with self.assertRaises(ValueError):
            EmbedderConfig(truncate_dim=0)
        with self.assertRaises(ValueError):
            EmbedderConfig(truncate_dim=-10)
        config = EmbedderConfig(truncate_dim=256)
        self.assertEqual(config.truncate_dim, 256)


class TestEnhancedEmbedder(unittest.TestCase):

    def setUp(self):
        self.temp_model_dir_obj = tempfile.TemporaryDirectory(
            prefix="test_embedder_models_"
        )
        self.models_dir = Path(self.temp_model_dir_obj.name)

        self.data_manager_patcher = patch(MOCK_EMBEDDER_DATAMANAGER_PATH)
        self.mock_data_manager = self.data_manager_patcher.start()
        self.mock_data_manager.get_data_paths.return_value = {
            "models": str(self.models_dir)
        }

        self.hf_hub_download_patcher = patch(MOCK_EMBEDDER_HF_HUB_DOWNLOAD)
        self.mock_snapshot_download = self.hf_hub_download_patcher.start()

        self.st_constructor_patcher = patch(MOCK_EMBEDDER_SENTENCE_TRANSFORMER_CLS)
        self.MockSentenceTransformer = self.st_constructor_patcher.start()

        self.mock_st_model_instance = MagicMock(spec=SentenceTransformer)
        self.mock_st_model_instance.max_seq_length = 512
        self.mock_st_model_instance.encode.return_value = np.array(
            [[0.1, 0.2, 0.3]], dtype=np.float32
        )
        # Make get_sentence_embedding_dimension return a valid int for default cases
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = 384
        self.MockSentenceTransformer.return_value = self.mock_st_model_instance

        self.shutdown_event = threading.Event()

        # Patch the logger used within EnhancedEmbedder
        self.logger_patcher = patch(
            "llamasearch.core.embedder.logger", MagicMock(spec=logging.Logger)
        )
        self.mock_embedder_logger = self.logger_patcher.start()

    def tearDown(self):
        self.data_manager_patcher.stop()
        self.hf_hub_download_patcher.stop()
        self.st_constructor_patcher.stop()
        self.temp_model_dir_obj.cleanup()
        self.logger_patcher.stop()

    def test_init_default_config(self):
        embedder = EnhancedEmbedder()
        self.mock_snapshot_download.assert_called_once_with(
            repo_id=DEFAULT_MODEL_NAME,
            cache_dir=self.models_dir,
            local_files_only=True,
            local_dir_use_symlinks=False,
        )
        self.MockSentenceTransformer.assert_called_once_with(
            DEFAULT_MODEL_NAME,
            device="cpu",
            cache_folder=str(self.models_dir),
            trust_remote_code=True,
        )
        self.assertIsNotNone(embedder.model)
        self.assertEqual(embedder.config.batch_size, DEFAULT_CPU_BATCH_SIZE)

    def test_init_custom_params(self):
        custom_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        embedder = EnhancedEmbedder(
            model_name=custom_name, max_length=256, batch_size=4, truncate_dim=128
        )
        self.mock_snapshot_download.assert_called_once_with(
            repo_id=custom_name,
            cache_dir=self.models_dir,
            local_files_only=True,
            local_dir_use_symlinks=False,
        )
        self.MockSentenceTransformer.assert_called_once_with(
            custom_name,
            device="cpu",
            cache_folder=str(self.models_dir),
            trust_remote_code=True,
            truncate_dim=128,
        )
        self.assertEqual(embedder.config.max_length, 256)
        self.assertEqual(self.mock_st_model_instance.max_seq_length, 256)
        self.assertEqual(embedder.config.batch_size, 4)
        self.assertEqual(embedder.config.truncate_dim, 128)

    def test_init_invalid_truncate_dim_warning(self):
        embedder = EnhancedEmbedder(truncate_dim=-5)
        self.mock_embedder_logger.warning.assert_any_call(
            "Ignoring invalid truncate_dim value: -5. Using model default."
        )
        self.assertIsNone(embedder.config.truncate_dim)

    def test_load_model_not_found_locally_raises_error(self):
        self.mock_snapshot_download.side_effect = ModelNotFoundError("Not found")
        with self.assertRaises(ModelNotFoundError):
            EnhancedEmbedder()

    @patch.object(os, "cpu_count", return_value=4)
    @patch("torch.set_num_threads")
    @patch("torch.get_num_threads", return_value=8)
    def test_init_sets_torch_threads(
        self, mock_get_threads, mock_set_threads, mock_cpu_count
    ):
        EnhancedEmbedder()
        mock_set_threads.assert_called_once_with(2)  # 4 // 2

    def test_embed_strings_success(self):
        embedder = EnhancedEmbedder()
        docs = ["doc1", "doc2"]
        expected_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        # Ensure the mocked encode returns the expected dimension for this test
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = 2
        self.mock_st_model_instance.encode.return_value = expected_embeddings

        embeddings = embedder.embed_strings(docs)

        self.mock_st_model_instance.encode.assert_called_once_with(
            docs, batch_size=len(docs), show_progress_bar=False, convert_to_numpy=True
        )
        self.assertTrue(np.array_equal(embeddings, expected_embeddings))

    def test_embed_strings_empty_input(self):
        embedder = EnhancedEmbedder()
        embeddings = embedder.embed_strings([], show_progress=False)
        self.mock_st_model_instance.encode.assert_not_called()
        expected_dim = embedder.get_embedding_dimension()  # This will be 384 from setUp
        self.assertEqual(embeddings.shape, (0, expected_dim if expected_dim else 0))

    def test_embed_strings_model_not_loaded(self):
        embedder = EnhancedEmbedder()
        embedder.model = None
        with self.assertRaisesRegex(RuntimeError, "model is not loaded"):
            embedder.embed_strings(["test"])

    def test_embed_strings_truncation_warning(self):
        embedder = EnhancedEmbedder(max_length=5)
        long_doc = "This is a very long document that will surely be_truncated" * 10
        docs = [long_doc, "short doc"]

        # Ensure encode is mocked to simulate actual call structure
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = (
            2  # Match dummy output
        )
        self.mock_st_model_instance.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4]]
        )

        embedder.embed_strings(docs)
        self.mock_embedder_logger.warning.assert_any_call(
            f"Truncated 1 input strings > ~{5*6} chars for embedding."
        )

        truncated_first_doc = long_doc[: 5 * 6]
        self.mock_st_model_instance.encode.assert_called_once_with(
            [truncated_first_doc, "short doc"],
            batch_size=len(docs),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def test_embed_strings_batching(self):
        embedder = EnhancedEmbedder(batch_size=2)
        strings_to_embed = ["s1", "s2", "s3", "s4", "s5"]

        call_batches = []
        # Make the side effect return arrays of the correct (mocked) dimension
        mock_dim = (
            self.mock_st_model_instance.get_sentence_embedding_dimension.return_value
        )

        def batched_encode_effect(texts, batch_size, **kwargs):
            call_batches.append(list(texts))
            return np.random.rand(len(texts), mock_dim).astype(np.float32)

        self.mock_st_model_instance.encode.side_effect = batched_encode_effect

        embeddings = embedder.embed_strings(strings_to_embed, show_progress=False)

        self.assertEqual(self.mock_st_model_instance.encode.call_count, 3)
        self.assertEqual(call_batches[0], ["s1", "s2"])
        self.assertEqual(call_batches[1], ["s3", "s4"])
        self.assertEqual(call_batches[2], ["s5"])
        self.assertEqual(embeddings.shape, (5, mock_dim))

    def test_embed_strings_shutdown_event(self):
        embedder = EnhancedEmbedder()
        shutdown_event_instance = threading.Event()
        embedder.set_shutdown_event(shutdown_event_instance)

        strings = ["s1", "s2", "s3", "s4"]
        embedder.config.batch_size = 1

        call_count = 0
        # Ensure the mock dimension is consistent
        mock_dim = (
            self.mock_st_model_instance.get_sentence_embedding_dimension.return_value
        )

        processed_embeddings = []

        def encode_with_shutdown_check(texts, **kwargs):
            nonlocal call_count, processed_embeddings
            if shutdown_event_instance.is_set():  # Check before processing
                # Simulate SentenceTransformer raising an error or returning empty if interrupted
                # For this test, we'll assume the loop in embed_strings breaks.
                # The SUT's loop checks _shutdown_event *before* calling encode for a batch.
                return np.array([])

            call_count += 1
            current_batch_embedding = np.array(
                [[0.1 + call_count] * mock_dim], dtype=np.float32
            )
            processed_embeddings.append(current_batch_embedding)

            if call_count == 1:  # After processing "s1", set shutdown
                shutdown_event_instance.set()
            return current_batch_embedding

        self.mock_st_model_instance.encode.side_effect = encode_with_shutdown_check

        embeddings_interrupted = embedder.embed_strings(strings, show_progress=False)

        # The loop in embed_strings checks shutdown *before* calling encode for a batch.
        # So, "s1" gets encoded. For "s2", shutdown is set, loop breaks.
        self.assertEqual(call_count, 1)  # "s1" processed
        self.assertEqual(embeddings_interrupted.shape[0], 1)  # Only "s1" embedding
        self.assertTrue(shutdown_event_instance.is_set())

    def test_embed_string_success(self):
        embedder = EnhancedEmbedder()
        # Ensure consistent dimension with get_embedding_dimension
        mock_dim = (
            self.mock_st_model_instance.get_sentence_embedding_dimension.return_value
        )
        expected_embedding = np.array([0.1] * mock_dim, dtype=np.float32)
        self.mock_st_model_instance.encode.return_value = expected_embedding.reshape(
            1, -1
        )

        embedding = embedder.embed_string("hello")

        self.mock_st_model_instance.encode.assert_called_once_with(
            "hello", show_progress_bar=False, convert_to_numpy=True
        )
        self.assertIsNotNone(embedding)
        self.assertTrue(np.array_equal(embedding, expected_embedding))  # type: ignore

    def test_embed_string_empty(self):
        embedder = EnhancedEmbedder()
        embedding = embedder.embed_string("")
        self.assertIsNone(embedding)
        self.mock_st_model_instance.encode.assert_not_called()

    def test_get_embedding_dimension_with_truncation(self):
        embedder = EnhancedEmbedder(truncate_dim=128)
        self.assertEqual(embedder.get_embedding_dimension(), 128)

    def test_get_embedding_dimension_from_model_method(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = 768
        self.assertEqual(embedder.get_embedding_dimension(), 768)

    def test_get_embedding_dimension_from_model_config_hidden_size(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = None

        # Test with dict config
        self.mock_st_model_instance.config = {"hidden_size": 512}
        self.assertEqual(embedder.get_embedding_dimension(), 512)

        # Test with PretrainedConfig object
        mock_config_obj = PretrainedConfig(
            hidden_size=256
        )  # Use actual PretrainedConfig
        self.mock_st_model_instance.config = mock_config_obj
        self.assertEqual(embedder.get_embedding_dimension(), 256)

    def test_get_embedding_dimension_from_model_config_d_model(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = None

        # Test with dict config
        self.mock_st_model_instance.config = {"d_model": 513, "hidden_size": None}
        self.assertEqual(embedder.get_embedding_dimension(), 513)

        # Test with PretrainedConfig object
        mock_config_obj = PretrainedConfig(d_model=257)  # Use actual PretrainedConfig
        self.mock_st_model_instance.config = mock_config_obj
        self.assertEqual(embedder.get_embedding_dimension(), 257)

    def test_get_embedding_dimension_from_direct_attribute(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = None
        self.mock_st_model_instance.config = None
        self.mock_st_model_instance._embedding_size = 192
        self.assertEqual(embedder.get_embedding_dimension(), 192)

    def test_get_embedding_dimension_not_found(self):
        embedder = EnhancedEmbedder()
        self.mock_st_model_instance.get_sentence_embedding_dimension.return_value = None
        self.mock_st_model_instance.config = None
        # Ensure these attributes don't exist on the mock for this test
        for attr in [
            "_embedding_size",
            "embedding_dim",
            "dim",
            "output_embedding_dimension",
        ]:
            if hasattr(self.mock_st_model_instance, attr):
                delattr(self.mock_st_model_instance, attr)
        self.mock_st_model_instance.encode.return_value = None
        self.assertIsNone(embedder.get_embedding_dimension())

    def test_close_method(self):
        embedder = EnhancedEmbedder()
        embedder.set_shutdown_event(self.shutdown_event)
        self.assertIsNotNone(embedder.model)

        embedder.close()

        self.assertTrue(self.shutdown_event.is_set())
        self.assertIsNone(embedder.model)


if __name__ == "__main__":
    unittest.main()
