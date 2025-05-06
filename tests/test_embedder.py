import unittest
from unittest.mock import patch, MagicMock, call, ANY
import numpy as np
import torch
import os
import threading
from pathlib import Path
from pydantic import ValidationError

# Modules to test
from llamasearch.core.embedder import (
    EmbedderConfig,
    EnhancedEmbedder,
    DEFAULT_MODEL_NAME,
    DEVICE_TYPE,
    MXBAI_QUERY_PROMPT_NAME,
    ModelNotFoundError,
)

# Conditional import from the module itself
try:
    from transformers.configuration_utils import PretrainedConfig
except ImportError:
    PretrainedConfig = None # type: ignore

# Helper mock classes for HardwareInfo
class MockCPUInfo:
    def __init__(self, physical_cores=None):
        self.physical_cores = physical_cores

class MockMemoryInfo:
    def __init__(self, available_gb=0):
        self.available_gb = available_gb

class MockHardwareInfo:
    def __init__(self, available_ram_gb=16, physical_cores=4):
        self.memory = MockMemoryInfo(available_gb=available_ram_gb)
        self.cpu = MockCPUInfo(physical_cores=physical_cores)

# This is the model name that will be identified as "mxbai" type
MXBAI_TEST_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
OTHER_TEST_MODEL_NAME = "sentence-transformers/other-model"


class TestEmbedderConfig(unittest.TestCase):
    def test_default_initialization(self):
        config = EmbedderConfig()
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.device, DEVICE_TYPE)
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.batch_size, 32)
        self.assertIsNone(config.truncate_dim)

    def test_custom_initialization(self):
        config = EmbedderConfig(
            model_name="custom-model",
            max_length=256,
            batch_size=64,
            truncate_dim=128,
        )
        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.truncate_dim, 128)
        self.assertEqual(config.device, DEVICE_TYPE) # Should remain 'cpu'

    @patch("llamasearch.core.embedder.detect_hardware_info")
    def test_from_hardware_high_ram(self, mock_detect_hw):
        mock_detect_hw.return_value = MockHardwareInfo(available_ram_gb=32)
        config = EmbedderConfig.from_hardware()
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)

    @patch("llamasearch.core.embedder.detect_hardware_info")
    def test_from_hardware_mid_ram(self, mock_detect_hw):
        mock_detect_hw.return_value = MockHardwareInfo(available_ram_gb=20)
        config = EmbedderConfig.from_hardware()
        self.assertEqual(config.batch_size, 32)

    @patch("llamasearch.core.embedder.detect_hardware_info")
    def test_from_hardware_low_ram(self, mock_detect_hw):
        mock_detect_hw.return_value = MockHardwareInfo(available_ram_gb=10)
        config = EmbedderConfig.from_hardware()
        self.assertEqual(config.batch_size, 16)

    @patch("llamasearch.core.embedder.detect_hardware_info")
    def test_from_hardware_very_low_ram(self, mock_detect_hw):
        mock_detect_hw.return_value = MockHardwareInfo(available_ram_gb=5)
        config = EmbedderConfig.from_hardware()
        self.assertEqual(config.batch_size, 8)

    @patch("llamasearch.core.embedder.detect_hardware_info")
    def test_from_hardware_custom_model(self, mock_detect_hw):
        mock_detect_hw.return_value = MockHardwareInfo(available_ram_gb=16)
        config = EmbedderConfig.from_hardware(model_name="custom/model")
        self.assertEqual(config.model_name, "custom/model")
        self.assertEqual(config.batch_size, 32) # Based on 16GB RAM

    def test_truncate_dim_validator_valid(self):
        config = EmbedderConfig(truncate_dim=256)
        self.assertEqual(config.truncate_dim, 256)

    def test_truncate_dim_validator_none(self):
        config = EmbedderConfig(truncate_dim=None)
        self.assertIsNone(config.truncate_dim)

    def test_truncate_dim_validator_invalid_zero(self):
        with self.assertRaises(ValidationError) as context:
            EmbedderConfig(truncate_dim=0)
        self.assertIn("truncate_dim must be a positive integer if set", str(context.exception))

    def test_truncate_dim_validator_invalid_negative(self):
        with self.assertRaises(ValidationError) as context:
            EmbedderConfig(truncate_dim=-10)
        self.assertIn("truncate_dim must be a positive integer if set", str(context.exception))


@patch("llamasearch.core.embedder.logger", MagicMock()) # Global mock for logger
class TestEnhancedEmbedder(unittest.TestCase):

    def _create_mock_st_model(self, dim=384, throws_on_prompt_name=False, no_get_dim_method=False):
        mock_model = MagicMock(spec=True) # spec=SentenceTransformer) # Using spec=True for flexibility
        mock_model.encode = MagicMock()

        # Mocking for get_embedding_dimension
        if no_get_dim_method:
            if hasattr(mock_model, 'get_sentence_embedding_dimension'):
                delattr(mock_model, 'get_sentence_embedding_dimension')
        else:
            mock_model.get_sentence_embedding_dimension = MagicMock(return_value=dim)
        
        # Simplified config for dimension fallback
        mock_model.config = {'hidden_size': dim} 
        mock_model.max_seq_length = 512 # Default

        def mock_encode_fn(sentences, batch_size, show_progress_bar, convert_to_numpy, prompt_name=None):
            if throws_on_prompt_name and prompt_name:
                raise ValueError(f"Prompt name '{prompt_name}' not found in model.")
            
            if isinstance(sentences, str): # single string
                return np.random.rand(dim).astype(np.float32)
            # list of strings
            return np.random.rand(len(sentences), dim).astype(np.float32)

        mock_model.encode.side_effect = mock_encode_fn
        return mock_model

    def setUp(self):
        # Mock hardware info
        self.mock_hw_info = MockHardwareInfo(available_ram_gb=16, physical_cores=4)
        self.patch_detect_hw = patch("llamasearch.core.embedder.detect_hardware_info", return_value=self.mock_hw_info)
        self.mock_detect_hw = self.patch_detect_hw.start()

        # Mock data_manager
        self.mock_data_manager = MagicMock()
        self.mock_data_manager.get_data_paths.return_value = {"models": "mock/models/path"}
        self.patch_data_manager = patch("llamasearch.core.embedder.data_manager", self.mock_data_manager)
        self.mock_data_manager_instance = self.patch_data_manager.start()

        # Mock SentenceTransformer
        self.mock_st_class = MagicMock()
        self.mock_st_instance = self._create_mock_st_model()
        self.mock_st_class.return_value = self.mock_st_instance
        self.patch_st_class = patch("llamasearch.core.embedder.SentenceTransformer", self.mock_st_class)
        self.mock_st_constructor = self.patch_st_class.start()

        # Mock snapshot_download
        self.patch_snapshot_download = patch("llamasearch.core.embedder.snapshot_download")
        self.mock_snapshot_download = self.patch_snapshot_download.start()
        self.mock_snapshot_download.return_value = None # Simulate model found locally

        # Mock torch functionalities
        self.patch_torch_get_threads = patch("torch.get_num_threads", side_effect=[4, 2]) # Initial, After set
        self.mock_torch_get_threads = self.patch_torch_get_threads.start()
        self.patch_torch_set_threads = patch("torch.set_num_threads")
        self.mock_torch_set_threads = self.patch_torch_set_threads.start()
        
        self.patch_torch_mps_is_available = patch("torch.backends.mps.is_available", return_value=False)
        self.mock_torch_mps_is_available = self.patch_torch_mps_is_available.start()
        self.patch_torch_mps_empty_cache = patch("torch.mps.empty_cache")
        self.mock_torch_mps_empty_cache = self.patch_torch_mps_empty_cache.start()


        # Mock gc
        self.patch_gc_collect = patch("gc.collect")
        self.mock_gc_collect = self.patch_gc_collect.start()

        # Mock os.cpu_count
        self.patch_os_cpu_count = patch("os.cpu_count", return_value=4)
        self.mock_os_cpu_count = self.patch_os_cpu_count.start()

        # Mock Path.mkdir
        self.patch_path_mkdir = patch("pathlib.Path.mkdir")
        self.mock_path_mkdir = self.patch_path_mkdir.start()
        
        # Mock tqdm
        self.patch_tqdm = patch("llamasearch.core.embedder.tqdm")
        self.mock_tqdm_constructor = self.patch_tqdm.start()
        self.mock_tqdm_instance = MagicMock()
        self.mock_tqdm_constructor.return_value = self.mock_tqdm_instance

        # Mock PretrainedConfig from embedder's scope
        self.patch_embedder_pretrained_config = patch("llamasearch.core.embedder.PretrainedConfig", PretrainedConfig)
        self.mock_embedder_pretrained_config = self.patch_embedder_pretrained_config.start()


    def tearDown(self):
        self.patch_detect_hw.stop()
        self.patch_data_manager.stop()
        self.patch_st_class.stop()
        self.patch_snapshot_download.stop()
        self.patch_torch_get_threads.stop()
        self.patch_torch_set_threads.stop()
        self.patch_torch_mps_is_available.stop()
        self.patch_torch_mps_empty_cache.stop()
        self.patch_gc_collect.stop()
        self.patch_os_cpu_count.stop()
        self.patch_path_mkdir.stop()
        self.patch_tqdm.stop()
        self.patch_embedder_pretrained_config.stop()


    def test_init_default_config(self):
        embedder = EnhancedEmbedder()
        self.assertEqual(embedder.config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(embedder.config.batch_size, 32) # From MockHardwareInfo (16GB)
        self.mock_st_constructor.assert_called_once_with(
            DEFAULT_MODEL_NAME,
            device=DEVICE_TYPE,
            cache_folder="mock/models/path",
            trust_remote_code=True,
        )
        self.assertIsNotNone(embedder.model)

    def test_init_custom_params_override_hardware(self):
        embedder = EnhancedEmbedder(
            model_name="custom/model", max_length=256, batch_size=128, truncate_dim=100
        )
        self.assertEqual(embedder.config.model_name, "custom/model")
        self.assertEqual(embedder.config.max_length, 256)
        self.assertEqual(embedder.config.batch_size, 128) # User override
        self.assertEqual(embedder.config.truncate_dim, 100)
        self.mock_st_constructor.assert_called_once_with(
            "custom/model",
            device=DEVICE_TYPE,
            cache_folder="mock/models/path",
            trust_remote_code=True,
            truncate_dim=100,
        )

    def test_init_truncate_dim_valid_and_invalid(self):
        # Valid
        EnhancedEmbedder(truncate_dim=256)
        # Last call, because other tests also call constructor
        args, kwargs = self.mock_st_constructor.call_args_list[-1]
        self.assertEqual(kwargs.get("truncate_dim"), 256)

        # Invalid (0) - should be ignored and use model default (None for truncate_dim)
        EnhancedEmbedder(truncate_dim=0)
        args, kwargs = self.mock_st_constructor.call_args_list[-1]
        self.assertNotIn("truncate_dim", kwargs) # Should not pass it if invalid
        
        # Invalid (-1)
        EnhancedEmbedder(truncate_dim=-1)
        args, kwargs = self.mock_st_constructor.call_args_list[-1]
        self.assertNotIn("truncate_dim", kwargs)

    def test_init_datamanager_path_failure(self):
        self.mock_data_manager.get_data_paths.side_effect = Exception("DM Error")
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_dir", return_value=True): # Ensure default path seems to exist
            embedder = EnhancedEmbedder()
            expected_fallback_path = Path(".") / ".llamasearch" / "models"
            self.assertEqual(embedder.models_dir, expected_fallback_path)
            self.mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_load_model_snapshot_download_local_not_found(self):
        from huggingface_hub.errors import LocalEntryNotFoundError
        self.mock_snapshot_download.side_effect = LocalEntryNotFoundError("Not found")
        with self.assertRaises(ModelNotFoundError):
            EnhancedEmbedder()

    def test_load_model_snapshot_download_entry_not_found(self):
        from huggingface_hub.errors import EntryNotFoundError
        self.mock_snapshot_download.side_effect = EntryNotFoundError("Not found")
        with self.assertRaises(ModelNotFoundError):
            EnhancedEmbedder()
            
    def test_load_model_snapshot_download_file_not_found(self):
        self.mock_snapshot_download.side_effect = FileNotFoundError("Not found")
        with self.assertRaises(ModelNotFoundError):
            EnhancedEmbedder()

    def test_load_model_snapshot_download_other_exception(self):
        self.mock_snapshot_download.side_effect = Exception("Other snapshot error")
        with self.assertRaises(ModelNotFoundError): # It's caught and re-raised as ModelNotFoundError
            EnhancedEmbedder()

    def test_load_model_sentence_transformer_exception(self):
        self.mock_st_constructor.side_effect = RuntimeError("ST Load Error")
        with self.assertRaises(RuntimeError) as context:
            EnhancedEmbedder()
        self.assertIn("Failed to load embedder model", str(context.exception))
        self.assertIn("ST Load Error", str(context.exception))

    def test_load_model_torch_thread_settings(self):
        self.mock_hw_info.cpu.physical_cores = 8
        self.mock_torch_get_threads.side_effect = [10, 4] # Initial 10, after set, target 8/2 = 4
        EnhancedEmbedder()
        self.mock_torch_set_threads.assert_called_once_with(4) # 8 physical / 2

    def test_load_model_torch_thread_settings_no_physical_cores(self):
        self.mock_hw_info.cpu.physical_cores = None
        self.mock_os_cpu_count.return_value = 6 # os.cpu_count() fallback
        self.mock_torch_get_threads.side_effect = [10, 3] # Initial 10, after set, target 6/2 = 3
        EnhancedEmbedder()
        self.mock_torch_set_threads.assert_called_once_with(3)

    def test_load_model_torch_thread_settings_no_physical_no_os_count(self):
        self.mock_hw_info.cpu.physical_cores = None
        self.mock_os_cpu_count.return_value = None # os.cpu_count() fallback
        self.mock_torch_get_threads.side_effect = [10, 1] # Initial 10, after set, target default 2 / 2 = 1
        EnhancedEmbedder()
        self.mock_torch_set_threads.assert_called_once_with(1) # Default 2 cores / 2

    def test_set_shutdown_event(self):
        embedder = EnhancedEmbedder()
        event = threading.Event()
        embedder.set_shutdown_event(event)
        self.assertIs(embedder._shutdown_event, event)

    def test_should_use_prompt_name(self):
        embedder = EnhancedEmbedder(model_name=MXBAI_TEST_MODEL_NAME)
        self.assertTrue(embedder._should_use_prompt_name("query"))
        self.assertFalse(embedder._should_use_prompt_name("document"))

        embedder_other = EnhancedEmbedder(model_name=OTHER_TEST_MODEL_NAME)
        self.assertFalse(embedder_other._should_use_prompt_name("query"))
        self.assertFalse(embedder_other._should_use_prompt_name("document"))

    def test_embed_strings_model_not_loaded(self):
        embedder = EnhancedEmbedder()
        embedder.model = None # Simulate failed load
        with self.assertRaises(RuntimeError) as context:
            embedder.embed_strings(["test"])
        self.assertIn("model is not loaded", str(context.exception))

    def test_embed_strings_empty_input(self):
        embedder = EnhancedEmbedder()
        result = embedder.embed_strings([])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0,)) # As per np.array([], dtype=np.float32)

    def test_embed_strings_success_document_type(self):
        embedder = EnhancedEmbedder(model_name=OTHER_TEST_MODEL_NAME)
        texts = ["doc1", "doc2"]
        expected_embeddings = np.random.rand(2, 384).astype(np.float32)
        self.mock_st_instance.encode.return_value = expected_embeddings
        
        result = embedder.embed_strings(texts, input_type="document")
        
        self.mock_st_instance.encode.assert_called_once_with(
            texts, batch_size=len(texts), show_progress_bar=False, convert_to_numpy=True # prompt_name should not be there
        )
        self.assertTrue(np.array_equal(result, expected_embeddings))
        self.mock_tqdm_instance.update.assert_called_with(len(texts))
        self.mock_tqdm_instance.close.assert_called_once()

    def test_embed_strings_success_query_type_mxbai_model(self):
        embedder = EnhancedEmbedder(model_name=MXBAI_TEST_MODEL_NAME)
        texts = ["query1", "query2"]
        expected_embeddings = np.random.rand(2, 384).astype(np.float32)
        self.mock_st_instance.encode.return_value = expected_embeddings

        result = embedder.embed_strings(texts, input_type="query")
        
        self.mock_st_instance.encode.assert_called_once_with(
            texts, batch_size=len(texts), show_progress_bar=False, convert_to_numpy=True,
            prompt_name=MXBAI_QUERY_PROMPT_NAME
        )
        self.assertTrue(np.array_equal(result, expected_embeddings))
        
    def test_embed_strings_no_progress_bar(self):
        embedder = EnhancedEmbedder()
        texts = ["text1"]
        embedder.embed_strings(texts, show_progress=False)
        self.mock_tqdm_constructor.assert_called_with(total=ANY, desc=ANY, unit=ANY, disable=True)


    def test_embed_strings_input_truncation(self):
        embedder = EnhancedEmbedder(max_length=10) # max_length for config
        self.mock_st_instance.max_seq_length = 10 # Set on mock model
        
        long_text = "a" * (10 * 6 + 100) # max_length * 6 (char estimate) + more
        short_text = "short"
        
        expected_truncated_text = long_text[:10*6]
        
        embedder.embed_strings([long_text, short_text])
        
        # Check that encode was called with the truncated text
        # The mock_st_instance.encode gets called per batch
        call_args_list = self.mock_st_instance.encode.call_args_list
        self.assertEqual(len(call_args_list), 1) # Assuming batch_size > 2 for this test
        called_with_texts = call_args_list[0][0][0]
        self.assertEqual(called_with_texts[0], expected_truncated_text)
        self.assertEqual(called_with_texts[1], short_text)

    def test_embed_strings_batching(self):
        embedder = EnhancedEmbedder(batch_size=2) # Config batch size
        texts = ["t1", "t2", "t3", "t4", "t5"]
        
        # Mock encode to return based on input length
        def batched_encode_effect(sents, batch_size, **kwargs):
            return np.random.rand(len(sents), 384).astype(np.float32)
        self.mock_st_instance.encode.side_effect = batched_encode_effect

        result = embedder.embed_strings(texts)
        
        self.assertEqual(self.mock_st_instance.encode.call_count, 3) # 5 texts, batch_size 2 -> 2, 2, 1
        self.assertEqual(len(self.mock_st_instance.encode.call_args_list[0][0][0]), 2) # First batch
        self.assertEqual(len(self.mock_st_instance.encode.call_args_list[1][0][0]), 2) # Second batch
        self.assertEqual(len(self.mock_st_instance.encode.call_args_list[2][0][0]), 1) # Third batch
        self.assertEqual(result.shape, (5, 384))
        self.mock_tqdm_instance.update.assert_has_calls([call(2), call(2), call(1)])


    def test_embed_strings_shutdown_interrupt(self):
        embedder = EnhancedEmbedder(batch_size=1)
        shutdown_event = threading.Event()
        embedder.set_shutdown_event(shutdown_event)
        
        texts = ["t1", "t2", "t3"]
        
        # Simulate shutdown after the first item
        processed_count = 0
        def interrupt_encode_effect(sents, **kwargs):
            nonlocal processed_count
            processed_count += len(sents)
            if processed_count >= 1:
                shutdown_event.set()
            return np.random.rand(len(sents), 384).astype(np.float32)

        self.mock_st_instance.encode.side_effect = interrupt_encode_effect
        result = embedder.embed_strings(texts)
        
        self.assertEqual(result.shape[0], 1) # Only first item processed
        self.assertTrue(shutdown_event.is_set())

    def test_embed_strings_model_encode_returns_empty_batch(self):
        embedder = EnhancedEmbedder()
        self.mock_st_instance.encode.return_value = np.array([], dtype=np.float32)
        result = embedder.embed_strings(["test1", "test2"])
        # Falls back to get_embedding_dimension to shape the empty array
        expected_dim = embedder.get_embedding_dimension()
        self.assertEqual(result.shape, (0, expected_dim if expected_dim else 0))


    def test_embed_strings_model_encode_exception(self):
        embedder = EnhancedEmbedder()
        self.mock_st_instance.encode.side_effect = Exception("Encode error")
        # Mock get_embedding_dimension for the error return path
        embedder.get_embedding_dimension = MagicMock(return_value=128)

        result = embedder.embed_strings(["test"])
        
        self.assertEqual(result.shape, (0, 128)) # Empty array with expected dim
        self.mock_tqdm_instance.close.assert_called_once()


    def test_embed_strings_model_encode_prompt_name_value_error(self):
        # This test ensures the specific ValueError for prompt_name is handled
        # and logged differently (less verbosely, as it's an expected model incompatibility)
        self.mock_st_instance = self._create_mock_st_model(throws_on_prompt_name=True)
        self.mock_st_constructor.return_value = self.mock_st_instance

        embedder = EnhancedEmbedder(model_name=MXBAI_TEST_MODEL_NAME) # MXBAI model to trigger prompt
        embedder.get_embedding_dimension = MagicMock(return_value=128) # For error path

        result = embedder.embed_strings(["query1"], input_type="query")
        self.assertEqual(result.shape, (0, 128))
        # Here, one would check log messages if caplog was used.
        # The `logger.error` should be called with `exc_info=False`.


    def test_embed_documents_alias(self):
        embedder = EnhancedEmbedder()
        embedder.embed_strings = MagicMock()
        docs = ["doc1"]
        embedder.embed_documents(docs, show_progress=False)
        embedder.embed_strings.assert_called_once_with(
            docs, input_type="document", show_progress=False
        )

    def test_embed_string_success(self):
        embedder = EnhancedEmbedder()
        text = "single text"
        expected_embedding = np.random.rand(384).astype(np.float32)
        self.mock_st_instance.encode.return_value = expected_embedding # encode returns 1D for single string
        
        result = embedder.embed_string(text)
        
        self.mock_st_instance.encode.assert_called_once_with(
            text, show_progress_bar=False, convert_to_numpy=True
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 1)
        self.assertTrue(np.array_equal(result, expected_embedding))

    def test_embed_string_success_query_type_mxbai_model(self):
        embedder = EnhancedEmbedder(model_name=MXBAI_TEST_MODEL_NAME)
        text = "a query"
        expected_embedding = np.random.rand(384).astype(np.float32)
        self.mock_st_instance.encode.return_value = expected_embedding

        result = embedder.embed_string(text, input_type="query")
        self.mock_st_instance.encode.assert_called_once_with(
            text, show_progress_bar=False, convert_to_numpy=True,
            prompt_name=MXBAI_QUERY_PROMPT_NAME
        )
        self.assertTrue(np.array_equal(result, expected_embedding))

    def test_embed_string_model_not_loaded(self):
        embedder = EnhancedEmbedder()
        embedder.model = None
        self.assertIsNone(embedder.embed_string("test"))

    def test_embed_string_empty_input(self):
        embedder = EnhancedEmbedder()
        self.assertIsNone(embedder.embed_string(""))

    def test_embed_string_shutdown_interrupt(self):
        embedder = EnhancedEmbedder()
        shutdown_event = threading.Event()
        shutdown_event.set() # Pre-set shutdown
        embedder.set_shutdown_event(shutdown_event)
        self.assertIsNone(embedder.embed_string("test"))

    def test_embed_string_input_truncation(self):
        embedder = EnhancedEmbedder(max_length=5)
        self.mock_st_instance.max_seq_length = 5
        long_text = "abc" * (5 * 6) # text longer than 5*6
        expected_truncated = long_text[:5*6]
        
        embedder.embed_string(long_text)
        self.mock_st_instance.encode.assert_called_once_with(
            expected_truncated, show_progress_bar=False, convert_to_numpy=True
        )

    def test_embed_string_model_encode_returns_none(self):
        embedder = EnhancedEmbedder()
        self.mock_st_instance.encode.return_value = None
        self.assertIsNone(embedder.embed_string("test"))

    def test_embed_string_model_encode_exception(self):
        embedder = EnhancedEmbedder()
        self.mock_st_instance.encode.side_effect = Exception("Encode single error")
        self.assertIsNone(embedder.embed_string("test"))
        
    def test_embed_string_model_encode_returns_2d_array(self):
        embedder = EnhancedEmbedder()
        text = "single text"
        # Simulate model returning (1, dim) for a single string
        expected_1d_embedding = np.random.rand(384).astype(np.float32)
        mock_2d_return = expected_1d_embedding.reshape(1, -1)
        self.mock_st_instance.encode.return_value = mock_2d_return
        
        result = embedder.embed_string(text)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 1) # Should be converted back to 1D
        self.assertTrue(np.array_equal(result, expected_1d_embedding))

    def test_get_embedding_dimension_model_not_loaded(self):
        embedder = EnhancedEmbedder()
        embedder.model = None
        self.assertIsNone(embedder.get_embedding_dimension())

    def test_get_embedding_dimension_with_truncate_dim_config(self):
        embedder = EnhancedEmbedder(truncate_dim=128)
        self.assertEqual(embedder.get_embedding_dimension(), 128)

    def test_get_embedding_dimension_from_st_method(self):
        embedder = EnhancedEmbedder() # mock_st_instance has get_sentence_embedding_dimension
        self.mock_st_instance.get_sentence_embedding_dimension.return_value = 768
        self.assertEqual(embedder.get_embedding_dimension(), 768)

    def test_get_embedding_dimension_from_st_method_invalid_return(self):
        embedder = EnhancedEmbedder()
        self.mock_st_instance.get_sentence_embedding_dimension.return_value = "not an int"
        # Fallback to config, then direct attributes. Our mock has config {'hidden_size': 384}
        self.assertEqual(embedder.get_embedding_dimension(), 384)

        self.mock_st_instance.get_sentence_embedding_dimension.return_value = 0 # non-positive
        self.assertEqual(embedder.get_embedding_dimension(), 384)


    def test_get_embedding_dimension_from_model_config_dict(self):
        embedder = EnhancedEmbedder()
        # Remove get_sentence_embedding_dimension if it exists on the mock for this test
        if hasattr(embedder.model, 'get_sentence_embedding_dimension'):
            delattr(embedder.model, 'get_sentence_embedding_dimension')
        embedder.model.config = {"hidden_size": 512}
        self.assertEqual(embedder.get_embedding_dimension(), 512)
        
        embedder.model.config = {"d_model": 256}
        self.assertEqual(embedder.get_embedding_dimension(), 256)

    @unittest.skipIf(PretrainedConfig is None, "transformers not installed or PretrainedConfig not available")
    def test_get_embedding_dimension_from_model_config_pretrainedconfig(self):
        embedder = EnhancedEmbedder()
        if hasattr(embedder.model, 'get_sentence_embedding_dimension'):
            delattr(embedder.model, 'get_sentence_embedding_dimension')
        
        mock_hf_config = MagicMock(spec=PretrainedConfig)
        mock_hf_config.hidden_size = 768
        embedder.model.config = mock_hf_config
        self.assertEqual(embedder.get_embedding_dimension(), 768)

        mock_hf_config_dmodel = MagicMock(spec=PretrainedConfig)
        mock_hf_config_dmodel.hidden_size = None # Test fallback to d_model
        mock_hf_config_dmodel.d_model = 1024
        embedder.model.config = mock_hf_config_dmodel
        self.assertEqual(embedder.get_embedding_dimension(), 1024)

    def test_get_embedding_dimension_from_model_config_pretrainedconfig_module_not_available(self):
        # Patch PretrainedConfig to be None in the embedder's scope for this test
        with patch('llamasearch.core.embedder.PretrainedConfig', None):
            embedder = EnhancedEmbedder() # Re-init to pick up patched PretrainedConfig
            # Ensure that the model.config is an object, not a dict for this specific path test
            # But it won't be an 'instanceof PretrainedConfig' if PretrainedConfig is None
            class SomeOtherConfig:
                hidden_size = 768
            
            if hasattr(embedder.model, 'get_sentence_embedding_dimension'):
                delattr(embedder.model, 'get_sentence_embedding_dimension')
            embedder.model.config = SomeOtherConfig()
            
            # It should skip the PretrainedConfig check and fall to direct attributes or None
            # Our default mock_st_instance doesn't have direct attributes, so should be None if other paths failed.
            # Let's force it to have a direct attribute for clarity
            embedder.model._embedding_size = 123
            self.assertEqual(embedder.get_embedding_dimension(), 123)

    def test_get_embedding_dimension_from_direct_attributes(self):
        embedder = EnhancedEmbedder()
        # Remove other ways to get dim
        if hasattr(embedder.model, 'get_sentence_embedding_dimension'):
            delattr(embedder.model, 'get_sentence_embedding_dimension')
        embedder.model.config = {} # Empty config

        direct_attrs = ['_embedding_size', 'embedding_dim', 'dim', 'output_embedding_dimension']
        for i, attr_name in enumerate(direct_attrs):
            # Reset other direct attributes
            for prev_attr in direct_attrs:
                if hasattr(embedder.model, prev_attr):
                    delattr(embedder.model, prev_attr)
            
            setattr(embedder.model, attr_name, 100 + i)
            self.assertEqual(embedder.get_embedding_dimension(), 100 + i)

    def test_get_embedding_dimension_not_found(self):
        embedder = EnhancedEmbedder()
        # Make sure all paths fail
        if hasattr(embedder.model, 'get_sentence_embedding_dimension'):
            delattr(embedder.model, 'get_sentence_embedding_dimension')
        embedder.model.config = {}
        direct_attrs = ['_embedding_size', 'embedding_dim', 'dim', 'output_embedding_dimension']
        for attr_name in direct_attrs:
            if hasattr(embedder.model, attr_name):
                delattr(embedder.model, attr_name)
        self.assertIsNone(embedder.get_embedding_dimension())

    def test_get_embedding_dimension_exception_during_retrieval(self):
        embedder = EnhancedEmbedder()
        embedder.model.get_sentence_embedding_dimension.side_effect = Exception("Dim error")
        self.assertIsNone(embedder.get_embedding_dimension()) # Should be caught

    def test_close_sets_shutdown_deletes_model_gcs(self):
        shutdown_event = MagicMock(spec=threading.Event)
        embedder = EnhancedEmbedder()
        embedder.set_shutdown_event(shutdown_event)
        initial_model = embedder.model
        
        embedder.close()
        
        shutdown_event.set.assert_called_once()
        self.assertIsNone(embedder.model)
        self.assertIsNotNone(initial_model) # Make sure it was there before
        self.mock_gc_collect.assert_called() # _try_gc calls it


    def test_try_gc_calls_gc_collect(self):
        embedder = EnhancedEmbedder()
        embedder._try_gc()
        self.mock_gc_collect.assert_called_once()

    def test_try_gc_calls_mps_empty_cache_if_available(self):
        self.mock_torch_mps_is_available.return_value = True
        embedder = EnhancedEmbedder()
        embedder._try_gc()
        self.mock_torch_mps_empty_cache.assert_called_once()
        self.mock_gc_collect.assert_called_once()

    def test_try_gc_mps_not_available(self):
        self.mock_torch_mps_is_available.return_value = False
        embedder = EnhancedEmbedder()
        embedder._try_gc()
        self.mock_torch_mps_empty_cache.assert_not_called()
        self.mock_gc_collect.assert_called_once()


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)