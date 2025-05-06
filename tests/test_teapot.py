# tests/test_teapot.py

import unittest
import tempfile
import shutil
import gc
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY, call

# External library mocks needed for type hints or direct patching
import torch # Keep torch for mocking torch internals

# Project-specific imports (ensure these paths are correct for your structure)
from llamasearch.core.teapot import (
    TeapotONNXModelInfo,
    TeapotONNXLLM,
    load_teapot_onnx_llm,
    _determine_onnx_provider,
    _detect_available_onnx_suffix,
    # _select_onnx_quantization, # Not tested directly
    REQUIRED_ONNX_BASENAMES,
    TEAPOT_BASE_FILES,
    ONNX_SUBFOLDER,
    # SETTINGS_FILENAME, <-- REMOVED (Was incorrect import)
    # DEFAULT_SUBDIRS, <-- REMOVED (Was incorrect import)
    QUANTIZATION_SUFFIXES_PRIORITY,
    TEAPOT_REPO_ID
)
from llamasearch.protocols import ModelInfo
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.hardware import HardwareInfo # Keep for type hints if needed

# Mock objects for external libraries being replaced
MockORTModel = MagicMock()
MockTokenizer = MagicMock()
MockTorchDevice = MagicMock(spec=torch.device)
MockTorchTensor = MagicMock(spec=torch.Tensor)

# --- Test TeapotONNXModelInfo Class ---
class TestTeapotONNXModelInfo(unittest.TestCase):
    """Tests for the TeapotONNXModelInfo data class."""

    def test_properties_fp32(self):
        info = TeapotONNXModelInfo(TEAPOT_REPO_ID, "", 1024)
        self.assertEqual(info.model_id, f"{TEAPOT_REPO_ID}-onnx-fp32")
        self.assertEqual(info.model_engine, "onnx_teapot")
        self.assertEqual(info.description, "Teapot ONNX model (fp32 quantization)")
        self.assertEqual(info.context_length, 1024)

    def test_properties_quantized(self):
        info = TeapotONNXModelInfo(TEAPOT_REPO_ID, "_int8", 512)
        self.assertEqual(info.model_id, f"{TEAPOT_REPO_ID}-onnx-int8")
        self.assertEqual(info.model_engine, "onnx_teapot")
        self.assertEqual(info.description, "Teapot ONNX model (int8 quantization)")
        self.assertEqual(info.context_length, 512)

# --- Tests for Helper Functions ---
class TestDetermineProvider(unittest.TestCase):
    """Tests for the _determine_onnx_provider helper function."""

    @patch('llamasearch.core.teapot.onnxruntime.get_available_providers')
    def test_prefer_cpu(self, mock_get_providers):
        mock_get_providers.return_value = ["CPUExecutionProvider", "CoreMLExecutionProvider"]
        provider, opts = _determine_onnx_provider(preferred_provider="CPUExecutionProvider")
        self.assertEqual(provider, "CPUExecutionProvider")
        self.assertIsNone(opts)

    @patch('llamasearch.core.teapot.onnxruntime.get_available_providers')
    def test_prefer_cuda_available(self, mock_get_providers):
        mock_get_providers.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        provider, opts = _determine_onnx_provider(preferred_provider="CUDAExecutionProvider", preferred_options={"device_id": 1})
        self.assertEqual(provider, "CUDAExecutionProvider")
        self.assertEqual(opts, {"device_id": 1})

    @patch('llamasearch.core.teapot.onnxruntime.get_available_providers')
    def test_prefer_cuda_unavailable_fallback_cpu(self, mock_get_providers):
        mock_get_providers.return_value = ["CPUExecutionProvider", "CoreMLExecutionProvider"]
        provider, opts = _determine_onnx_provider(preferred_provider="CUDAExecutionProvider")
        self.assertEqual(provider, "CPUExecutionProvider")
        self.assertIsNone(opts)

    @patch('llamasearch.core.teapot.onnxruntime.get_available_providers')
    def test_auto_select_cuda(self, mock_get_providers):
        mock_get_providers.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        provider, opts = _determine_onnx_provider()
        self.assertEqual(provider, "CUDAExecutionProvider")
        self.assertEqual(opts, {"device_id": 0})

    @patch('llamasearch.core.teapot.onnxruntime.get_available_providers')
    def test_auto_select_coreml(self, mock_get_providers):
        mock_get_providers.return_value = ["CPUExecutionProvider", "CoreMLExecutionProvider"]
        provider, opts = _determine_onnx_provider()
        self.assertEqual(provider, "CoreMLExecutionProvider")
        self.assertIsNone(opts)

    @patch('llamasearch.core.teapot.onnxruntime.get_available_providers')
    def test_auto_select_cpu_only(self, mock_get_providers):
        mock_get_providers.return_value = ["CPUExecutionProvider"]
        provider, opts = _determine_onnx_provider()
        self.assertEqual(provider, "CPUExecutionProvider")
        self.assertIsNone(opts)

    @patch('llamasearch.core.teapot.onnxruntime.get_available_providers')
    def test_auto_select_priority_cuda_over_coreml(self, mock_get_providers):
        mock_get_providers.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"]
        provider, opts = _determine_onnx_provider()
        self.assertEqual(provider, "CUDAExecutionProvider")
        self.assertEqual(opts, {"device_id": 0})

class TestDetectAvailableSuffix(unittest.TestCase):
    """Tests for the _detect_available_onnx_suffix helper function."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_teapot_detect_"))
        self.onnx_dir = self.temp_dir / ONNX_SUBFOLDER
        self.onnx_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_onnx_files(self, suffix):
        for basename in REQUIRED_ONNX_BASENAMES:
            (self.onnx_dir / f"{basename}{suffix}.onnx").touch()

    def test_no_onnx_dir(self):
        shutil.rmtree(self.onnx_dir)
        self.assertIsNone(_detect_available_onnx_suffix(self.onnx_dir))

    def test_no_files(self):
        self.assertIsNone(_detect_available_onnx_suffix(self.onnx_dir))

    def test_incomplete_set(self):
        (self.onnx_dir / f"{REQUIRED_ONNX_BASENAMES[0]}_int8.onnx").touch()
        self.assertIsNone(_detect_available_onnx_suffix(self.onnx_dir))

    def test_complete_fp32(self):
        self._create_onnx_files("")
        self.assertEqual(_detect_available_onnx_suffix(self.onnx_dir), "")

    def test_complete_int8(self):
        self._create_onnx_files("_int8")
        self.assertEqual(_detect_available_onnx_suffix(self.onnx_dir), "_int8")

    def test_multiple_complete_priority_fp32(self):
        self._create_onnx_files("")
        self._create_onnx_files("_int8")
        self.assertEqual(_detect_available_onnx_suffix(self.onnx_dir), "")

    def test_multiple_complete_priority_fp16(self):
        self._create_onnx_files("_fp16")
        self._create_onnx_files("_int8")
        self.assertEqual(_detect_available_onnx_suffix(self.onnx_dir), "_fp16")

    def test_multiple_complete_priority_int8_over_q4(self):
        self._create_onnx_files("_q4")
        self._create_onnx_files("_int8")
        self.assertEqual(_detect_available_onnx_suffix(self.onnx_dir), "_int8")

# --- Tests for TeapotONNXLLM Class ---
@patch('llamasearch.core.teapot.data_manager') # Mock data_manager interaction
@patch('llamasearch.core.teapot.setup_logging') # Prevent actual logging setup
class TestTeapotONNXLLM(unittest.TestCase):

    def setUp(self):
        self.mock_model_internal = MagicMock(spec=MockORTModel)
        self.mock_tokenizer_internal = MagicMock(spec=MockTokenizer)
        self.mock_tokenizer_internal.decode = MagicMock(return_value=" Decoded Output")

        mock_input_tensor = MagicMock(spec=MockTorchTensor)
        mock_input_tensor.shape = [1, 10]
        mock_input_tensor.to.return_value = mock_input_tensor
        self.mock_tokenizer_internal.return_value = {"input_ids": mock_input_tensor}

        mock_output_tensor = MagicMock(spec=MockTorchTensor)
        mock_output_tensor.device = MockTorchDevice(type='cpu')
        mock_output_tensor.__len__ = MagicMock(return_value=25)
        self.mock_model_internal.generate.return_value = [mock_output_tensor]


    
    
# --- Tests for load_teapot_onnx_llm Function ---
@patch('llamasearch.core.teapot.data_manager')
@patch('llamasearch.core.teapot._determine_onnx_provider')
@patch('llamasearch.core.teapot._detect_available_onnx_suffix')
@patch('llamasearch.core.teapot.ORTModelForSeq2SeqLM.from_pretrained')
@patch('llamasearch.core.teapot.AutoTokenizer.from_pretrained')
@patch('llamasearch.core.teapot.gc.collect')
@patch('llamasearch.core.teapot.torch.cuda')
@patch('llamasearch.core.teapot.setup_logging')
class TestLoadTeapotONNXLLM(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_teapot_load_"))
        self.models_dir = self.temp_dir / "models"
        self.active_dir = self.models_dir / "active_teapot"
        self.onnx_dir = self.active_dir / ONNX_SUBFOLDER
        self.onnx_dir.mkdir(parents=True)

        for fname in TEAPOT_BASE_FILES:
            if fname != ONNX_SUBFOLDER:
                 (self.active_dir / fname).touch()

        # Keep mock_data_mgr instance specific if needed, or use the class-level patch
        # self.mock_dm = mock_data_mgr # Use the argument from @patch
        # self.mock_dm.get_data_paths.return_value = {"models": str(self.models_dir)}

        # Store mock loaders for potential assertion checks
        self.mock_model_loader_instance = MockORTModel()
        self.mock_tokenizer_loader_instance = MockTokenizer()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_onnx_files(self, suffix):
        for basename in REQUIRED_ONNX_BASENAMES:
            (self.onnx_dir / f"{basename}{suffix}.onnx").touch()

    def test_load_success_auto_detect(self, mock_setup_logging, mock_torch_cuda, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_mgr):
        mock_data_mgr.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = "_int8"
        self._create_onnx_files("_int8")
        mock_mod_loader.return_value = self.mock_model_loader_instance
        mock_tok_loader.return_value = self.mock_tokenizer_loader_instance

        llm = load_teapot_onnx_llm(onnx_quantization="auto")

        self.assertIsInstance(llm, TeapotONNXLLM)
        self.assertEqual(llm.model_info.model_id, f"{TEAPOT_REPO_ID}-onnx-int8")
        mock_detect_suffix.assert_called_once_with(self.onnx_dir)
        expected_enc_fn = f"{ONNX_SUBFOLDER}/encoder_model_int8.onnx"
        expected_dec_fn = f"{ONNX_SUBFOLDER}/decoder_model_int8.onnx"
        expected_dec_past_fn = f"{ONNX_SUBFOLDER}/decoder_with_past_model_int8.onnx"
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            encoder_file_name=expected_enc_fn, decoder_file_name=expected_dec_fn,
            decoder_with_past_file_name=expected_dec_past_fn, export=False,
            provider="CPUExecutionProvider", provider_options=None, use_io_binding=False,
            local_files_only=True
        )
        mock_tok_loader.assert_called_once_with(self.active_dir, use_fast=True, local_files_only=True)

    def test_load_success_user_preference_exists(self, mock_setup_logging, mock_torch_cuda, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_mgr):
        mock_data_mgr.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CUDAExecutionProvider", {"device_id": 0}) # Test CUDA provider
        self._create_onnx_files("_fp16")
        mock_mod_loader.return_value = self.mock_model_loader_instance
        mock_tok_loader.return_value = self.mock_tokenizer_loader_instance

        llm = load_teapot_onnx_llm(onnx_quantization="fp16", preferred_provider="CUDAExecutionProvider")

        self.assertIsInstance(llm, TeapotONNXLLM)
        self.assertEqual(llm.model_info.model_id, f"{TEAPOT_REPO_ID}-onnx-fp16")
        mock_detect_suffix.assert_not_called()
        expected_enc_fn = f"{ONNX_SUBFOLDER}/encoder_model_fp16.onnx"
        expected_dec_fn = f"{ONNX_SUBFOLDER}/decoder_model_fp16.onnx"
        expected_dec_past_fn = f"{ONNX_SUBFOLDER}/decoder_with_past_model_fp16.onnx"
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            encoder_file_name=expected_enc_fn, decoder_file_name=expected_dec_fn,
            decoder_with_past_file_name=expected_dec_past_fn, export=False,
            provider="CUDAExecutionProvider", provider_options={"device_id": 0}, use_io_binding=True, # use_io_binding is True for CUDA
            local_files_only=True
        )

    def test_load_fallback_user_preference_missing(self, mock_setup_logging, mock_torch_cuda, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_mgr):
        mock_data_mgr.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        self._create_onnx_files("_q4")
        mock_detect_suffix.return_value = "_q4"
        mock_mod_loader.return_value = self.mock_model_loader_instance
        mock_tok_loader.return_value = self.mock_tokenizer_loader_instance

        llm = load_teapot_onnx_llm(onnx_quantization="fp16") # Prefer fp16 (missing)

        self.assertIsInstance(llm, TeapotONNXLLM)
        self.assertEqual(llm.model_info.model_id, f"{TEAPOT_REPO_ID}-onnx-q4")
        mock_detect_suffix.assert_called_once_with(self.onnx_dir)
        expected_enc_fn = f"{ONNX_SUBFOLDER}/encoder_model_q4.onnx"
        # ... check call args ...
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            encoder_file_name=expected_enc_fn, decoder_file_name=ANY,
            decoder_with_past_file_name=ANY, export=False,
            provider="CPUExecutionProvider", provider_options=None, use_io_binding=False,
            local_files_only=True
        )


    def test_load_fail_no_complete_set(self, mock_setup_logging, mock_torch_cuda, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_mgr):
        mock_data_mgr.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = None # Simulate detection failure

        with self.assertRaisesRegex(ModelNotFoundError, "No complete ONNX model set found"):
            load_teapot_onnx_llm(onnx_quantization="auto")
        mock_detect_suffix.assert_called_once_with(self.onnx_dir)

    
    def test_load_fail_model_loader_exception(self, mock_setup_logging, mock_torch_cuda, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_mgr):
        mock_data_mgr.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = "" # fp32
        self._create_onnx_files("")
        error_msg = "ONNX Loading Failed"
        mock_mod_loader.side_effect = Exception(error_msg)

        with self.assertRaisesRegex(RuntimeError, f"Failed to load Teapot ONNX model .* {error_msg}"):
            load_teapot_onnx_llm(onnx_quantization="fp32")

        mock_gc.assert_called_once()
        # mock_torch_cuda.empty_cache.assert_called_once() # Check based on provider potentially used

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)