# FIX: tests/test_onnx_model.py
# Added assert llm is not None before checking llm._provider

# tests/test_onnx_model.py (CPU-Only Version, Fixes Applied)

import unittest
import tempfile
import shutil

import logging
from pathlib import Path
from unittest.mock import patch, MagicMock  

import torch
from optimum.onnxruntime import ORTModelForCausalLM

from transformers.tokenization_utils import PreTrainedTokenizer

from llamasearch.core.onnx_model import (
    GenericONNXModelInfo,
    GenericONNXLLM,
    load_onnx_llm,
    _determine_onnx_provider,
    _detect_available_onnx_suffix,
    _select_onnx_quantization,
    MODEL_ONNX_BASENAME,
    ONNX_SUBFOLDER,
    QWEN3_BASE_FILES,
    QWEN3_REPO_ID,
    QWEN3_CONTEXT_LENGTH,
)

from llamasearch.exceptions import ModelNotFoundError  
from llamasearch.hardware import HardwareInfo, CPUInfo, MemoryInfo


# --- Test GenericONNXModelInfo Class ---
# (No changes needed)
class TestGenericONNXModelInfo(unittest.TestCase):
    def test_properties_qwen3_fp32(self):
        info = GenericONNXModelInfo(QWEN3_REPO_ID, "", QWEN3_CONTEXT_LENGTH)
        self.assertEqual(info.model_id, "Qwen3-0.6B-ONNX-onnx-fp32")
        self.assertEqual(info.model_engine, "onnx_causal")
        self.assertEqual(
            info.description,
            f"Generic ONNX Causal LM ({QWEN3_REPO_ID}, fp32 quantization)",
        )
        self.assertEqual(info.context_length, QWEN3_CONTEXT_LENGTH)

    def test_properties_qwen3_quantized(self):
        info = GenericONNXModelInfo(QWEN3_REPO_ID, "_int8", QWEN3_CONTEXT_LENGTH)
        self.assertEqual(info.model_id, "Qwen3-0.6B-ONNX-onnx-int8")
        self.assertEqual(
            info.description,
            f"Generic ONNX Causal LM ({QWEN3_REPO_ID}, int8 quantization)",
        )


# --- Tests for Helper Functions (CPU-Only) ---
class TestDetermineProviderCPU(unittest.TestCase):
    @patch("llamasearch.core.onnx_model.onnxruntime.get_available_providers")
    def test_always_returns_cpu(self, mock_get):
        mock_get.return_value = [
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
        ]
        p, o = _determine_onnx_provider()
        self.assertEqual(p, "CPUExecutionProvider")
        self.assertIsNone(o)
        p, o = _determine_onnx_provider("CPUExecutionProvider")
        self.assertEqual(p, "CPUExecutionProvider")
        self.assertIsNone(o)
        p, o = _determine_onnx_provider("CUDAExecutionProvider", {"device_id": 0})
        self.assertEqual(p, "CPUExecutionProvider")
        self.assertIsNone(o)
        p, o = _determine_onnx_provider("CoreMLExecutionProvider")
        self.assertEqual(p, "CPUExecutionProvider")
        self.assertIsNone(o)

    @patch("llamasearch.core.onnx_model.onnxruntime.get_available_providers")
    def test_raises_if_cpu_missing(self, mock_get):
        mock_get.return_value = ["CUDAExecutionProvider"]  # CPU missing
        with self.assertRaisesRegex(RuntimeError, "CPU provider is missing"):
            _determine_onnx_provider()


@patch("llamasearch.core.onnx_model.detect_hardware_info")
class TestSelectQuantizationCPU(unittest.TestCase):
    def _mock_hw(self, ram_gb, avx2):
        # --- FIX: Added frequency_mhz=None ---
        return HardwareInfo(
            cpu=CPUInfo(
                logical_cores=4,
                physical_cores=2,
                architecture="x64",
                model_name="test",
                supports_avx2=avx2,
                frequency_mhz=None,
            ),
            memory=MemoryInfo(
                total_gb=ram_gb,
                available_gb=ram_gb * 0.8,
                used_gb=ram_gb * 0.2,
                percent_used=20.0,
            ),
        )
        # --- END FIX ---

    def test_cpu_high_ram(self, m):
        m.return_value = self._mock_hw(16.0, True)
        self.assertEqual(_select_onnx_quantization("auto"), "")

    def test_cpu_medium_ram_fp16(self, m):
        m.return_value = self._mock_hw(5.0, True)
        self.assertEqual(_select_onnx_quantization("auto"), "_fp16")

    def test_cpu_lower_ram_int8_avx2(self, m):
        m.return_value = self._mock_hw(4.0, True)
        self.assertEqual(_select_onnx_quantization("auto"), "_int8")

    def test_cpu_lower_ram_int8_no_avx2(self, m):
        m.return_value = self._mock_hw(4.0, False)
        self.assertEqual(_select_onnx_quantization("auto"), "_int8")

    def test_cpu_low_ram_q4(self, m):
        m.return_value = self._mock_hw(3.6, True)
        self.assertEqual(_select_onnx_quantization("auto"), "_q4")

    def test_cpu_very_low_ram_q4f16(self, m):
        m.return_value = self._mock_hw(3.1, True)
        self.assertEqual(_select_onnx_quantization("auto"), "_q4f16")

    def test_cpu_ultra_low_ram_bnb4(self, m):
        m.return_value = self._mock_hw(2.5, True)
        self.assertEqual(_select_onnx_quantization("auto"), "_bnb4")

    def test_user_preference_valid(self, m):
        m.return_value = self._mock_hw(16.0, True)
        self.assertEqual(_select_onnx_quantization("fp16"), "_fp16")

    def test_user_preference_invalid(self, m):
        m.return_value = self._mock_hw(16.0, True)
        self.assertEqual(_select_onnx_quantization("bad"), "")


# TestDetectAvailableSuffixQwen remains the same
class TestDetectAvailableSuffixQwen(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp(prefix="test_qwen_"))
        self.od = self.td / ONNX_SUBFOLDER
        self.od.mkdir()

    def tearDown(self):
        shutil.rmtree(self.td)

    def _cf(self, s):
        (self.od / f"{MODEL_ONNX_BASENAME}{s}.onnx").touch()
        ((self.od / f"{MODEL_ONNX_BASENAME}.onnx_data").touch() if s == "" else None)

    def test_no_dir(self):
        shutil.rmtree(self.od)
        self.assertIsNone(_detect_available_onnx_suffix(self.od))

    def test_empty_dir(self):
        self.assertIsNone(_detect_available_onnx_suffix(self.od))

    def test_fp32_ok(self):
        self._cf("")
        self.assertEqual(_detect_available_onnx_suffix(self.od), "")

    def test_fp32_nodata(self):
        (self.od / f"{MODEL_ONNX_BASENAME}.onnx").touch()
        self.assertIsNone(_detect_available_onnx_suffix(self.od))

    def test_int8_ok(self):
        self._cf("_int8")
        self.assertEqual(_detect_available_onnx_suffix(self.od), "_int8")

    def test_multi_pref_fp16(self):
        self._cf("_fp16")
        self._cf("_int8")
        self.assertEqual(_detect_available_onnx_suffix(self.od), "_fp16")

    def test_multi_pref_int8_q4(self):
        self._cf("_q4")
        self._cf("_int8")
        self.assertEqual(_detect_available_onnx_suffix(self.od), "_int8")

    def test_multi_pref_quant_q4(self):
        self._cf("_q4")
        self._cf("_quantized")
        self.assertEqual(_detect_available_onnx_suffix(self.od), "_quantized")

    def test_lowest_bnb4(self):
        self._cf("_bnb4")
        self.assertEqual(_detect_available_onnx_suffix(self.od), "_bnb4")


# --- Tests for GenericONNXLLM Class (CPU-Only) ---
# Mock the actual classes used internally
@patch("llamasearch.core.onnx_model.torch.no_grad")
@patch("llamasearch.core.onnx_model.gc.collect")
class TestGenericONNXLLMCPU(unittest.TestCase):
    def setUp(self):
        self.mock_model_internal = MagicMock(spec=ORTModelForCausalLM)
        self.mock_tokenizer_internal = MagicMock(spec=PreTrainedTokenizer)
        self.mock_tokenizer_internal.eos_token_id = 12345
        self.mock_tokenizer_internal.decode = MagicMock(return_value=" generated text")
        self.cpu_device = torch.device("cpu")
        self.mock_model_internal.device = self.cpu_device
        self.mock_input_ids = torch.tensor([[101, 2054, 2003, 102]])
        self.mock_tokenizer_internal.return_value = MagicMock(
            to=MagicMock(return_value={"input_ids": self.mock_input_ids})
        )
        self.mock_output_ids = torch.tensor([[101, 2054, 2003, 102, 500, 600, 700]])
        self.mock_model_internal.generate.return_value = MagicMock(
            to=MagicMock(return_value=[self.mock_output_ids])
        )
        self.llm = GenericONNXLLM(
            model=self.mock_model_internal,
            tokenizer=self.mock_tokenizer_internal,
            model_repo_id=QWEN3_REPO_ID,
            quant_suffix="_test",
            provider="CPUExecutionProvider",
            provider_options=None,
        )

    def test_init(self, mock_gc, mock_no_grad):
        self.assertTrue(self.llm._is_loaded)
        self.assertEqual(self.llm.device, self.cpu_device)
        self.assertEqual(self.llm._provider, "CPUExecutionProvider")

    def test_generate_success(self, mock_gc, mock_no_grad):
        response, metadata = self.llm.generate("prompt")
        self.assertEqual(response, "generated text")
        self.mock_tokenizer_internal.return_value.to.assert_called_with(self.cpu_device)
        self.mock_model_internal.generate.return_value[0].to.assert_called_with("cpu")

    def test_unload(self, mock_gc, mock_no_grad):
        self.llm.unload()
        self.assertFalse(self.llm._is_loaded)
        self.assertIsNone(self.llm._model)
        self.assertIsNone(self.llm._tokenizer)
        mock_gc.assert_called_once()


# --- Tests for load_onnx_llm Function (CPU-Only) ---
@patch("llamasearch.core.onnx_model.data_manager")
@patch("llamasearch.core.onnx_model._determine_onnx_provider")
@patch("llamasearch.core.onnx_model._detect_available_onnx_suffix")
@patch("llamasearch.core.onnx_model.ORTModelForCausalLM.from_pretrained")
@patch("llamasearch.core.onnx_model.AutoTokenizer.from_pretrained")
@patch("llamasearch.core.onnx_model.gc.collect")
class TestLoadONNXLLMCPU(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_load_cpu_"))
        self.models_dir = self.temp_dir / "models"
        self.active_dir = self.models_dir / "active_model"
        self.onnx_dir = self.active_dir / ONNX_SUBFOLDER
        self.onnx_dir.mkdir(parents=True)
        for fname in QWEN3_BASE_FILES:
            (self.active_dir / fname).touch()
        self.mock_model_instance = MagicMock(spec=ORTModelForCausalLM)
        self.mock_tokenizer_instance = MagicMock(spec=PreTrainedTokenizer)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_onnx_file(self, suffix):
        (self.onnx_dir / f"{MODEL_ONNX_BASENAME}{suffix}.onnx").touch()
        if suffix == "":
            (self.onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data").touch()

    def test_load_success_forces_cpu(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_detect_suffix,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {
            "models": str(self.models_dir)
        }
        mock_determine_provider.return_value = (
            "CPUExecutionProvider",
            None,
        )  # Simulate force CPU
        mock_detect_suffix.return_value = "_int8"
        self._create_onnx_file("_int8")
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        llm = load_onnx_llm(
            onnx_quantization="auto", preferred_provider="CUDAExecutionProvider"
        )  # Request CUDA

        self.assertIsInstance(llm, GenericONNXLLM)
        mock_determine_provider.assert_called_once_with("CPUExecutionProvider", None)
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            file_name=f"{ONNX_SUBFOLDER}/model_int8.onnx",
            export=False,
            provider="CPUExecutionProvider",
            provider_options=None,
            use_io_binding=False,
            local_files_only=True,
            trust_remote_code=False,
        )

    def test_load_fail_no_active_dir(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_detect_suffix,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {
            "models": str(self.models_dir)
        }
        shutil.rmtree(self.active_dir)
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        with self.assertRaisesRegex(ModelNotFoundError, "Active dir .* DNE."):
            load_onnx_llm()

    def test_load_fail_no_onnx_files(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_detect_suffix,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {
            "models": str(self.models_dir)
        }
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = None
        with self.assertRaisesRegex(ModelNotFoundError, "No complete ONNX model found"):
            load_onnx_llm()


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
    logging.disable(logging.NOTSET)
