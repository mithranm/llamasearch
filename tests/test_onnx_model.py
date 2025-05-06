# tests/test_onnx_model.py (CPU-Only Version, Refactored for Llama-3.2 & Optimum)

import unittest
import tempfile
import shutil
import logging
import re
from pathlib import Path
from unittest.mock import patch, MagicMock, call, ANY

import torch
from optimum.onnxruntime import ORTModelForCausalLM
# Use PreTrainedTokenizerBase for mocking and type hinting
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding


from llamasearch.core.onnx_model import (
    GenericONNXModelInfo,
    GenericONNXLLM,
    load_onnx_llm,
    _determine_onnx_provider,
    _detect_available_onnx_suffix,
    MODEL_ONNX_BASENAME,
    ONNX_SUBFOLDER,
    ONNX_MODEL_REPO_ID,
    ONNX_MODEL_CONTEXT_LENGTH,
    ONNX_REQUIRED_LOAD_FILES,
)

from llamasearch.exceptions import ModelNotFoundError, SetupError


# --- Test GenericONNXModelInfo Class ---
class TestGenericONNXModelInfo(unittest.TestCase):
    def test_properties_llama3_fp32(self):
        info = GenericONNXModelInfo(ONNX_MODEL_REPO_ID, "", ONNX_MODEL_CONTEXT_LENGTH)
        self.assertEqual(info.model_id, "Llama-3.2-1B-Instruct-onnx-fp32")
        self.assertEqual(info.model_engine, "onnx_causal")
        self.assertEqual(
            info.description,
            f"Generic ONNX Causal LM ({ONNX_MODEL_REPO_ID}, fp32 quantization)",
        )
        self.assertEqual(info.context_length, ONNX_MODEL_CONTEXT_LENGTH)

    def test_properties_llama3_quantized(self):
        info = GenericONNXModelInfo(ONNX_MODEL_REPO_ID, "_int8", ONNX_MODEL_CONTEXT_LENGTH)
        self.assertEqual(info.model_id, "Llama-3.2-1B-Instruct-onnx-int8")
        self.assertEqual(
            info.description,
            f"Generic ONNX Causal LM ({ONNX_MODEL_REPO_ID}, int8 quantization)",
        )


# --- Tests for Helper Functions (CPU-Only) ---
class TestDetermineProviderCPU(unittest.TestCase):
    @patch("llamasearch.core.onnx_model.onnxruntime.get_available_providers")
    def test_always_returns_cpu(self, mock_get):
        mock_get.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        p, o = _determine_onnx_provider()
        self.assertEqual(p, "CPUExecutionProvider")
        self.assertIsNone(o)
        p, o = _determine_onnx_provider("CUDAExecutionProvider") # Request ignored
        self.assertEqual(p, "CPUExecutionProvider")
        self.assertIsNone(o)

    @patch("llamasearch.core.onnx_model.onnxruntime.get_available_providers")
    def test_raises_if_cpu_missing(self, mock_get):
        mock_get.return_value = ["CUDAExecutionProvider"]
        with self.assertRaisesRegex(RuntimeError, "ONNX Runtime CPU provider is missing."):
            _determine_onnx_provider()


class TestDetectAvailableSuffixLlama(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp(prefix="test_llama_"))
        self.od = self.td / ONNX_SUBFOLDER
        self.od.mkdir()
    def tearDown(self):
        shutil.rmtree(self.td)
    def _cf(self, s):
        (self.od / f"{MODEL_ONNX_BASENAME}{s}.onnx").touch()
        if s == "":
            (self.od / f"{MODEL_ONNX_BASENAME}.onnx_data").touch()
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

    @patch('pathlib.Path.glob')
    def test_scan_error(self, mock_glob):
        """Test handling of OS error during directory scan."""
        mock_glob.side_effect = OSError("Permission denied")
        (self.od / "model_fp16.onnx").touch()
        result = _detect_available_onnx_suffix(self.od)
        self.assertIsNone(result)
        mock_glob.assert_called_once_with(f"{MODEL_ONNX_BASENAME}*.onnx")

    def test_unsupported_suffix_found(self):
        """Test behavior when only unsupported suffix files exist."""
        (self.od / "model_weird.onnx").touch()
        result = _detect_available_onnx_suffix(self.od)
        self.assertIsNone(result)


# --- Tests for GenericONNXLLM Class ---
@patch("llamasearch.core.onnx_model.torch.no_grad")
@patch("llamasearch.core.onnx_model.gc.collect")
class TestGenericONNXLLMCPU(unittest.TestCase):
    def setUp(self):
        self.mock_model_internal = MagicMock(spec=ORTModelForCausalLM)
        self.mock_model_internal.config = MagicMock()
        self.mock_model_internal.config.eos_token_id = 128001

        self.mock_tokenizer_internal = MagicMock(spec=PreTrainedTokenizerBase)
        self.mock_tokenizer_internal.eos_token_id = 128001

        self.mock_tokenizer_internal.apply_chat_template.return_value = "formatted user prompt"
        self.mock_input_ids = torch.tensor([[1, 2, 3, 4]])
        self.mock_attention_mask = torch.tensor([[1, 1, 1, 1]])
        self.batch_encoding_return = BatchEncoding({'input_ids': self.mock_input_ids, 'attention_mask': self.mock_attention_mask})
        # <<< FIX: Use configure_mock for __call__ >>>
        self.mock_tokenizer_internal.configure_mock(**{'__call__.return_value': self.batch_encoding_return})

        self.mock_tokenizer_internal.decode.return_value = " generated text"
        self.cpu_device = torch.device("cpu")
        self.batch_encoding_return.to = MagicMock(return_value=self.batch_encoding_return)
        self.mock_output_ids_full = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        self.mock_model_internal.generate.return_value = self.mock_output_ids_full
        self.llm = GenericONNXLLM(
            model=self.mock_model_internal,
            tokenizer=self.mock_tokenizer_internal,
            model_repo_id=ONNX_MODEL_REPO_ID,
            quant_suffix="_test",
            provider="CPUExecutionProvider",
            provider_options=None,
            context_length=ONNX_MODEL_CONTEXT_LENGTH
        )

    def test_init(self, mock_gc, mock_no_grad_cm):
        self.assertIsNotNone(self.llm._model)
        self.assertIsNotNone(self.llm._tokenizer)
        self.assertEqual(self.llm.device, self.cpu_device)
        self.assertEqual(self.llm._provider, "CPUExecutionProvider")

    def test_generate_success(self, mock_gc, mock_no_grad_cm):
        prompt_text = "user prompt"
        response, metadata = self.llm.generate(prompt_text)
        self.assertEqual(response, "generated text")
        self.mock_tokenizer_internal.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
        )
        # <<< FIX: Check call_args tuple explicitly >>>
        self.assertEqual(self.mock_tokenizer_internal.call_args, call("formatted user prompt", return_tensors="pt"))
        self.batch_encoding_return.to.assert_called_once_with(self.cpu_device)
        self.mock_model_internal.generate.assert_called_once()
        _, generate_kwargs = self.mock_model_internal.generate.call_args
        self.assertTrue(torch.equal(generate_kwargs["input_ids"], self.mock_input_ids))
        self.assertTrue(torch.equal(generate_kwargs["attention_mask"], self.mock_attention_mask))
        self.assertEqual(generate_kwargs["max_new_tokens"], 512)
        self.assertEqual(generate_kwargs["temperature"], 0.7)
        self.assertEqual(generate_kwargs["top_p"], 0.9)
        self.assertEqual(generate_kwargs["repetition_penalty"], 1.0)
        self.assertEqual(generate_kwargs["do_sample"], True)
        self.assertEqual(generate_kwargs["pad_token_id"], self.mock_tokenizer_internal.eos_token_id)
        self.mock_tokenizer_internal.decode.assert_called_once()
        args_decode, decode_kwargs = self.mock_tokenizer_internal.decode.call_args
        self.assertTrue(torch.equal(args_decode[0], torch.tensor([5, 6, 7])))
        self.assertEqual(decode_kwargs, {'skip_special_tokens': True})
        self.assertEqual(metadata.get("output_token_count"), 3)
        self.assertEqual(metadata.get("input_token_count"), 4)

    def test_generate_unloaded(self, mock_gc, mock_no_grad_cm):
        self.llm.unload()
        with self.assertRaisesRegex(RuntimeError, "Model or tokenizer not loaded"):
            self.llm.generate("test")
        mock_gc.assert_called_once()

    def test_generate_tokenizer_returns_dict(self, mock_gc, mock_no_grad_cm):
        plain_dict_return = {'input_ids': self.mock_input_ids, 'attention_mask': self.mock_attention_mask}
        # <<< FIX: Use configure_mock >>>
        self.mock_tokenizer_internal.configure_mock(**{'__call__.return_value': plain_dict_return})

        response, metadata = self.llm.generate("test dict return")

        self.assertEqual(response, "generated text")
        self.mock_model_internal.generate.assert_called_once()
        _, generate_kwargs = self.mock_model_internal.generate.call_args
        self.assertEqual(generate_kwargs["input_ids"].device, self.cpu_device)

    def test_generate_tokenizer_returns_invalid(self, mock_gc, mock_no_grad_cm):
        # <<< FIX: Use configure_mock >>>
        self.mock_tokenizer_internal.configure_mock(**{'__call__.return_value': 123})
        response, metadata = self.llm.generate("test invalid return")
        self.assertTrue(response.startswith("Error: Tokenizer failed unexpectedly"))
        self.assertEqual(metadata.get("error"), "Tokenizer output type error")
        self.mock_model_internal.generate.assert_not_called()

    def test_generate_model_returns_invalid(self, mock_gc, mock_no_grad_cm):
        self.mock_model_internal.generate.return_value = None
        response, metadata = self.llm.generate("test invalid model output")
        self.assertTrue(response.startswith("Error: Unexpected output format from LLM"))
        self.assertTrue(metadata.get("error", "").startswith("Unexpected LLM output type"))
        self.mock_tokenizer_internal.decode.assert_not_called()

    def test_generate_no_new_tokens(self, mock_gc, mock_no_grad_cm):
        self.mock_model_internal.generate.return_value = self.mock_input_ids
        self.mock_tokenizer_internal.decode.return_value = ""

        with self.assertLogs(logger='llamasearch.core.onnx_model', level='WARNING') as cm:
             response, metadata = self.llm.generate("test no new tokens")

        self.assertEqual(response, "")
        self.assertEqual(metadata.get("output_token_count"), 0)
        self.assertEqual(metadata.get("input_token_count"), 4)
        self.assertTrue(any("LLM generated no new tokens" in msg for msg in cm.output))
        self.mock_tokenizer_internal.decode.assert_called_once()
        args_decode, _ = self.mock_tokenizer_internal.decode.call_args
        self.assertTrue(torch.equal(args_decode[0], torch.tensor([], dtype=torch.long)))

    def test_generate_model_error(self, mock_gc, mock_no_grad_cm):
        self.mock_model_internal.generate.side_effect = RuntimeError("ONNX Runtime Error")
        response, metadata = self.llm.generate("test model error")
        self.assertTrue(response.startswith("LLM Error: ONNX Runtime Error"))
        self.assertEqual(metadata.get("error"), "ONNX Runtime Error")
        self.mock_tokenizer_internal.decode.assert_not_called()

    def test_generate_decode_error(self, mock_gc, mock_no_grad_cm):
        self.mock_tokenizer_internal.decode.side_effect = ValueError("Decoding failed")
        response, metadata = self.llm.generate("test decode error")
        self.assertTrue(response.startswith("LLM Error: Decoding failed"))
        self.assertEqual(metadata.get("error"), "Decoding failed")
        self.mock_model_internal.generate.assert_called_once()
        self.mock_tokenizer_internal.decode.assert_called_once()

    def test_unload(self, mock_gc, mock_no_grad_cm):
        self.llm.unload()
        self.assertIsNone(self.llm._model)
        self.assertIsNone(self.llm._tokenizer)
        mock_gc.assert_called_once()


# --- Tests for load_onnx_llm Function (Modified) ---
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
        for fname in ONNX_REQUIRED_LOAD_FILES:
             (self.active_dir / fname).touch()
        self.mock_model_instance = MagicMock(spec=ORTModelForCausalLM)
        self.mock_tokenizer_instance = MagicMock(spec=PreTrainedTokenizerBase)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_onnx_file(self, suffix):
        (self.onnx_dir / f"{MODEL_ONNX_BASENAME}{suffix}.onnx").touch()
        if suffix == "":
            (self.onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data").touch()

    def test_load_success_detects_correct_suffix(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test successful load when files for a specific suffix exist."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = "_int8"
        self._create_onnx_file("_int8")
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        llm = load_onnx_llm()
        self.assertIsNotNone(llm)
        self.assertIsInstance(llm, GenericONNXLLM)
        if llm is not None:
            self.assertIn("-int8", llm.model_info.model_id)
        else:
            self.fail("LLM instance was None despite assertion")

        mock_determine_provider.assert_called_once_with(None, None)
        mock_detect_suffix.assert_called_once_with(self.onnx_dir)
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            file_name=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}_int8.onnx",
            export=False, provider="CPUExecutionProvider", provider_options=None, use_io_binding=False, local_files_only=True, trust_remote_code=False,
        )
        mock_tok_loader.assert_called_once_with(
            self.active_dir, use_fast=True, local_files_only=True, trust_remote_code=False,
        )

    def test_load_success_user_preference(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test successful load when user specifies a valid quantization."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        self._create_onnx_file("_fp16")
        self._create_onnx_file("_int8")
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        llm = load_onnx_llm(onnx_quantization="fp16")

        self.assertIsNotNone(llm)
        self.assertIsInstance(llm, GenericONNXLLM)
        if llm is not None:
            self.assertIn("-fp16", llm.model_info.model_id)
        else:
            self.fail("LLM instance was None despite assertion")

        mock_detect_suffix.assert_not_called()
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            file_name=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}_fp16.onnx",
            export=False, provider="CPUExecutionProvider", provider_options=None, use_io_binding=False, local_files_only=True, trust_remote_code=False,
        )
        mock_tok_loader.assert_called_once_with(
            self.active_dir, use_fast=True, local_files_only=True, trust_remote_code=False,
        )

    def test_load_user_preference_missing_fallback(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test fallback to detection when preferred quant file is missing."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        self._create_onnx_file("_int8")
        mock_detect_suffix.return_value = "_int8"
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        with self.assertLogs(logger='llamasearch.core.onnx_model', level='WARNING') as cm:
            llm = load_onnx_llm(onnx_quantization="fp16")

        self.assertTrue(any("Preferred ONNX file 'model_fp16.onnx' missing" in msg for msg in cm.output))
        self.assertIsNotNone(llm)
        if llm is not None:
            self.assertIn("-int8", llm.model_info.model_id)
        else:
            self.fail("LLM instance was None despite assertion")

        mock_detect_suffix.assert_called_once_with(self.onnx_dir)
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            file_name=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}_int8.onnx",
            export=False, provider="CPUExecutionProvider", provider_options=None, use_io_binding=False, local_files_only=True, trust_remote_code=False,
        )

    def test_load_user_preference_invalid_fallback(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test fallback to detection when preferred quant is invalid."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        self._create_onnx_file("")
        mock_detect_suffix.return_value = ""
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        with self.assertLogs(logger='llamasearch.core.onnx_model', level='WARNING') as cm:
            llm = load_onnx_llm(onnx_quantization="bad_pref")

        self.assertTrue(any("Invalid preference 'bad_pref'" in msg for msg in cm.output))
        self.assertIsNotNone(llm)
        if llm is not None:
            self.assertIn("-fp32", llm.model_info.model_id)
        else:
             self.fail("LLM instance was None despite assertion")

        mock_detect_suffix.assert_called_once_with(self.onnx_dir)
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            file_name=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx",
            export=False, provider="CPUExecutionProvider", provider_options=None, use_io_binding=False, local_files_only=True, trust_remote_code=False,
        )

    def test_load_fail_no_active_dir(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        shutil.rmtree(self.active_dir)
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        expected_regex = re.escape(f"Active dir '{self.active_dir}' DNE.")
        with self.assertRaisesRegex(ModelNotFoundError, expected_regex):
            load_onnx_llm()

    def test_load_fail_no_onnx_files_detected(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test failure when detector finds no valid ONNX files."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = None
        expected_regex = re.escape(f"No complete ONNX model found in {self.onnx_dir}.")
        with self.assertRaisesRegex(ModelNotFoundError, expected_regex):
            load_onnx_llm()
        mock_detect_suffix.assert_called_once_with(self.onnx_dir)

    def test_load_fail_fp32_missing_data_file(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test failure when fp32 model file exists but data file is missing."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = ""
        (self.onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx").touch()

        expected_regex = re.escape(f"Required model/tokenizer files missing for suffix '': ['{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data']. Run setup.")
        with self.assertRaisesRegex(ModelNotFoundError, expected_regex):
            load_onnx_llm()
        mock_detect_suffix.assert_called_once_with(self.onnx_dir)

    def test_load_fail_missing_required_root_file(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test failure when a root config/tokenizer file is missing."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = ""
        self._create_onnx_file("")
        missing_file_name = ""
        if ONNX_REQUIRED_LOAD_FILES:
            missing_file_name = ONNX_REQUIRED_LOAD_FILES[0]
            (self.active_dir / missing_file_name).unlink()

        expected_regex = re.escape(f"Required model/tokenizer files missing for suffix '': ['{missing_file_name}']. Run setup.")
        with self.assertRaisesRegex(ModelNotFoundError, expected_regex):
            load_onnx_llm()
        mock_detect_suffix.assert_called_once_with(self.onnx_dir)

    def test_load_fail_models_path_missing(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test failure if data_manager doesn't provide 'models' path."""
        mock_data_manager.get_data_paths.return_value = {"index": "/fake/index"}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        with self.assertRaisesRegex(RuntimeError, r"Failed load ONNX Causal LM \(SetupError\): Models path missing\."):
            load_onnx_llm()
        mock_determine_provider.assert_called_once_with(None, None)

    def test_load_fail_model_loader_returns_none(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test failure if ORTModelForCausalLM.from_pretrained returns None."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = ""
        self._create_onnx_file("")
        mock_mod_loader.return_value = None
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        with self.assertRaisesRegex(RuntimeError, r"Failed load ONNX Causal LM \(TypeError\): Expected ORTModelForCausalLM, got <class 'NoneType'>"):
            load_onnx_llm()
        mock_mod_loader.assert_called_once()
        mock_tok_loader.assert_not_called()
        mock_gc.assert_called_once()

    def test_load_fail_tokenizer_loader_returns_none(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test failure if AutoTokenizer.from_pretrained returns None."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = ""
        self._create_onnx_file("")
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = None

        with self.assertRaisesRegex(RuntimeError, r"Failed load ONNX Causal LM \(TypeError\): Expected PreTrainedTokenizerBase, got <class 'NoneType'>"):
             load_onnx_llm()
        mock_mod_loader.assert_called_once()
        mock_tok_loader.assert_called_once()
        mock_gc.assert_called_once()

    def test_load_fail_generic_loader_error(
        self, mock_gc, mock_tok_loader, mock_mod_loader, mock_detect_suffix, mock_determine_provider, mock_data_manager,
    ):
        """Test cleanup on generic exception during loading."""
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_detect_suffix.return_value = ""
        self._create_onnx_file("")
        mock_mod_loader.side_effect = RuntimeError("Disk read error")
        mock_llm_instance = MagicMock(spec=GenericONNXLLM)

        with patch("llamasearch.core.onnx_model.GenericONNXLLM", return_value=mock_llm_instance) as mock_constructor:
            with self.assertRaisesRegex(RuntimeError, "Failed load ONNX Causal LM.*Disk read error"):
                load_onnx_llm()

        mock_mod_loader.assert_called_once()
        mock_tok_loader.assert_not_called()
        mock_constructor.assert_not_called()
        mock_gc.assert_called_once()


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
    logging.disable(logging.NOTSET)