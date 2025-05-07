# tests/test_onnx_model.py

import logging  # Import logging
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
# Removed ORTModelForCausalLM import as we only mock it
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)

# Import the specific functions/classes being tested or needed for setup
from llamasearch.core.onnx_model import (MODEL_ONNX_BASENAME,
                                         ONNX_MODEL_CONTEXT_LENGTH,
                                         ONNX_MODEL_REPO_ID,
                                         ONNX_REQUIRED_LOAD_FILES,
                                         ONNX_SUBFOLDER, GenericONNXLLM,
                                         GenericONNXModelInfo,
                                         _check_onnx_files_exist,
                                         _determine_onnx_provider,
                                         load_onnx_llm)
from llamasearch.exceptions import ModelNotFoundError

# Patch setup_logging for this test module
MOCK_ONNX_SETUP_LOGGING_TARGET = "llamasearch.utils.setup_logging"
mock_onnx_logger_global_instance = MagicMock(spec=logging.Logger)
onnx_global_logger_patcher = patch(
    MOCK_ONNX_SETUP_LOGGING_TARGET, return_value=mock_onnx_logger_global_instance
)


def setUpModule():
    onnx_global_logger_patcher.start()


def tearDownModule():
    onnx_global_logger_patcher.stop()


# --- Test GenericONNXModelInfo Class ---
class TestGenericONNXModelInfo(unittest.TestCase):
    def test_properties_llama3_fp32(self):
        info = GenericONNXModelInfo(ONNX_MODEL_REPO_ID, ONNX_MODEL_CONTEXT_LENGTH)
        self.assertEqual(info.model_id, "Llama-3.2-1B-Instruct-onnx-fp32")
        self.assertEqual(info.model_engine, "onnx_causal")
        self.assertEqual(
            info.description,
            f"Generic ONNX Causal LM ({ONNX_MODEL_REPO_ID}, fp32)",
        )
        self.assertEqual(info.context_length, ONNX_MODEL_CONTEXT_LENGTH)


# --- Tests for Helper Functions (CPU-Only) ---
class TestDetermineProviderCPU(unittest.TestCase):
    @patch("llamasearch.core.onnx_model.onnxruntime.get_available_providers")
    def test_always_returns_cpu(self, mock_get):
        mock_get.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        p, o = _determine_onnx_provider()
        self.assertEqual(p, "CPUExecutionProvider")
        self.assertIsNone(o)

    @patch("llamasearch.core.onnx_model.onnxruntime.get_available_providers")
    def test_raises_if_cpu_missing(self, mock_get):
        mock_get.return_value = ["CUDAExecutionProvider"]
        with self.assertRaisesRegex(
            RuntimeError, "ONNX Runtime CPU provider is missing."
        ):
            _determine_onnx_provider()


class TestCheckONNXFilesExist(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp(prefix="test_check_"))
        self.od = self.td / ONNX_SUBFOLDER
        self.od.mkdir()

    def tearDown(self):
        shutil.rmtree(self.td)

    def _cf(self, name):
        (self.od / name).touch()

    def test_no_dir(self):
        shutil.rmtree(self.od)
        self.assertFalse(_check_onnx_files_exist(self.od))

    def test_empty_dir(self):
        self.assertFalse(_check_onnx_files_exist(self.od))

    def test_both_exist(self):
        self._cf(f"{MODEL_ONNX_BASENAME}.onnx")
        self._cf(f"{MODEL_ONNX_BASENAME}.onnx_data")
        self.assertTrue(_check_onnx_files_exist(self.od))

    def test_only_model_exists(self):
        self._cf(f"{MODEL_ONNX_BASENAME}.onnx")
        self.assertFalse(_check_onnx_files_exist(self.od))

    def test_only_data_exists(self):
        self._cf(f"{MODEL_ONNX_BASENAME}.onnx_data")
        self.assertFalse(_check_onnx_files_exist(self.od))

    def test_wrong_names(self):
        self._cf("other_model.onnx")
        self._cf("other_model.onnx_data")
        self.assertFalse(_check_onnx_files_exist(self.od))


# --- Tests for GenericONNXLLM Class ---
@patch("llamasearch.core.onnx_model.torch.no_grad")
@patch("llamasearch.core.onnx_model.gc.collect")
class TestGenericONNXLLMCPU(unittest.TestCase):
    def setUp(self):
        # Reset the global mock logger instance before each test in this class
        mock_onnx_logger_global_instance.reset_mock()

        # Define mock BatchEncoding instance *first*
        self.mock_input_ids = torch.tensor([[1, 2, 3, 4]])
        self.mock_attention_mask = torch.tensor([[1, 1, 1, 1]])
        self.mock_batch_encoding_instance = BatchEncoding(
            {
                "input_ids": self.mock_input_ids,
                "attention_mask": self.mock_attention_mask,
            }
        )
        # Mock the .to() method on the instance
        self.mock_batch_encoding_instance.to = MagicMock(
            return_value=self.mock_batch_encoding_instance
        )

        # Now define mocks that might use the instance above
        self.mock_model_internal = MagicMock()  # Removed spec
        self.mock_model_internal.generate = MagicMock()
        self.mock_model_internal.config = MagicMock()
        self.mock_model_internal.config.eos_token_id = 128001

        self.mock_tokenizer_internal = MagicMock(
            spec=PreTrainedTokenizerBase
        )  # Add spec
        # Make the tokenizer mock itself callable and return the BatchEncoding instance
        self.mock_tokenizer_internal.return_value = self.mock_batch_encoding_instance
        self.mock_tokenizer_internal.eos_token_id = 128001
        self.mock_tokenizer_internal.apply_chat_template = MagicMock(
            return_value="formatted user prompt"
        )
        self.mock_tokenizer_internal.decode = MagicMock(return_value=" generated text")

        self.cpu_device = torch.device("cpu")

        # Set model generate return value
        self.mock_output_ids_full = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        self.mock_model_internal.generate.return_value = self.mock_output_ids_full

        # Initialize the class under test
        self.llm = GenericONNXLLM(
            model=self.mock_model_internal,
            tokenizer=self.mock_tokenizer_internal,
            model_repo_id=ONNX_MODEL_REPO_ID,
            provider="CPUExecutionProvider",
            provider_options=None,
            context_length=ONNX_MODEL_CONTEXT_LENGTH,
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
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Assert call to tokenizer mock
        self.mock_tokenizer_internal.assert_called_once()
        self.assertEqual(
            self.mock_tokenizer_internal.call_args[0], ("formatted user prompt",)
        )
        self.assertEqual(
            self.mock_tokenizer_internal.call_args[1], {"return_tensors": "pt"}
        )

        self.mock_batch_encoding_instance.to.assert_called_once_with(self.cpu_device)  # type: ignore

        self.mock_model_internal.generate.assert_called_once()
        generate_kwargs = self.mock_model_internal.generate.call_args.kwargs
        self.assertTrue(torch.equal(generate_kwargs["input_ids"], self.mock_input_ids))
        self.assertTrue(
            torch.equal(generate_kwargs["attention_mask"], self.mock_attention_mask)
        )

        self.assertEqual(generate_kwargs["max_new_tokens"], 512)
        self.assertEqual(generate_kwargs["temperature"], 0.7)
        self.assertEqual(generate_kwargs["top_p"], 0.9)
        self.assertEqual(generate_kwargs["repetition_penalty"], 1.0)
        self.assertEqual(generate_kwargs["do_sample"], True)
        self.assertEqual(generate_kwargs["pad_token_id"], 128001)

        self.mock_tokenizer_internal.decode.assert_called_once()
        decode_args = self.mock_tokenizer_internal.decode.call_args.args
        decode_kwargs = self.mock_tokenizer_internal.decode.call_args.kwargs

        expected_decoded_ids = torch.tensor(
            [5, 6, 7], device=self.cpu_device, dtype=torch.long
        )
        self.assertTrue(torch.equal(decode_args[0], expected_decoded_ids))
        self.assertEqual(decode_kwargs, {"skip_special_tokens": True})

        self.assertEqual(metadata.get("output_token_count"), 3)
        self.assertEqual(metadata.get("input_token_count"), 4)
        self.assertNotIn("error", metadata)

    def test_generate_unloaded(self, mock_gc, mock_no_grad_cm):
        self.llm.unload()
        with self.assertRaisesRegex(RuntimeError, "Model or tokenizer not loaded"):
            self.llm.generate("test")
        mock_gc.assert_called_once()

    def test_generate_tokenizer_returns_invalid(self, mock_gc, mock_no_grad_cm):
        self.mock_tokenizer_internal.return_value = 123  # Simulate invalid return

        response, metadata = self.llm.generate("test invalid return")
        self.assertTrue(response.startswith("Error: Tokenizer failed unexpectedly"))
        self.assertIn(
            "Tokenizer output type/structure error", metadata.get("error", "")
        )
        self.mock_model_internal.generate.assert_not_called()

    def test_generate_model_returns_invalid(self, mock_gc, mock_no_grad_cm):
        self.mock_model_internal.generate.return_value = None
        response, metadata = self.llm.generate("test invalid model output")
        self.assertTrue(response.startswith("Error: Unexpected output format from LLM"))
        self.assertIn(
            "Unexpected LLM output type <class 'NoneType'>", metadata.get("error", "")
        )
        self.mock_tokenizer_internal.decode.assert_not_called()

    def test_generate_no_new_tokens(self, mock_gc, mock_no_grad_cm):
        self.mock_model_internal.generate.return_value = self.mock_input_ids
        self.mock_tokenizer_internal.decode.return_value = ""

        with self.assertLogs(
            logger="llamasearch.core.onnx_model", level="WARNING"
        ) as cm:
            response, metadata = self.llm.generate("test no new tokens")

        self.assertIn(
            "LLM generated no new tokens or output tensor shape invalid.", cm.output[0]
        )
        self.assertEqual(response, "")
        self.assertEqual(metadata.get("output_token_count"), 0)
        self.assertEqual(metadata.get("input_token_count"), 4)
        self.mock_tokenizer_internal.decode.assert_called_once()
        args_decode, _ = self.mock_tokenizer_internal.decode.call_args
        expected_empty_tensor = torch.tensor(
            [], dtype=torch.long, device=self.cpu_device
        )
        self.assertTrue(torch.equal(args_decode[0], expected_empty_tensor))

    def test_generate_model_error(self, mock_gc, mock_no_grad_cm):
        self.mock_model_internal.generate.side_effect = RuntimeError(
            "ONNX Runtime Error"
        )
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


# --- Tests for load_onnx_llm Function ---
@patch("llamasearch.core.onnx_model.data_manager")
@patch("llamasearch.core.onnx_model._determine_onnx_provider")
@patch("llamasearch.core.onnx_model._check_onnx_files_exist")
@patch("llamasearch.core.onnx_model.ORTModelForCausalLM.from_pretrained")
@patch("llamasearch.core.onnx_model.AutoTokenizer.from_pretrained")
@patch("llamasearch.core.onnx_model.gc.collect")
class TestLoadONNXLLMCPU(unittest.TestCase):
    def setUp(self):
        # Reset global mock logger
        mock_onnx_logger_global_instance.reset_mock()
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_load_cpu_")
        self.temp_dir = Path(self.temp_dir_obj.name)
        self.models_dir = self.temp_dir / "models"
        self.active_dir = self.models_dir / "active_model"
        self.onnx_dir = self.active_dir / ONNX_SUBFOLDER
        self.onnx_dir.mkdir(parents=True)
        for fname in ONNX_REQUIRED_LOAD_FILES:
            (self.active_dir / fname).touch()
        # Use ORTModelForCausalLM for spec if available, otherwise generic MagicMock
        try:
            from optimum.onnxruntime import ORTModelForCausalLM

            self.mock_model_instance = MagicMock(spec=ORTModelForCausalLM)
        except ImportError:
            self.mock_model_instance = MagicMock()  # Fallback
        self.mock_tokenizer_instance = MagicMock(spec=PreTrainedTokenizerBase)

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_load_success(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_check_files.return_value = True
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        llm = load_onnx_llm()
        self.assertIsNotNone(llm)
        self.assertIsInstance(llm, GenericONNXLLM)
        if llm:
            self.assertEqual("-fp32", llm.model_info.model_id.split("onnx")[-1])
        else:
            self.fail("LLM instance is None")

        mock_determine_provider.assert_called_once()
        mock_check_files.assert_called_once_with(self.onnx_dir)
        mock_mod_loader.assert_called_once_with(
            self.active_dir,
            file_name=f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx",
            export=False,
            provider="CPUExecutionProvider",
            provider_options=None,
            use_io_binding=False,
            local_files_only=True,
            trust_remote_code=False,
        )
        mock_tok_loader.assert_called_once_with(
            self.active_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )

    def test_load_fail_no_active_dir(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        if self.active_dir.exists():
            shutil.rmtree(self.active_dir)
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        expected_regex = re.escape(f"Active dir '{self.active_dir}' DNE.")
        with self.assertRaisesRegex(ModelNotFoundError, expected_regex):
            load_onnx_llm()
        mock_check_files.assert_not_called()

    def test_load_fail_onnx_files_missing(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_check_files.return_value = False
        expected_regex = re.escape(
            f"Required ONNX files (model.onnx and model.onnx_data) not found in {self.onnx_dir}. Run setup."
        )
        with self.assertRaisesRegex(ModelNotFoundError, expected_regex):
            load_onnx_llm()
        mock_check_files.assert_called_once_with(self.onnx_dir)
        mock_mod_loader.assert_not_called()
        mock_tok_loader.assert_not_called()

    def test_load_fail_missing_required_root_file(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_check_files.return_value = True
        missing_file_name = ""
        if ONNX_REQUIRED_LOAD_FILES:
            missing_file_name = ONNX_REQUIRED_LOAD_FILES[0]
            file_to_remove = self.active_dir / missing_file_name
            if file_to_remove.exists():
                file_to_remove.unlink()
        else:
            self.fail("ONNX_REQUIRED_LOAD_FILES list is empty")

        expected_regex = re.escape(
            f"Required root files missing: ['{missing_file_name}']. Run setup."
        )
        with self.assertRaisesRegex(ModelNotFoundError, expected_regex):
            load_onnx_llm()
        mock_check_files.assert_called_once_with(self.onnx_dir)
        mock_mod_loader.assert_not_called()
        mock_tok_loader.assert_not_called()

    def test_load_fail_models_path_missing(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"index": "/fake/index"}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed load ONNX Causal LM \(SetupError\): Models path missing.",
        ):
            load_onnx_llm()
        mock_determine_provider.assert_called_once()
        mock_check_files.assert_not_called()

    def test_load_fail_model_loader_returns_none(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_check_files.return_value = True
        mock_mod_loader.return_value = None
        mock_tok_loader.return_value = self.mock_tokenizer_instance

        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed load ONNX Causal LM \(TypeError\): Expected ORTModelForCausalLM, got <class 'NoneType'>",
        ):
            load_onnx_llm()
        mock_mod_loader.assert_called_once()
        # Tokenizer is not loaded if model loading fails first
        mock_tok_loader.assert_not_called()
        mock_gc.assert_called_once()

    def test_load_fail_tokenizer_loader_returns_none(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_check_files.return_value = True
        mock_mod_loader.return_value = self.mock_model_instance
        mock_tok_loader.return_value = None

        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed load ONNX Causal LM \(TypeError\): Expected PreTrainedTokenizerBase, got <class 'NoneType'>",
        ):
            load_onnx_llm()
        mock_mod_loader.assert_called_once()
        mock_tok_loader.assert_called_once()
        mock_gc.assert_called_once()

    def test_load_fail_generic_loader_error(
        self,
        mock_gc,
        mock_tok_loader,
        mock_mod_loader,
        mock_check_files,
        mock_determine_provider,
        mock_data_manager,
    ):
        mock_data_manager.get_data_paths.return_value = {"models": str(self.models_dir)}
        mock_determine_provider.return_value = ("CPUExecutionProvider", None)
        mock_check_files.return_value = True
        mock_mod_loader.side_effect = RuntimeError("Disk read error")

        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed load ONNX Causal LM \(RuntimeError\): Disk read error",
        ):
            load_onnx_llm()

        mock_mod_loader.assert_called_once()
        mock_tok_loader.assert_not_called()
        mock_gc.assert_called_once()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
