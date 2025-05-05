# src/llamasearch/core/teapot.py

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import onnxruntime
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizer

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.hardware import HardwareInfo, detect_hardware_info
from llamasearch.protocols import LLM, ModelInfo
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

# --- Constants ---
TEAPOT_REPO_ID = "teapotai/teapotllm"
ONNX_SUBFOLDER = "onnx"
REQUIRED_ONNX_BASENAMES = ["encoder_model", "decoder_model", "decoder_with_past_model"]

TEAPOT_BASE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",  # T5 Tokenizer specific
]
# --- END ADDED Constant Definition ---


# --- Concrete ModelInfo Implementation ---
class TeapotONNXModelInfo(ModelInfo):
    """Implementation of ModelInfo protocol for Teapot ONNX."""

    def __init__(self, model_id: str, quant_suffix: str, context_len: int):
        self._model_id_base = model_id
        self._quant_suffix = quant_suffix
        self._context_len = context_len

    @property
    def model_id(self) -> str:
        quant_str = self._quant_suffix.lstrip("_") if self._quant_suffix else "fp32"
        return f"{self._model_id_base}-onnx-{quant_str}"

    @property
    def model_engine(self) -> str:
        return "onnx_teapot"

    @property
    def description(self) -> str:
        quant_str = self._quant_suffix.lstrip("_") if self._quant_suffix else "fp32"
        return f"Teapot ONNX model ({quant_str} quantization)"

    @property
    def context_length(self) -> int:
        return self._context_len


# --- Helper functions ---
def _determine_onnx_provider(
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Determines the best available ONNX Runtime provider."""
    provider = preferred_provider or "CPUExecutionProvider"
    options = preferred_options if preferred_options is not None else {}
    available_providers = onnxruntime.get_available_providers()
    logger.debug(f"Available ONNX Runtime providers: {available_providers}")
    if preferred_provider:
        if preferred_provider not in available_providers:
            logger.warning(
                f"Preferred provider '{preferred_provider}' not available. Falling back to CPU."
            )
            provider, options = "CPUExecutionProvider", {}
        else:
            provider = preferred_provider
            logger.info(f"Using preferred ONNX provider: {provider}")
    else:
        if "CUDAExecutionProvider" in available_providers:
            provider, options["device_id"] = (
                "CUDAExecutionProvider",
                options.get("device_id", 0),
            )
        elif "ROCMExecutionProvider" in available_providers:
            provider, options["device_id"] = (
                "ROCMExecutionProvider",
                options.get("device_id", 0),
            )
        elif "CoreMLExecutionProvider" in available_providers:
            provider = "CoreMLExecutionProvider"
        else:
            provider = "CPUExecutionProvider"
        logger.info(f"Auto-selecting available ONNX provider: {provider}")
    return provider, options if options else None


def _select_onnx_quantization(
    hw: HardwareInfo,
    onnx_provider: str,
    onnx_provider_opts: Optional[Dict[str, Any]],
    preference: str,
) -> str:
    """Selects the most appropriate ONNX quantization suffix."""
    preference_map = {
        p: f"_{p}" for p in ["fp16", "int8", "q4", "q4f16", "bnb4", "uint8"]
    }
    preference_map["fp32"] = ""
    if preference != "auto":
        if preference in preference_map:
            logger.info(f"Using user-preferred ONNX quantization: {preference}")
            return preference_map[preference]
        else:
            logger.warning(
                f"Invalid quantization preference '{preference}'. Falling back to 'auto'."
            )
    logger.info("Performing automatic ONNX quantization selection...")
    req_fp32_gb, req_fp16_gb, req_int8_gb, req_q4_gb = 8.0, 5.0, 3.0, 2.0
    ram_gb, has_avx2 = hw.memory.available_gb, hw.cpu.supports_avx2
    logger.info(
        f"System Info - Available RAM: {ram_gb:.1f} GB, AVX2: {has_avx2}, Provider: {onnx_provider}"
    )
    is_gpu = (
        "CUDAExecutionProvider" in onnx_provider
        or "ROCMExecutionProvider" in onnx_provider
    )
    is_coreml = "CoreMLExecutionProvider" in onnx_provider
    selected_quant = "_bnb4"
    if is_gpu:
        if ram_gb >= req_fp16_gb + 5.0:
            selected_quant = "_fp16"
        elif ram_gb >= req_int8_gb + 2.0:
            selected_quant = "_int8"
        elif ram_gb >= req_q4_gb + 1.0:
            selected_quant = "_q4"
        else:
            selected_quant = "_bnb4"
        logger.info(f"GPU Selection based on System RAM: {selected_quant}")
    elif is_coreml:
        selected_quant = "_int8"
        logger.info(f"CoreML Selection: {selected_quant}")
    else:  # CPU Logic
        if ram_gb >= req_fp32_gb:
            selected_quant = ""
        elif ram_gb >= req_fp16_gb:
            selected_quant = "_fp16"
        elif ram_gb >= req_int8_gb:
            selected_quant = "_int8"
            if not has_avx2:
                logger.warning(
                    "Selecting INT8 on CPU without detected AVX2 support. Performance might be suboptimal."
                )
        elif ram_gb >= req_q4_gb:
            selected_quant = "_q4"
        else:
            selected_quant = "_bnb4"
        logger.info(f"CPU Selection based on RAM/AVX2: {selected_quant}")
    return selected_quant


# --- TeapotONNXLLM Wrapper Class ---
class TeapotONNXLLM(LLM):
    """Wraps the loaded Teapot ONNX model and tokenizer."""

    def __init__(
        self,
        model: Any,  # Keep Any type hint
        tokenizer: PreTrainedTokenizer,
        quant_suffix: str,
        provider: str,
        provider_options: Optional[Dict[str, Any]],
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._info = TeapotONNXModelInfo(
            TEAPOT_REPO_ID, quant_suffix, 1024 # Hardcoded, but this is the actual teapot max length
        )
        self._is_loaded = True
        self._provider = provider
        self._provider_options = provider_options
        logger.info(
            f"Initialized {self.model_info.model_id} wrapper on device '{self.device}' with provider {self._provider}"
        )

    @property
    def model_info(self) -> ModelInfo:
        """Returns model metadata."""
        return self._info

    @property
    def device(self) -> torch.device:
        """Determines the effective device the model is running on."""
        # --- Add assertion ---
        assert self._is_loaded and self._model is not None, (
            "Model must be loaded to access device."
        )
        # --- End assertion ---
        if hasattr(self._model, "device") and isinstance(
            self._model.device, torch.device
        ):
            return self._model.device
        provider_name = self._provider
        if provider_name == "CUDAExecutionProvider":
            return torch.device("cuda")
        if provider_name == "ROCMExecutionProvider":
            return torch.device("rocm")
        if provider_name == "CoreMLExecutionProvider":
            return (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        return torch.device("cpu")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[str, Any]:
        """Generates text using the loaded ONNX model."""
        if not self._is_loaded:
            return "Error: Model not loaded.", {"error": "Model not loaded"}
        # --- Add assertions ---
        assert self._model is not None, "Model cannot be None during generation."
        assert self._tokenizer is not None, (
            "Tokenizer cannot be None during generation."
        )
        # --- End assertions ---
        try:
            target_device = self.device
            max_input_length = self.model_info.context_length - max_tokens
            if max_input_length <= 0:
                logger.warning(
                    f"max_tokens ({max_tokens}) exceeds context length ({self.model_info.context_length}). Truncating severely."
                )
                max_input_length = max(10, self.model_info.context_length // 2)

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
                padding=False,
            ).to(target_device)

            gen_kwargs = {"max_new_tokens": max_tokens, **kwargs}
            # Sampling logic
            if temperature > 0.0 or (top_p is not None and top_p < 1.0):
                gen_kwargs["do_sample"] = True
                if temperature > 0.0:
                    gen_kwargs["temperature"] = temperature
                if top_p is not None and top_p < 1.0:
                    gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False
            if repeat_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = repeat_penalty

            input_ids = inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("Tokenizer did not return a Tensor for input_ids")
            in_tokens = input_ids.shape[1]
            logger.debug(
                f"Generating ONNX. Input Tokens:{in_tokens}, Device:{target_device}, Kwargs:{gen_kwargs}"
            )

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            out_ids = outputs[0]
            raw_result = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            logger.debug(f"Raw decoded result (before strip): '{raw_result}'")
            result = raw_result.replace(prompt, "").strip()
            logger.debug(f"Final result (after strip): '{result}'")
            out_tokens = out_ids.shape[0] if isinstance(out_ids, torch.Tensor) else 0

            return result, {
                "output_token_count": out_tokens,
                "input_token_count": in_tokens,
            }
        except Exception as e:
            logger.error(f"ONNX generation error: {e}", exc_info=True)
            return f"Error during generation: {e}", {"error": str(e)}

    def load(self) -> bool:
        """Returns loading status (model is loaded on init)."""
        return self._is_loaded

    def unload(self) -> None:
        """Unloads the model and attempts garbage collection."""
        # Fast exit when running on CPU – avoids known hang in ORT finaliser
        if getattr(self, "_provider", "CPUExecutionProvider") == "CPUExecutionProvider":
            logger.info("TeapotONNXLLM unload skipped directly (CPU provider).")
            self._model = None
            self._tokenizer = None
            self._is_loaded = False
            gc.collect()
            return

        logger.info(f"Unloading TeapotONNXLLM ({self.model_info.model_id})...")
        # Get device type *before* deleting the model
        dev_type = "cpu"  # Default
        try:
            if self._model is not None:
                dev_type = self.device.type  # Get device type if model exists
        except Exception:
            logger.warning("Could not determine device type before unloading.")

        try:
            if hasattr(self, "_model"):
                del self._model
            if hasattr(self, "_tokenizer"):
                del self._tokenizer
        except Exception as e:
            logger.error(f"Error deleting internal model/tokenizer references: {e}")

        # Set internal attributes to None
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        gc.collect()
        if dev_type == "cuda" and torch.cuda.is_available():
            logger.debug("Clearing CUDA cache after unloading TeapotONNXLLM.")
            torch.cuda.empty_cache()
        logger.info("TeapotONNXLLM unloaded.")


# --- Loader Function ---
def load_teapot_onnx_llm(
    onnx_quantization: str = "auto",
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Optional[TeapotONNXLLM]:
    """Loads the Teapot ONNX model from the assembled 'active_teapot' directory."""
    logger.info(
        f"--- Initializing Teapot ONNX LLM (Quant Pref: {onnx_quantization}) ---"
    )
    onnx_model, tokenizer = None, None
    try:
        hw_info = detect_hardware_info()
        provider, options = _determine_onnx_provider(
            preferred_provider, preferred_options
        )
        quant_suffix = _select_onnx_quantization(
            hw_info, provider, options, onnx_quantization
        )
        logger.info(
            f"Loading with ONNX Provider: {provider}, Target Quant Suffix: '{quant_suffix}'"
        )

        paths = data_manager.get_data_paths()
        models_dir_str = paths.get("models")
        if not models_dir_str:
            raise SetupError(
                "Models directory path not found in data_manager settings."
            )
        model_cache_dir = Path(models_dir_str)
        active_model_dir = model_cache_dir / "active_teapot"

        logger.info(
            f"Attempting to load from active model directory: {active_model_dir}"
        )

        # Verification of active_model_dir content
        if not active_model_dir.is_dir():
            raise ModelNotFoundError(
                f"Active directory '{active_model_dir}' not found. Please run 'llamasearch-setup'."
            )

        # --- Corrected: Use TEAPOT_BASE_FILES constant ---
        required_files_rel = list(TEAPOT_BASE_FILES)  # Use the defined constant
        # --- End Correction ---
        onnx_sub = active_model_dir / ONNX_SUBFOLDER
        if not onnx_sub.is_dir():
            raise ModelNotFoundError(
                f"ONNX subfolder missing in {active_model_dir}. Please run 'llamasearch-setup'."
            )
        for basename in REQUIRED_ONNX_BASENAMES:
            onnx_fname = f"{basename}{quant_suffix}.onnx"
            required_files_rel.append(f"{ONNX_SUBFOLDER}/{onnx_fname}")

        missing_files = []
        for req_file_rel in required_files_rel:
            req_file_abs = active_model_dir / req_file_rel
            if not req_file_abs.exists():
                missing_files.append(req_file_rel)
        if missing_files:
            logger.error(
                f"Required model files missing in {active_model_dir}: {missing_files}"
            )
            raise ModelNotFoundError(
                f"Required files missing in '{active_model_dir}'. Please run 'llamasearch-setup'."
            )

        # Load from the verified 'active_teapot' directory
        logger.debug(f"Loading ONNX model components from {active_model_dir}...")
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            active_model_dir,
            export=False,
            provider=provider,
            provider_options=options,
            use_io_binding=(
                "CUDAExecutionProvider" in provider
                or "ROCMExecutionProvider" in provider
            ),
            library_name="transformers",
            local_files_only=True,
        )
        logger.debug(f"Loading tokenizer from {active_model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(
            active_model_dir, use_fast=True, local_files_only=True
        )

        logger.info(
            "Teapot ONNX model and tokenizer loaded successfully from active directory."
        )
        llm_instance = TeapotONNXLLM(
            onnx_model, tokenizer, quant_suffix, provider, options
        )
        return llm_instance

    except ModelNotFoundError:
        logger.error(
            "ModelNotFoundError occurred during LLM init. Setup might be needed."
        )
        raise
    except Exception as e:
        logger.error(
            f"Failed during Teapot ONNX LLM initialization: {e}", exc_info=True
        )
        if onnx_model is not None:
            del onnx_model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"Failed to load Teapot ONNX model ({e.__class__.__name__}): {e}"
        ) from e
