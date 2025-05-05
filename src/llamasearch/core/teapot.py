# src/llamasearch/core/teapot.py

import gc

# <<< Removed unused logging import >>>
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import onnxruntime
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizer

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError

# <<< Removed unused detect_hardware_info import >>>
from llamasearch.hardware import HardwareInfo  # Keep HardwareInfo for type hints
from llamasearch.protocols import LLM, ModelInfo
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

# --- Constants ---
TEAPOT_REPO_ID = "teapotai/teapotllm"
ONNX_SUBFOLDER = "onnx"
REQUIRED_ONNX_BASENAMES = ["encoder_model", "decoder_model", "decoder_with_past_model"]
# Order matters for auto-detection fallback (prefer less quantized if available)
# "" (fp32) is highest priority, "_bnb4" is lowest.
QUANTIZATION_SUFFIXES_PRIORITY = [
    "",
    "_fp16",
    "_int8",
    "_q4",
    "_q4f16",
    "_uint8",
    "_bnb4",
]

TEAPOT_BASE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",  # T5 Tokenizer specific
]


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
        # Order of preference: CUDA > ROCm > CoreML > CPU
        if "CUDAExecutionProvider" in available_providers:
            provider = "CUDAExecutionProvider"
            options["device_id"] = options.get("device_id", 0)
        elif "ROCMExecutionProvider" in available_providers:
            provider = "ROCMExecutionProvider"
            options["device_id"] = options.get("device_id", 0)
        elif "CoreMLExecutionProvider" in available_providers:
            # Prefer CoreML over CPU on macOS if available
            provider = "CoreMLExecutionProvider"
            logger.info("CoreML provider detected and selected.")
            # Remove potentially incompatible CPU options if CoreML is chosen
            options = {}
        else:
            provider = "CPUExecutionProvider"
        logger.info(f"Auto-selecting available ONNX provider: {provider}")

    # Log final decision
    logger.info(
        f"Final ONNX provider choice: {provider} with options: {options or '{}'}"
    )
    return provider, options if options else None


def _select_onnx_quantization(
    hw: HardwareInfo,  # HardwareInfo used for type hint, but detect_hardware_info import removed
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
    # Use TOTAL RAM for selection logic, as available RAM can fluctuate wildly
    ram_gb, has_avx2 = hw.memory.total_gb, hw.cpu.supports_avx2
    logger.info(
        f"System Info - Total RAM: {ram_gb:.1f} GB, AVX2: {has_avx2}, Provider: {onnx_provider}"
    )
    is_gpu = (
        "CUDAExecutionProvider" in onnx_provider
        or "ROCMExecutionProvider" in onnx_provider
    )
    is_coreml = "CoreMLExecutionProvider" in onnx_provider
    selected_quant = "_bnb4"  # Default lowest

    if is_gpu:
        # GPU RAM is the primary constraint, but we don't detect it reliably yet.
        # Fallback to system RAM as a rough proxy.
        logger.warning(
            "GPU detected, but GPU RAM check not implemented. Using system RAM as proxy."
        )
        if ram_gb >= req_fp16_gb + 5.0:  # Higher buffer for GPU
            selected_quant = "_fp16"
        elif ram_gb >= req_int8_gb + 2.0:
            selected_quant = "_int8"
        elif ram_gb >= req_q4_gb + 1.0:
            selected_quant = (
                "_q4"  # q4 often performs better than bnb4 on GPU if RAM allows
            )
        else:
            selected_quant = "_bnb4"
        logger.info(f"GPU Selection (using System RAM proxy): {selected_quant}")
    elif is_coreml:
        # CoreML usually handles quantization well, int8 is often a good balance
        # but let's try q4 first as it's smaller
        if ram_gb >= req_int8_gb:
            selected_quant = "_int8"  # Prefer int8 if enough RAM
        elif ram_gb >= req_q4_gb:
            selected_quant = "_q4"
        else:
            selected_quant = "_bnb4"  # Fallback
        logger.info(f"CoreML Selection: {selected_quant}")
    else:  # CPU Logic (using TOTAL RAM)
        if ram_gb >= req_fp32_gb + 2.0:  # Extra buffer for OS etc.
            selected_quant = ""  # fp32
        elif ram_gb >= req_fp16_gb + 1.0:
            selected_quant = "_fp16"
        elif ram_gb >= req_int8_gb + 0.5:
            selected_quant = "_int8"
            if not has_avx2:
                logger.warning(
                    "Selecting INT8 on CPU without detected AVX2 support. Performance might be suboptimal."
                )
        elif ram_gb >= req_q4_gb:
            selected_quant = "_q4"
        else:
            selected_quant = "_bnb4"  # Lowest fallback
        logger.info(f"CPU Selection based on TOTAL RAM/AVX2: {selected_quant}")
    return selected_quant


# --- NEW Helper to Detect Existing Quantization ---
def _detect_available_onnx_suffix(onnx_dir: Path) -> Optional[str]:
    """
    Detects the best available quantization suffix based on files present.
    Returns the suffix string (e.g., "_int8", "_q4", "" for fp32) or None if no complete set found.
    """
    if not onnx_dir.is_dir():
        logger.error(f"ONNX directory does not exist: {onnx_dir}")
        return None

    available_suffixes = set()
    try:
        # Find all potential encoder files to deduce suffixes
        for item in onnx_dir.glob("encoder_model*.onnx"):
            if item.is_file():
                name = item.name
                if name.startswith("encoder_model") and name.endswith(".onnx"):
                    # Extract suffix part (e.g., _int8, _q4, or empty for fp32)
                    suffix = name.replace("encoder_model", "").replace(".onnx", "")
                    available_suffixes.add(suffix)
    except Exception as e:
        logger.error(f"Error scanning ONNX directory {onnx_dir}: {e}", exc_info=True)
        return None

    if not available_suffixes:
        logger.warning(f"No potential ONNX model files found in {onnx_dir}")
        return None

    logger.debug(f"Found potential suffixes in {onnx_dir}: {available_suffixes}")

    # Check for complete sets based on priority
    for suffix in QUANTIZATION_SUFFIXES_PRIORITY:
        if suffix in available_suffixes:
            is_complete = True
            for basename in REQUIRED_ONNX_BASENAMES:
                onnx_fname = f"{basename}{suffix}.onnx"
                if not (onnx_dir / onnx_fname).exists():
                    logger.debug(
                        f"Suffix '{suffix}' is incomplete: Missing {onnx_fname}"
                    )
                    is_complete = False
                    break
            if is_complete:
                logger.info(f"Detected complete ONNX model set with suffix: '{suffix}'")
                return (
                    suffix  # Return the first complete suffix found based on priority
                )

    logger.error(
        f"No complete set of ONNX model files found in {onnx_dir} for any detected suffix."
    )
    return None


# --- TeapotONNXLLM Wrapper Class ---
class TeapotONNXLLM(LLM):
    """Wraps the loaded Teapot ONNX model and tokenizer."""

    def __init__(
        self,
        model: Any,  # Keep Any type hint
        tokenizer: PreTrainedTokenizer,
        quant_suffix: str,  # The suffix actually loaded
        provider: str,
        provider_options: Optional[Dict[str, Any]],
    ):
        self._model = model
        self._tokenizer = tokenizer
        # Use the actual loaded quant_suffix for ModelInfo
        self._info = TeapotONNXModelInfo(
            TEAPOT_REPO_ID,
            quant_suffix,
            1024,  # Hardcoded, but this is the actual teapot max length
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
        # ORTModel might not have a direct .device attribute reflecting the ONNX provider target
        # We rely on the provider string used during initialization
        provider_name = self._provider
        if provider_name == "CUDAExecutionProvider":
            return torch.device(
                f"cuda:{self._provider_options.get('device_id', 0)}"
                if self._provider_options
                else "cuda:0"
            )
        if provider_name == "ROCMExecutionProvider":
            # ROCm support in PyTorch might vary, assume basic device mapping
            return torch.device(
                f"rocm:{self._provider_options.get('device_id', 0)}"
                if self._provider_options
                else "rocm:0"
            )
        if provider_name == "CoreMLExecutionProvider":
            # CoreML runs on the Neural Engine, conceptually map to MPS if available, else CPU
            # Check if MPS is available and built
            is_mps_available = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            is_mps_built = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            )
            if is_mps_available and is_mps_built:
                return torch.device("mps")
            else:
                return torch.device("cpu")
        # Default to CPU for CPUExecutionProvider or unknown providers
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
            target_device = self.device  # Get the effective device
            # Some providers might need inputs on CPU even if compute happens elsewhere
            # For simplicity, let's try putting inputs on the target device first
            input_device = target_device

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
            )

            # Move inputs to the target device
            try:
                inputs = inputs.to(input_device)
            except Exception as move_err:
                logger.warning(
                    f"Could not move inputs to {input_device}, keeping on CPU: {move_err}"
                )
                input_device = torch.device("cpu")  # Fallback to CPU if move fails
                inputs = inputs.to(input_device)

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
                f"Generating ONNX. Input Tokens:{in_tokens}, Target Device:{target_device}, Input Device: {input_device}, Kwargs:{gen_kwargs}"
            )

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            out_ids = outputs[0]
            # Move outputs back to CPU for decoding if they aren't already
            if isinstance(out_ids, torch.Tensor) and out_ids.device != torch.device(
                "cpu"
            ):
                out_ids = out_ids.to("cpu")

            raw_result = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            logger.debug(f"Raw decoded result (before strip): '{raw_result}'")
            # Strip the original prompt text more robustly
            result = raw_result
            # <<< FIX: E701 Multiple statements on one line >>>
            if result.startswith(prompt):
                result = result[len(prompt) :]
            # <<< END FIX >>>
            result = result.strip()

            logger.debug(f"Final result (after strip): '{result}'")
            # <<< FIX: E701 Multiple statements on one line >>>
            out_tokens = 0
            if isinstance(out_ids, torch.Tensor):
                out_tokens = len(out_ids)
            # <<< END FIX >>>

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
        logger.info(f"Unloading TeapotONNXLLM ({self.model_info.model_id})...")
        # Get device type *before* deleting the model
        dev_type = "cpu"  # Default
        try:
            if self._model is not None:
                dev_type = self.device.type  # Get device type if model exists
        except Exception:
            logger.warning("Could not determine device type before unloading.")

        # Explicitly delete the model and tokenizer references *before* GC
        try:
            if hasattr(self, "_model") and self._model is not None:
                logger.debug("Deleting internal _model reference...")
                del self._model
            else:
                logger.debug("_model already None or missing.")
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                logger.debug("Deleting internal _tokenizer reference...")
                del self._tokenizer
            else:
                logger.debug("_tokenizer already None or missing.")
        except Exception as e:
            logger.error(f"Error deleting internal model/tokenizer references: {e}")
        finally:
            # Ensure attributes are None even if del failed
            self._model = None
            self._tokenizer = None
            self._is_loaded = False

        # Force garbage collection
        logger.debug("Running garbage collection...")
        gc.collect()

        # Clear GPU caches if applicable
        if dev_type == "cuda" and torch.cuda.is_available():
            logger.debug("Clearing CUDA cache after unloading TeapotONNXLLM.")
            torch.cuda.empty_cache()
        elif dev_type == "mps":
            is_mps_available = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            is_mps_built = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            )
            if is_mps_available and is_mps_built:
                try:
                    logger.debug("Attempting to clear MPS cache...")
                    torch.mps.empty_cache()
                except Exception as mps_err:
                    logger.warning(f"Could not clear MPS cache: {mps_err}")

        logger.info("TeapotONNXLLM unloaded.")


# --- Loader Function ---
def load_teapot_onnx_llm(
    onnx_quantization: str = "auto",  # Preference, not strict requirement
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Optional[TeapotONNXLLM]:
    """
    Loads the Teapot ONNX model from the assembled 'active_teapot' directory.
    Detects the available quantization if preference is 'auto' or unavailable.
    """
    logger.info(
        f"--- Initializing Teapot ONNX LLM (Quant Pref: {onnx_quantization}) ---"
    )
    onnx_model, tokenizer = None, None
    try:
        # Determine provider first
        provider, options = _determine_onnx_provider(
            preferred_provider, preferred_options
        )
        logger.info(f"Using ONNX Provider: {provider}")

        # Locate model directory
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

        # Basic directory verification
        if not active_model_dir.is_dir():
            raise ModelNotFoundError(
                f"Active directory '{active_model_dir}' not found. Run setup."
            )
        onnx_sub = active_model_dir / ONNX_SUBFOLDER
        if not onnx_sub.is_dir():
            raise ModelNotFoundError(
                f"ONNX subfolder missing in {active_model_dir}. Run setup."
            )

        # --- Determine the quantization suffix to load ---
        suffix_to_load: Optional[str] = None
        preference_map = {
            p: f"_{p}" for p in ["fp16", "int8", "q4", "q4f16", "bnb4", "uint8"]
        }
        preference_map["fp32"] = ""

        if onnx_quantization != "auto":
            preferred_suffix = preference_map.get(onnx_quantization)
            if preferred_suffix is not None:
                # Check if the preferred suffix actually exists
                is_complete = True
                for basename in REQUIRED_ONNX_BASENAMES:
                    onnx_fname = f"{basename}{preferred_suffix}.onnx"
                    if not (onnx_sub / onnx_fname).exists():
                        logger.warning(
                            f"Preferred quantization '{onnx_quantization}' (suffix '{preferred_suffix}') files missing ({onnx_fname}). Falling back to detection."
                        )
                        is_complete = False
                        break
                if is_complete:
                    logger.info(
                        f"Using user-preferred ONNX quantization: '{onnx_quantization}' (suffix '{preferred_suffix}')"
                    )
                    suffix_to_load = preferred_suffix
            else:
                logger.warning(
                    f"Invalid quantization preference '{onnx_quantization}'. Falling back to detection."
                )

        # If no valid preference or preference was 'auto', detect available suffix
        if suffix_to_load is None:
            logger.info("Detecting available ONNX quantization in active directory...")
            suffix_to_load = _detect_available_onnx_suffix(onnx_sub)
            if suffix_to_load is None:
                raise ModelNotFoundError(
                    f"No complete ONNX model set found in {onnx_sub}. Run setup, potentially with '--force'."
                )
            loaded_quant_str = suffix_to_load.lstrip("_") if suffix_to_load else "fp32"
            logger.info(
                f"Detected and selected quantization suffix: '{suffix_to_load}' ({loaded_quant_str})"
            )

        # --- Verify all required files exist with the chosen suffix ---
        required_files_rel: List[str] = list(TEAPOT_BASE_FILES)
        for basename in REQUIRED_ONNX_BASENAMES:
            onnx_fname = f"{basename}{suffix_to_load}.onnx"
            required_files_rel.append(f"{ONNX_SUBFOLDER}/{onnx_fname}")

        missing_files = []
        for req_file_rel in required_files_rel:
            req_file_abs = active_model_dir / req_file_rel
            if not req_file_abs.exists():
                missing_files.append(req_file_rel)
        if missing_files:
            logger.error(
                f"Required model files missing in {active_model_dir} for suffix '{suffix_to_load}': {missing_files}"
            )
            raise ModelNotFoundError(
                f"Required files missing for suffix '{suffix_to_load}' in '{active_model_dir}'. Run setup."
            )

        # --- Load the model and tokenizer ---
        logger.debug(
            f"Loading ONNX model components from {active_model_dir} with suffix '{suffix_to_load}'..."
        )

        # <<< FIX: Pass filenames as explicit keyword arguments >>>
        encoder_fn = f"{ONNX_SUBFOLDER}/encoder_model{suffix_to_load}.onnx"
        decoder_fn = f"{ONNX_SUBFOLDER}/decoder_model{suffix_to_load}.onnx"
        decoder_past_fn = (
            f"{ONNX_SUBFOLDER}/decoder_with_past_model{suffix_to_load}.onnx"
        )
        logger.debug(
            f"ORTModel filenames: enc='{encoder_fn}', dec='{decoder_fn}', dec_past='{decoder_past_fn}'"
        )

        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            active_model_dir,
            encoder_file_name=encoder_fn,
            decoder_file_name=decoder_fn,
            decoder_with_past_file_name=decoder_past_fn,
            # Removed **onnx_file_paths
            export=False,
            provider=provider,
            provider_options=options,
            use_io_binding=(
                "CUDAExecutionProvider" in provider
                or "ROCMExecutionProvider" in provider
            ),
            local_files_only=True,  # Ensure it only uses local files
        )
        # <<< END FIX >>>

        logger.debug(f"Loading tokenizer from {active_model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(
            active_model_dir, use_fast=True, local_files_only=True
        )

        logger.info(
            f"Teapot ONNX model (suffix '{suffix_to_load}') and tokenizer loaded successfully."
        )
        # Pass the suffix_to_load to the wrapper
        llm_instance = TeapotONNXLLM(
            onnx_model, tokenizer, suffix_to_load, provider, options
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
        # --- Cleanup ---
        if onnx_model is not None:
            del onnx_model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # --- End Cleanup ---
        raise RuntimeError(
            f"Failed to load Teapot ONNX model ({e.__class__.__name__}): {e}"
        ) from e
