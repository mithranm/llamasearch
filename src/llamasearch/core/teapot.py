# src/llamasearch/core/teapot.py

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import onnxruntime
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer, PreTrainedTokenizer

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.hardware import detect_hardware_info # Import detect_hardware_info for selection logic
from llamasearch.protocols import LLM, ModelInfo
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

# --- Constants ---
TEAPOT_REPO_ID = "teapotai/teapotllm"
ONNX_SUBFOLDER = "onnx"
REQUIRED_ONNX_BASENAMES = ["encoder_model", "decoder_model", "decoder_with_past_model"]
QUANTIZATION_SUFFIXES_PRIORITY = [
    "",
    "_fp16",
    "_int8",
    "_q4",
    "_q4f16",
    "_uint8",
    "_bnb4",
]
TEAPOT_CONTEXT_LENGTH = 1024 # Define the expected context length
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
    # Parameters kept for potential future extension, but ignored internally
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Determines the ONNX Runtime provider. Always returns CPU for this setup."""
    provider = "CPUExecutionProvider"
    options = {} # No specific options needed for CPU generally
    available_providers = onnxruntime.get_available_providers()
    logger.debug(f"Available ONNX Runtime providers: {available_providers}")
    if "CPUExecutionProvider" not in available_providers:
         # This should practically never happen with standard onnxruntime install
         logger.critical("FATAL: CPUExecutionProvider not found in ONNX Runtime!")
         raise RuntimeError("ONNX Runtime CPU provider is missing.")

    logger.info(f"Forcing ONNX provider to CPU: {provider}")
    return provider, options if options else None


def _select_onnx_quantization(
    # hw: HardwareInfo, # hw parameter no longer used directly, get info inside
    # onnx_provider: str, # Always CPU
    # onnx_provider_opts: Optional[Dict[str, Any]], # Always None for CPU
    preference: str,
) -> str:
    """
    Selects the most appropriate ONNX quantization suffix for CPU,
    considering RAM requirements including embedder overhead.
    """
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

    logger.info("Performing automatic ONNX quantization selection for CPU...")

    # --- Increased Memory Requirements (Estimate ~2GB overhead for embedder + OS) ---
    embedder_overhead_gb = 2.0
    req_fp32_gb = 8.0 + embedder_overhead_gb # ~10 GB
    req_fp16_gb = 5.0 + embedder_overhead_gb # ~7 GB
    req_int8_gb = 3.0 + embedder_overhead_gb # ~5 GB
    req_q4_gb = 2.0 + embedder_overhead_gb # ~4 GB
    # ------------------------------------------------------------------------

    hw_info = detect_hardware_info() # Get hardware info here
    # Use TOTAL RAM for selection logic, as available RAM can fluctuate wildly
    ram_gb, has_avx2 = hw_info.memory.total_gb, hw_info.cpu.supports_avx2
    onnx_provider = "CPUExecutionProvider" # Explicitly state CPU for logging clarity

    logger.info(
        f"System Info - Total RAM: {ram_gb:.1f} GB, AVX2: {has_avx2}, Provider: {onnx_provider}"
    )
    logger.info(
        f"Quantization RAM Thresholds (incl. ~{embedder_overhead_gb:.1f}GB overhead): "
        f"FP32>={req_fp32_gb:.1f}, FP16>={req_fp16_gb:.1f}, INT8>={req_int8_gb:.1f}, Q4>={req_q4_gb:.1f}"
    )

    selected_quant = "_bnb4"  # Lowest fallback

    # CPU Logic (using TOTAL RAM and higher thresholds)
    if ram_gb >= req_fp32_gb:
        selected_quant = ""  # fp32
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
        selected_quant = "_bnb4" # Lowest fallback (might still struggle on very low RAM)
        if ram_gb < req_q4_gb:
            logger.warning(f"Total RAM ({ram_gb:.1f} GB) is below the minimum recommended threshold ({req_q4_gb:.1f} GB) for Q4. Performance may be severely impacted.")

    logger.info(f"CPU Selection based on TOTAL RAM/AVX2: {selected_quant}")
    return selected_quant


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
        for item in onnx_dir.glob("encoder_model*.onnx"):
            if item.is_file():
                name = item.name
                if name.startswith("encoder_model") and name.endswith(".onnx"):
                    suffix = name.replace("encoder_model", "").replace(".onnx", "")
                    available_suffixes.add(suffix)
    except Exception as e:
        logger.error(f"Error scanning ONNX directory {onnx_dir}: {e}", exc_info=True)
        return None

    if not available_suffixes:
        logger.warning(f"No potential ONNX model files found in {onnx_dir}")
        return None

    logger.debug(f"Found potential suffixes in {onnx_dir}: {available_suffixes}")

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
                return suffix

    logger.error(
        f"No complete set of ONNX model files found in {onnx_dir} for any detected suffix."
    )
    return None


# --- TeapotONNXLLM Wrapper Class ---
class TeapotONNXLLM(LLM):
    """Wraps the loaded Teapot ONNX model and tokenizer for CPU execution."""

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        quant_suffix: str,
        provider: str, # Will always be CPUExecutionProvider
        provider_options: Optional[Dict[str, Any]], # Will always be None or {}
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._info = TeapotONNXModelInfo(
            TEAPOT_REPO_ID,
            quant_suffix,
            TEAPOT_CONTEXT_LENGTH,
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
        """Returns the device (always CPU)."""
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
        """Generates text using the loaded ONNX model on CPU."""
        if not self._is_loaded:
            return "Error: Model not loaded.", {"error": "Model not loaded"}
        assert self._model is not None, "Model cannot be None during generation."
        assert self._tokenizer is not None, (
            "Tokenizer cannot be None during generation."
        )
        try:
            # For CPU, input and target device are the same
            target_device = self.device
            input_device = target_device

            model_context_length = self.model_info.context_length
            buffer = 10
            max_input_length = model_context_length - max_tokens - buffer
            if max_input_length <= 0:
                 logger.warning(
                     f"max_tokens ({max_tokens}) requested is too large for model context ({model_context_length}). "
                     f"Reducing max_input_length significantly."
                 )
                 max_input_length = max(10, model_context_length // 2)

            logger.debug(
                f"Tokenizing prompt. Model Context: {model_context_length}, Max New Tokens: {max_tokens}, "
                f"Calculated Max Input Tokens: {max_input_length}"
            )

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
                padding=False,
            ).to(input_device) # Send to CPU

            input_ids = inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                 raise TypeError("Tokenizer did not return a Tensor for input_ids")
            actual_input_tokens = input_ids.shape[1]
            logger.debug(f"Actual input tokens after tokenization/truncation: {actual_input_tokens}")

            gen_kwargs = {
                 "max_new_tokens": max_tokens,
                 **kwargs
            }
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

            logger.debug(
                f"Generating ONNX on CPU. Kwargs:{gen_kwargs}"
            )

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            out_ids = outputs[0].to("cpu") # Ensure output is on CPU
            raw_result = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            logger.debug(f"Raw decoded result (before strip): '{raw_result}'")

            result = raw_result
            if result.startswith(prompt):
                result = result[len(prompt) :]
            result = result.strip()
            logger.debug(f"Final result (after strip): '{result}'")

            out_tokens = len(out_ids)

            return result, {
                "output_token_count": out_tokens,
                "input_token_count": actual_input_tokens,
            }
        except Exception as e:
            logger.error(f"ONNX generation error: {e}", exc_info=True)
            return f"Error during generation: {e}", {"error": str(e)}

    def load(self) -> bool:
        """Returns loading status (model is loaded on init)."""
        return self._is_loaded

    def unload(self) -> None:
        """Unloads the model and attempts garbage collection (CPU only)."""
        logger.info(f"Unloading TeapotONNXLLM ({self.model_info.model_id})...")

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
            self._model = None
            self._tokenizer = None
            self._is_loaded = False

        logger.debug("Running garbage collection...")
        gc.collect()
        # No GPU/MPS cache clearing needed for CPU

        logger.info("TeapotONNXLLM unloaded.")


# --- Loader Function ---
def load_teapot_onnx_llm(
    onnx_quantization: str = "auto",
    # Parameters kept for potential future extension, but ignored internally
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Optional[TeapotONNXLLM]:
    """
    Loads the Teapot ONNX model from the assembled 'active_teapot' directory
    for CPU execution. Detects the available quantization if preference is 'auto'
    or unavailable.
    """
    logger.info(
        f"--- Initializing Teapot ONNX LLM for CPU (Quant Pref: {onnx_quantization}) ---"
    )
    onnx_model, tokenizer = None, None
    try:
        # Determine provider (will always be CPU)
        provider, options = _determine_onnx_provider(None, None) # Pass None to force CPU
        logger.info(f"Using ONNX Provider: {provider}")

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

        if not active_model_dir.is_dir():
            raise ModelNotFoundError(
                f"Active directory '{active_model_dir}' not found. Run setup."
            )
        onnx_sub = active_model_dir / ONNX_SUBFOLDER
        if not onnx_sub.is_dir():
            raise ModelNotFoundError(
                f"ONNX subfolder missing in {active_model_dir}. Run setup."
            )

        # Select quantization (CPU specific logic)
        # hw_info = detect_hardware_info() # No longer needed here, called inside select
        suffix_to_load: Optional[str] = _select_onnx_quantization(onnx_quantization)

        # --- Determine the quantization suffix to load ---
        preference_map = {
            p: f"_{p}" for p in ["fp16", "int8", "q4", "q4f16", "bnb4", "uint8"]
        }
        preference_map["fp32"] = ""

        if onnx_quantization != "auto":
            preferred_suffix = preference_map.get(onnx_quantization)
            if preferred_suffix is not None:
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
            export=False,
            provider=provider, # Will be CPUExecutionProvider
            provider_options=options, # Will be {} or None
            use_io_binding=False, # IO Binding not relevant for CPU
            local_files_only=True,
        )

        logger.debug(f"Loading tokenizer from {active_model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(
            active_model_dir, use_fast=True, local_files_only=True
        )

        logger.info(
            f"Teapot ONNX model (suffix '{suffix_to_load}') and tokenizer loaded successfully."
        )
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
        if onnx_model is not None:
            del onnx_model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        # No GPU cache clearing needed
        raise RuntimeError(
            f"Failed to load Teapot ONNX model ({e.__class__.__name__}): {e}"
        ) from e