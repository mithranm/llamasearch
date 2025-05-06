# src/llamasearch/core/onnx_model.py (CPU-Only Refactor with Llama-3.2-1B & Pyright/Ruff Fixes)

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import onnxruntime
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, BatchEncoding
# Removed unused imports

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.hardware import detect_hardware_info
from llamasearch.protocols import LLM, ModelInfo
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

# --- Constants --- (Updated for Llama-3.2-1B)
ONNX_MODEL_REPO_ID = "onnx-community/Llama-3.2-1B-Instruct"
ONNX_SUBFOLDER = "onnx"
MODEL_ONNX_BASENAME = "model"
QUANTIZATION_SUFFIXES_PRIORITY = [
    "",
    "_fp16",
    "_int8",
    "_quantized",
    "_q4",
    "_q4f16",
    "_uint8",
    "_bnb4",
]
ONNX_MODEL_CONTEXT_LENGTH = 128000  # Context length for Llama-3.2-1B
# Define the minimal required base files for loading check
# (Download logic in setup.py will grab all root files)
ONNX_REQUIRED_LOAD_FILES = [
    "config.json",
    "tokenizer.json",
]

# Restore original list for reference/potential future use, though not used for download anymore
_ONNX_ORIGINAL_BASE_FILES_ = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json"
]


# --- GenericONNXModelInfo Class (No change needed) ---
class GenericONNXModelInfo(ModelInfo):
    """Implementation of ModelInfo protocol for Generic ONNX Causal LMs."""

    def __init__(self, model_repo_id: str, quant_suffix: str, context_len: int):
        self._model_repo_id = model_repo_id
        self._quant_suffix = quant_suffix
        self._context_len = context_len

    @property
    def model_id(self) -> str:
        quant_str = self._quant_suffix.lstrip("_") if self._quant_suffix else "fp32"
        base_name = self._model_repo_id.split("/")[-1]
        return f"{base_name}-onnx-{quant_str}"

    @property
    def model_engine(self) -> str:
        return "onnx_causal"

    @property
    def description(self) -> str:
        quant_str = self._quant_suffix.lstrip("_") if self._quant_suffix else "fp32"
        return (
            f"Generic ONNX Causal LM ({self._model_repo_id}, {quant_str} quantization)"
        )

    @property
    def context_length(self) -> int:
        return self._context_len


# --- Helper functions (No change needed) ---
def _determine_onnx_provider(
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Determines the ONNX Runtime provider. FORCES CPU."""
    available_providers = onnxruntime.get_available_providers()
    logger.debug(f"Available ONNX Runtime providers: {available_providers}")

    selected_provider = "CPUExecutionProvider"

    if "CPUExecutionProvider" not in available_providers:
        logger.critical("FATAL: CPUExecutionProvider not found in ONNX Runtime!")
        raise RuntimeError("ONNX Runtime CPU provider is missing.")

    if preferred_provider and preferred_provider != "CPUExecutionProvider":
        logger.warning(
            f"Requested provider '{preferred_provider}' ignored. Forcing CPUExecutionProvider."
        )

    logger.info(f"Using ONNX provider: {selected_provider}")
    return selected_provider, None


def _select_onnx_quantization(
    preference: str,
) -> str:
    """Selects the most appropriate ONNX quantization suffix for CPU."""
    preference_map = {p.lstrip("_"): p for p in QUANTIZATION_SUFFIXES_PRIORITY}
    preference_map["fp32"] = ""

    if preference != "auto":
        if preference in preference_map:
            logger.info(f"Using user-preferred ONNX quantization: {preference}")
            return preference_map[preference]
        else:
            logger.warning(
                f"Invalid preference '{preference}'. Falling back to 'auto'."
            )

    logger.info("Auto-selecting ONNX quantization for CPU...")

    embedder_overhead_gb = 2.0
    req_fp32_gb = 4.0 + embedder_overhead_gb
    req_fp16_gb = 2.5 + embedder_overhead_gb
    req_int8_gb = 1.7 + embedder_overhead_gb
    req_q4_gb = 1.5 + embedder_overhead_gb
    req_q4f16_gb = 1.0 + embedder_overhead_gb

    hw_info = detect_hardware_info()
    ram_gb, has_avx2 = hw_info.memory.total_gb, hw_info.cpu.supports_avx2
    onnx_provider = "CPUExecutionProvider"

    logger.info(
        f"System: RAM={ram_gb:.1f}GB, AVX2={has_avx2}, Provider={onnx_provider}"
    )
    logger.info(
        f"RAM Thresholds (+{embedder_overhead_gb:.1f}GB): FP32>={req_fp32_gb:.1f}, FP16>={req_fp16_gb:.1f}, INT8>={req_int8_gb:.1f}, Q4>={req_q4_gb:.1f}, Q4F16>={req_q4f16_gb:.1f}"
    )

    selected_quant = "_bnb4"

    if ram_gb >= req_fp32_gb:
        selected_quant = ""
    elif ram_gb >= req_fp16_gb:
        selected_quant = "_fp16"
    elif ram_gb >= req_int8_gb:
        selected_quant = "_int8"
        if not has_avx2:
            logger.warning("INT8 on CPU without AVX2 may be suboptimal.")
    elif ram_gb >= req_q4_gb:
        selected_quant = "_q4"
    elif ram_gb >= req_q4f16_gb:
        selected_quant = "_q4f16"

    if selected_quant == "_bnb4" and ram_gb < req_q4f16_gb:
        logger.warning(
            f"RAM ({ram_gb:.1f}GB) < min recommended ({req_q4f16_gb:.1f}GB). Performance may suffer."
        )

    logger.info(f"CPU Auto Selection: '{selected_quant}'")
    return selected_quant


def _detect_available_onnx_suffix(onnx_dir: Path) -> Optional[str]:
    """Detects the best available ONNX suffix based on files in onnx_dir."""
    if not onnx_dir.is_dir():
        logger.error(f"ONNX directory does not exist: {onnx_dir}")
        return None

    available_suffixes = set()
    try:
        for item in onnx_dir.glob(f"{MODEL_ONNX_BASENAME}*.onnx"):
            if item.is_file():
                name = item.name
                if name.startswith(MODEL_ONNX_BASENAME) and name.endswith(".onnx"):
                    suffix = name.replace(MODEL_ONNX_BASENAME, "").replace(".onnx", "")
                    if suffix == "":
                        if (onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data").exists():
                            available_suffixes.add(suffix)
                        else:
                            logger.debug(
                                f"Found model.onnx but missing model.onnx_data in {onnx_dir}. Skipping fp32."
                            )
                    else:
                        available_suffixes.add(suffix)
    except Exception as e:
        logger.error(f"Error scanning ONNX dir {onnx_dir}: {e}", exc_info=True)
        return None

    if not available_suffixes:
        logger.warning(f"No potential ONNX files found in {onnx_dir}")
        return None
    logger.debug(f"Found potential suffixes in {onnx_dir}: {available_suffixes}")

    for suffix in QUANTIZATION_SUFFIXES_PRIORITY:
        if suffix in available_suffixes:
            logger.info(f"Detected available ONNX suffix: '{suffix}' in {onnx_dir}")
            return suffix

    logger.error(f"No supported ONNX suffix found in {onnx_dir}.")
    return None


# --- GenericONNXLLM Wrapper Class (Modified) ---
class GenericONNXLLM(LLM):
    """Wraps a loaded Generic ONNX Causal LM model and tokenizer for CPU execution."""

    _model: Optional[ORTModelForCausalLM] = None
    _tokenizer: Optional[PreTrainedTokenizerBase] = None
    _model_repo_id: str
    _quant_suffix: str
    _provider: str
    _provider_options: Optional[Dict[str, Any]]

    def __init__(
        self,
        model: ORTModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        model_repo_id: str,
        quant_suffix: str,
        provider: str,
        provider_options: Optional[Dict[str, Any]],
        context_length: int,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_repo_id = model_repo_id
        self._quant_suffix = quant_suffix
        self._provider = provider
        self._provider_options = provider_options
        self._context_length = context_length

    @property
    def model_info(self) -> ModelInfo:
        return GenericONNXModelInfo(
            self._model_repo_id, self._quant_suffix, self._context_length
        )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded. Call load() first.")

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty, 
            "do_sample": temperature > 0, 
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        gen_kwargs.update(filtered_kwargs)

        logger.debug(f"Generation kwargs: {gen_kwargs}")

        try:
            # --- Apply Chat Template --- 
            logger.debug("Applying chat template...")
            # Structure the input prompt as a user message
            messages = [{"role": "user", "content": prompt}]
            # Apply the chat template defined in the tokenizer's config
            # Cast to str because tokenize=False ensures a string return type, satisfying Pyright.
            formatted_prompt_str = cast(str, self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # Don't tokenize yet, just format the string
                add_generation_prompt=True,  # Add the prompt for the assistant's turn
            ))
            logger.debug(f"Formatted prompt for LLM: '{formatted_prompt_str}'") # Log for verification
            # -------------------------

            logger.debug("Tokenizing formatted prompt...")
            # Tokenize the *formatted* string prompt
            inputs = self._tokenizer(formatted_prompt_str, return_tensors="pt") 
            if not isinstance(inputs, BatchEncoding):
                logger.error(f"Tokenizer returned unexpected type: {type(inputs)}")
                # Attempt to handle if it's just a dict containing tensors
                if isinstance(inputs, dict) and 'input_ids' in inputs and 'attention_mask' in inputs:
                    logger.warning("Tokenizer output is dict, attempting to proceed.")
                    inputs = BatchEncoding(inputs) # Wrap in BatchEncoding
                else:
                    return "Error: Tokenizer failed unexpectedly", {"error": "Tokenizer output type error"} # Return Dict
            inputs = inputs.to(self.device) # Move to device after check/fix
            logger.debug(f"Tokenization successful. Input keys: {inputs.keys()}")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

            output_ids = outputs[0]
            actual_input_len = len(input_ids[0])
            generated_ids = output_ids[actual_input_len:] 

            if self._tokenizer is None:
                raise RuntimeError("Tokenizer became unavailable before decoding.")

            result_full_text = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            out_tokens = len(generated_ids)
            logger.debug(
                f"Decoded full generated text ({out_tokens} tokens): '{result_full_text}'"
            )

            metadata = {
                "output_token_count": out_tokens,
                "input_token_count": actual_input_len,
            }
            return result_full_text, metadata

        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            return "Error: Generation failed unexpectedly", {"error": str(e)} # Return Dict

    def load(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def unload(self) -> None:
        """Unloads the model and attempts garbage collection (CPU focus)."""
        logger.info(f"Unloading GenericONNXLLM ({self.model_info.model_id})...")
        try:
            if self._model is not None:
                logger.debug(f"Deleting ONNX model ({self._model.__class__.__name__})")
                del self._model
                self._model = None
            if self._tokenizer is not None:
                logger.debug(f"Deleting tokenizer ({self._tokenizer.__class__.__name__})")
                del self._tokenizer
                self._tokenizer = None
        except Exception as e:
            logger.error(f"Error deleting refs: {e}")
        finally:
            self._model, self._tokenizer = None, None

        logger.debug("Running garbage collection...")
        gc.collect()
        logger.info("GenericONNXLLM unloaded.")


# --- Loader Function (Modified) ---
def load_onnx_llm(
    onnx_quantization: str = "auto",
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Optional[LLM]:
    """Loads a Generic ONNX Causal LM model (CPU-Only) using Optimum."""
    logger.info("--- Initializing ONNX Causal LM (CPU-Only) --- ")
    onnx_model: Optional[ORTModelForCausalLM] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    llm_instance: Optional[GenericONNXLLM] = None

    try:
        provider, options = _determine_onnx_provider("CPUExecutionProvider", None)
        logger.info(f"Using ONNX Provider: {provider}")

        paths = data_manager.get_data_paths()
        models_dir_str = paths.get("models")
        if not models_dir_str:
            raise SetupError("Models path missing.")
        model_cache_dir = Path(models_dir_str)
        active_model_dir = model_cache_dir / "active_model"
        logger.info(f"Loading from active model directory: {active_model_dir}")
        if not active_model_dir.is_dir():
            raise ModelNotFoundError(f"Active dir '{active_model_dir}' DNE.")

        suffix_to_load: Optional[str] = None
        preference_map = {p.lstrip("_"): p for p in QUANTIZATION_SUFFIXES_PRIORITY}
        preference_map["fp32"] = ""

        if onnx_quantization != "auto":
            preferred_suffix = preference_map.get(onnx_quantization)
            if preferred_suffix is not None:
                onnx_subfolder_path = active_model_dir / ONNX_SUBFOLDER
                model_filename = f"{MODEL_ONNX_BASENAME}{preferred_suffix}.onnx"
                if (onnx_subfolder_path / model_filename).exists():
                    logger.info(f"Using preferred ONNX quant: '{onnx_quantization}'")
                    suffix_to_load = preferred_suffix
                else:
                    logger.warning(
                        f"Pref quant file '{model_filename}' missing in {onnx_subfolder_path}. Detecting."
                    )
            else:
                logger.warning(f"Invalid pref '{onnx_quantization}'. Detecting.")

        if suffix_to_load is None:
            logger.info("Detecting available ONNX quantization...")
            onnx_subfolder_path = active_model_dir / ONNX_SUBFOLDER
            suffix_to_load = _detect_available_onnx_suffix(onnx_subfolder_path)
            if suffix_to_load is None:
                raise ModelNotFoundError(
                    f"No complete ONNX model found in {onnx_subfolder_path}."
                )
            loaded_quant_str = suffix_to_load.lstrip("_") if suffix_to_load else "fp32"
            logger.info(
                f"Detected/selected suffix: '{suffix_to_load}' ({loaded_quant_str})"
            )

        # Check for required files for loading (minimal set)
        missing_files = []
        for req_file_rel in ONNX_REQUIRED_LOAD_FILES:
            if not (active_model_dir / req_file_rel).exists():
                missing_files.append(req_file_rel)

        selected_model_rel_path = (
            f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{suffix_to_load}.onnx"
        )
        if not (active_model_dir / selected_model_rel_path).exists():
            missing_files.append(selected_model_rel_path)

        # Check for ONNX data file only if it's expected (fp32)
        onnx_data_file_rel = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data"
        if suffix_to_load == "" and not (active_model_dir / onnx_data_file_rel).exists():
            missing_files.append(onnx_data_file_rel)

        # Final check on collected missing files
        if missing_files:
            raise ModelNotFoundError(
                f"Required model/tokenizer files missing for suffix '{suffix_to_load}': {missing_files}. Run setup."
            )

        logger.debug(f"Loading ONNX Causal LM from {active_model_dir} using Optimum...")
        onnx_model_loaded = ORTModelForCausalLM.from_pretrained(
            active_model_dir,
            file_name=selected_model_rel_path,
            export=False,
            provider=provider,
            provider_options=options,
            use_io_binding=False,
            local_files_only=True,
            trust_remote_code=False,
        )
        onnx_model = cast(ORTModelForCausalLM, onnx_model_loaded)
        logger.debug(f"Loading tokenizer from {active_model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(
            active_model_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        logger.info(
            f"ONNX Causal LM (suffix '{suffix_to_load}') and tokenizer loaded via Optimum."
        )

        # FIX: Add check for None before instantiating
        if onnx_model is None or tokenizer is None:
            raise RuntimeError("ONNX Model or Tokenizer failed to load properly.")

        llm_instance = GenericONNXLLM(
            onnx_model,
            tokenizer,
            ONNX_MODEL_REPO_ID,
            suffix_to_load,
            provider,
            options,
            ONNX_MODEL_CONTEXT_LENGTH
        )
        return llm_instance

    except ModelNotFoundError:
        logger.error("ModelNotFoundError during LLM init.")
        raise
    except Exception as e:
        logger.error(f"Failed ONNX LLM init: {e}", exc_info=True)
        if onnx_model is not None and hasattr(onnx_model, '__del__'):
            del onnx_model
        if tokenizer is not None and hasattr(tokenizer, '__del__'):
            del tokenizer
        if llm_instance is not None:
            llm_instance.unload()
            del llm_instance

        gc.collect()
        raise RuntimeError(
            f"Failed load ONNX Causal LM ({e.__class__.__name__}): {e}"
        ) from e
