# src/llamasearch/core/onnx_model.py (CPU-Only Refactor with Llama-3.2 & Pyright/Ruff Fixes)

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
# <<< REMOVED hardware imports as _select_onnx_quantization is removed >>>
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


# --- Helper functions ---
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
    # Always return None for options for CPU provider currently
    return selected_provider, None


def _detect_available_onnx_suffix(onnx_dir: Path) -> Optional[str]:
    """Detects the best available ONNX suffix based on files in onnx_dir."""
    if not onnx_dir.is_dir():
        logger.error(f"ONNX directory does not exist: {onnx_dir}")
        return None

    available_suffixes = set()
    try:
        for item in onnx_dir.glob(f"{MODEL_ONNX_BASENAME}*.onnx"):
            # Make sure it's actually a file before proceeding
            if not item.is_file(): # Added check
                continue
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
        return None # Return None on error during scan

    if not available_suffixes:
        logger.warning(f"No potential ONNX files found in {onnx_dir}")
        return None
    logger.debug(f"Found potential suffixes in {onnx_dir}: {available_suffixes}")

    for suffix in QUANTIZATION_SUFFIXES_PRIORITY:
        if suffix in available_suffixes:
            logger.info(f"Detected available ONNX suffix: '{suffix}' in {onnx_dir}")
            return suffix

    # This part is reached only if files were found, but none matched the priority list
    logger.error(f"No supported ONNX suffix found among available files in {onnx_dir}.")
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
    _context_length: int # Renamed from context_length to avoid conflict with property

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

        # Ensure tokenizer has eos_token_id attribute before using it
        tokenizer_eos_token_id = getattr(self._tokenizer, 'eos_token_id', None)
        if tokenizer_eos_token_id is None:
             # Attempt to get it from config if not directly on tokenizer
             model_config = getattr(self._model, 'config', None)
             if model_config:
                tokenizer_eos_token_id = getattr(model_config, 'eos_token_id', None)

             if tokenizer_eos_token_id is None:
                 logger.warning("Could not determine eos_token_id for tokenizer. Generation might behave unexpectedly.")
                 # Provide a default or raise error depending on desired behavior
                 # For now, let's try proceeding without setting pad_token_id explicitly
                 # raise ValueError("EOS token ID not found on tokenizer or model config.")

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty,
            "do_sample": temperature > 0,
            # Set pad_token_id only if found
            **({"pad_token_id": tokenizer_eos_token_id} if tokenizer_eos_token_id is not None else {})
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        gen_kwargs.update(filtered_kwargs)

        logger.debug(f"Generation kwargs: {gen_kwargs}")

        try:
            # --- Apply Chat Template ---
            logger.debug("Applying chat template...")
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt_str = cast(str, self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ))
            logger.debug(f"Formatted prompt for LLM: '{formatted_prompt_str}'")
            # -------------------------

            logger.debug("Tokenizing formatted prompt...")
            inputs = self._tokenizer(formatted_prompt_str, return_tensors="pt")
            # Handle different return types from tokenizer robustly
            if isinstance(inputs, dict):
                 if 'input_ids' in inputs and 'attention_mask' in inputs:
                     logger.debug("Tokenizer returned dict, wrapping in BatchEncoding.")
                     inputs = BatchEncoding(inputs)
                 else:
                     logger.error(f"Tokenizer returned dict missing required keys: {inputs.keys()}")
                     return "Error: Tokenizer failed (missing keys)", {"error": "Tokenizer output dict missing keys"}
            elif not isinstance(inputs, BatchEncoding):
                 logger.error(f"Tokenizer returned unexpected type: {type(inputs)}")
                 return "Error: Tokenizer failed unexpectedly", {"error": "Tokenizer output type error"}

            inputs = inputs.to(self.device)
            logger.debug(f"Tokenization successful. Input keys: {inputs.keys()}")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

            # Check output type
            if isinstance(outputs, torch.Tensor):
                output_ids_tensor = outputs
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                logger.warning("Model output was wrapped in a sequence, using first element.")
                output_ids_tensor = outputs[0]
            else:
                logger.error(f"Unexpected output type from model.generate: {type(outputs)}")
                return "Error: Unexpected output format from LLM", {"error": f"Unexpected LLM output type {type(outputs)}"}

            # Slice generated part
            actual_input_len = input_ids.shape[1]
            if output_ids_tensor.ndim < 2 or output_ids_tensor.shape[1] <= actual_input_len: # Check ndim too
                logger.warning("LLM generated no new tokens or output tensor shape invalid.")
                generated_ids = torch.tensor([], dtype=torch.long, device=self.device) # Empty tensor on correct device
            else:
                generated_ids = output_ids_tensor[0, actual_input_len:]

            if self._tokenizer is None:
                # This check remains, though unlikely to trigger if initial check passed
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
            return f"LLM Error: {e}", {"error": str(e)}

    def load(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def unload(self) -> None:
        """Unloads the model and attempts garbage collection (CPU focus)."""
        logger.info(f"Unloading GenericONNXLLM ({self.model_info.model_id})...")
        try:
            if self._model is not None:
                logger.debug(f"Deleting ONNX model ({self._model.__class__.__name__})")
                # Explicitly delete to help GC, especially with complex objects
                del self._model
                self._model = None
            if self._tokenizer is not None:
                logger.debug(f"Deleting tokenizer ({self._tokenizer.__class__.__name__})")
                del self._tokenizer
                self._tokenizer = None
        except Exception as e:
            logger.error(f"Error deleting model/tokenizer references: {e}") # Error deleting refs
        finally:
            # Ensure they are None even if deletion failed
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
    active_model_dir : Optional[Path] = None # Track active dir for cleanup

    try:
        # Determine provider (forcing CPU)
        # <<< FIX: Pass arguments correctly >>>
        provider, options = _determine_onnx_provider(preferred_provider, preferred_options)
        logger.info(f"Using ONNX Provider: {provider}")

        # Get model directory path
        paths = data_manager.get_data_paths()
        models_dir_str = paths.get("models")
        if not models_dir_str:
            raise SetupError("Models path missing.")
        model_cache_dir = Path(models_dir_str)
        active_model_dir = model_cache_dir / "active_model" # Assign to tracked var
        logger.info(f"Loading from active model directory: {active_model_dir}")
        if not active_model_dir.is_dir():
            raise ModelNotFoundError(f"Active dir '{active_model_dir}' DNE.")

        onnx_subfolder_path = active_model_dir / ONNX_SUBFOLDER
        suffix_to_load: Optional[str] = None
        preference_map = {p.lstrip("_"): p for p in QUANTIZATION_SUFFIXES_PRIORITY}
        preference_map["fp32"] = ""

        # Handle user preference for quantization
        if onnx_quantization != "auto":
            preferred_suffix = preference_map.get(onnx_quantization)
            if preferred_suffix is not None:
                model_filename = f"{MODEL_ONNX_BASENAME}{preferred_suffix}.onnx"
                # Check if the *specific preferred* model file exists
                if (onnx_subfolder_path / model_filename).exists():
                    # Additionally check for .onnx_data if fp32 is preferred
                    if preferred_suffix == "" and not (onnx_subfolder_path / f"{MODEL_ONNX_BASENAME}.onnx_data").exists():
                         logger.warning(f"Preferred FP32 ONNX model '{model_filename}' found, but missing data file. Detecting best available.")
                    else:
                         logger.info(f"Using preferred ONNX quant: '{onnx_quantization}' (suffix '{preferred_suffix}')")
                         suffix_to_load = preferred_suffix
                else:
                    logger.warning(
                        f"Preferred ONNX file '{model_filename}' missing in {onnx_subfolder_path}. Detecting best available."
                    )
            else:
                logger.warning(f"Invalid preference '{onnx_quantization}'. Detecting best available.")

        # Detect best available suffix if no valid preference was found/used
        if suffix_to_load is None:
            logger.info("Detecting best available ONNX quantization...")
            suffix_to_load = _detect_available_onnx_suffix(onnx_subfolder_path)
            if suffix_to_load is None:
                raise ModelNotFoundError(
                    f"No complete ONNX model found in {onnx_subfolder_path}."
                )
            loaded_quant_str = suffix_to_load.lstrip("_") if suffix_to_load else "fp32"
            logger.info(
                f"Detected/selected suffix: '{suffix_to_load}' ({loaded_quant_str})"
            )

        # Check for required files for loading (tokenizer, config)
        missing_files = []
        for req_file_rel in ONNX_REQUIRED_LOAD_FILES:
            if not (active_model_dir / req_file_rel).exists():
                missing_files.append(req_file_rel)

        # Check for the selected ONNX model file
        selected_model_rel_path = (
            f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}{suffix_to_load}.onnx"
        )
        if not (active_model_dir / selected_model_rel_path).exists():
            missing_files.append(selected_model_rel_path)

        # Check for ONNX data file *only* if fp32 was selected
        onnx_data_file_rel = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx_data"
        if suffix_to_load == "" and not (active_model_dir / onnx_data_file_rel).exists():
            missing_files.append(onnx_data_file_rel)

        # Final check on collected missing files
        if missing_files:
            raise ModelNotFoundError(
                f"Required model/tokenizer files missing for suffix '{suffix_to_load}': {missing_files}. Run setup."
            )

        # --- Load Model and Tokenizer ---
        logger.debug(f"Loading ONNX Causal LM from {active_model_dir} using Optimum...")
        onnx_model_loaded = ORTModelForCausalLM.from_pretrained(
            active_model_dir,
            file_name=selected_model_rel_path, # Use relative path within active_model_dir
            export=False, # Already exported
            provider=provider,
            provider_options=options, # Pass determined options (likely None for CPU)
            use_io_binding=False, # Generally False for CPU
            local_files_only=True, # Must be local
            trust_remote_code=False, # Safer default
        )
        # Ensure the loaded object is of the expected type
        if not isinstance(onnx_model_loaded, ORTModelForCausalLM):
             raise TypeError(f"Expected ORTModelForCausalLM, got {type(onnx_model_loaded)}")
        onnx_model = onnx_model_loaded # Assign after type check

        logger.debug(f"Loading tokenizer from {active_model_dir}...")
        tokenizer_loaded = AutoTokenizer.from_pretrained(
            active_model_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False, # Safer default
        )
        if not isinstance(tokenizer_loaded, PreTrainedTokenizerBase):
             raise TypeError(f"Expected PreTrainedTokenizerBase, got {type(tokenizer_loaded)}")
        tokenizer = tokenizer_loaded # Assign after type check

        logger.info(
            f"ONNX Causal LM (suffix '{suffix_to_load}') and tokenizer loaded via Optimum."
        )

        # Create LLM instance (already checked model/tokenizer are not None implicitly by loaders)
        llm_instance = GenericONNXLLM(
            onnx_model,
            tokenizer,
            ONNX_MODEL_REPO_ID, # Store the base repo ID
            suffix_to_load, # Store the specific suffix loaded
            provider,
            options,
            ONNX_MODEL_CONTEXT_LENGTH # Store context length
        )
        return llm_instance

    except ModelNotFoundError as e:
        logger.error(f"ModelNotFoundError during LLM init: {e}")
        # Reraise specific error for setup/calling code to handle
        raise
    except Exception as e:
        logger.error(f"Failed ONNX LLM init: {e}", exc_info=True)
        # --- Cleanup partially loaded resources on error ---
        if onnx_model is not None and hasattr(onnx_model, '__del__'):
            try:
                del onnx_model
            except Exception as del_e:
                 logger.warning(f"Error deleting loaded ONNX model during cleanup: {del_e}")
        if tokenizer is not None and hasattr(tokenizer, '__del__'):
            try:
                 del tokenizer
            except Exception as del_e:
                 logger.warning(f"Error deleting loaded tokenizer during cleanup: {del_e}")
        if llm_instance is not None:
            # Unload potentially partially created instance
            llm_instance.unload()
            try:
                 del llm_instance
            except Exception as del_e:
                 logger.warning(f"Error deleting LLM instance during cleanup: {del_e}")

        gc.collect() # Attempt garbage collection
        # Reraise a generic error indicating load failure
        raise RuntimeError(
            f"Failed load ONNX Causal LM ({e.__class__.__name__}): {e}"
        ) from e