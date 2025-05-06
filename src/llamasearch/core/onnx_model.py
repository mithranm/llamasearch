# src/llamasearch/core/onnx_model.py

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import onnxruntime
import torch
# Removed ORTModelForCausalLM import as we only mock it
from optimum.onnxruntime import ORTModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.protocols import LLM, ModelInfo
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

# --- Constants --- (Updated for Llama-3.2-1B, FP32 ONLY)
ONNX_MODEL_REPO_ID = "onnx-community/Llama-3.2-1B-Instruct"
ONNX_SUBFOLDER = "onnx"
MODEL_ONNX_BASENAME = "model"  # Always use the base name
# Removed QUANTIZATION_SUFFIXES_PRIORITY
ONNX_MODEL_CONTEXT_LENGTH = 128000  # Context length for Llama-3.2-1B
ONNX_REQUIRED_LOAD_FILES = [
    "config.json",
    "tokenizer.json",
]

# Files to download and copy during setup
_ONNX_ORIGINAL_BASE_FILES_ = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


# --- GenericONNXModelInfo Class ---
class GenericONNXModelInfo(ModelInfo):
    """Implementation of ModelInfo protocol for Generic ONNX Causal LMs (FP32 only)."""

    def __init__(self, model_repo_id: str, context_len: int):
        self._model_repo_id = model_repo_id
        self._context_len = context_len

    @property
    def model_id(self) -> str:
        base_name = self._model_repo_id.split("/")[-1]
        return f"{base_name}-onnx-fp32"  # Hardcoded FP32

    @property
    def model_engine(self) -> str:
        return "onnx_causal"

    @property
    def description(self) -> str:
        return f"Generic ONNX Causal LM ({self._model_repo_id}, fp32)"  # Hardcoded FP32

    @property
    def context_length(self) -> int:
        return self._context_len


# --- Helper functions ---
def _determine_onnx_provider() -> Tuple[str, Optional[Dict[str, Any]]]:
    """Determines the ONNX Runtime provider. FORCES CPU."""
    available_providers = onnxruntime.get_available_providers()
    logger.debug(f"Available ONNX Runtime providers: {available_providers}")

    selected_provider = "CPUExecutionProvider"

    if "CPUExecutionProvider" not in available_providers:
        logger.critical("FATAL: CPUExecutionProvider not found in ONNX Runtime!")
        raise RuntimeError("ONNX Runtime CPU provider is missing.")

    logger.info(f"Using ONNX provider: {selected_provider}")
    return selected_provider, None


def _check_onnx_files_exist(onnx_dir: Path) -> bool:
    """Checks if the required model.onnx and model.onnx_data files exist."""
    if not onnx_dir.is_dir():
        logger.error(f"ONNX directory does not exist: {onnx_dir}")
        return False

    model_file = onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx"
    data_file = onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data"

    model_exists = model_file.is_file()
    data_exists = data_file.is_file()

    if model_exists and data_exists:
        logger.debug(
            f"Found required ONNX files: {model_file.name} and {data_file.name}"
        )
        return True
    else:
        if not model_exists:
            logger.warning(f"Required file missing: {model_file}")
        if not data_exists:
            logger.warning(f"Required file missing: {data_file}")
        return False


# --- GenericONNXLLM Wrapper Class ---
class GenericONNXLLM(LLM):
    """Wraps a loaded Generic ONNX Causal LM model and tokenizer for CPU execution."""

    _model: Optional[ORTModelForCausalLM] = None
    _tokenizer: Optional[PreTrainedTokenizerBase] = None
    _model_repo_id: str
    _provider: str
    _provider_options: Optional[Dict[str, Any]]
    _context_length: int

    def __init__(
        self,
        model: ORTModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        model_repo_id: str,
        provider: str,
        provider_options: Optional[Dict[str, Any]],
        context_length: int,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_repo_id = model_repo_id
        self._provider = provider
        self._provider_options = provider_options
        self._context_length = context_length

    @property
    def model_info(self) -> ModelInfo:
        # Removed quant_suffix
        return GenericONNXModelInfo(self._model_repo_id, self._context_length)

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

        # Determine EOS token ID safely
        tokenizer_eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if tokenizer_eos_token_id is None:
            model_config = getattr(self._model, "config", None)
            if model_config:
                tokenizer_eos_token_id = getattr(model_config, "eos_token_id", None)
            if tokenizer_eos_token_id is None:
                logger.warning(
                    "Could not determine eos_token_id for tokenizer. Generation might behave unexpectedly."
                )

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repeat_penalty,
            "do_sample": temperature > 0, # Sample if temperature > 0
            **(
                {"pad_token_id": tokenizer_eos_token_id}
                if tokenizer_eos_token_id is not None
                else {}
            ),
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        gen_kwargs.update(filtered_kwargs)
        logger.debug(f"Generation kwargs: {gen_kwargs}")

        try:
            logger.debug("Applying chat template...")
            messages = [{"role": "user", "content": prompt}]
            # Ensure tokenizer is callable
            if not callable(self._tokenizer.apply_chat_template):
                 raise TypeError("Tokenizer's apply_chat_template is not callable.")
            if not callable(self._tokenizer):
                 raise TypeError("Tokenizer is not callable for input tokenization.")

            formatted_prompt_str = cast(
                str,
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )
            logger.debug(f"Formatted prompt for LLM: '{formatted_prompt_str}'")

            logger.debug("Tokenizing formatted prompt...")
            # Pass explicitly to CPU
            inputs = self._tokenizer(formatted_prompt_str, return_tensors="pt")
            
            # Check if inputs has 'to' method before calling
            if hasattr(inputs, "to") and callable(inputs.to):
                inputs = inputs.to(self.device)
            else:
                # This case can happen if tokenizer returns something unexpected (e.g. not BatchEncoding)
                # Or if the returned object doesn't have a 'to' method (e.g. already on CPU, or a simple dict)
                logger.debug(f"Tokenizer output (type: {type(inputs)}) does not have 'to' method or is not callable, assuming CPU or correct device.")


            if (
                not isinstance(inputs, BatchEncoding)
                or not hasattr(inputs, "input_ids")
                or not hasattr(inputs, "attention_mask")
            ):
                logger.error(
                    f"Tokenizer returned unexpected type or structure: {type(inputs)}"
                )
                return "Error: Tokenizer failed unexpectedly", {
                    "error": "Tokenizer output type/structure error"
                }

            logger.debug(f"Tokenization successful. Input keys: {list(inputs.keys())}") # Use list() for safety
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Ensure model is callable
            if not callable(self._model.generate):
                raise TypeError("Model's generate method is not callable.")

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
                )

            if isinstance(outputs, torch.Tensor):
                output_ids_tensor = outputs
            elif (
                isinstance(outputs, (list, tuple))
                and len(outputs) > 0
                and isinstance(outputs[0], torch.Tensor)
            ):
                logger.warning(
                    "Model output was wrapped in a sequence, using first element."
                )
                output_ids_tensor = outputs[0]
            else:
                logger.error(
                    f"Unexpected output type from model.generate: {type(outputs)}"
                )
                return "Error: Unexpected output format from LLM", {
                    "error": f"Unexpected LLM output type {type(outputs)}"
                }

            actual_input_len = input_ids.shape[1]
            if (
                output_ids_tensor.ndim < 2
                or output_ids_tensor.shape[1] <= actual_input_len
            ):
                logger.warning(
                    "LLM generated no new tokens or output tensor shape invalid."
                )
                generated_ids = torch.tensor([], dtype=torch.long, device=self.device)
            else:
                generated_ids = output_ids_tensor[0, actual_input_len:]

            if self._tokenizer is None: # Should not happen given initial check
                raise RuntimeError("Tokenizer became unavailable before decoding.")
            if not callable(self._tokenizer.decode):
                raise TypeError("Tokenizer's decode method is not callable.")


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
        logger.info(f"Unloading GenericONNXLLM ({self.model_info.model_id})...")
        try:
            if self._model is not None:
                logger.debug(f"Deleting ONNX model ({self._model.__class__.__name__})")
                del self._model
                self._model = None
            if self._tokenizer is not None:
                logger.debug(
                    f"Deleting tokenizer ({self._tokenizer.__class__.__name__})"
                )
                del self._tokenizer
                self._tokenizer = None
        except Exception as e:
            logger.error(f"Error deleting model/tokenizer references: {e}")
        finally:
            self._model, self._tokenizer = None, None
        logger.debug("Running garbage collection...")
        gc.collect()
        logger.info("GenericONNXLLM unloaded.")


# --- Loader Function ---
def load_onnx_llm() -> Optional[LLM]:
    """Loads the base Generic ONNX Causal LM model (CPU-Only) using Optimum."""
    logger.info("--- Initializing ONNX Causal LM (CPU-Only, FP32) --- ")
    onnx_model: Optional[ORTModelForCausalLM] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    llm_instance: Optional[GenericONNXLLM] = None
    active_model_dir: Optional[Path] = None

    try:
        provider, options = _determine_onnx_provider()
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

        onnx_subfolder_path = active_model_dir / ONNX_SUBFOLDER

        # Check if the required base ONNX files exist
        if not _check_onnx_files_exist(onnx_subfolder_path):
            raise ModelNotFoundError(
                f"Required ONNX files (model.onnx and model.onnx_data) not found in {onnx_subfolder_path}. Run setup."
            )

        missing_root_files = []
        for req_file_rel in ONNX_REQUIRED_LOAD_FILES:
            if not (active_model_dir / req_file_rel).exists():
                missing_root_files.append(req_file_rel)

        if missing_root_files:
            raise ModelNotFoundError(
                f"Required root files missing: {missing_root_files}. Run setup."
            )

        selected_model_rel_path = f"{ONNX_SUBFOLDER}/{MODEL_ONNX_BASENAME}.onnx"
        logger.info(f"Loading ONNX model: {selected_model_rel_path}")

        logger.debug(f"Loading ONNX Causal LM from {active_model_dir} using Optimum...")
        onnx_model_loaded = ORTModelForCausalLM.from_pretrained(
            active_model_dir,
            file_name=selected_model_rel_path,  # Load base model
            export=False,
            provider=provider,
            provider_options=options,
            use_io_binding=False,
            local_files_only=True,
            trust_remote_code=False,  # Explicitly set to False
        )
        if not isinstance(onnx_model_loaded, ORTModelForCausalLM):
            raise TypeError(
                f"Expected ORTModelForCausalLM, got {type(onnx_model_loaded)}"
            )
        onnx_model = onnx_model_loaded

        logger.debug(f"Loading tokenizer from {active_model_dir}...")
        tokenizer_loaded = AutoTokenizer.from_pretrained(
            active_model_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,  # Explicitly set to False
        )
        if not isinstance(tokenizer_loaded, PreTrainedTokenizerBase):
            raise TypeError(
                f"Expected PreTrainedTokenizerBase, got {type(tokenizer_loaded)}"
            )
        tokenizer = tokenizer_loaded

        logger.info("ONNX Causal LM (FP32) and tokenizer loaded via Optimum.")

        llm_instance = GenericONNXLLM(
            onnx_model,
            tokenizer,
            ONNX_MODEL_REPO_ID,
            provider,
            options,
            ONNX_MODEL_CONTEXT_LENGTH,
        )
        return llm_instance

    except ModelNotFoundError as e:
        logger.error(f"ModelNotFoundError during LLM init: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed ONNX LLM init: {e}", exc_info=True)
        # Cleanup attempts
        if onnx_model is not None and hasattr(onnx_model, "__del__"):
            try:
                del onnx_model
            except Exception:
                pass
        if tokenizer is not None and hasattr(tokenizer, "__del__"):
            try:
                del tokenizer
            except Exception:
                pass
        if llm_instance is not None:
            try:
                llm_instance.unload()
            except Exception:
                pass
            try:
                del llm_instance
            except Exception:
                pass
        gc.collect()
        raise RuntimeError(
            f"Failed load ONNX Causal LM ({e.__class__.__name__}): {e}"
        ) from e