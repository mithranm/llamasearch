# src/llamasearch/core/embedder.py
"""
Embedder using SentenceTransformer strictly on CPU, configured for
all-mini-lm or other models. Includes optional dimension truncation.
Conditionally uses prompt_name based on model type.
"""

import gc
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Try importing transformers config for type checking, but don't fail if not installed
try:
    from transformers.configuration_utils import PretrainedConfig
except ImportError:
    PretrainedConfig = None  # type: ignore

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CPU_BATCH_SIZE = 16

DEVICE_TYPE = "cpu"
InputType = Literal["query", "document"]


class EmbedderConfig(BaseModel):
    model_name: str = Field(default=DEFAULT_MODEL_NAME)
    device: str = Field(default=DEVICE_TYPE, description="Device (always 'cpu')")
    max_length: int = Field(default=512, gt=0, description="Max sequence length")
    batch_size: int = Field(
        default=DEFAULT_CPU_BATCH_SIZE,
        gt=0,
        description="Batch size for encoding (CPU optimized)",
    )
    truncate_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional dimension to truncate embeddings to (e.g., 512).",
    )

    @field_validator("truncate_dim")
    @classmethod
    def check_truncate_dim(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("truncate_dim must be a positive integer if set")
        return v


class EnhancedEmbedder:
    def embed_documents(self, docs: list, show_progress: bool = True):
        return self.embed_strings(
            docs, input_type="document", show_progress=show_progress
        )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 0,
        batch_size: int = 0,
        truncate_dim: Optional[int] = None,
    ):
        config_data: Dict[str, Any] = {
            "model_name": model_name,
            "device": DEVICE_TYPE,
            "max_length": max_length if max_length > 0 else 512,
            "batch_size": batch_size if batch_size > 0 else DEFAULT_CPU_BATCH_SIZE,
            "truncate_dim": (
                truncate_dim if truncate_dim is not None and truncate_dim > 0 else None
            ),
        }

        if batch_size > 0:
            logger.info(f"Using user-provided batch size: {batch_size}")
        else:
            logger.info(f"Using default CPU batch size: {DEFAULT_CPU_BATCH_SIZE}")

        if truncate_dim is not None:
            if truncate_dim > 0:
                logger.info(
                    f"Setting embedding truncation dimension to: {truncate_dim}"
                )
            else:
                logger.warning(
                    f"Ignoring invalid truncate_dim value: {truncate_dim}. Using model default."
                )
                config_data["truncate_dim"] = None

        self.config = EmbedderConfig(**config_data)
        logger.info(f"Final Embedder Config (CPU-Only): {self.config.model_dump()}")

        try:
            paths = data_manager.get_data_paths()
            models_dir_str = paths.get("models")
            if not models_dir_str:
                raise ValueError(
                    "Models directory path not found in data manager settings."
                )
            self.models_dir = Path(models_dir_str)
            logger.info(f"Embedder using models directory: {self.models_dir}")
        except Exception as e:
            logger.error(
                f"Failed to get models directory: {e}. Using default.", exc_info=True
            )
            self.models_dir = Path(".") / ".llamasearch" / "models"
            self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SentenceTransformer] = None
        self._shutdown_event: Optional[threading.Event] = None
        logger.debug("Embedder: Calling _load_model()...")
        self._load_model()

    def set_shutdown_event(self, event: threading.Event):
        self._shutdown_event = event
        logger.debug("Shutdown event set for embedder.")

    def _load_model(self):
        logger.debug("Embedder: Entered _load_model()")
        model_name = self.config.model_name
        target_device = self.config.device
        logger.info(f"Preparing to load embedder model: {model_name} onto CPU")

        try:
            logger.debug(
                f"Checking for model '{model_name}' in cache: {self.models_dir}"
            )
            snapshot_download(
                repo_id=model_name,
                cache_dir=self.models_dir,
                local_files_only=True,
                local_dir_use_symlinks=False,
            )
            logger.info(f"Model '{model_name}' found locally in cache.")
        except (EntryNotFoundError, LocalEntryNotFoundError, FileNotFoundError):
            logger.error(
                f"Embedder model '{model_name}' not found locally. Run 'llamasearch-setup'."
            )
            raise ModelNotFoundError(
                f"Embedder model '{model_name}' not found locally."
            )
        except Exception as e:
            logger.error(
                f"Error checking model cache for '{model_name}': {e}", exc_info=True
            )
            raise ModelNotFoundError(f"Error accessing embedder model cache: {e}")

        try:
            physical_cores = os.cpu_count() or 2
            num_threads = max(1, physical_cores // 2)
            current_threads = torch.get_num_threads()
            if current_threads != num_threads:
                torch.set_num_threads(num_threads)
                logger.info(
                    f"Set torch global CPU threads from {current_threads} to {torch.get_num_threads()} (based on {physical_cores} logical cores)"
                )

            model_kwargs: Dict[str, Any] = {
                "device": target_device,
                "cache_folder": str(self.models_dir),
                "trust_remote_code": True,
            }
            if self.config.truncate_dim is not None:
                logger.info(
                    f"Applying truncation dimension: {self.config.truncate_dim}"
                )
                model_kwargs["truncate_dim"] = self.config.truncate_dim

            logger.info(
                f"Loading embedding model '{model_name}' onto CPU with kwargs: {model_kwargs}"
            )
            model = SentenceTransformer(model_name, **model_kwargs)
            model.max_seq_length = self.config.max_length
            logger.debug(f"Model max_seq_length set to {model.max_seq_length}")
            self.model = model
            logger.info("Embedder model loaded successfully on CPU.")

        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error loading embedding model '{model_name}' from cache onto CPU: {e}",
                exc_info=True,
            )
            self.model = None
            raise RuntimeError(
                f"Failed to load embedder model '{model_name}' on CPU: {e}"
            ) from e

    def _should_use_prompt_name(self, input_type: InputType) -> bool:
        return False

    def embed_strings(
        self,
        strings: List[str],
        input_type: InputType = "document",
        show_progress: bool = True,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Cannot embed strings, model is not loaded.")

        # Handle empty input correctly
        if not strings:
            emb_dim = self.get_embedding_dimension()
            # Return an empty 2D array with the correct second dimension
            return np.empty((0, emb_dim if emb_dim else 0), dtype=np.float32)

        model_max_len = self.config.max_length
        max_input_chars = model_max_len * 6
        truncated_strings = []
        num_truncated = 0
        for s in strings:
            if len(s) > max_input_chars:
                truncated_strings.append(s[:max_input_chars])
                num_truncated += 1
            else:
                truncated_strings.append(s)
        if num_truncated > 0:
            logger.warning(
                f"Truncated {num_truncated} input strings > ~{max_input_chars} chars for embedding."
            )

        all_embeddings_list = []
        interrupted = False
        input_texts = truncated_strings

        logger.debug(
            f"Starting CPU embedding of {len(input_texts)} strings (type: {input_type}) BSize: {self.config.batch_size}"
        )
        progress_bar = tqdm(
            total=len(input_texts),
            desc="Generating embeddings (CPU)",
            unit="text",
            disable=not show_progress,
        )

        base_encode_kwargs: Dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }

        try:
            for i in range(0, len(input_texts), self.config.batch_size):
                if self._shutdown_event and self._shutdown_event.is_set():
                    interrupted = True
                    logger.info("Shutdown during embedding loop.")
                    break
                batch = input_texts[i : i + self.config.batch_size]
                if not batch:  # Should not happen if input_texts is not empty
                    continue

                current_batch_encode_kwargs = base_encode_kwargs.copy()
                # Override batch_size for the current actual batch if it's smaller
                current_batch_encode_kwargs["batch_size"] = len(batch)

                batch_embeddings = self.model.encode(
                    batch, **current_batch_encode_kwargs
                )

                if batch_embeddings is not None and batch_embeddings.size > 0:
                    if batch_embeddings.dtype != np.float32:
                        batch_embeddings = batch_embeddings.astype(np.float32)
                    all_embeddings_list.append(batch_embeddings)
                    progress_bar.update(len(batch))
                else:
                    logger.warning(
                        f"Embedding batch {i // self.config.batch_size} returned empty."
                    )

            progress_bar.close()
            if interrupted:
                logger.warning("Embedding process was interrupted.")
            if not all_embeddings_list:
                logger.warning("Embedding process yielded no results.")
                emb_dim = self.get_embedding_dimension()
                return np.empty((0, emb_dim if emb_dim else 0), dtype=np.float32)

            combined = np.vstack(all_embeddings_list)
            result = np.ascontiguousarray(combined, dtype=np.float32)
        except Exception as e:
            progress_bar.close()
            if not (self._shutdown_event and self._shutdown_event.is_set()):
                if (
                    isinstance(e, ValueError)
                    and "Prompt name" in str(e)
                    and "not found" in str(e)
                ):
                    logger.error(
                        f"Model '{self.config.model_name}' likely does not support prompt_name parameter. Error: {e}",
                        exc_info=False,
                    )
                else:
                    logger.error(
                        f"Error during CPU encode (type: {input_type}): {e}",
                        exc_info=True,
                    )
            emb_dim = self.get_embedding_dimension()
            return np.empty((0, emb_dim if emb_dim else 0), dtype=np.float32)
        finally:
            self._try_gc()

        processed_count = result.shape[0]
        expected_count = len(truncated_strings) if not interrupted else processed_count
        final_dim = (
            result.shape[1] if result.ndim == 2 and result.shape[1] > 0 else "N/A"
        )
        if processed_count != expected_count:
            logger.warning(
                f"Expected {expected_count} embeddings, got {processed_count}."
            )
        else:
            logger.info(
                f"Generated {result.shape[0]} embeddings (CPU, {input_type}) dim {final_dim}{' (interrupted)' if interrupted else ''}."
            )
        return result

    def _try_gc(self):
        gc.collect()
        try:
            if (
                torch.backends.mps.is_available()
            ):  # Should always be False as we force CPU
                torch.mps.empty_cache()
        except AttributeError:
            pass  # mps not available

    def embed_string(
        self, text: str, input_type: InputType = "document"
    ) -> Optional[np.ndarray]:
        if self.model is None:
            logger.error("Cannot embed string, model not loaded.")
            return None
        if not text:
            logger.warning("Attempted to embed an empty string.")
            return None
        if self._shutdown_event and self._shutdown_event.is_set():
            logger.debug("Shutdown detected in embed_string.")
            return None

        model_max_len = self.config.max_length
        max_input_chars = model_max_len * 6
        if len(text) > max_input_chars:
            text = text[:max_input_chars]
            logger.debug("Truncated single string for embedding.")

        encode_kwargs: Dict[str, Any] = {
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }

        try:
            input_text = text
            embedding = self.model.encode(input_text, **encode_kwargs)
            if embedding is not None:
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
                if embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)
                return embedding[0] if embedding.shape[0] == 1 else embedding
            else:
                logger.warning(f"Model returned None embedding for: '{text[:50]}...'")
                return None
        except Exception as e:
            if not (self._shutdown_event and self._shutdown_event.is_set()):
                if (
                    isinstance(e, ValueError)
                    and "Prompt name" in str(e)
                    and "not found" in str(e)
                ):
                    logger.error(
                        f"Model '{self.config.model_name}' likely does not support prompt_name parameter. Error: {e}",
                        exc_info=False,
                    )
                else:
                    logger.error(
                        f"Error embedding single string ({input_type}) on CPU: {e}",
                        exc_info=True,
                    )
            return None

    def get_embedding_dimension(self) -> Optional[int]:
        if self.model is None:
            logger.warning("Cannot get embedding dimension, no model loaded.")
            return None
        if self.config.truncate_dim is not None:
            logger.debug(f"Using configured truncate_dim: {self.config.truncate_dim}")
            return self.config.truncate_dim
        try:
            get_dim_method = getattr(
                self.model, "get_sentence_embedding_dimension", None
            )
            if callable(get_dim_method):
                dimension = get_dim_method()
                if isinstance(dimension, int) and dimension > 0:
                    logger.debug(
                        f"Using model's default dimension via method: {dimension}"
                    )
                    return dimension
                else:
                    logger.warning(
                        f"Method 'get_sentence_embedding_dimension' returned non-positive or non-integer: {dimension}"
                    )

            model_config: Any = getattr(self.model, "config", None)
            if model_config is not None:
                if isinstance(model_config, dict):
                    dimension = model_config.get("hidden_size") or model_config.get(
                        "d_model"
                    )
                    if isinstance(dimension, int) and dimension > 0:
                        logger.debug(
                            f"Using dimension from model config dict: {dimension}"
                        )
                        return dimension
                elif PretrainedConfig is not None and isinstance(
                    model_config, PretrainedConfig
                ):
                    if (
                        hasattr(model_config, "hidden_size")
                        and isinstance(getattr(model_config, "hidden_size", None), int)
                        and model_config.hidden_size > 0
                    ):
                        dimension = model_config.hidden_size
                        logger.debug(
                            f"Using dimension from PretrainedConfig attribute 'hidden_size': {dimension}"
                        )
                        return dimension
                    elif (
                        hasattr(model_config, "d_model")
                        and isinstance(getattr(model_config, "d_model", None), int)
                        and model_config.d_model > 0
                    ):
                        dimension = model_config.d_model
                        logger.debug(
                            f"Using dimension from PretrainedConfig attribute 'd_model': {dimension}"
                        )
                        return dimension
                else:
                    logger.debug(
                        f"Model config found (type: {type(model_config)}), but dimension attribute not found or invalid."
                    )

            direct_dim = (
                getattr(self.model, "_embedding_size", None)
                or getattr(self.model, "embedding_dim", None)
                or getattr(self.model, "dim", None)
                or getattr(self.model, "output_embedding_dimension", None)
            )
            if isinstance(direct_dim, int) and direct_dim > 0:
                logger.debug(
                    f"Using dimension from direct model attribute: {direct_dim}"
                )
                return direct_dim

            logger.warning(
                f"Could not determine embedding dimension for model {self.config.model_name}."
            )
            return None

        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}", exc_info=True)
            return None

    def close(self):
        logger.info("Closing Embedder and releasing CPU resources...")
        if self._shutdown_event and not self._shutdown_event.is_set():
            self._shutdown_event.set()
        if self.model is not None:
            try:
                del self.model
                self.model = None
                logger.debug("Embedder model deleted (CPU).")
            except Exception as e:
                logger.warning(f"Error deleting model during close: {e}")
        self._try_gc()
        logger.info("Embedder closed.")
