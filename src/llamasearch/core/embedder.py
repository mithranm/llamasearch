# src/llamasearch/core/embedder.py
"""
Embedder using SentenceTransformer strictly on CPU, configured for
mixedbread-ai/mxbai-embed-large-v1. Includes optional dimension truncation.
"""

import gc
import os
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError
from llamasearch.hardware import HardwareInfo, detect_hardware_info  # CPU/Mem only
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

# --- Updated Default Model ---
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# --- End Update ---

DEVICE_TYPE = "cpu"  # Hardcode to CPU
InputType = Literal["query", "document"]
MXBAI_QUERY_PROMPT_NAME = "query"  # Standard prompt name for mxbai queries


class EmbedderConfig(BaseModel):
    """Configuration for the CPU-only embedder using mxbai-embed-large-v1."""

    model_name: str = Field(default=DEFAULT_MODEL_NAME)
    device: str = Field(default=DEVICE_TYPE, description="Device (always 'cpu')")
    max_length: int = Field(default=512, gt=0, description="Max sequence length")
    batch_size: int = Field(
        default=32, gt=0, description="Batch size for encoding (CPU optimized)"
    )
    # --- Added truncate_dim ---
    truncate_dim: Optional[int] = Field(
        default=None,
        gt=0,  # Must be positive if set
        description="Optional dimension to truncate embeddings to (e.g., 512).",
    )
    # --- Removed cpu_config_kwargs and query_prompt_name (using standard 'query') ---

    @validator("truncate_dim")
    def check_truncate_dim(cls, v):
        # Optional validation if needed, e.g., max value based on model
        if v is not None and v <= 0:
            raise ValueError("truncate_dim must be a positive integer if set")
        return v

    @classmethod
    def from_hardware(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbedderConfig":
        """Create a CPU-optimized configuration based on RAM."""
        hw: HardwareInfo = detect_hardware_info()  # Gets CPU/Mem info
        config_data: Dict[str, Any] = {
            "model_name": model_name,
            "device": DEVICE_TYPE,
            # Keep defaults for other fields unless overridden later
        }

        # Set batch size based on available RAM
        ram_gb = hw.memory.available_gb
        if ram_gb > 30:
            config_data["batch_size"] = 64
        elif ram_gb > 15:
            config_data["batch_size"] = 32
        elif ram_gb > 7:
            config_data["batch_size"] = 16
        else:
            config_data["batch_size"] = 8
        logger.info(
            f"Available RAM: {ram_gb:.1f} GB. Auto CPU batch size: {config_data['batch_size']}."
        )

        final_config = cls(**config_data)
        logger.info(f"CPU-Optimized EmbedderConfig: {final_config.dict()}")
        return final_config


class EnhancedEmbedder:
    """Embedder optimized for CPU using mxbai-embed-large-v1."""

    def embed_documents(self, docs: list, show_progress: bool = True):
        """Alias for embed_strings with input_type='document'."""
        return self.embed_strings(
            docs, input_type="document", show_progress=show_progress
        )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 0,
        batch_size: int = 0,
        truncate_dim: Optional[int] = None,  # Allow user override
    ):
        base_config = EmbedderConfig.from_hardware(model_name)
        config_data: Dict[str, Any] = base_config.dict()

        # Apply user overrides
        if model_name != DEFAULT_MODEL_NAME:
            config_data["model_name"] = model_name
        if max_length > 0:
            config_data["max_length"] = max_length
        if batch_size > 0:
            logger.info(
                f"Overriding auto batch size ({config_data['batch_size']}) with user value: {batch_size}"
            )
            config_data["batch_size"] = batch_size
        if truncate_dim is not None:  # Allow setting truncate_dim
            if truncate_dim > 0:
                logger.info(
                    f"Setting embedding truncation dimension to: {truncate_dim}"
                )
                config_data["truncate_dim"] = truncate_dim
            else:
                logger.warning(
                    f"Ignoring invalid truncate_dim value: {truncate_dim}. Using model default."
                )
                config_data["truncate_dim"] = None  # Ensure it's None if invalid

        self.config = EmbedderConfig(**config_data)
        logger.info(f"Final Embedder Config (CPU-Only): {self.config.dict()}")

        try:
            self.models_dir = Path(data_manager.get_data_paths()["models"])
            logger.info(f"Embedder using models directory: {self.models_dir}")
        except Exception as e:
            logger.error(
                f"Failed to get models directory: {e}. Using default.", exc_info=True
            )
            self.models_dir = Path(".") / ".llamasearch" / "models"
            self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[SentenceTransformer] = None
        self._shutdown_event: Optional[threading.Event] = None
        self._load_model()

    def set_shutdown_event(self, event: threading.Event):
        self._shutdown_event = event
        logger.debug("Shutdown event set for embedder.")

    def _load_model(self):
        """Load the embedding model onto the CPU, with optional truncation."""
        model_name = self.config.model_name
        target_device = self.config.device  # Always "cpu"
        logger.info(f"Preparing to load embedder model: {model_name} onto CPU")

        # Model cache check
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
        except (EntryNotFoundError, LocalEntryNotFoundError):
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
            # Set reasonable torch threads for CPU
            hw = detect_hardware_info()
            physical_cores = (
                hw.cpu.physical_cores
                if hw.cpu.physical_cores
                else (os.cpu_count() or 2)
            )
            num_threads = max(1, physical_cores // 2)
            current_threads = torch.get_num_threads()
            if current_threads != num_threads:
                torch.set_num_threads(num_threads)
                logger.info(
                    f"Set torch global CPU threads from {current_threads} to {torch.get_num_threads()}"
                )

            # Prepare kwargs for SentenceTransformer initialization
            model_kwargs: Dict[str, Any] = {
                "device": target_device,  # cpu
                "cache_folder": str(self.models_dir),
                "trust_remote_code": True,  # May or may not be needed by mxbai, but safe to include
            }
            # --- Add truncate_dim if specified ---
            if self.config.truncate_dim is not None:
                logger.info(
                    f"Applying truncation dimension: {self.config.truncate_dim}"
                )
                model_kwargs["truncate_dim"] = self.config.truncate_dim
            # --- End truncate_dim ---

            # --- Removed CPU compatibility flags ---

            logger.info(
                f"Loading embedding model '{model_name}' onto CPU with kwargs: {model_kwargs}"
            )
            model = SentenceTransformer(model_name, **model_kwargs)

            # Set max_length after loading
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

    def embed_strings(
        self,
        strings: List[str],
        input_type: InputType = "document",
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings on CPU, using prompt_name="query" for queries."""
        if self.model is None:
            raise RuntimeError("Cannot embed strings, model is not loaded.")
        if not strings:
            return np.array([], dtype=np.float32)

        # Truncation check
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
                f"Truncated {num_truncated} strings > ~{max_input_chars} chars."
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

        # Prepare encode kwargs based on input type
        encode_kwargs: Dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }  # Normalize is often default, but explicit is fine
        if input_type == "query":
            logger.debug(f"Using query prompt_name: '{MXBAI_QUERY_PROMPT_NAME}'")
            encode_kwargs["prompt_name"] = (
                MXBAI_QUERY_PROMPT_NAME  # Use standard name for mxbai
            )

        try:
            # Iterate manually for shutdown check
            for i in range(0, len(input_texts), self.config.batch_size):
                if self._shutdown_event and self._shutdown_event.is_set():
                    interrupted = True
                    logger.info("Shutdown during embedding loop.")
                    break
                batch = input_texts[i : i + self.config.batch_size]
                if not batch:
                    continue

                current_batch_encode_kwargs = encode_kwargs.copy()
                current_batch_encode_kwargs["batch_size"] = len(
                    batch
                )  # Use actual batch size

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
                logger.error(
                    f"Error during CPU encode (type: {input_type}): {e}", exc_info=True
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
        """Try to run garbage collection."""
        gc.collect()
        try:  # Check MPS just in case
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except AttributeError:
            pass

    def embed_string(
        self, text: str, input_type: InputType = "document"
    ) -> Optional[np.ndarray]:
        """Embed a single string on CPU, using prompt_name="query" for queries."""
        if self.model is None:
            logger.error("Cannot embed string, model not loaded.")
            return None
        if not text:
            logger.warning("Attempted to embed an empty string.")
            return None
        if self._shutdown_event and self._shutdown_event.is_set():
            logger.debug("Shutdown detected in embed_string.")
            return None

        # Truncation
        model_max_len = self.config.max_length
        max_input_chars = model_max_len * 6
        if len(text) > max_input_chars:
            text = text[:max_input_chars]
            logger.debug("Truncated single string.")

        # Prepare encode kwargs
        encode_kwargs: Dict[str, Any] = {
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        if input_type == "query":
            encode_kwargs["prompt_name"] = MXBAI_QUERY_PROMPT_NAME

        try:
            input_text = text  # No instruction wrapper
            embedding = self.model.encode(input_text, **encode_kwargs)
            if embedding is not None:
                return (
                    embedding.astype(np.float32)
                    if embedding.dtype != np.float32
                    else embedding
                )
            else:
                logger.warning(f"Model returned None embedding for: '{text[:50]}...'")
                return None
        except Exception as e:
            if not (self._shutdown_event and self._shutdown_event.is_set()):
                logger.error(
                    f"Error embedding single string ({input_type}) on CPU: {e}",
                    exc_info=True,
                )
            return None

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns the embedding dimension (considers truncation)."""
        if self.model is None:
            logger.warning("Cannot get embedding dimension, no model loaded.")
            return None
        # --- Prioritize configured truncate_dim ---
        if self.config.truncate_dim is not None:
            logger.debug(f"Using configured truncate_dim: {self.config.truncate_dim}")
            return self.config.truncate_dim
        # --- Fallback to model's reported dimension ---
        try:
            get_dim_method = getattr(
                self.model, "get_sentence_embedding_dimension", None
            )
            if callable(get_dim_method):
                dimension = get_dim_method()
                if isinstance(dimension, int):
                    logger.debug(f"Using model's default dimension: {dimension}")
                    return dimension
                else:
                    logger.warning(
                        f"Method 'get_sentence_embedding_dimension' returned non-integer: {dimension}"
                    )
                    return None
            else:
                logger.warning(
                    "Model object does not have a callable 'get_sentence_embedding_dimension' method."
                )
                return None
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}", exc_info=True)
            return None

    def close(self):
        """Release resources and free memory."""
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
