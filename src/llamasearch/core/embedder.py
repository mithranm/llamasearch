# src/llamasearch/core/embedder.py
"""
Simplified embedder optimized for CPU using SentenceTransformer batching.
Includes checks for model existence and shutdown event.
"""

import gc
import os
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any # Added Dict, Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError
from llamasearch.hardware import HardwareInfo, detect_hardware_info
from llamasearch.utils import setup_logging

logger = setup_logging(__name__, use_qt_handler=True)

DEFAULT_MODEL_NAME = "teapotai/teapotembedding"
DEVICE_TYPE = "cpu"


class EmbedderConfig(BaseModel):
    """Configuration for the embedder optimized for CPU."""

    model_name: str = DEFAULT_MODEL_NAME
    device: str = Field(default=DEVICE_TYPE, description="Device (always CPU)")
    max_length: int = Field(default=512, gt=0, description="Max sequence length")
    batch_size: int = Field(default=32, gt=0, description="Batch size for encoding")
    instruction: str = Field(
        "Given a passage, represent its content for retrieval.",
        description="Instruction for E5 models",
    )
    # Removed auto_optimize, num_workers, threads_per_worker

    @classmethod
    def from_hardware(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbedderConfig":
        """Create an optimized CPU configuration based on detected hardware."""
        hw: HardwareInfo = detect_hardware_info()
        # <<< FIX: Initialize config_data with correct types >>>
        config_data: Dict[str, Any] = {
            "model_name": model_name,
            "device": DEVICE_TYPE,
            "batch_size": 32, # Default batch size
            # Initialize other fields if needed, though Pydantic will use defaults
            # "max_length": 512,
            # "instruction": "Given a passage, represent its content for retrieval.",
        }
        logger.info("Configuring Embedder for CPU based on detected hardware.")
        # Set batch size based on available RAM
        if hw.memory.available_gb > 30:
            config_data["batch_size"] = 64
        elif hw.memory.available_gb > 15:
            config_data["batch_size"] = 32
        elif hw.memory.available_gb > 7:
            config_data["batch_size"] = 16
        else:
            config_data["batch_size"] = 8
        logger.info(
            f"Available RAM: {hw.memory.available_gb:.1f} GB. Auto batch size: {config_data['batch_size']}."
        )
        # Pydantic handles the int type correctly here
        final_config = cls(**config_data)
        logger.info(f"Auto-optimized EmbedderConfig: {final_config.dict()}")
        return final_config


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create an instruction template for the E5 instruct model."""
    if task_description and not task_description.endswith((".", "!", "?", ":")):
        task_description += ":"
    return f"Instruct: {task_description}\nQuery: {query}"


class EnhancedEmbedder:
    """Embedder optimized for CPU. Uses SentenceTransformer batching. Checks shutdown event."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 0,  # Allow override, 0 means use default/auto
        batch_size: int = 0,  # Allow override, 0 means use default/auto
        instruction: str = "",  # Allow override
    ):
        # Start with hardware-optimized config
        base_config = EmbedderConfig.from_hardware(model_name)
        # <<< FIX: Initialize config_data correctly >>>
        config_data: Dict[str, Any] = base_config.dict()

        # Apply user overrides if provided and valid
        if model_name != DEFAULT_MODEL_NAME:
            config_data["model_name"] = model_name
        if max_length > 0: # Only override if positive int provided
            config_data["max_length"] = max_length
        if batch_size > 0: # Only override if positive int provided
            config_data["batch_size"] = batch_size
        if instruction: # Override if non-empty string provided
            config_data["instruction"] = instruction

        # Pydantic handles int conversion and validation here
        self.config = EmbedderConfig(**config_data)
        logger.info(f"Final Embedder Config: {self.config.dict()}")

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
        """Allows the parent application to pass a shutdown event."""
        self._shutdown_event = event
        logger.debug("Shutdown event set for embedder.")

    def _load_model(self):
        """Load the embedding model onto the CPU after checking existence."""
        model_name = self.config.model_name
        logger.info(f"Preparing to load embedder model: {model_name} onto CPU")

        try:
            logger.debug(
                f"Checking for model '{model_name}' in cache: {self.models_dir}"
            )
            # Use local_dir_use_symlinks=False for better Windows compatibility
            snapshot_download(
                repo_id=model_name, cache_dir=self.models_dir, local_files_only=True,
                local_dir_use_symlinks=False
            )
            logger.info(f"Model '{model_name}' found locally in cache.")
        except (EntryNotFoundError, LocalEntryNotFoundError):
            logger.error(
                f"Embedder model '{model_name}' not found in cache: {self.models_dir}"
            )
            raise ModelNotFoundError(
                f"Embedder model '{model_name}' not found locally. "
                f"Please run 'llamasearch-setup'."
            )
        except Exception as e:
            logger.error(
                f"Error checking model cache for '{model_name}': {e}", exc_info=True
            )
            raise ModelNotFoundError(f"Error accessing embedder model cache: {e}")

        try:
            # Set reasonable torch threads (e.g., half of physical cores)
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

            logger.info(
                f"Loading embedding model '{model_name}' onto device '{DEVICE_TYPE}'"
            )
            model = SentenceTransformer(
                model_name,
                device=DEVICE_TYPE,
                cache_folder=str(self.models_dir),
            )
            # Set max_seq_length on the loaded model using the config value
            model.max_seq_length = self.config.max_length
            logger.debug(f"Model max_seq_length set to {model.max_seq_length}")

            self.model = model

        except ModelNotFoundError:
            raise  # Re-raise if check failed
        except Exception as e:
            logger.error(
                f"Error loading embedding model '{model_name}' from cache: {e}",
                exc_info=True,
            )
            self.model = None
            raise RuntimeError(
                f"Failed to load embedder model '{model_name}': {e}"
            ) from e

    def embed_strings(
        self, strings: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of strings using CPU."""
        if self.model is None:
            raise RuntimeError("Cannot embed strings, model is not loaded.")
        if not strings:
            return np.array([], dtype=np.float32)

        # Truncation check (heuristic based on max_length)
        model_max_len = self.config.max_length
        max_input_chars = model_max_len * 6  # Rough estimate
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
                f"Truncated {num_truncated} strings > ~{max_input_chars} chars (based on max_length {model_max_len})."
            )

        all_embeddings_list = []
        interrupted = False

        # Apply instruction wrapper
        input_texts = [
            get_detailed_instruct(self.config.instruction, text)
            for text in truncated_strings
        ]

        logger.debug(
            f"Starting embedding of {len(input_texts)} strings with batch size {self.config.batch_size}"
        )
        progress_bar = tqdm(
            total=len(input_texts),
            desc="Generating embeddings (CPU)",
            unit="text",
            disable=not show_progress,
        )

        try:
            # Iterate through batches manually to allow shutdown check
            for i in range(0, len(input_texts), self.config.batch_size):
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.info("Shutdown detected during embedding loop.")
                    interrupted = True
                    break

                batch = input_texts[i : i + self.config.batch_size]
                if not batch:
                    continue

                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),  # Process the actual batch size
                    show_progress_bar=False,  # Use external progress bar
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Assuming normalization
                )

                if batch_embeddings is not None and batch_embeddings.size > 0:
                    # Ensure float32 type
                    if batch_embeddings.dtype != np.float32:
                        batch_embeddings = batch_embeddings.astype(np.float32)
                    all_embeddings_list.append(batch_embeddings)
                    progress_bar.update(len(batch))
                else:
                    logger.warning(
                        f"Embedding batch {i // self.config.batch_size} returned empty result."
                    )

            progress_bar.close()

            if interrupted:
                logger.warning("Embedding process was interrupted by shutdown.")

            if not all_embeddings_list:
                logger.warning("Embedding process yielded no results.")
                emb_dim = self.get_embedding_dimension()
                # Return shape (0, dim) or (0, 0) if dim is unknown
                return np.empty((0, emb_dim if emb_dim else 0), dtype=np.float32)

            # Combine results
            combined = np.vstack(all_embeddings_list)
            result = np.ascontiguousarray(combined, dtype=np.float32)

        except Exception as e:
            progress_bar.close()
            if not (self._shutdown_event and self._shutdown_event.is_set()):
                logger.error(
                    f"Error during SentenceTransformer encode: {e}", exc_info=True
                )
            # Return empty on error
            emb_dim = self.get_embedding_dimension()
            return np.empty((0, emb_dim if emb_dim else 0), dtype=np.float32)
        finally:
            self._try_gc()  # Attempt GC after processing

        processed_count = result.shape[0]
        expected_count = (
            len(truncated_strings) if not interrupted else processed_count
        )  # Adjust expected count if interrupted
        if processed_count != expected_count:
            logger.warning(
                f"Expected {expected_count} embeddings, but got {processed_count}. Some batches may have failed."
            )
        else:
            logger.info(
                f"Generated {result.shape[0]} embeddings with dimension {result.shape[1]}{' (interrupted)' if interrupted else ''}."
            )

        return result

    def _try_gc(self):
        """Try to run garbage collection."""
        gc.collect()
        # Conditional CUDA/MPS cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try: # Check for MPS availability safely
            if torch.backends.mps.is_available():
                 torch.mps.empty_cache()
        except AttributeError:
             pass # Ignore if mps backend doesn't exist

    def embed_string(self, text: str) -> Optional[np.ndarray]:
        """Embed a single string using the CPU model."""
        if self.model is None:
            logger.error("Cannot embed string, model is not loaded.")
            return None
        if not text:
            logger.warning("Attempted to embed an empty string.")
            return None

        # --- Shutdown check ---
        if self._shutdown_event and self._shutdown_event.is_set():
            logger.debug("Shutdown detected in embed_string.")
            return None

        # --- Truncation ---
        model_max_len = self.config.max_length
        max_input_chars = model_max_len * 6
        if len(text) > max_input_chars:
            text = text[:max_input_chars]
            logger.debug(f"Truncated single string to {max_input_chars} chars.")

        try:
            input_text = get_detailed_instruct(self.config.instruction, text)
            embedding = self.model.encode(
                input_text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if embedding is not None:
                # Ensure float32 type
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
                logger.error(f"Error embedding single string: {e}", exc_info=True)
            return None

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns the embedding dimension of the loaded CPU model."""
        if self.model is None:
            logger.warning("Cannot get embedding dimension, no model loaded.")
            return None
        try:
            # Use getattr for safe access in case method doesn't exist
            get_dim_method = getattr(self.model, 'get_sentence_embedding_dimension', None)
            if callable(get_dim_method):
                dimension = get_dim_method()
                if isinstance(dimension, int):
                    return dimension
                else:
                    logger.warning(f"Method 'get_sentence_embedding_dimension' returned non-integer value: {dimension}")
                    return None
            else:
                logger.warning("Model object does not have 'get_sentence_embedding_dimension' method.")
                return None
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return None

    def close(self):
        """Release resources and free memory."""
        logger.info("Closing Embedder and releasing resources...")
        if self._shutdown_event and not self._shutdown_event.is_set():
            logger.debug("Signalling shutdown during embedder close.")
            self._shutdown_event.set()

        if self.model is not None:
            try:
                del self.model
                self.model = None
                logger.debug("Embedder model deleted.")
            except Exception as e:
                logger.warning(f"Error deleting model during close: {e}")
        self._try_gc()
        logger.info("Embedder closed.")