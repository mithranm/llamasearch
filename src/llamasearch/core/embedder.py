"""
Enhanced embedder optimized for CPU with multi-threading.
Includes checks for model existence before loading.
"""

import numpy as np
import torch
import gc
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from multiprocessing.synchronize import Lock as SyncLock
from pydantic import BaseModel, Field, validator
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from llamasearch.exceptions import ModelNotFoundError
from llamasearch.data_manager import data_manager  # To get models path

# Import hardware detection (assuming it provides CPU info)
from llamasearch.hardware import detect_hardware_info, HardwareInfo
from llamasearch.utils import setup_logging

logger = setup_logging(__name__)  # Get logger using utility

DEFAULT_MODEL_NAME = "teapotai/teapotembedding"
DEVICE_TYPE = "cpu" # Hardcoded for CPU-only


class EmbedderConfig(BaseModel):
    """Configuration for the embedder optimized for CPU."""

    model_name: str = DEFAULT_MODEL_NAME
    device: str = Field(default=DEVICE_TYPE) # Always CPU
    max_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=8, gt=0)
    auto_optimize: bool = True
    num_workers: int = Field(default=1, ge=1)
    instruction: str = "Given a passage, represent its content for retrieval."
    threads_per_worker: int = Field(default=1, ge=1)
    # use_half_precision removed, not relevant for CPU

    @validator("num_workers")
    def validate_workers(cls, v, values):
        """Ensure a reasonable number of workers based on CPU cores."""
        is_auto_optimizing = values.get("auto_optimize", True)
        cpu_limit = max(1, (os.cpu_count() or 4) // 2) # Default reasonable limit
        limit = v

        if v > cpu_limit:
            limit = cpu_limit
            if is_auto_optimizing:
                logger.debug(f"Auto-limiting CPU workers from {v} to {limit}.")
            else:
                logger.warning(
                    f"Provided CPU workers ({v}) exceeds reasonable limit ({limit}), capping."
                )

        final_workers = max(1, limit)
        if final_workers != v:
            logger.info(
                f"Adjusted num_workers from {v} to {final_workers} for CPU."
            )
        return final_workers

    @classmethod
    def from_hardware(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbedderConfig":
        """Create an optimized CPU configuration based on detected hardware."""
        hw: HardwareInfo = detect_hardware_info()
        config_data = {
            "model_name": model_name,
            "auto_optimize": True,
            "device": DEVICE_TYPE # Always CPU
            }

        logger.info("Configuring for CPU based on detected hardware.")
        # Use physical cores for worker calculation
        physical_cores = hw.cpu.physical_cores if hw.cpu.physical_cores else (os.cpu_count() or 2) # Fallback if detection fails
        
        # Aim for roughly half the physical cores as workers, min 1, max 8
        cpu_workers = min(8, max(1, physical_cores // 2))
        config_data["num_workers"] = cpu_workers

        # Distribute remaining cores as threads per worker
        threads_per = min(
            4, # Limit threads per worker to avoid excessive context switching
            max(
                1,
                physical_cores // cpu_workers if cpu_workers > 0 else physical_cores,
            ),
        )
        config_data["threads_per_worker"] = threads_per
        logger.info(
            f"Auto CPU config: {cpu_workers} workers, {threads_per} threads per worker."
        )

        # Set batch size based on available RAM
        if hw.memory.available_gb > 30:
            config_data["batch_size"] = 32
        elif hw.memory.available_gb > 15:
            config_data["batch_size"] = 16
        elif hw.memory.available_gb > 7:
            config_data["batch_size"] = 8
        else:
            config_data["batch_size"] = 4 # Lower batch size for low RAM
        logger.info(
            f"Available RAM: {hw.memory.available_gb:.1f} GB. Auto batch size: {config_data['batch_size']}."
        )

        final_config = cls(**config_data)
        logger.info(
            f"Auto-optimized EmbedderConfig: device={final_config.device}, batch_size={final_config.batch_size}, "
            f"workers={final_config.num_workers}, threads_per_worker={final_config.threads_per_worker}"
        )
        return final_config


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create an instruction template for the E5 instruct model."""
    if task_description and not task_description.endswith((".", "!", "?", ":")):
        task_description += ":"
    return f"Instruct: {task_description}\nQuery: {query}"


class EnhancedEmbedder:
    """
    Enhanced embedder optimized for CPU with multi-threading.
    Includes checks for model existence before loading.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        # device parameter removed - always CPU
        max_length: int = 512,
        batch_size: int = 0,
        auto_optimize: bool = True,
        num_workers: int = 0,
        instruction: str = "",
    ):
        config_data = {"model_name": model_name, "auto_optimize": auto_optimize}

        if auto_optimize:
            logger.info("Auto-optimizing embedder configuration based on hardware (CPU)...")
            base_config = EmbedderConfig.from_hardware(model_name)
            config_data.update(base_config.dict())
            # Ensure device is CPU even if base_config somehow missed it
            config_data["device"] = DEVICE_TYPE
        else:
             logger.info("Using provided embedder configuration (auto_optimize=False).")
             config_data["device"] = DEVICE_TYPE # Force CPU
             config_data["batch_size"] = batch_size or 8
             config_data["num_workers"] = num_workers or 1
             config_data["instruction"] = instruction or EmbedderConfig().instruction
             # Calculate threads_per_worker if not auto-optimized
             hw = detect_hardware_info()
             physical_cores = hw.cpu.physical_cores if hw.cpu.physical_cores else (os.cpu_count() or 2)
             config_data["threads_per_worker"] = config_data.get(
                 "threads_per_worker", # Keep if provided
                 min(
                     4,
                     max(
                         1,
                         physical_cores // config_data["num_workers"]
                         if config_data["num_workers"] > 0
                         else physical_cores,
                     ),
                 ),
             )


        # Override auto-config with specific user inputs if provided
        if batch_size > 0:
            config_data["batch_size"] = batch_size
        if num_workers > 0:
            config_data["num_workers"] = num_workers
        if max_length != 512:
            config_data["max_length"] = max_length
        if instruction:
            config_data["instruction"] = instruction
        else:
            # Ensure instruction is set even if not provided and not auto-optimizing
            config_data["instruction"] = config_data.get(
                "instruction", EmbedderConfig().instruction
            )

        # Final config object creation
        self.config = EmbedderConfig(**config_data)

        # Ensure final config reflects CPU only settings
        if self.config.device != DEVICE_TYPE:
             logger.warning(f"Overriding configured device '{self.config.device}' to '{DEVICE_TYPE}'.")
             self.config.device = DEVICE_TYPE


        try:
            self.models_dir = Path(data_manager.get_data_paths()["models"])
            logger.info(f"Embedder using models directory: {self.models_dir}")
        except Exception as e:
            logger.error(
                f"Failed to get models directory from data_manager: {e}. Using default relative path.",
                exc_info=True,
            )
            self.models_dir = Path(".") / ".llamasearch" / "models"
            self.models_dir.mkdir(parents=True, exist_ok=True)

        # Rough check if max_length is feasible before loading the full model
        try:
            temp_model_check = SentenceTransformer(
                self.config.model_name, cache_folder=str(self.models_dir)
            )
            # Check if setting max_seq_length works (might raise error if model doesn't support it well)
            temp_model_check.max_seq_length = self.config.max_length
            logger.debug(f"Model seems compatible with max_length {self.config.max_length}")
            del temp_model_check
        except Exception as e:
            logger.warning(
                f"Could not verify max_seq_length ({self.config.max_length}) on temp model check: {e}. Using model default if needed."
            )
            # Don't fail here, let the main loading handle it


        # Models dictionary will hold only one model for CPU
        self.model: Optional[SentenceTransformer] = None
        self.model_lock: Optional[SyncLock] = None
        self._lock_initialized = False

        logger.info(
            f"Final Embedder Config: model={self.config.model_name}, device={self.config.device}, "
            f"batch_size={self.config.batch_size}, workers={self.config.num_workers}, "
            f"threads_per_worker={self.config.threads_per_worker}"
        )

        self._load_model()  # Load the single CPU model

    def _initialize_lock(self):
        """Initialize lock safely for multiprocessing/threading."""
        if not self._lock_initialized and self.config.num_workers > 1:
            try:
                # Use lock for multi-worker CPU case for thread safety
                logger.debug("Initializing multiprocessing lock for embedder worker.")
                # Use spawn context if available for better isolation, otherwise default
                try:
                    mp_context = torch.multiprocessing.get_context("spawn")
                except Exception:
                    logger.warning("Spawn context not available, using default multiprocessing context for lock.")
                    import multiprocessing as mp # Fallback
                    mp_context = mp.get_context()

                self.model_lock = mp_context.Lock()
                self._lock_initialized = True
                logger.debug("Lock initialized.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize multiprocessing lock: {e}. Proceeding without lock (potential race conditions).",
                    exc_info=True,
                )
                self.model_lock = None
                self._lock_initialized = True # Mark as initialized even if failed
        elif self.config.num_workers <= 1:
            self.model_lock = None # No lock needed for single worker
            self._lock_initialized = True

    def _load_model(self):
        """Load the embedding model onto the CPU after checking existence."""
        model_name = self.config.model_name
        logger.info(f"Preparing to load embedder model: {model_name} onto CPU")

        # --- CHECK MODEL EXISTENCE ---
        try:
            logger.debug(
                f"Checking for model '{model_name}' in cache: {self.models_dir}"
            )
            snapshot_download(
                repo_id=model_name,
                cache_dir=self.models_dir,
                local_files_only=True,  # Check cache only
            )
            logger.info(f"Model '{model_name}' found locally in cache.")
        except (EntryNotFoundError, LocalEntryNotFoundError):
            logger.error(
                f"Embedder model '{model_name}' not found in cache directory: {self.models_dir}"
            )
            raise ModelNotFoundError(
                f"Embedder model '{model_name}' not found locally. "
                f"Please run 'llamasearch-setup' or ensure the model exists at the specified path."
            )
        except Exception as e:
            logger.error(
                f"Error checking model cache for '{model_name}': {e}", exc_info=True
            )
            raise ModelNotFoundError(
                f"Error accessing embedder model '{model_name}' cache. "
                f"Check permissions or run 'llamasearch-setup'. Error: {e}"
            )

        # --- PROCEED WITH LOADING (if check passed) ---
        try:
            # Set torch threads based on workers and threads_per_worker
            total_threads = self.config.num_workers * self.config.threads_per_worker
            logical_cores = os.cpu_count() or 1
            effective_threads = max(1, min(total_threads, logical_cores))

            # Only set threads if > 0, avoid setting 0 threads
            if effective_threads > 0 :
                current_threads = torch.get_num_threads()
                # Only change if necessary
                if current_threads != effective_threads:
                    torch.set_num_threads(effective_threads)
                    logger.info(
                        f"Set torch global CPU threads from {current_threads} to: {torch.get_num_threads()}"
                    )
                else:
                     logger.info(f"Torch global CPU threads already set to {current_threads}.")

            logger.info(
                f"Loading embedding model '{model_name}' onto device '{DEVICE_TYPE}'"
            )
            model = SentenceTransformer(
                model_name,
                device=DEVICE_TYPE,
                cache_folder=str(self.models_dir),  # Explicitly specify cache folder
            )
            model.max_seq_length = self.config.max_length
            logger.debug(
                f"Model {model_name} on {DEVICE_TYPE} max_seq_length set to {model.max_seq_length}"
            )

            # No FP16 conversion for CPU
            self.model = model

            # Initialize lock after model loading attempt
            self._initialize_lock()

        except ModelNotFoundError:
            raise  # Re-raise if check failed earlier
        except Exception as e:
            logger.error(
                f"Error loading embedding model '{model_name}' from cache: {e}",
                exc_info=True,
            )
            self.model = None
            self.model_lock = None
            raise RuntimeError(
                f"Failed to load embedder model '{model_name}' from cache: {e}"
            ) from e

    def _get_model_and_lock(self) -> tuple:
        """Return the single CPU model instance and its lock."""
        self._initialize_lock() # Ensure lock is initialized if needed
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")
        return self.model, self.model_lock

    def _embed_batch_task(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts using the single CPU model (task for executor)."""
        if not texts:
            return np.array([], dtype=np.float32)
        try:
            model, lock = self._get_model_and_lock()
            input_texts = [
                get_detailed_instruct(self.config.instruction, text) for text in texts
            ]
            embeddings = None
            acquired_lock = False

            # Acquire lock only if it exists (i.e., num_workers > 1)
            if lock:
                acquired_lock = lock.acquire(timeout=15) # Timeout for lock acquisition
                if not acquired_lock:
                    # If lock fails, log warning and return empty array for this batch
                    logger.warning(
                        f"Worker failed to acquire lock for batch of size {len(texts)}, skipping."
                    )
                    return np.array([], dtype=np.float32) # Return empty, indicates failure for this batch

            try:
                # Process the batch within the lock (if acquired)
                effective_batch_size = min(len(input_texts), self.config.batch_size)
                embeddings = model.encode(
                    input_texts,
                    batch_size=effective_batch_size,
                    show_progress_bar=False, # Progress handled by outer loop
                    convert_to_numpy=True,
                    normalize_embeddings=True, # Assuming normalization is desired
                )
                # Ensure float32 type
                if embeddings is not None and embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
            except Exception as e:
                logger.error(
                    f"Error during model.encode on CPU for batch size {len(texts)}: {e}",
                    exc_info=True,
                )
                # Return empty on error, indicating failure for this batch
                return np.array([], dtype=np.float32)
            finally:
                # Release lock if it was acquired
                if acquired_lock and lock:
                    lock.release()

            # Return results, handle None case
            return (
                embeddings if embeddings is not None else np.array([], dtype=np.float32)
            )
        except Exception as e:
            # Catch errors in getting model/lock or other unexpected issues
            logger.error(
                f"Error processing batch (size {len(texts)}) in worker task: {e}",
                exc_info=True,
            )
            return np.array([], dtype=np.float32) # Return empty on major task error


    def embed_strings(
        self, strings: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of strings using CPU workers."""
        if self.model is None:
             raise RuntimeError("Cannot embed strings, model is not loaded.")
        if not strings:
            return np.array([], dtype=np.float32)

        # --- String Truncation ---
        model_max_len = self.config.max_length
        try:
            # Use actual model max length if available
            model_max_len = self.model.max_seq_length or model_max_len
        except Exception:
            pass # Use config value if model property access fails
        
        # Estimate max characters (heuristic, depends on tokenization)
        # Using 6 chars/token as a generous upper bound estimate
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
                f"Truncated {num_truncated} strings longer than estimated {max_input_chars} chars "
                f"(based on model max_length {model_max_len}). Adjust max_length if needed."
            )

        # --- Embedding Process ---
        total = len(truncated_strings)
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers

        progress_bar = tqdm(
            total=total,
            desc="Generating embeddings (CPU)",
            unit="text",
            disable=not show_progress or total <= batch_size, # Disable for small jobs or if requested
        )
        
        all_embeddings_list = [] # Store results from futures

        # Use ThreadPoolExecutor for parallel batch processing on CPU
        # It handles the case num_workers=1 correctly (runs sequentially)
        logger.debug(f"Running embedding with {num_workers} worker(s).")
        executor_factory = ThreadPoolExecutor # Standard choice for I/O bound or GIL-releasing tasks like ST encode

        with executor_factory(max_workers=num_workers) as executor:
            futures = []
            # Submit batches to the executor
            for i in range(0, total, batch_size):
                batch = truncated_strings[i : min(i + batch_size, total)]
                if batch: # Ensure non-empty batch
                    futures.append(
                        executor.submit(self._embed_batch_task, batch)
                    )
            
            # Collect results as they complete
            for fut in as_completed(futures):
                try:
                    result_emb = fut.result()
                    # Check if the result is valid before appending
                    if result_emb is not None and isinstance(result_emb, np.ndarray) and result_emb.size > 0:
                        all_embeddings_list.append(result_emb)
                        progress_bar.update(result_emb.shape[0]) # Update by number of embeddings processed
                    elif result_emb is not None and isinstance(result_emb, np.ndarray) and result_emb.size == 0:
                        # Log if a worker explicitly returned an empty array (e.g., due to lock failure)
                        logger.debug("Worker task returned an empty result array.")
                    else:
                         logger.warning(
                            f"Embedding worker thread returned unexpected result type: {type(result_emb)}"
                        )
                except Exception as e:
                    # Catch errors from the future itself (e.g., task raised exception not caught internally)
                    logger.error(
                        f"Error retrieving result from embedding worker future: {e}",
                        exc_info=True,
                    )
                
                # Periodic garbage collection during long processes
                if len(all_embeddings_list) % 20 == 0: # Every 20 completed batches
                     self._try_gc()


        progress_bar.close()

        # --- Combine Results ---
        if not all_embeddings_list:
            logger.warning("Embedding process yielded no valid results.")
            emb_dim = self.get_embedding_dimension()
            # Return empty array with correct shape if possible
            return np.empty((0, emb_dim) if emb_dim else 0, dtype=np.float32)

        # Initialize valid_embeddings *before* the try block
        valid_embeddings = []
        try:
            # Filter out any remaining Nones or empty arrays just in case
            valid_embeddings = [
                emb
                for emb in all_embeddings_list
                if isinstance(emb, np.ndarray) and emb.size > 0
            ]
            if not valid_embeddings:
                logger.error("No valid numpy arrays found in embedding results after filtering.")
                emb_dim = self.get_embedding_dimension()
                return np.empty((0, emb_dim) if emb_dim else 0, dtype=np.float32)

            # Combine the valid embeddings
            combined = np.vstack(valid_embeddings)
            # Ensure contiguous array with float32 type
            result = np.ascontiguousarray(combined, dtype=np.float32)

        except ValueError as e:
            # Handle potential shape mismatches during vstack
            logger.error(
                f"Error during vstack of embeddings (shape mismatch?): {e}",
                exc_info=True,
            )
            shapes = [emb.shape for emb in valid_embeddings] # Use the filtered list
            logger.debug(f"Shapes of collected embeddings: {shapes}")
            emb_dim = self.get_embedding_dimension()
            return np.empty((0, emb_dim) if emb_dim else 0, dtype=np.float32) # Return empty on error

        self._try_gc() # Final GC
        if result.shape[0] != total:
            logger.warning(f"Expected {total} embeddings, but got {result.shape[0]}. Some batches might have failed.")
        else:
             logger.info(
                f"Generated {result.shape[0]} embeddings with dimension {result.shape[1]}."
            )
        return result

    def _try_gc(self):
        """Try to run garbage collection."""
        gc.collect()
        # No CUDA cache to empty

    def embed_string(self, text: str) -> Optional[np.ndarray]:
        """Embed a single string using the CPU model."""
        if self.model is None:
             logger.error("Cannot embed string, model is not loaded.")
             return None
        if not text:
            logger.warning("Attempted to embed an empty string.")
            return None # Return None for empty input

        # --- String Truncation (same logic as embed_strings) ---
        model_max_len = self.config.max_length
        try:
            model_max_len = self.model.max_seq_length or model_max_len
        except Exception:
            pass
        max_input_chars = model_max_len * 6
        if len(text) > max_input_chars:
            original_len = len(text)
            text = text[:max_input_chars]
            logger.debug(f"Truncated single string from {original_len} to {max_input_chars} chars.")

        # --- Embedding ---
        try:
            # Get model and lock (lock is likely None for single string, but use the helper)
            model, lock = self._get_model_and_lock()
            input_text = get_detailed_instruct(self.config.instruction, text)

            embedding = None
            acquired_lock = False
            if lock: # Check if lock exists (might if called concurrently)
                acquired_lock = lock.acquire(timeout=5) # Short timeout for single embed
                if not acquired_lock:
                     logger.warning("Failed lock for single string embed, returning None.")
                     return None

            try:
                 # Encode the single string
                 embedding = model.encode(
                     input_text,
                     show_progress_bar=False,
                     convert_to_numpy=True,
                     normalize_embeddings=True,
                 )
            except Exception as e:
                logger.error(f"Error during single string encode: {e}", exc_info=True)
                return None # Return None on encoding error
            finally:
                 if acquired_lock and lock:
                     lock.release()

            # Process the result
            if embedding is not None:
                # Ensure correct dtype
                return (
                    embedding.astype(np.float32)
                    if embedding.dtype != np.float32
                    else embedding
                )
            else:
                # Handle case where model returns None
                logger.warning(f"Model returned None embedding for single string: '{text[:50]}...'")
                return None
        except Exception as e:
            # Catch errors in getting model/lock or other issues
            logger.error(f"Error embedding single string: {e}", exc_info=True)
            return None

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns the embedding dimension of the loaded CPU model."""
        if self.model is None:
            logger.warning("Cannot get embedding dimension, no model loaded.")
            return None
        try:
            # Directly access the model's method
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return None

    def embed_batch(self, strings: List[str]) -> np.ndarray:
        """Legacy alias for embed_strings."""
        logger.debug("embed_batch called, redirecting to embed_strings.")
        return self.embed_strings(strings)

    def similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between two sets of normalized embeddings."""
        if not isinstance(embeddings1, np.ndarray) or not isinstance(
            embeddings2, np.ndarray
        ):
            raise TypeError("Inputs must be numpy arrays.")
        
        # Handle empty inputs gracefully
        if embeddings1.size == 0 or embeddings2.size == 0:
             # Return empty array with correct dimensions if possible
             # If one is (0, D) and other is (N, D), result should be (0, N)
             # If one is (N, D) and other is (0, D), result should be (N, 0)
             # If both are empty, result is (0, 0)
            return np.empty(
                (embeddings1.shape[0], embeddings2.shape[0]), dtype=np.float32
            )
            
        # Reshape 1D arrays to 2D
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
            
        # Dimension check
        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: {embeddings1.shape[1]} != {embeddings2.shape[1]}"
            )
            
        # Calculate dot product (cosine similarity for normalized vectors)
        # Ensure float32 for potentially better performance/consistency
        sim = np.dot(embeddings1.astype(np.float32), embeddings2.astype(np.float32).T)
        
        # Clip values to handle potential floating point inaccuracies
        return np.clip(sim, -1.0, 1.0)

    def close(self):
        """Release resources and free memory."""
        logger.info("Closing CPU Embedder and releasing resources...")
        if self.model is not None:
             try:
                 # Explicitly delete the model object
                 del self.model
                 self.model = None
                 logger.debug("Embedder model deleted.")
             except Exception as e:
                 logger.warning(f"Error deleting model during close: {e}")
        
        # Clear lock reference
        self.model_lock = None
        self._lock_initialized = False
        
        # Run final garbage collection
        self._try_gc()
        logger.info("CPU Embedder closed.")