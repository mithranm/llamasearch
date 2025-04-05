# llamasearch/core/enhanced_embedder.py

import numpy as np
import logging
import torch
import gc
import time
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from .resource_manager import get_resource_manager

# Instead of using torch.multiprocessing.Lock (a runtime variable),
# we import the Lock type from the standard library.
from multiprocessing.synchronize import Lock as SyncLock

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EnhancedEmbedder:
    """
    Enhanced embedder with multi-threading and multi-GPU support.
    
    Features:
    - Uses thread pools for parallel processing
    - Supports distributing work across multiple GPUs
    - Dynamic batching based on available resources
    - Uses half precision on CUDA for faster inference and lower memory usage
    - Enables cuDNN benchmarking on CUDA
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: Optional[int] = None,
        auto_optimize: bool = True,
        num_workers: Optional[int] = None
    ):
        """
        Initialize the embedder with dynamic hardware detection.
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Get resource manager for hardware optimization
        self.resource_manager = get_resource_manager(auto_optimize=auto_optimize)
        
        if auto_optimize:
            config = self.resource_manager.get_embedding_config()
            self.device = device or config.get("device", "cpu")
            self.batch_size = batch_size or config.get("batch_size", 32)
            if self.device.startswith("cuda") and torch.cuda.device_count() <= 1:
                self.num_workers = 1
            else:
                self.num_workers = num_workers or config.get("num_workers", 2)
            self.threads_per_worker = config.get("threads_per_worker", 1)
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.batch_size = batch_size or 8
            self.num_workers = num_workers or 1
            self.threads_per_worker = 1

        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        
        self.models: Dict[int, SentenceTransformer] = {}
        self.model_locks: Dict[int, Optional[SyncLock]] = {}
        
        logger.info(f"Loading primary embedding model: {model_name} on {self.device}")
        self._load_primary_model()
        
        logger.info(f"EnhancedEmbedder initialized with device={self.device}, batch_size={self.batch_size}, "
                    f"workers={self.num_workers}, threads_per_worker={self.threads_per_worker}")

    def _load_primary_model(self):
        """Load the primary embedding model."""
        try:
            if self.device == "cpu":
                torch.set_num_threads(self.threads_per_worker)
            self.models[0] = SentenceTransformer(model_name_or_path=self.model_name, device=self.device)
            if self.device.startswith("cuda"):
                try:
                    self.models[0].half()
                    logger.info("Converted primary model to FP16 for CUDA acceleration.")
                except Exception as e:
                    logger.warning(f"Could not convert model to FP16: {e}")
            self.model_locks[0] = torch.multiprocessing.Lock() if self.num_workers > 1 else None

            if self.num_workers > 1 and self.device.startswith("cuda") and torch.cuda.device_count() > 1:
                for gpu_id in range(1, min(self.num_workers, torch.cuda.device_count())):
                    device_str = f"cuda:{gpu_id}"
                    logger.info(f"Loading additional model on {device_str}")
                    self.models[gpu_id] = SentenceTransformer(model_name_or_path=self.model_name, device=device_str)
                    try:
                        self.models[gpu_id].half()
                        logger.info(f"Converted model on {device_str} to FP16.")
                    except Exception as e:
                        logger.warning(f"Could not convert model on {device_str} to FP16: {e}")
                    self.model_locks[gpu_id] = torch.multiprocessing.Lock()
        except Exception as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise

    def _get_model_for_worker(self, worker_id: int = 0):
        """Return the model and lock for a given worker."""
        if self.device.startswith("cuda") and torch.cuda.device_count() > 1:
            gpu_id = worker_id % torch.cuda.device_count()
            return self.models[gpu_id], self.model_locks[gpu_id]
        else:
            return self.models[0], self.model_locks.get(0)

    def _embed_batch_with_worker(self, texts: List[str], worker_id: int) -> np.ndarray:
        """Embed a batch of texts using a specific worker."""
        if not texts:
            return np.array([])
        model, lock = self._get_model_for_worker(worker_id)
        if lock:
            lock.acquire()
        try:
            with torch.no_grad():
                embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings
        finally:
            if lock:
                lock.release()

    def embed_strings(self, strings: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of strings using multiple workers.
        """
        if not strings:
            return np.array([])
        truncated = [s if len(s) <= 2000 else s[:2000] for s in strings]
        total = len(truncated)
        batch_size = self.batch_size
        progress_bar = tqdm(total=total, desc="Generating embeddings", unit="text") if show_progress and total > batch_size else None

        if self.num_workers == 1:
            all_embeddings = []
            for i in range(0, total, batch_size):
                batch = truncated[i:i+batch_size]
                emb = self._embed_batch_with_worker(batch, worker_id=0)
                all_embeddings.append(emb)
                if progress_bar:
                    progress_bar.update(len(batch))
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    if torch.cuda.is_available() and self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i in range(0, total, batch_size):
                    batch = truncated[i:i+batch_size]
                    worker_id = (i // batch_size) % self.num_workers
                    futures.append(executor.submit(self._embed_batch_with_worker, batch, worker_id))
                all_embeddings = []
                for fut in as_completed(futures):
                    all_embeddings.append(fut.result())
                    if progress_bar:
                        progress_bar.update(batch_size)
        if progress_bar:
            progress_bar.close()
        if not all_embeddings:
            return np.array([])
        combined = np.vstack(all_embeddings)
        result = np.ascontiguousarray(combined, dtype=np.float32)
        gc.collect()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    def _process_worker_batches(self, batches: List[List[str]], worker_id: int, progress_bar=None) -> List[np.ndarray]:
        results = []
        for batch in batches:
            emb = self._embed_batch_with_worker(batch, worker_id)
            results.append(emb)
            if progress_bar:
                progress_bar.update(len(batch))
        return results

    def embed_string(self, text: str) -> np.ndarray:
        """Embed a single string."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        if len(text) > 2000:
            text = text[:2000]
            logger.debug("Truncated string to 2000 chars.")
        model, lock = self._get_model_for_worker(0)
        if lock:
            lock.acquire()
        try:
            with torch.no_grad():
                embedding = model.encode(text, convert_to_numpy=True)
            return embedding
        finally:
            if lock:
                lock.release()

    def embed_batch(self, strings: List[str]) -> np.ndarray:
        """Legacy alias for embed_strings."""
        return self.embed_strings(strings)

    def close(self):
        """Release resources."""
        self.models.clear()
        self.model_locks.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    test_texts = ["sample " * (5 + (i % 50)) for i in range(100)]
    embedder = EnhancedEmbedder(auto_optimize=True)
    start = time.time()
    embeddings = embedder.embed_strings(test_texts)
    duration = time.time() - start
    print(f"Generated {len(embeddings)} embeddings in {duration:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    single_embedding = embedder.embed_string("This is a test")
    print(f"Single embedding shape: {single_embedding.shape}")
    embedder.close()
