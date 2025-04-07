# llamasearch/core/embedder.py

import numpy as np
import logging
import torch
import gc
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Instead of using torch.multiprocessing.Lock (a runtime variable),
# we import the Lock type from the standard library.
from multiprocessing.synchronize import Lock as SyncLock

from .resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Create an instruction template for the E5 instruct model.
    
    Args:
        task_description: The description of the task
        query: The query text
        
    Returns:
        Formatted instruction text
    """
    return f'Instruct: {task_description}\nQuery: {query}'

class EnhancedEmbedder:
    """
    Enhanced embedder with multi-threading and multi-GPU support using multilingual E5 instruct model.
    
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
        device: str = "",
        max_length: int = 512,
        batch_size: int = 0,
        auto_optimize: bool = True,
        num_workers: int = 0,
        instruction: str = "",
        embedding_config: dict = {}
    ):
        """
        Initialize the embedder with dynamic hardware detection.
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set default instruction for embedding retrieval
        self.instruction = instruction if instruction else "Given a passage, represent its content for retrieval."
        
        # Get resource manager for hardware optimization
        self.resource_manager = get_resource_manager(auto_optimize=auto_optimize)
        
        if auto_optimize:
            config = self.resource_manager.get_embedding_config()
            self.device = device if device else config.get("device", "cpu")
            self.batch_size = batch_size if batch_size else config.get("batch_size", 32)
            if self.device.startswith("cuda") and torch.cuda.device_count() <= 1:
                self.num_workers = 1
            else:
                self.num_workers = num_workers if num_workers else config.get("num_workers", 2)
            self.threads_per_worker = config.get("threads_per_worker", 1)
        else:
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            self.batch_size = batch_size if batch_size else 8
            self.num_workers = num_workers if num_workers else 1
            self.threads_per_worker = 1

        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        
        self.models: Dict[int, SentenceTransformer] = {}
        self.model_locks: Dict[int, SyncLock | None] = {}
        
        logger.info(f"Loading primary embedding model: {model_name} on {self.device}")
        self._load_primary_model()
        
        logger.info(f"EnhancedEmbedder initialized with device={self.device}, batch_size={self.batch_size}, "
                    f"workers={self.num_workers}, threads_per_worker={self.threads_per_worker}")

    def _load_primary_model(self):
        """Load the primary embedding model."""
        try:
            if self.device == "cpu":
                torch.set_num_threads(self.threads_per_worker)
            
            # Load Sentence Transformer model
            model = SentenceTransformer(self.model_name, device=self.device)
            
            if self.device.startswith("cuda"):
                try:
                    # Apply half precision if on CUDA
                    model.half()
                    logger.info("Converted primary model to FP16 for CUDA acceleration.")
                except Exception as e:
                    logger.warning(f"Could not convert model to FP16: {e}")
            
            # Store model
            self.models[0] = model
            self.model_locks[0] = torch.multiprocessing.Lock() if self.num_workers > 1 else None

            # Load additional models on multi-GPU systems
            if self.num_workers > 1 and self.device.startswith("cuda") and torch.cuda.device_count() > 1:
                for gpu_id in range(1, min(self.num_workers, torch.cuda.device_count())):
                    device_str = f"cuda:{gpu_id}"
                    logger.info(f"Loading additional model on {device_str}")
                    
                    # Load model for this GPU
                    model = SentenceTransformer(self.model_name, device=device_str)
                    
                    try:
                        model.half()  # Convert to FP16
                        logger.info(f"Converted model on {device_str} to FP16.")
                    except Exception as e:
                        logger.warning(f"Could not convert model on {device_str} to FP16: {e}")
                    
                    self.models[gpu_id] = model
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
        
        # Create instruction-based inputs
        input_texts = [get_detailed_instruct(self.instruction, text) for text in texts]
        
        if lock:
            lock.acquire()
            
        try:
            # Use sentence_transformers encode method
            embeddings = model.encode(
                input_texts,
                batch_size=min(len(input_texts), self.batch_size),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
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
        
        # Truncate very long strings to avoid excessive tokenization time
        truncated = [s if len(s) <= 5000 else s[:5000] for s in strings]
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

    def embed_string(self, text: str) -> np.ndarray:
        """Embed a single string."""
        
        # Truncate very long text
        if len(text) > 2000:
            text = text[:2000]
            logger.debug("Truncated string to 2000 chars.")
        
        # Get model
        model, lock = self._get_model_for_worker(0)
        
        # Format with instruction
        input_text = get_detailed_instruct(self.instruction, text)
        
        if lock:
            lock.acquire()
            
        try:
            # Use sentence_transformers encode method
            embedding = model.encode(
                input_text,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding
        finally:
            if lock:
                lock.release()

    def embed_batch(self, strings: List[str]) -> np.ndarray:
        """Legacy alias for embed_strings."""
        return self.embed_strings(strings)
        
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between two sets of embeddings."""
        return embeddings1 @ embeddings2.T

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
    
    # Test with different languages
    test_texts = [
        "sample document about machine learning",
        "文档示例关于机器学习",
        "exemple de document sur l'apprentissage automatique",
        "пример документа о машинном обучении",
        "muestras de documentos sobre aprendizaje automático"
    ]
    
    embedder = EnhancedEmbedder(auto_optimize=True)
    
    start = time.time()
    embeddings = embedder.embed_strings(test_texts)
    duration = time.time() - start
    
    print(f"Generated {len(embeddings)} embeddings in {duration:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test single embedding
    single_embedding = embedder.embed_string("This is a test")
    print(f"Single embedding shape: {single_embedding.shape}")
    
    # Calculate similarity matrix using the new similarity method
    similarity = embedder.similarity(embeddings, embeddings)
    print("Similarity matrix:")
    print(similarity)
    
    embedder.close()