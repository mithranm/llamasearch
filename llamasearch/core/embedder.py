# llamasearch/core/enhanced_embedder.py

import numpy as np
import logging
import torch
import gc
import time
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from .resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EnhancedEmbedder:
    """
    Enhanced embedder with multi-threading and multi-GPU support.
    
    Features:
    - Uses thread pools for parallel processing
    - Supports distributing work across multiple GPUs
    - Dynamic batching based on available resources
    - Granular memory management to prevent OOM errors
    - Progress tracking with tqdm
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

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            max_length: Maximum token length for input text
            batch_size: Batch size for processing multiple texts (None for auto)
            auto_optimize: Whether to automatically optimize settings based on hardware
            num_workers: Number of worker threads (None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Get resource manager for hardware optimization
        self.resource_manager = get_resource_manager(auto_optimize=auto_optimize)
        
        # Use resource manager to get optimal configuration
        if auto_optimize:
            config = self.resource_manager.get_embedding_config()
            
            # Use provided values if specified, otherwise use optimized values
            self.device = device or config["device"]
            self.batch_size = batch_size or config["batch_size"]
            self.num_workers = num_workers or (2 if config["multi_process"] else 1)
            self.threads_per_worker = config.get("threads_per_worker", 1)
        else:
            # Use provided values or reasonable defaults
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.batch_size = batch_size or 8
            self.num_workers = num_workers or 1
            self.threads_per_worker = 1
        
        # Initialize models (potentially multiple for parallelization)
        self.models = {}
        self.model_locks = {}
        
        # Load primary model
        logger.info(f"Loading primary embedding model: {model_name} on {self.device}")
        self._load_primary_model()
        
        # Log configuration
        logger.info(f"EnhancedEmbedder initialized with device={self.device}, batch_size={self.batch_size}, "
                   f"workers={self.num_workers}, threads_per_worker={self.threads_per_worker}")

    def _load_primary_model(self):
        """Load the primary embedding model."""
        try:
            # Set thread count for efficient CPU utilization when using CPU-only
            if self.device == "cpu":
                torch.set_num_threads(self.threads_per_worker)
                
            # Load the model on the primary device
            self.models[0] = SentenceTransformer(model_name_or_path=self.model_name, device=self.device)
            self.model_locks[0] = torch.multiprocessing.Lock() if self.num_workers > 1 else None
            
            # If multiple workers are requested and we have multiple GPUs, load additional models
            if self.num_workers > 1 and self.device == "cuda" and torch.cuda.device_count() > 1:
                for gpu_id in range(1, min(self.num_workers, torch.cuda.device_count())):
                    logger.info(f"Loading additional model on CUDA device {gpu_id}")
                    
                    # Create model on specific GPU
                    device = f"cuda:{gpu_id}"
                    self.models[gpu_id] = SentenceTransformer(model_name_or_path=self.model_name, device=device)
                    self.model_locks[gpu_id] = torch.multiprocessing.Lock()
        except Exception as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise

    def _get_model_for_worker(self, worker_id: int = 0):
        """Get the appropriate model for a specific worker."""
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            # Distribute across available GPUs
            gpu_id = worker_id % torch.cuda.device_count()
            return self.models[gpu_id], self.model_locks[gpu_id]
        else:
            # Use the primary model
            return self.models[0], self.model_locks[0]

    def _embed_batch_with_worker(self, texts: List[str], worker_id: int) -> np.ndarray:
        """Embed a batch of texts using a specific worker."""
        if not texts:
            return np.array([])
            
        model, lock = self._get_model_for_worker(worker_id)
        
        # Use lock if provided (for multi-worker scenarios)
        if lock:
            lock.acquire()
            
        try:
            # Process the batch with the model
            with torch.no_grad():
                embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings
        finally:
            # Ensure lock is released even if an error occurs
            if lock:
                lock.release()

    def embed_strings(self, strings: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of strings using multiple workers.
        
        Args:
            strings: List of text strings to embed
            show_progress: Whether to show a progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not strings:
            return np.array([])
            
        # Truncate very long texts to prevent OOM
        truncated_strings = []
        for text in strings:
            if len(text) > 2000:
                truncated_strings.append(text[:2000])
            else:
                truncated_strings.append(text)
                
        # Prepare batches
        total_samples = len(truncated_strings)
        batch_size = self.batch_size
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        # Set up progress tracking
        progress_bar = None
        if show_progress and total_samples > batch_size:
            progress_bar = tqdm(total=total_samples, desc="Generating embeddings", unit="text")
        
        # Single worker case (simpler)
        if self.num_workers == 1:
            all_embeddings = []
            
            for i in range(0, total_samples, batch_size):
                batch = truncated_strings[i:i+batch_size]
                embeddings = self._embed_batch_with_worker(batch, worker_id=0)
                all_embeddings.append(embeddings)
                
                # Update progress
                if progress_bar:
                    progress_bar.update(len(batch))
                    
                # Periodic garbage collection for long sequences
                if i > 0 and i % (batch_size * 10) == 0:
                    gc.collect()
                    if torch.cuda.is_available() and self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        
        # Multi-worker case
        else:
            # Create a thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                # Submit batch jobs to workers
                for worker_id in range(self.num_workers):
                    # Create batches for this worker (interleaved)
                    worker_batches = []
                    for i in range(worker_id, num_batches, self.num_workers):
                        start_idx = i * batch_size
                        end_idx = min(start_idx + batch_size, total_samples)
                        if end_idx > start_idx:
                            worker_batches.append(truncated_strings[start_idx:end_idx])
                    
                    # Skip if no batches for this worker
                    if not worker_batches:
                        continue
                        
                    # Create a job for this worker with all its batches
                    futures.append(executor.submit(self._process_worker_batches, 
                                                 worker_batches, worker_id, progress_bar))
                
                # Collect results from all workers
                all_embeddings = []
                for future in as_completed(futures):
                    worker_embeddings = future.result()
                    all_embeddings.extend(worker_embeddings)
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
            
        # Combine all embeddings and ensure contiguous array
        if not all_embeddings:
            return np.array([])
            
        combined = np.vstack(all_embeddings)
        result = np.ascontiguousarray(combined, dtype=np.float32)
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        return result

    def _process_worker_batches(self, batches: List[List[str]], worker_id: int, 
                               progress_bar=None) -> List[np.ndarray]:
        """Process multiple batches with a single worker."""
        results = []
        
        for batch in batches:
            # Process the batch
            embeddings = self._embed_batch_with_worker(batch, worker_id)
            results.append(embeddings)
            
            # Update progress
            if progress_bar:
                progress_bar.update(len(batch))
        
        return results

    def embed_string(self, text: str) -> np.ndarray:
        """Embed a single string."""
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
            
        # Truncate very long text
        if len(text) > 2000:
            text = text[:2000]
            logger.debug("Truncated string to 2000 chars to reduce memory usage")
            
        # Get embedding using the primary model
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
        """Legacy method to maintain compatibility with existing code."""
        return self.embed_strings(strings)

    def close(self):
        """Release resources."""
        # Clear all models
        self.models.clear()
        self.model_locks.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Testing/benchmark code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create sample texts of varying lengths
    test_texts = []
    for i in range(100):
        # Generate texts of different lengths
        word_count = 5 + (i % 50)
        test_texts.append(" ".join(["sample"] * word_count))
        
    # Test with auto-optimization
    embedder = EnhancedEmbedder(auto_optimize=True)
    
    # Benchmark time
    start_time = time.time()
    embeddings = embedder.embed_strings(test_texts)
    end_time = time.time()
    
    print(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test single string embedding
    single = embedder.embed_string("This is a test")
    print(f"Single embedding shape: {single.shape}")
    
    embedder.close()