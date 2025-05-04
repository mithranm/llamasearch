"""
Enhanced embedder with multi-threading and multi-GPU optimization.

This module provides a high-performance text embedding system that
automatically configures itself for optimal performance based on
the available hardware.
"""

import numpy as np
import logging
import torch
import gc
import time
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from multiprocessing.synchronize import Lock as SyncLock
from pydantic import BaseModel, Field, validator

# Import hardware detection
from killeraiagent.hardware import detect_hardware_capabilities, AcceleratorType

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kaia.embedder")

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"


class EmbedderConfig(BaseModel):
    """Configuration for the embedder with hardware optimization."""
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "cpu"
    max_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=8, gt=0)
    auto_optimize: bool = True
    num_workers: int = Field(default=1, ge=1)
    instruction: str = "Given a passage, represent its content for retrieval."
    threads_per_worker: int = Field(default=1, ge=1)
    use_half_precision: bool = False
    
    @validator('num_workers')
    def validate_workers(cls, v, values):
        """Ensure a reasonable number of workers."""
        if v > 8 and not values.get('device', '').startswith('cuda'):
            # Limit number of CPU workers to avoid oversubscription
            return 8
        return v
    
    @classmethod
    def from_hardware(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbedderConfig":
        """
        Create an optimized configuration based on detected hardware.
        
        Args:
            model_name: The name of the embedding model to use
            
        Returns:
            Optimized EmbedderConfig
        """
        # Detect hardware capabilities
        hw = detect_hardware_capabilities()
        
        # Start with default config
        config = cls(model_name=model_name)
        
        # Configure based on detected hardware
        if hw.primary_acceleration == AcceleratorType.CUDA and hw.cuda.available:
            config.device = "cuda"
            # Scale batch size with available GPU memory
            if hw.cuda.devices:
                free_memory_gb = hw.cuda.devices[0]["free_memory_mb"] / 1024
                config.batch_size = min(32, int(free_memory_gb * 4))
            config.num_workers = min(hw.cuda.device_count, 4)
            config.use_half_precision = True
            
        elif hw.primary_acceleration == AcceleratorType.METAL and hw.metal.available:
            config.device = "mps"
            # Conservative batch sizes for Metal
            config.batch_size = min(24, int(hw.memory.total_gb))
            config.num_workers = 1  # Metal only supports one GPU
            
        else:  # CPU or other
            # Set threads based on available cores
            config.threads_per_worker = min(4, max(1, hw.cpu.physical_cores // 2))
            
            # Set batch size based on available memory
            if hw.memory.total_gb > 32:
                config.batch_size = 16
            elif hw.memory.total_gb > 16:
                config.batch_size = 12
            elif hw.memory.total_gb > 8:
                config.batch_size = 8
            else:
                config.batch_size = 4
                
            # Set workers based on cores
            config.num_workers = min(4, max(1, hw.cpu.physical_cores // 2))
            
        return config


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
        device: str = "",
        max_length: int = 512,
        batch_size: int = 0,
        auto_optimize: bool = True,
        num_workers: int = 0,
        instruction: str = "",
    ):
        """
        Initialize the embedder with dynamic hardware detection.
        
        Args:
            model_name: Name/path of the model to use
            device: Device to use (cpu, cuda, mps, etc.). Empty for auto-detection
            max_length: Maximum sequence length
            batch_size: Batch size for inference. 0 for auto-detection
            auto_optimize: Whether to automatically optimize based on hardware
            num_workers: Number of worker threads. 0 for auto-detection
            instruction: Custom instruction for the embedding model
        """
        # Create config
        if auto_optimize:
            # Auto-detect optimal configuration
            self.config = EmbedderConfig.from_hardware(model_name)
            
            # Override with provided values
            if device:
                self.config.device = device
            if batch_size > 0:
                self.config.batch_size = batch_size
            if num_workers > 0:
                self.config.num_workers = num_workers
            if max_length != 512:
                self.config.max_length = max_length
            if instruction:
                self.config.instruction = instruction
        else:
            # Use provided values directly
            self.config = EmbedderConfig(
                model_name=model_name,
                device=device or "cpu",
                max_length=max_length,
                batch_size=batch_size or 8,
                auto_optimize=False,
                num_workers=num_workers or 1,
                instruction=instruction or EmbedderConfig().instruction
            )
        
        # Enable cuDNN benchmarking for CUDA
        if self.config.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        
        # Initialize model storage
        self.models: Dict[int, SentenceTransformer] = {}
        self.model_locks: Dict[int, Optional[SyncLock]] = {}
        
        logger.info(f"Loading embedding model: {self.config.model_name} on {self.config.device}")
        logger.info(f"Configuration: batch_size={self.config.batch_size}, "
                    f"workers={self.config.num_workers}, "
                    f"threads_per_worker={self.config.threads_per_worker}")
        
        # Load models
        self._load_models()

    def _load_models(self):
        """Load embedding models based on configuration."""
        try:
            # Configure CPU threads if using CPU
            if self.config.device == "cpu":
                torch.set_num_threads(self.config.threads_per_worker)
            
            # Load primary model
            model = SentenceTransformer(
                self.config.model_name, 
                device=self.config.device
            )
            
            # Apply half precision for CUDA if enabled
            if self.config.device.startswith("cuda") and self.config.use_half_precision:
                try:
                    model.half()
                    logger.info("Using FP16 precision for faster CUDA inference")
                except Exception as e:
                    logger.warning(f"Could not convert model to FP16: {e}")
            
            # Store model
            self.models[0] = model
            self.model_locks[0] = torch.multiprocessing.Lock() if self.config.num_workers > 1 else None

            # Load additional models on multi-GPU systems
            if (self.config.num_workers > 1 and 
                self.config.device.startswith("cuda") and 
                torch.cuda.device_count() > 1):
                
                for gpu_id in range(1, min(self.config.num_workers, torch.cuda.device_count())):
                    device_str = f"cuda:{gpu_id}"
                    logger.info(f"Loading additional model on {device_str}")
                    
                    # Load model for this GPU
                    model = SentenceTransformer(self.config.model_name, device=device_str)
                    
                    if self.config.use_half_precision:
                        try:
                            model.half()
                            logger.info(f"Using FP16 precision on {device_str}")
                        except Exception as e:
                            logger.warning(f"Could not convert model on {device_str} to FP16: {e}")
                    
                    self.models[gpu_id] = model
                    self.model_locks[gpu_id] = torch.multiprocessing.Lock()
                    
        except Exception as e:
            logger.error(f"Error loading model '{self.config.model_name}': {e}")
            raise

    def _get_model_for_worker(self, worker_id: int = 0) -> tuple:
        """
        Return the model and lock for a given worker.
        
        Args:
            worker_id: ID of the worker thread
            
        Returns:
            Tuple of (model, lock)
        """
        if (self.config.device.startswith("cuda") and 
            torch.cuda.device_count() > 1 and 
            len(self.models) > 1):
            gpu_id = worker_id % torch.cuda.device_count()
            return self.models[gpu_id], self.model_locks[gpu_id]
        else:
            return self.models[0], self.model_locks.get(0)

    def _embed_batch_with_worker(self, texts: List[str], worker_id: int) -> np.ndarray:
        """
        Embed a batch of texts using a specific worker.
        
        Args:
            texts: List of strings to embed
            worker_id: ID of the worker thread
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        model, lock = self._get_model_for_worker(worker_id)
        
        # Create instruction-based inputs
        input_texts = [get_detailed_instruct(self.config.instruction, text) for text in texts]
        
        if lock:
            lock.acquire()
            
        try:
            # Use sentence_transformers encode method
            embeddings = model.encode(
                input_texts,
                batch_size=min(len(input_texts), self.config.batch_size),
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
        
        Args:
            strings: List of strings to embed
            show_progress: Whether to show a progress bar
            
        Returns:
            Numpy array of embeddings with shape (len(strings), embedding_dim)
        """
        if not strings:
            return np.array([])
        
        # Truncate very long strings to avoid excessive tokenization time
        truncated = [s if len(s) <= 5000 else s[:5000] for s in strings]
        total = len(truncated)
        batch_size = self.config.batch_size
        progress_bar = tqdm(total=total, desc="Generating embeddings", unit="text") if show_progress and total > batch_size else None

        if self.config.num_workers == 1:
            # Single worker mode
            all_embeddings = []
            for i in range(0, total, batch_size):
                batch = truncated[i:i+batch_size]
                emb = self._embed_batch_with_worker(batch, worker_id=0)
                all_embeddings.append(emb)
                if progress_bar:
                    progress_bar.update(len(batch))
                # Periodically free memory
                if i % (batch_size * 10) == 0:
                    gc.collect()
                    if torch.cuda.is_available() and self.config.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        else:
            # Multi-worker mode
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []
                for i in range(0, total, batch_size):
                    batch = truncated[i:i+batch_size]
                    worker_id = (i // batch_size) % self.config.num_workers
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
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available() and self.config.device.startswith("cuda"):
            torch.cuda.empty_cache()
            
        return result

    def embed_string(self, text: str) -> np.ndarray:
        """
        Embed a single string.
        
        Args:
            text: String to embed
            
        Returns:
            Numpy array embedding vector
        """
        # Truncate very long text
        if len(text) > 2000:
            text = text[:2000]
            logger.debug("Truncated string to 2000 chars.")
        
        # Get model
        model, lock = self._get_model_for_worker(0)
        
        # Format with instruction
        input_text = get_detailed_instruct(self.config.instruction, text)
        
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
        """
        Calculate cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix with shape (len(embeddings1), len(embeddings2))
        """
        return embeddings1 @ embeddings2.T

    def close(self):
        """Release resources and free memory."""
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
    
    # Auto-optimize based on hardware
    embedder = EnhancedEmbedder(auto_optimize=True)
    
    start = time.time()
    embeddings = embedder.embed_strings(test_texts)
    duration = time.time() - start
    
    print(f"Generated {len(embeddings)} embeddings in {duration:.2f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test single embedding
    single_embedding = embedder.embed_string("This is a test")
    print(f"Single embedding shape: {single_embedding.shape}")
    
    # Calculate similarity matrix
    similarity = embedder.similarity(embeddings, embeddings)
    print("Similarity matrix:")
    print(similarity)
    
    embedder.close()