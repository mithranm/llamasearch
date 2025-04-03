"""
Optimized embedding module with dynamic batch size and hardware detection
"""

import numpy as np
import logging
import torch
import gc
import platform
import os
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def detect_hardware_capabilities():
    """
    Detect available hardware acceleration and memory capacity
    to determine optimal batch size.
    
    Returns:
        dict: Hardware capabilities information
    """
    capabilities = {
        "device": "cpu",
        "recommended_batch_size": 4,  # Default conservative value
        "max_length": 512,
        "memory_optimized": False
    }
    
    # Check for CUDA
    if torch.cuda.is_available():
        capabilities["device"] = "cuda"
        capabilities["cuda_device_count"] = torch.cuda.device_count()
        capabilities["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        # Estimate available GPU memory (in GB)
        try:
            free_memory = torch.cuda.mem_get_info(0)[0] / (1024**3)  # Convert bytes to GB
            total_memory = torch.cuda.mem_get_info(0)[1] / (1024**3)
            capabilities["cuda_free_memory_gb"] = free_memory
            capabilities["cuda_total_memory_gb"] = total_memory
            
            # Dynamically set batch size based on available GPU memory
            if free_memory > 10:  # More than 10GB free
                capabilities["recommended_batch_size"] = 32
            elif free_memory > 6:  # 6-10GB free
                capabilities["recommended_batch_size"] = 16
            elif free_memory > 2:  # 2-6GB free
                capabilities["recommended_batch_size"] = 8
            else:  # Less than 2GB
                capabilities["recommended_batch_size"] = 4
        except:
            # Fallback if mem_get_info is not available
            capabilities["recommended_batch_size"] = 8
    
    # Check for Apple Silicon (M1/M2/M3) with Metal support
    elif platform.system() == "Darwin" and platform.processor() == "arm":
        capabilities["device"] = "mps" if torch.backends.mps.is_available() else "cpu"
        capabilities["is_apple_silicon"] = True
        
        # For Apple Silicon, we'll use a heuristic since direct memory query isn't available
        try:
            # Try to get total system memory as a proxy
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                system_mem_gb = int(result.stdout.strip()) / (1024**3)
                capabilities["system_memory_gb"] = system_mem_gb
                
                # Set batch size based on system memory
                if system_mem_gb > 32:  # 32+ GB system RAM
                    capabilities["recommended_batch_size"] = 24
                elif system_mem_gb > 16:  # 16-32GB system RAM
                    capabilities["recommended_batch_size"] = 16
                elif system_mem_gb > 8:  # 8-16GB system RAM
                    capabilities["recommended_batch_size"] = 8
                else:  # Less than 8GB
                    capabilities["recommended_batch_size"] = 4
        except:
            # Fallback if sysctl fails
            capabilities["recommended_batch_size"] = 8
    
    # For CPU-only systems, determine based on available system memory
    else:
        try:
            import psutil
            system_mem_gb = psutil.virtual_memory().total / (1024**3)
            capabilities["system_memory_gb"] = system_mem_gb
            
            # Set batch size based on system memory
            if system_mem_gb > 32:  # 32+ GB system RAM
                capabilities["recommended_batch_size"] = 16
            elif system_mem_gb > 16:  # 16-32GB system RAM
                capabilities["recommended_batch_size"] = 8
            elif system_mem_gb > 8:  # 8-16GB system RAM
                capabilities["recommended_batch_size"] = 4
            else:  # Less than 8GB
                capabilities["recommended_batch_size"] = 2
        except:
            # Fallback if psutil is not available
            capabilities["recommended_batch_size"] = 4
    
    # Log the detected capabilities
    logger.info(f"Detected hardware capabilities: {capabilities}")
    return capabilities


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: Optional[int] = None,
        auto_optimize: bool = True,
    ):
        """
        Initialize the embedder with the sentence-transformers model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cpu', 'cuda', or 'mps')
            max_length: Maximum token length for input text
            batch_size: Batch size for processing multiple texts (set to None for auto)
            auto_optimize: Whether to automatically optimize settings based on hardware
        """
        try:
            self.max_length = max_length
            self.auto_optimize = auto_optimize
            
            # Ensure batch_size is always initialized to a valid value
            default_batch_size = 8
            
            # Auto-detect optimal settings based on hardware
            if auto_optimize:
                self.capabilities = detect_hardware_capabilities()
                self.device = device or self.capabilities["device"]
                
                # Use the recommended batch size if batch_size is None
                if batch_size is None:
                    self.batch_size = self.capabilities["recommended_batch_size"]
                else:
                    # Use the specified batch_size if provided
                    self.batch_size = batch_size
            else:
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                # Use the specified batch_size or default if not auto-optimizing
                self.batch_size = batch_size if batch_size is not None else default_batch_size
            
            logger.info(f"Loading sentence-transformer model: {model_name}")
            logger.info(f"Using device: {self.device}, batch size: {self.batch_size}")
            
            self.model = SentenceTransformer(model_name_or_path=model_name, device=self.device)
            
            # Perform a quick benchmark to fine-tune batch size if auto_optimize is enabled
            if auto_optimize and batch_size is None:  # Only benchmark if we're auto-detecting batch size
                self._benchmark_and_adjust()
                
            logger.info(f"Loaded sentence-transformer model {model_name} on {self.device}")
            logger.info(f"Using batch size: {self.batch_size}")
            
        except (OSError, ValueError) as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            raise
    
    def _benchmark_and_adjust(self):
        """Perform a quick benchmark to find the optimal batch size"""
        logger.info("Running embedding benchmark to optimize batch size...")
        
        # Create test data with varying lengths
        test_strings = [
            "This is a short test sentence.",
            "Here's a medium length test sentence with more words to process by the model.",
            "This is a longer test sentence that contains multiple phrases and clauses to better test the embedding performance with longer texts that need to be processed by the transformer model in batches."
        ] * 5  # Repeat to get a reasonable sample size
        
        # Test different batch sizes to find optimal
        # Ensure all values are integers and > 0
        start_batch = max(1, self.batch_size // 2)
        double_batch = min(64, self.batch_size * 2)
        
        batch_sizes_to_test = [
            start_batch,          # Half the recommended
            self.batch_size,      # The recommended
            double_batch          # Double the recommended (with upper limit)
        ]
        
        best_batch_size = self.batch_size
        best_throughput = 0
        
        for test_batch_size in batch_sizes_to_test:
            # Skip duplicates
            if test_batch_size in batch_sizes_to_test[:batch_sizes_to_test.index(test_batch_size)]:
                continue
                
            try:
                # Test embedding with this batch size
                start_time = time.time()
                with torch.no_grad():
                    for i in range(0, len(test_strings), test_batch_size):
                        batch = test_strings[i:i+test_batch_size]
                        _ = self.model.encode(batch, convert_to_numpy=True)
                
                end_time = time.time()
                elapsed = end_time - start_time
                throughput = len(test_strings) / elapsed
                
                logger.info(f"Batch size {test_batch_size}: {throughput:.2f} samples/sec, {elapsed:.4f}s total")
                
                # Update best batch size if this one is faster
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = test_batch_size
            
            except RuntimeError as e:
                # Memory error, skip this batch size
                logger.warning(f"Batch size {test_batch_size} failed: {e}")
                # Try to recover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Update batch size to the optimal one
        logger.info(f"Selected optimal batch size: {best_batch_size} ({best_throughput:.2f} samples/sec)")
        self.batch_size = best_batch_size

    def embed_string(self, string: str) -> np.ndarray:
        """
        Get embedding for a single string.

        Args:
            string: Input text string

        Returns:
            Numpy array containing the embedding
        """
        if not isinstance(string, str):
            raise TypeError("Input must be a string.")

        logger.debug(f"Embedding single string ({len(string)} chars)")

        # Process the string to enforce a max length
        if len(string) > 2000:
            string = string[:2000]
            logger.debug("Truncated string to 2000 chars to reduce memory usage")

        # Get embedding with minimal memory management
        with torch.no_grad():
            embedding = self.model.encode(string, convert_to_numpy=True)

        # Clean up memory only if absolutely necessary
        if self.device == "cpu" or len(string) > 1000:
            gc.collect()
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.empty_cache()

        return embedding

    def embed_batch(self, strings: list) -> np.ndarray:
        """
        Get embeddings for a batch of strings, optimizing for available hardware.

        Args:
            strings: List of input text strings

        Returns:
            Numpy array containing embeddings
        """
        if not strings:
            return np.array([])

        logger.info(f"Embedding batch of {len(strings)} strings with batch size {self.batch_size}")

        # For memory efficiency, process in optimized batches
        all_embeddings = []

        with tqdm(
            total=len(strings), desc="Generating embeddings", unit="text"
        ) as pbar:
            for i in range(0, len(strings), self.batch_size):
                batch = strings[i:i+self.batch_size]
                
                # Truncate very long texts to reduce memory pressure
                truncated_batch = []
                for text in batch:
                    if len(text) > 2000:
                        truncated_batch.append(text[:2000])
                    else:
                        truncated_batch.append(text)

                # Get embeddings for this batch
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        truncated_batch, convert_to_numpy=True
                    )

                all_embeddings.append(batch_embeddings)

                # Force garbage collection conditionally - not after every batch
                # Only do garbage collection periodically to improve performance
                if i % (self.batch_size * 5) == 0 and i > 0:
                    gc.collect()
                    if torch.cuda.is_available() and self.device == "cuda":
                        torch.cuda.empty_cache()

                # Update progress bar
                pbar.update(len(batch))

        # Combine all into a numpy array
        combined = np.vstack(all_embeddings)

        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

        return combined


if __name__ == "__main__":
    try:
        # Configure logging for standalone testing
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Test with auto-optimization
        embedder = Embedder(auto_optimize=True)

        # Test embedding
        text = "This is an example sentence."
        embedding = embedder.embed_string(text)
        logger.info(f"Embedding shape: {embedding.shape}")

        # Test batch embedding with optimized settings
        batch = ["First test sentence.", "Second test sentence."] * 5
        batch_embeddings = embedder.embed_batch(batch)
        logger.info(f"Batch embedding shape: {batch_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error: {e}")