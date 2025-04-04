# llamasearch/core/resource_manager.py

import os
import logging
import platform
import threading
import concurrent.futures
import psutil
import torch
from typing import Dict, Any, List, Optional, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class SafeThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """ThreadPoolExecutor that exposes its max_workers attribute safely."""
    
    @property
    def max_workers(self) -> int:
        """Get the maximum number of workers in this pool."""
        # In Python 3.7+, max_workers is stored as _max_workers
        # This property provides safe access to it
        if hasattr(self, "_max_workers"):
            return self._max_workers
        # Fallback for compatibility
        return len(self._threads) if hasattr(self, "_threads") else 1


class HardwareProfile:
    """Detects and provides information about the system's hardware capabilities."""
    
    def __init__(self):
        # CPU information
        self.cpu_count = os.cpu_count() or 4
        self.physical_cores = psutil.cpu_count(logical=False) or 2
        self.logical_cores = psutil.cpu_count(logical=True) or 4
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # GPU information
        self.has_cuda = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_cuda else 0
        self.gpu_info = []
        
        # Detect Apple Silicon
        self.is_apple_silicon = (
            platform.system() == "Darwin" and platform.processor() == "arm"
        )
        self.has_mps = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        
        # Detailed GPU information gathering
        if self.has_cuda:
            for i in range(self.gpu_count):
                try:
                    gpu_props = torch.cuda.get_device_properties(i)
                    total_memory = gpu_props.total_memory
                    # Get free memory if possible
                    try:
                        free_memory = (
                            torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                        )
                    except:
                        # If memory_reserved is not available, use a percentage estimate
                        free_memory = total_memory * 0.8  # Assume ~80% free
                    
                    self.gpu_info.append({
                        "name": gpu_props.name,
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        "total_memory_gb": total_memory / (1024**3),
                        "free_memory_gb": free_memory / (1024**3),
                        "multi_processor_count": gpu_props.multi_processor_count
                    })
                except Exception as e:
                    logger.warning(f"Error getting properties for GPU {i}: {e}")
        
        # Log the detected hardware profile
        self._log_hardware_profile()
    
    def _log_hardware_profile(self):
        """Log detected hardware capabilities."""
        logger.info(
            f"CPU: {self.logical_cores} logical cores ({self.physical_cores} physical)"
        )
        logger.info(
            f"Memory: {self.total_memory_gb:.1f}GB total, {self.available_memory_gb:.1f}GB available"
        )
        
        if self.has_cuda:
            for i, gpu in enumerate(self.gpu_info):
                logger.info(
                    f"GPU {i}: {gpu['name']}, "
                    f"{gpu['free_memory_gb']:.1f}GB/{gpu['total_memory_gb']:.1f}GB free, "
                    f"Compute {gpu['compute_capability']}"
                )
        elif self.has_mps:
            logger.info("GPU: Apple Silicon with Metal Performance Shaders support")
        else:
            logger.info("GPU: None detected")
    
    def get_optimal_worker_count(
        self,
        memory_per_worker_gb: float = 1.0, 
        cpu_intensive: bool = True
    ) -> int:
        """
        Calculate the optimal number of worker threads based on hardware.
        
        Args:
            memory_per_worker_gb: Estimated memory usage per worker in GB
            cpu_intensive: Whether the task is CPU-intensive (True) or I/O-bound (False)
            
        Returns:
            int: Recommended number of worker threads
        """
        # Base number on CPU cores
        if cpu_intensive:
            # For CPU-intensive tasks, typically use physical_cores + 1 or 2
            base_workers = min(self.physical_cores + 1, self.logical_cores)
        else:
            # For I/O-bound tasks, we can use more threads
            base_workers = min(self.logical_cores * 2, 32)  # Cap at 32 to avoid oversubscription
        
        # Adjust based on memory constraints
        if memory_per_worker_gb > 0:
            memory_limited_workers = int(
                self.available_memory_gb / memory_per_worker_gb * 0.8
            )  # 80% of capacity
            workers = min(base_workers, memory_limited_workers)
        else:
            workers = base_workers
            
        # Ensure at least one worker
        return max(1, workers)
    
    def get_optimal_embedding_config(self) -> Dict[str, Any]:
        """
        Determine optimal configuration for an embedding model.
        
        Returns:
            Dict with config including device, batch_size, etc.
        """
        config = {
            "device": "cpu",
            "batch_size": 4,
            "max_sequence_length": 512,
            "multi_process": False,
            "threads_per_worker": 1
        }
        
        # If high-end CUDA device
        if self.has_cuda and self.gpu_info and self.gpu_info[0]["free_memory_gb"] > 4.0:
            config["device"] = "cuda"
            # For example, scale batch size with free GPU memory
            config["batch_size"] = min(32, int(self.gpu_info[0]["free_memory_gb"] * 4))
            config["multi_process"] = (self.gpu_count > 1)
            
        # Apple Silicon MPS
        elif self.has_mps:
            config["device"] = "mps"
            # More conservative batch sizes for MPS
            if self.total_memory_gb > 16:
                config["batch_size"] = 24
            elif self.total_memory_gb > 8:
                config["batch_size"] = 16
            else:
                config["batch_size"] = 8
                
        # Otherwise, CPU
        else:
            if self.total_memory_gb > 32:
                config["batch_size"] = 16
                config["threads_per_worker"] = 4
            elif self.total_memory_gb > 16:
                config["batch_size"] = 12
                config["threads_per_worker"] = 3
            elif self.total_memory_gb > 8:
                config["batch_size"] = 8
                config["threads_per_worker"] = 2
            else:
                config["batch_size"] = 4
                config["threads_per_worker"] = 1
                
            # Possibly enable multiprocess if we have a high-core CPU
            config["multi_process"] = (self.physical_cores > 4)
        
        return config


class ResourceManager:
    """
    Manages computational resources and provides thread pools for parallel processing.
    Optimizes resource allocation based on available hardware and task requirements.
    """
    
    def __init__(self, auto_optimize: bool = True):
        self.hardware = HardwareProfile()
        self.auto_optimize = auto_optimize
        
        # Thread pools for different task types
        self._executor_pools: Dict[str, SafeThreadPoolExecutor] = {}
        self._executor_locks: Dict[str, threading.RLock] = {}
        self._thread_local = threading.local()
        
        # Create default thread pools if auto_optimize is enabled
        if auto_optimize:
            self._create_default_pools()
            
        logger.info(f"ResourceManager initialized with auto_optimize={auto_optimize}")
    
    def _create_default_pools(self):
        """Create default thread pools based on hardware detection."""
        # I/O pool for file operations
        io_workers = self.hardware.get_optimal_worker_count(
            memory_per_worker_gb=0.2, cpu_intensive=False
        )
        self._executor_pools["io"] = SafeThreadPoolExecutor(
            max_workers=io_workers, 
            thread_name_prefix="io_worker"
        )
        logger.info(f"Created I/O thread pool with {io_workers} workers")
        
        # CPU pool for compute-intensive tasks
        cpu_workers = self.hardware.get_optimal_worker_count(
            memory_per_worker_gb=0.5, cpu_intensive=True
        )
        self._executor_pools["cpu"] = SafeThreadPoolExecutor(
            max_workers=cpu_workers, 
            thread_name_prefix="cpu_worker"
        )
        logger.info(f"Created CPU thread pool with {cpu_workers} workers")
        
        # Create locks for each pool
        for pool_name in self._executor_pools:
            self._executor_locks[pool_name] = threading.RLock()
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get optimal configuration for embedding based on hardware profile."""
        return self.hardware.get_optimal_embedding_config()
    
    def get_executor(self, pool_type: str = "cpu") -> SafeThreadPoolExecutor:
        """
        Get a (Safe)ThreadPoolExecutor for the specified task type.
        
        Args:
            pool_type: Type of pool ('io', 'cpu', etc.)
            
        Returns:
            SafeThreadPoolExecutor instance
        """
        if pool_type not in self._executor_pools:
            # We’ll lock on a global basis to ensure we don't create multiple pools concurrently
            with threading.RLock():
                if pool_type not in self._executor_pools:
                    if pool_type == "io":
                        workers = self.hardware.get_optimal_worker_count(
                            memory_per_worker_gb=0.2, cpu_intensive=False
                        )
                    else:
                        workers = self.hardware.get_optimal_worker_count(
                            memory_per_worker_gb=0.5, cpu_intensive=True
                        )
                    self._executor_pools[pool_type] = SafeThreadPoolExecutor(
                        max_workers=workers,
                        thread_name_prefix=f"{pool_type}_worker"
                    )
                    self._executor_locks[pool_type] = threading.RLock()
                    
                    logger.info(
                        f"Created on-demand '{pool_type}' thread pool with {workers} workers"
                    )
        
        return self._executor_pools[pool_type]
    
    def parallel_map(
        self,
        func: Callable[[T], R],
        items: List[T],
        pool_type: str = "cpu",
        chunk_size: Optional[int] = None
    ) -> List[R]:
        """
        Apply a function to each item in parallel using the specified thread pool.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            pool_type: Type of thread pool to use ('cpu', 'io', etc.)
            chunk_size: Optional chunk size for the map operation (None for auto)
            
        Returns:
            List of results in the same order as `items`
        """
        executor = self.get_executor(pool_type)
        
        # Auto-determine chunk_size if not specified
        if chunk_size is None or chunk_size < 1:
            if pool_type == "io":
                # Smaller chunks for I/O-bound tasks => keep all workers busy
                chunk_size = max(1, len(items) // (executor.max_workers * 4))
            else:
                # For CPU tasks, bigger chunks => reduce overhead
                chunk_size = max(1, len(items) // executor.max_workers)
            
            # clamp chunk_size to a reasonable upper bound (avoid huge chunk sizes)
            chunk_size = min(chunk_size, 100)
        
        results = list(executor.map(func, items, chunksize=chunk_size))
        return results
    
    def close(self):
        """Shutdown all thread pool executors."""
        for name, executor in self._executor_pools.items():
            logger.info(f"Shutting down '{name}' thread pool")
            executor.shutdown(wait=True)
        self._executor_pools.clear()
        self._executor_locks.clear()
        logger.info("All thread pools shut down.")


# Global (singleton) instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager(auto_optimize: bool = True) -> ResourceManager:
    """
    Retrieve or create the global ResourceManager instance.
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager(auto_optimize=auto_optimize)
    return _resource_manager


# Simple test block if run as main
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    manager = get_resource_manager()
    config = manager.get_embedding_config()
    print("Optimal embedding config:", config)
    
    # A quick test of parallel_map
    def square(x: int) -> int:
        return x * x
    
    data = list(range(100))
    results = manager.parallel_map(square, data, pool_type="cpu")
    print("Parallel map results (len):", len(results))
    
    manager.close()
