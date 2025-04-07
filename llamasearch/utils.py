import os
import json
import logging
import numpy as np
from datetime import datetime
import time
from .setup_utils import find_project_root

def setup_logging(name, level=logging.INFO):
    """
    Set up logging to both console and file.
    """
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    # File handler with daily filename
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"{name.split('.')[-1]}_{today}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    # Console handler: changed to INFO so you see ingestion logs, etc.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # <--- was WARNING, now INFO
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that gracefully handles numpy types (e.g. float32, int64)
    and other non-JSON-serializable types.
    """
    def default(self, o):
        # Convert float32/float64 -> float
        if isinstance(o, np.floating):
            return float(o)
        # Convert int64/int32 -> int
        if isinstance(o, np.integer):
            return int(o)
        # Convert array -> list (optional)
        if isinstance(o, np.ndarray):
            return o.tolist()
        # Convert set -> list
        if isinstance(o, set):
            return list(o)
        return super().default(o)

def log_query(query: str, chunks: list, response: str, debug_info: dict, full_logging: bool = False) -> str:
    """
    Logs the query along with optimized chunk information to save disk space and improve performance.
    
    Args:
        query: The search query
        chunks: List of chunks returned by the search
        response: The generated response
        debug_info: Additional debugging information
        full_logging: Whether to log full chunk data (default: False for storage efficiency)
        
    Returns:
        Path to the generated log file
    """
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Process chunks to reduce storage space if not full logging
    if not full_logging and chunks:
        simplified_chunks = []
        for chunk in chunks:
            # Create a simplified version with only essential info
            simplified_chunk = {
                "id": chunk.get("id", ""),
                "score": chunk.get("score", 0),
                "source": chunk.get("metadata", {}).get("source", "") if chunk.get("metadata") else "",
            }
            
            # Include a truncated version of the text for context
            if "text" in chunk:
                text = chunk["text"]
                simplified_chunk["text_preview"] = text[:2000] + "..." if len(text) > 2000 else text
            
            # Include entity information if available (useful for debugging entity queries)
            if chunk.get("metadata") and "entities" in chunk["metadata"]:
                simplified_chunk["entities"] = chunk["metadata"]["entities"]
                
            simplified_chunks.append(simplified_chunk)
        chunks_to_log = simplified_chunks
    else:
        chunks_to_log = chunks
    
    # Also optimize debug_info to reduce storage
    optimized_debug_info = {}
    if debug_info:
        # Always include timing information
        for key in ["retrieval_time", "generation_time", "total_time"]:
            if key in debug_info:
                optimized_debug_info[key] = debug_info[key]
        
        # Include intent analysis if available
        if "intent" in debug_info:
            optimized_debug_info["intent"] = debug_info["intent"]
            
        # Only include other debug info if full logging is enabled
        if full_logging:
            for key, value in debug_info.items():
                if key not in optimized_debug_info and key != "chunks":
                    optimized_debug_info[key] = value
    
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "chunks": chunks_to_log,
        "response": response,
        "debug_info": optimized_debug_info
    }
    
    # Use timestamp to create unique log file name
    log_file = os.path.join(logs_dir, f"query_{int(time.time())}.json")
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    except Exception as e:
        print(f"Error saving log: {e}")
        
    return log_file