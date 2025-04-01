# llamasearch/utils.py

import os
import json
import logging
import numpy as np
from datetime import datetime

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

    # Console handler only for WARNING+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

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
    Custom JSON encoder that gracefully handles numpy types (e.g. float32, int64).
    """

    def default(self, obj):
        # Convert float32 -> float
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        # Convert int64 -> int
        if isinstance(obj, np.integer):
            return int(obj)
        # Convert array -> list (optional)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def log_query(query, context_chunks, response, debug_info=None):
    """
    Log query, retrieved chunks, and generated response to a JSON file.
    Converting all np.float32 to python float so it won't crash.
    """
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"query_{timestamp}.json")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "context_chunks": context_chunks,
        "response": response,
    }
    if debug_info is not None:
        log_data["debug_info"] = debug_info

    with open(log_file, "w", encoding="utf-8") as f:
        # Use our custom NumpyEncoder
        json.dump(log_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return log_file
