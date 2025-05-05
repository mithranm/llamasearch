import os
import json
import logging
import numpy as np
from pathlib import Path
import time
from logging.handlers import RotatingFileHandler  # Use rotating file handler

# --- Add import for our Qt logging components ---
# Use a try-except block for environments where UI/Qt might not be installed (e.g., pure CLI usage)
try:
    # --- Corrected import path if qt_logging is inside ui ---
    from .ui.qt_logging import QtLogHandler, qt_log_emitter

    _qt_logging_available = True
except ImportError:
    _qt_logging_available = False
    # Define dummy types if Qt isn't available, so type checking doesn't fail later
    QtLogHandler = type(None)  # Use type(None) as a placeholder type
    qt_log_emitter = None


def is_dev_mode() -> bool:
    """Determine if running in development mode."""
    return os.environ.get("LLAMASEARCH_DEV_MODE", "").lower() in ("1", "true", "yes")


def get_llamasearch_dir() -> Path:
    """Return the base directory for LlamaSearch data."""
    from .data_manager import data_manager  # Local import

    return Path(data_manager.get_data_paths()["base"])


def setup_logging(name, level=logging.INFO, use_qt_handler=True):
    """
    Set up logging to console, file, and optionally Qt signal emitter.
    """
    from .data_manager import data_manager  # Local import

    try:
        log_path_str = data_manager.get_data_paths().get("logs")
        if not log_path_str:
            project_root = Path(data_manager.get_data_paths()["base"])
            logs_dir = project_root / "logs"
        else:
            logs_dir = Path(log_path_str)

        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "llamasearch.log"

        logger = logging.getLogger(name)
        # Prevent adding handlers multiple times if logger already exists
        if logger.hasHandlers():
            logger.setLevel(level)  # Ensure level is updated if changed
            return logger  # Return existing logger

        # Logger doesn't have handlers, configure it
        logger.setLevel(level)

        # --- File Handler (Rotating) ---
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(level)

        # --- Console Handler ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Console fixed at INFO

        # --- Qt Handler (Conditional) ---
        qt_handler = None
        if use_qt_handler and _qt_logging_available and qt_log_emitter is not None:
            qt_handler = QtLogHandler(qt_log_emitter)  # type: ignore[reportCallIssue] # Restored qt_log_emitter argument
            # --- Check qt_handler before calling setLevel ---
            if qt_handler: 
                qt_handler.setLevel(level)
        elif use_qt_handler and not _qt_logging_available:
            logging.warning("Qt logging handler requested but Qt components not found.")

        # --- Formatters ---
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(simple_formatter)
        # --- Corrected: Check if qt_handler was successfully created ---
        if qt_handler:
            qt_handler.setFormatter(simple_formatter)

        # --- Add Handlers ---
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # --- Corrected: Check if qt_handler was successfully created ---
        if qt_handler:
            logger.addHandler(qt_handler)

        logger.propagate = False  # Prevent duplicate logs in root logger

        # Special handling for noisy libraries if needed
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("markdown_it").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.INFO)
        logging.getLogger("onnxruntime").setLevel(logging.WARNING)

        return logger

    except Exception as e:
        # Fallback to basic config if setup fails
        logging.basicConfig(level=logging.INFO)
        logging.error(
            f"Failed to configure custom logging: {e}. Using basic config.",
            exc_info=True,
        )
        return logging.getLogger(name)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and sets."""

    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return list(o)
        return super().default(o)


def log_query(
    query: str,
    chunks: list,
    response: str,
    debug_info: dict,
    full_logging: bool = False,
) -> str:
    """Logs the query, optimized chunks, and response to a JSON Lines file."""
    from .data_manager import data_manager  # Local import

    logger = setup_logging(
        __name__, use_qt_handler=False
    )  # Don't need Qt handler for this specific log
    try:
        logs_dir = Path(data_manager.get_data_paths()["logs"])
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create/access logs directory: {e}. Skipping query log.")
        return ""

    chunks_to_log = chunks
    if not full_logging and chunks:
        simplified_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                sc = {k: chunk.get(k) for k in ["id", "score"] if k in chunk}
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    sc["source"] = chunk["metadata"].get("source", "")
                    if chunk["metadata"].get("entities"):
                        sc["entities"] = chunk["metadata"]["entities"]
                if "text" in chunk and isinstance(chunk["text"], str):
                    sc["text_preview"] = (
                        (chunk["text"][:200] + "...")
                        if len(chunk["text"]) > 200
                        else chunk["text"]
                    )
                simplified_chunks.append(sc)
            else:
                simplified_chunks.append(str(chunk))
        chunks_to_log = simplified_chunks

    optimized_debug_info = {}
    if isinstance(debug_info, dict):
        for key in ["retrieval_time", "generation_time", "total_time", "intent"]:
            if key in debug_info:
                optimized_debug_info[key] = debug_info[key]
        if full_logging:
            for key, value in debug_info.items():
                if key not in optimized_debug_info and key != "chunks":
                    optimized_debug_info[key] = value

    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "chunks": chunks_to_log,
        "response": response,
        "debug_info": optimized_debug_info,
    }

    log_file = logs_dir / "query_log.jsonl"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, cls=NumpyEncoder)
            f.write("\n")
        return str(log_file)
    except Exception as e:
        logger.error(f"Error saving query log to {log_file}: {e}")
        return ""
