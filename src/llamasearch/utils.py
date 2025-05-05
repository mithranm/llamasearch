# src/llamasearch/utils.py

import json
import logging
import logging.handlers  # Import handlers submodule
import os
import sys  # <<< Added sys import >>>
import time
from pathlib import Path
from typing import Optional

import numpy as np

# --- Global variable to hold the single QtLogHandler instance ---
_qt_log_handler_instance: Optional[logging.Handler] = None
_qt_logging_available = False

try:
    # <<< Check if submodule exists before importing from it >>>
    import llamasearch.ui.qt_logging

    QtLogHandler = llamasearch.ui.qt_logging.QtLogHandler
    qt_log_emitter = llamasearch.ui.qt_logging.qt_log_emitter
    _qt_logging_available = True
except (ImportError, AttributeError):
    QtLogHandler = None  # Assign None if not available
    qt_log_emitter = None
    _qt_logging_available = False


def is_dev_mode() -> bool:
    """Determine if running in development mode."""
    return os.environ.get("LLAMASEARCH_DEV_MODE", "").lower() in ("1", "true", "yes")


def get_llamasearch_dir() -> Path:
    """Return the base directory for LlamaSearch data."""
    from .data_manager import data_manager  # Local import

    base_path_str = data_manager.get_data_paths().get("base")
    if base_path_str:
        path = Path(base_path_str)
        path.mkdir(parents=True, exist_ok=True)  # Ensure exists
        return path
    else:
        # Fallback if data_manager somehow doesn't have 'base'
        fallback_path = Path.home() / ".llamasearch"
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path


def setup_logging(
    name="llamasearch", level=logging.INFO, use_qt_handler=False
) -> logging.Logger:
    """
    Set up logging to console, file, and optionally Qt signal emitter.
    Uses a root logger 'llamasearch' and returns child loggers.
    Manages a single QtLogHandler instance.
    """
    global _qt_log_handler_instance
    from .data_manager import data_manager  # Local import

    root_logger_name = "llamasearch"
    logger = logging.getLogger(name)  # Get the specific logger requested

    root_logger = logging.getLogger(root_logger_name)
    # Configure root logger only once
    if not root_logger.handlers:  # Check if handlers are already configured
        root_logger.setLevel(
            logging.DEBUG
        )  # Set root logger to DEBUG to capture everything

        try:
            log_path_str = data_manager.get_data_paths().get("logs")
            logs_dir = (
                Path(log_path_str) if log_path_str else get_llamasearch_dir() / "logs"
            )
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / "llamasearch.log"

            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            # Use a simpler format for console/Qt for better readability
            simple_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )

            # File Handler - Level set by overall 'level' initially, but logger level controls final output
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # File handler captures DEBUG and up
            root_logger.addHandler(file_handler)

            # Console Handler - Fixed DEBUG level unless debug mode explicitly sets child logger lower
            console_handler = logging.StreamHandler(
                sys.stdout
            )  # Use stdout for console
            console_handler.setFormatter(simple_formatter)
            console_handler.setLevel(logging.DEBUG)  # Console fixed at DEBUG by default
            root_logger.addHandler(console_handler)

            # Qt Handler
            # <<< FIX: Check QtLogHandler type correctly before isinstance >>>
            if (
                use_qt_handler
                and _qt_logging_available
                and qt_log_emitter is not None
                and QtLogHandler is not None
            ):
                if _qt_log_handler_instance is None:
                    _qt_log_handler_instance = QtLogHandler(
                        qt_log_emitter
                    )  # Pass emitter
                    _qt_log_handler_instance.setFormatter(simple_formatter)
                    _qt_log_handler_instance.setLevel(
                        logging.INFO
                    )  # Qt handler also starts at INFO
                    root_logger.addHandler(_qt_log_handler_instance)
                    logging.info("Attached QtLogHandler to root logger.")
                else:
                    # Ensure existing handler's level is appropriate (usually INFO)
                    # The level might be changed later by __main__ or app_logic based on debug flags
                    _qt_log_handler_instance.setLevel(logging.INFO)

            elif use_qt_handler and not _qt_logging_available:
                logging.warning(
                    "Qt logging requested but Qt components not found or failed to import."
                )

            # Prevent messages logged to child loggers from propagating to the root logger's handlers
            # if the child logger also has handlers (avoids duplicate messages).
            # Set propagate=False on the root logger if you *only* want handlers attached directly to it.
            # Set propagate=True (default) if you want messages to flow up to root handlers.
            # Let's keep propagation for flexibility unless issues arise.
            # root_logger.propagate = False

            # Silence noisy libraries by setting their log level higher
            noisy_libs = [
                "urllib3",
                "matplotlib",
                "PIL",
                "asyncio",
                "markdown_it",
                "sentence_transformers",
                "huggingface_hub",
                "onnxruntime",
                "chromadb",
                "hpack",
                "httpx",
                "watchfiles",
                "uvicorn",
                "spacy",
                "py_cpuinfo",
                "filelock",
                "multiprocessing",
            ]  # Added more
            for lib_name in noisy_libs:
                logging.getLogger(lib_name).setLevel(logging.WARNING)

        except Exception as e:
            # Fallback logging if setup fails
            logging.basicConfig(level=logging.INFO)
            logging.error(
                f"Failed custom logging setup: {e}. Using basic config.", exc_info=True
            )

    return logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types, sets, and Path objects."""

    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return list(o)  # Handle sets
        if isinstance(o, Path):
            return str(o)  # Handle Path objects
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="ignore")  # Handle bytes
        # Let the base class default method raise the TypeError
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

    # Use specific logger instance, disable Qt handler for this specific log action
    query_logger = setup_logging("llamasearch.query_log", use_qt_handler=False)
    try:
        log_path_str = data_manager.get_data_paths().get("logs")
        logs_dir = (
            Path(log_path_str) if log_path_str else get_llamasearch_dir() / "logs"
        )
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        query_logger.error(
            f"Cannot create/access logs directory: {e}. Skipping query log."
        )
        return ""

    chunks_to_log = (
        chunks  # Keep original chunks for now, consider simplification later if needed
    )
    # Basic simplification example (can be customized)
    if not full_logging and isinstance(chunks, list):
        simplified_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                sc = {
                    k: chunk.get(k)
                    for k in ["chunk_id", "score", "source_path", "filename"]
                    if k in chunk
                }
                if "document" in chunk and isinstance(chunk["document"], str):
                    sc["text_preview"] = (
                        (chunk["document"][:150] + "...")
                        if len(chunk["document"]) > 150
                        else chunk["document"]
                    )
                simplified_chunks.append(sc)
            else:  # Handle non-dict chunks if they occur
                simplified_chunks.append(str(chunk)[:200])
        chunks_to_log = simplified_chunks

    # Log essential debug info, add more if full_logging
    optimized_debug_info = {}
    if isinstance(debug_info, dict):
        for key in [
            "retrieval_time",
            "llm_generation_time",
            "total_query_time",
            "vector_results_count",
            "bm25_results_count",
            "final_prompt_len_tokens",
        ]:
            if key in debug_info:
                optimized_debug_info[key] = debug_info[key]
        if full_logging:
            # Add more details in full logging mode, avoid very large raw data if possible
            optimized_debug_info.update(
                {
                    k: v
                    for k, v in debug_info.items()
                    if k not in optimized_debug_info and k != "raw_llm_output"
                }
            )  # Exclude raw LLM output by default

    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "chunks_retrieved_details": chunks_to_log,  # Log simplified or full chunks
        "response": response,
        "debug_info": optimized_debug_info,
    }

    log_file = logs_dir / "query_log.jsonl"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            # <<< Use NumpyEncoder here >>>
            json.dump(log_data, f, ensure_ascii=False, cls=NumpyEncoder)
            f.write("\n")
        return str(log_file)
    except Exception as e:
        query_logger.error(f"Error saving query log to {log_file}: {e}")
        return ""
