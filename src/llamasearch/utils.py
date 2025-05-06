# src/llamasearch/utils.py

import json
import logging
import logging.handlers
import sys
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
except (ImportError, AttributeError, ModuleNotFoundError): # Catch ModuleNotFoundError too
    QtLogHandler = None  # Assign None if not available
    qt_log_emitter = None
    _qt_logging_available = False

# Removed is_dev_mode function

def get_llamasearch_dir() -> Path:
    """Return the base directory for LlamaSearch data (~/.llamasearch)."""
    # Simplified: Always return the default base path used by DataManager
    # This avoids potential inconsistencies if DataManager were somehow bypassed.
    path = Path.home() / ".llamasearch"
    path.mkdir(parents=True, exist_ok=True) # Ensure it exists
    return path


def setup_logging(
    name="llamasearch", level=logging.INFO, use_qt_handler=False
) -> logging.Logger:
    """
    Set up logging to console, file, and optionally Qt signal emitter.
    Uses a root logger 'llamasearch' and returns child loggers.
    Manages a single QtLogHandler instance and updates existing handler levels.
    """
    global _qt_log_handler_instance

    root_logger_name = "llamasearch"
    logger_to_return = logging.getLogger(name)  # Get the specific logger requested

    root_logger = logging.getLogger(root_logger_name)
    root_logger.setLevel(logging.DEBUG)  # Root logger always captures DEBUG and up

    # --- Formatters ---
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Configure Handlers (add if not present, otherwise update) ---
    try:
        base_dir = get_llamasearch_dir()
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "llamasearch.log"

        # --- File Handler ---
        # Check if a file handler for our log file already exists
        file_handler_exists = any(
            isinstance(h, logging.handlers.RotatingFileHandler) and
            getattr(h, 'baseFilename', '') == str(log_file)
            for h in root_logger.handlers
        )
        if not file_handler_exists:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # File handler always DEBUG
            root_logger.addHandler(file_handler)

        # --- Console Handler ---
        # Try to find an existing stream handler for stdout
        console_handler: Optional[logging.StreamHandler] = None
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) == sys.stdout:
                console_handler = h
                break
        
        if console_handler is None:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        console_handler.setLevel(level) # Set/Update console handler level

        # --- Qt Handler ---
        if use_qt_handler and _qt_logging_available and qt_log_emitter and QtLogHandler:
            if _qt_log_handler_instance is None:
                _qt_log_handler_instance = QtLogHandler(qt_log_emitter)
                _qt_log_handler_instance.setFormatter(simple_formatter)
                root_logger.addHandler(_qt_log_handler_instance)
                logging.info("Attached QtLogHandler to root logger.")
            # Always set/update Qt handler level if use_qt_handler is true
            if _qt_log_handler_instance: # Ensure it was created successfully
                 _qt_log_handler_instance.setLevel(level)
        elif use_qt_handler and not _qt_logging_available:
            logging.warning("Qt logging requested but Qt components not found.")

        # --- Silence Noisy Libraries (only once) ---
        if not getattr(root_logger, '_noisy_libs_silenced', False):
            noisy_libs = [
                "urllib3", "matplotlib", "PIL", "asyncio", "markdown_it",
                "sentence_transformers", "huggingface_hub", "onnxruntime",
                "chromadb", "hpack", "httpx", "watchfiles", "uvicorn", "spacy",
                "py_cpuinfo", "filelock", "multiprocessing", "whoosh",
            ]
            for lib_name in noisy_libs:
                logging.getLogger(lib_name).setLevel(logging.WARNING)
            root_logger._noisy_libs_silenced = True # Mark as done

    except Exception as e:
        # Fallback logging if setup fails
        logging.basicConfig(level=logging.INFO) # This resets handlers
        logging.error(
            f"Failed custom logging setup: {e}. Using basic config.", exc_info=True
        )
        # After basicConfig, the specific logger also needs its level set
        logger_to_return.setLevel(level)
        return logger_to_return


    # Set the level for the specific logger instance being requested/returned
    logger_to_return.setLevel(level)
    return logger_to_return


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
            return list(o)
        if isinstance(o, Path):
            return o.as_posix() # Use as_posix() for consistent serialization
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="ignore")
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
    # Use specific logger instance, disable Qt handler for this specific log action
    query_logger = setup_logging("llamasearch.query_log", use_qt_handler=False)
    try:
        logs_dir = get_llamasearch_dir() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        query_logger.error(
            f"Cannot create/access logs directory: {e}. Skipping query log."
        )
        return ""

    chunks_to_log = chunks # Start with the full list provided

    # Simplify chunks if not full logging
    if not full_logging and isinstance(chunks, list):
        simplified_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                sc = {
                    k: chunk.get(k)
                    for k in ["id", "score", "source_path", "filename", "original_chunk_index"]
                    if k in chunk and chunk.get(k) is not None
                }
                # Add preview only if chunk wasn't simplified earlier
                if "document" in chunk and isinstance(chunk["document"], str):
                     sc["text_preview"] = (
                         (chunk["document"][:150] + "...")
                         if len(chunk["document"]) > 150
                         else chunk["document"]
                     )
                elif "text_preview" in chunk: # Keep existing preview if present
                    sc["text_preview"] = chunk["text_preview"]

                simplified_chunks.append(sc)
            else: # Handle non-dict chunks if they occur
                # Ensure string conversion and preview
                str_chunk = str(chunk)
                simplified_chunks.append(
                    (str_chunk[:200] + "...") if len(str_chunk) > 200 else str_chunk
                )
        chunks_to_log = simplified_chunks

    # Log essential debug info, add more if full_logging
    optimized_debug_info = {}
    if isinstance(debug_info, dict):
        essential_keys = [
            "retrieval_time", "llm_generation_time", "total_query_processing_time",
            "vector_initial_results", "bm25_initial_results",
            "final_selected_chunk_count", "query_embedding_time",
            "final_context_content_token_count", "estimated_full_prompt_tokens" 
        ]
        for key in essential_keys:
            if key in debug_info:
                optimized_debug_info[key] = debug_info[key]

        if full_logging:
            optimized_debug_info.update(
                {
                    k: v
                    for k, v in debug_info.items()
                    if k not in optimized_debug_info and k not in ["raw_llm_output", "final_selected_chunk_details"] # Explicitly exclude these even in full
                }
            )


    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "chunks_retrieved_details": chunks_to_log,
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
        query_logger.error(f"Error saving query log to {log_file}: {e}")
        return ""