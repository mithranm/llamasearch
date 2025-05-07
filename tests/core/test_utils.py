# src/llamasearch/utils.py

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_qt_log_handler_instance: Optional[logging.Handler] = None
_qt_logging_available = False

try:
    from llamasearch.ui import qt_logging  # Relative import

    QtLogHandler = qt_logging.QtLogHandler
    qt_log_emitter = qt_logging.qt_log_emitter
    _qt_logging_available = True
except (ImportError, AttributeError, ModuleNotFoundError):
    QtLogHandler = None
    qt_log_emitter = None
    _qt_logging_available = False


def get_llamasearch_dir() -> Path:
    path = Path.home() / ".llamasearch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(
    name="llamasearch", level=logging.INFO, use_qt_handler=False
) -> logging.Logger:
    global _qt_log_handler_instance

    root_logger_name = "llamasearch"
    logger_to_return = logging.getLogger(name)

    root_logger = logging.getLogger(root_logger_name)
    # Set root logger level only if it's not already set or if a more verbose level is requested
    if root_logger.level == logging.NOTSET or level < root_logger.level:
        root_logger.setLevel(
            min(level, logging.DEBUG)
        )  # Ensure root captures at least DEBUG

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        base_dir = get_llamasearch_dir()
        logs_dir = base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)  # This call needs to succeed
        log_file = logs_dir / "llamasearch.log"

        file_handler_exists = any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            and hasattr(h, "baseFilename")  # Ensure baseFilename exists
            and Path(h.baseFilename).resolve() == log_file.resolve()
            for h in root_logger.handlers
        )
        if not file_handler_exists:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)

        console_handler: Optional[logging.StreamHandler] = None
        for h in root_logger.handlers:
            if (
                isinstance(h, logging.StreamHandler)
                and getattr(h, "stream", None) == sys.stdout
            ):
                console_handler = h
                break

        if console_handler is None:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        console_handler.setLevel(level)

        if use_qt_handler and _qt_logging_available and qt_log_emitter and QtLogHandler:
            if _qt_log_handler_instance is None:
                _qt_log_handler_instance = QtLogHandler(qt_log_emitter)
                _qt_log_handler_instance.setFormatter(simple_formatter)
                root_logger.addHandler(_qt_log_handler_instance)
                logging.getLogger(root_logger_name).info(
                    "Attached QtLogHandler to root logger."
                )  # Use root to log this
            if _qt_log_handler_instance:
                _qt_log_handler_instance.setLevel(level)
        elif use_qt_handler and not _qt_logging_available:
            logging.getLogger(root_logger_name).warning(
                "Qt logging requested but Qt components not found."
            )

        if not getattr(root_logger, "_noisy_libs_silenced", False):
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
                "whoosh",
            ]
            for lib_name in noisy_libs:
                logging.getLogger(lib_name).setLevel(logging.WARNING)
            setattr(root_logger, "_noisy_libs_silenced", True)

    except Exception as e:
        # Fallback logging should not reconfigure if already configured by a previous call
        if not root_logger.handlers:  # Only call basicConfig if no handlers are present
            logging.basicConfig(level=logging.INFO)
            logging.error(
                f"Failed custom logging setup: {e}. Using basic config.", exc_info=True
            )
        else:  # If handlers exist, just log the error
            root_logger.error(f"Error during logging setup: {e}", exc_info=True)

    logger_to_return.setLevel(level)
    return logger_to_return


class NumpyEncoder(json.JSONEncoder):
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
            return o.as_posix()
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="ignore")
        return super().default(o)


def log_query(
    query: str,
    chunks: list,
    response: str,
    debug_info: dict,
    full_logging: bool = False,
) -> str:
    query_logger = setup_logging("llamasearch.query_log", use_qt_handler=False)
    try:
        # Ensure logs_dir exists; setup_logging should handle this, but defensive check.
        # The primary get_llamasearch_dir ensures the base .llamasearch exists.
        # The logs_dir itself is created within setup_logging.
        # For this function, we assume setup_logging has run or will run successfully.
        logs_dir_path = get_llamasearch_dir() / "logs"
        if not logs_dir_path.is_dir():  # Check if it became a dir
            logs_dir_path.mkdir(parents=True, exist_ok=True)  # Try to create if missing
    except Exception as e:
        query_logger.error(
            f"Cannot create/access logs directory '{logs_dir_path}': {e}. Skipping query log."  # type: ignore
        )
        return ""

    chunks_to_log = chunks

    if not full_logging and isinstance(chunks, list):
        simplified_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                sc = {
                    k: chunk.get(k)
                    for k in [
                        "id",
                        "score",
                        "source_path",
                        "filename",
                        "original_chunk_index",
                    ]
                    if k in chunk and chunk.get(k) is not None
                }
                if "document" in chunk and isinstance(chunk["document"], str):
                    sc["text_preview"] = (
                        (chunk["document"][:150] + "...")
                        if len(chunk["document"]) > 150
                        else chunk["document"]
                    )
                elif "text_preview" in chunk:
                    sc["text_preview"] = chunk["text_preview"]

                simplified_chunks.append(sc)
            else:
                str_chunk = str(chunk)
                simplified_chunks.append(
                    (str_chunk[:200] + "...") if len(str_chunk) > 200 else str_chunk
                )
        chunks_to_log = simplified_chunks

    optimized_debug_info = {}
    if isinstance(debug_info, dict):
        essential_keys = [
            "retrieval_time",
            "llm_generation_time",
            "total_query_processing_time",
            "vector_initial_results",
            "bm25_initial_results",
            "final_selected_chunk_count",
            "query_embedding_time",
            "final_context_content_token_count",
            "estimated_full_prompt_tokens",
        ]
        for key in essential_keys:
            if key in debug_info:
                optimized_debug_info[key] = debug_info[key]

        if full_logging:
            optimized_debug_info.update(
                {
                    k: v
                    for k, v in debug_info.items()
                    if k not in optimized_debug_info
                    and k not in ["raw_llm_output", "final_selected_chunk_details"]
                }
            )

    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "chunks_retrieved_details": chunks_to_log,
        "response": response,
        "debug_info": optimized_debug_info,
    }

    log_file = logs_dir_path / "query_log.jsonl"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, cls=NumpyEncoder)
            f.write("\n")
        return str(log_file)
    except Exception as e:
        query_logger.error(f"Error saving query log to {log_file}: {e}")
        return ""
