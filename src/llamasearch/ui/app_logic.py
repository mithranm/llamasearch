# src/llamasearch/ui/app_logic.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- Qt Imports for Signals ---
from PySide6.QtCore import QObject, Signal  # Keep Signal for type hint clarity

from llamasearch.core.llmsearch import LLMSearch
from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.core.teapot import TeapotONNXLLM
from llamasearch.data_manager import data_manager

# --- Corrected: Import utils components ---
from llamasearch.utils import setup_logging, _qt_logging_available, QtLogHandler
from llamasearch.exceptions import ModelNotFoundError

logger = setup_logging(__name__, level=logging.INFO, use_qt_handler=True)


# --- Backend Signal Emitter ---
class AppLogicSignals(QObject):
    """Holds signals emitted by the backend logic."""

    status_updated = Signal(str, str)
    search_completed = Signal(str, bool)
    crawl_index_completed = Signal(str, bool)
    manual_index_completed = Signal(str, bool)
    removal_completed = Signal(str, bool)
    refresh_needed = Signal()
    settings_applied = Signal(str, str)


class LlamaSearchApp:
    """Backend logic handler for LlamaSearch GUI. Runs tasks in threads."""

    def __init__(self, requires_gpu: bool = False, debug: bool = False):
        self.debug = debug
        self.requires_gpu = requires_gpu
        self.data_paths = data_manager.get_data_paths()
        self.llm_search: Optional[LLMSearch] = None
        self.signals = AppLogicSignals()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="LlamaSearchWorker"
        )

        self._log("INFO", f"LlamaSearchApp initializing. Data paths: {self.data_paths}")

        self._current_config = {
            "model_id": "N/A",
            "model_engine": "N/A",
            "context_length": 0,
            "max_results": 3,
            "debug_mode": self.debug,
            "provider": "N/A",
            "quantization": "N/A",
        }
        self._initialize_llm_search()  # Sync init
        if self.llm_search:
            self._log("INFO", "LlamaSearchApp ready.")
        else:
            self._log("ERROR", "LlamaSearchApp init failed. Run 'llamasearch-setup'.")

    def _log(self, level: str, message: str):
        """Wrapper for standard logging."""
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)

    def _initialize_llm_search(self):
        """Initializes LLMSearch synchronously."""
        if self.llm_search:
            self._log("INFO", "Closing existing LLMSearch instance...")
            try:
                self.llm_search.close()
            except Exception as e:
                self._log("ERROR", f"Error closing previous LLMSearch: {e}")
            self.llm_search = None

        index_dir = Path(self.data_paths["index"])
        self._log("INFO", f"Attempting to initialize LLMSearch in: {index_dir}")
        try:
            self.llm_search = LLMSearch(
                storage_dir=index_dir,
                debug=self.debug,
                verbose=self.debug,
                max_results=self._current_config.get("max_results", 3),
            )
            if self.llm_search and self.llm_search.model:
                info = self.llm_search.model.model_info
                self._current_config["model_id"] = info.model_id
                self._current_config["model_engine"] = info.model_engine
                self._current_config["context_length"] = info.context_length
                if isinstance(self.llm_search.model, TeapotONNXLLM):
                    self._current_config["provider"] = getattr(
                        self.llm_search.model, "_provider", "N/A"
                    )
                    model_id_parts = info.model_id.split("-")
                    if len(model_id_parts) >= 3:
                        quant_part = model_id_parts[-1]
                        if quant_part in [
                            "fp32",
                            "fp16",
                            "int8",
                            "q4",
                            "q4f16",
                            "bnb4",
                            "uint8",
                        ]:
                            self._current_config["quantization"] = quant_part
                        else:
                            self._current_config["quantization"] = "unknown"
                    else:
                        self._current_config["quantization"] = "unknown"
                else:
                    self._current_config["provider"] = "N/A"
                    self._current_config["quantization"] = "N/A"
                self._log(
                    "INFO",
                    f"LLMSearch initialized: {info.model_id} (Provider: {self._current_config['provider']}, Quant: {self._current_config['quantization']}, Ctx: {info.context_length})",
                )
            else:
                self._log("ERROR", "LLMSearch initialized, but LLM component failed.")
                self.llm_search = None
                raise ModelNotFoundError("LLM component failed.")
        except ModelNotFoundError as e:
            error_msg = f"Model setup required: {e}. Run 'llamasearch-setup'."
            self._log("ERROR", error_msg)
            logger.error(error_msg)
            self.llm_search = None
        except Exception as e:
            self._log("ERROR", f"Unexpected error initializing LLMSearch: {e}")
            logger.error("LLMSearch unexpected init error", exc_info=True)
            self.llm_search = None

    # --- ASYNC TASK EXECUTION ---
    # --- Corrected: Changed type hint to Any to satisfy pyright with SignalInstance ---
    def _run_in_background(self, task_func, *args, completion_signal: Any):
        """Submits function to thread pool."""
        try:
            future = self.executor.submit(task_func, *args)
            # Pass the *instance* of the signal to the callback
            future.add_done_callback(
                lambda f: self._task_done_callback(f, completion_signal)
            )
        except Exception as e:
            logger.error(f"Failed to submit task: {e}", exc_info=True)
            completion_signal.emit(f"Task Submission Error: {e}", False)

    # --- Corrected: Changed type hint to Any ---
    def _task_done_callback(self, future, completion_signal: Any):
        """Handles task completion."""
        try:
            result_message, success = future.result()
            completion_signal.emit(result_message, success)
        except Exception as e:
            logger.error(f"Exception in background task: {e}", exc_info=True)
            completion_signal.emit(f"Task Error: {e}", False)

    # --- GUI ACTIONS ---
    def submit_search(self, query: str):
        if not self.llm_search:
            self.signals.search_completed.emit(
                "Search failed: LLMSearch not initialized.", False
            )
            return
        if not query:
            self.signals.search_completed.emit("Please enter a query.", False)
            return
        self._log("INFO", f"Submitting search: '{query[:50]}...'")
        self._run_in_background(
            self._execute_search_task,
            query,
            completion_signal=self.signals.search_completed,
        )

    def submit_crawl_and_index(
        self,
        root_urls: List[str],
        target_links: int,
        max_depth: int,
        keywords: Optional[List[str]],
    ):
        self._log("INFO", f"Submitting crawl & index task for {len(root_urls)} URLs...")
        self._run_in_background(
            self._execute_crawl_and_index_task,
            root_urls,
            target_links,
            max_depth,
            keywords,
            completion_signal=self.signals.crawl_index_completed,
        )

    def submit_manual_index(self, path_str: str):
        if not self.llm_search:
            self.signals.manual_index_completed.emit(
                "Indexing failed: LLMSearch not initialized.", False
            )
            return
        source_path = Path(path_str)
        if not source_path.exists():
            self.signals.manual_index_completed.emit(
                f"Error: Path does not exist: {path_str}", False
            )
            return
        self._log("INFO", f"Submitting manual index task for: {source_path}")
        self._run_in_background(
            self._execute_manual_index_task,
            path_str,
            completion_signal=self.signals.manual_index_completed,
        )

    def submit_removal(self, source_id_to_remove: str):
        if not self.llm_search or not self.llm_search.vectordb:
            self.signals.removal_completed.emit(
                "Error: Cannot remove, LLMSearch/VectorDB not ready.", False
            )
            return
        if not isinstance(source_id_to_remove, str) or not source_id_to_remove:
            self.signals.removal_completed.emit(
                "Error: Invalid source ID for removal.", False
            )
            return
        self._log(
            "INFO", f"Submitting removal task for source ID: {source_id_to_remove}"
        )
        self._run_in_background(
            self._execute_removal_task,
            source_id_to_remove,
            completion_signal=self.signals.removal_completed,
        )

    # --- WORKER METHODS ---
    def _execute_search_task(self, query: str) -> Tuple[str, bool]:
        self._log("DEBUG", f"Executing search task for: '{query[:50]}...'")
        try:
            if not self.llm_search:
                return "Search Error: LLMSearch instance lost.", False
            start_time = time.time()
            results = self.llm_search.llm_query(query, debug_mode=self.debug)
            duration = time.time() - start_time
            self._log("INFO", f"Search task completed in {duration:.2f} seconds.")
            response = results.get("formatted_response", "No response generated.")
            return response, True
        except Exception as e:
            self._log("ERROR", f"Search query task failed: {e}")
            return f"Error performing search: {e}", False

    def _execute_crawl_and_index_task(
        self,
        root_urls: List[str],
        target_links: int,
        max_depth: int,
        keywords: Optional[List[str]],
    ) -> Tuple[str, bool]:
        self._log("DEBUG", "Executing crawl & index task...")
        crawl_successful = False
        index_successful = False
        crawl_duration = 0.0
        index_duration = 0.0
        added_chunks = 0
        all_collected_urls = []
        success_count, fail_count = 0, 0
        total_start_time = time.time()
        crawl_dir_base = Path(self.data_paths["crawl_data"])
        raw_output_dir = crawl_dir_base / "raw"
        crawl_dir_base.mkdir(parents=True, exist_ok=True)
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        try:  # Crawl Phase
            for url in root_urls:
                single_crawl_start = time.time()
                self._log(
                    "INFO", f"Crawling URL: {url} (Tgt:{target_links}, D:{max_depth})"
                )
                try:
                    crawler = Crawl4AICrawler(
                        root_urls=[url],
                        base_crawl_dir=crawl_dir_base,
                        target_links=target_links,
                        max_depth=max_depth,
                        relevance_keywords=keywords,
                    )
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    collected_urls = loop.run_until_complete(crawler.run_crawl())
                    loop.close()
                    all_collected_urls.extend(collected_urls)
                    duration = time.time() - single_crawl_start
                    crawl_duration += duration
                    self._log(
                        "INFO",
                        f"Crawl {url} OK ({duration:.2f}s). Got {len(collected_urls)} pages.",
                    )
                    success_count += 1
                    crawl_successful = True
                except Exception as e:
                    duration = time.time() - single_crawl_start
                    crawl_duration += duration
                    self._log("ERROR", f"Crawl FAILED for {url} ({duration:.2f}s): {e}")
                    fail_count += 1
            if not crawl_successful and fail_count > 0:
                raise Exception("All crawl tasks failed.")
        except Exception as crawl_exc:
            total_duration = time.time() - total_start_time
            result_msg = f"Crawl phase FAILED after {total_duration:.2f}s: {crawl_exc}"
            self._log("ERROR", result_msg)
            return result_msg, False
        if crawl_successful:  # Indexing Phase
            try:
                self._log(
                    "INFO",
                    f"Crawl finished ({crawl_duration:.2f}s). Indexing '{raw_output_dir.name}'...",
                )
                if not self.llm_search:
                    raise Exception("LLMSearch not initialized.")
                indexing_start_time = time.time()
                added_chunks = self.llm_search.add_documents_from_directory(
                    raw_output_dir, recursive=True
                )
                index_duration = time.time() - indexing_start_time
                index_successful = True
                self._log(
                    "INFO",
                    f"Auto indexing OK ({index_duration:.2f}s). Added {added_chunks} chunks.",
                )
                self.signals.refresh_needed.emit()
            except Exception as index_exc:
                self._log("ERROR", f"Automatic indexing FAILED: {index_exc}")
                index_successful = False
        # Final Status
        total_duration = time.time() - total_start_time
        unique_collected = len(set(all_collected_urls))
        crawl_status = f"Crawled {success_count}/{len(root_urls)} URLs ({unique_collected} unique pages) in {crawl_duration:.2f}s."
        index_status = ""
        overall_success = crawl_successful
        if crawl_successful:
            if index_successful:
                index_status = (
                    f"Indexed {added_chunks} new chunks in {index_duration:.2f}s."
                )
            else:
                index_status = "Automatic indexing failed."
                overall_success = False
        final_message = (
            f"Finished ({total_duration:.2f}s). {crawl_status} {index_status}".strip()
        )
        self._log("INFO", final_message)
        return final_message, overall_success

    def _execute_manual_index_task(self, path_str: str) -> Tuple[str, bool]:
        self._log("DEBUG", f"Executing manual index task for: {path_str}")
        source_path = Path(path_str)
        try:
            start_time = time.time()
            added_count = 0
            if not self.llm_search:
                return "Indexing Error: LLMSearch instance lost.", False
            if source_path.is_file():
                if hasattr(self.llm_search, "add_document"):
                    added_count = self.llm_search.add_document(source_path)
                else:
                    self._log("ERROR", "add_document method not found.")
                    return "Error: Indexing logic unavailable.", False
            elif source_path.is_dir():
                self._log("INFO", f"Indexing directory recursively: {source_path}")
                added_count = self.llm_search.add_documents_from_directory(
                    source_path, recursive=True
                )
            else:
                return f"Error: Path is not file/dir: {source_path}", False
            duration = time.time() - start_time
            msg = f"Indexing '{source_path.name}' OK ({duration:.2f}s). Added {added_count} new chunks."
            self._log("INFO", msg)
            if added_count > 0:
                self.signals.refresh_needed.emit()
            return msg, True
        except Exception as e:
            msg = f"Error indexing {source_path.name}: {e}"
            self._log("ERROR", msg)
            return msg, False

    def _execute_removal_task(self, source_id_to_remove: str) -> Tuple[str, bool]:
        self._log("DEBUG", f"Executing removal task for: {source_id_to_remove}")
        try:
            if not self.llm_search or not self.llm_search.vectordb:
                return "Removal Error: LLMSearch/VectorDB lost.", False
            display_name = source_id_to_remove
            try:
                path_obj = None
                if source_id_to_remove:
                    path_obj = Path(source_id_to_remove)
                if path_obj and path_obj.exists():
                    display_name = path_obj.name
            except (TypeError, ValueError, OSError):
                pass
            self.llm_search.vectordb._remove_document(source_id_to_remove)
            msg = f"Successfully requested removal for source: {display_name}"
            self._log("INFO", msg)
            self.signals.refresh_needed.emit()
            return msg, True
        except Exception as e:
            display_name = source_id_to_remove
            try:
                path_obj = None
                if source_id_to_remove:
                    path_obj = Path(source_id_to_remove)
                if path_obj and path_obj.exists():
                    display_name = path_obj.name
            except (TypeError, ValueError, OSError):
                pass
            msg = f"Error removing item {display_name}: {e}"
            self._log("ERROR", msg)
            return msg, False

    # --- Other Methods ---
    def get_crawl_data_items(self) -> List[Dict[str, str]]:
        """Retrieves indexed source information (sync)."""
        items = []
        if not self.llm_search or not self.llm_search.vectordb:
            self._log("WARN", "Cannot get index items: Not ready.")
            return items
        vdb = self.llm_search.vectordb
        unique_sources: Dict[str, str] = {}
        try:
            global_lookup_path = (
                Path(self.data_paths["crawl_data"]) / "reverse_lookup.json"
            )
            global_lookup: Dict[str, str] = {}
            if global_lookup_path.exists():
                try:
                    with open(global_lookup_path, "r", encoding="utf-8") as f:
                        global_lookup = json.load(f)
                except Exception as e:
                    self._log("WARN", f"Could not load reverse lookup: {e}")
            if hasattr(vdb, "document_metadata") and isinstance(
                vdb.document_metadata, list
            ):
                all_source_ids = set()
                for meta in vdb.document_metadata:
                    source_id = meta.get("source")  # Can return None
                    # --- Corrected Check ---
                    if isinstance(source_id, str) and source_id:
                        all_source_ids.add(source_id)
                    elif (
                        source_id is not None
                    ):  # Log if not None and not a valid string
                        self._log(
                            "WARN", f"Found non-string/empty source ID: {source_id}"
                        )

                for source_id in all_source_ids:
                    display_name = source_id  # Default display name
                    # --- Add check: source_id must be a string here ---
                    if not isinstance(source_id, str):
                        self._log("WARN", f"Skipping non-string source ID: {source_id}")
                        continue
                    try:
                        # --- Corrected: source_id is now guaranteed to be a str ---
                        path_obj = Path(source_id)
                        potential_hash = path_obj.stem
                        is_local = False
                        try:
                            is_local = path_obj.exists()
                        except OSError:
                            pass  # Handle filesystem errors during check
                        if (
                            not is_local
                            and len(potential_hash) == 16
                            and potential_hash in global_lookup
                        ):
                            display_name = global_lookup[potential_hash]
                        elif is_local:
                            display_name = path_obj.name
                    except (TypeError, ValueError) as path_err:
                        self._log(
                            "WARN",
                            f"Could not interpret source_id '{source_id}': {path_err}",
                        )
                    # --- Corrected: source_id is guaranteed str ---
                    unique_sources[source_id] = display_name
                items = [
                    {"hash": src_id, "url": name}
                    for src_id, name in unique_sources.items()
                ]
                items.sort(key=lambda x: x["url"].lower())
                self._log("DEBUG", f"Found {len(items)} unique sources in index.")
            else:
                self._log("WARN", "VectorDB metadata missing/invalid.")
        except Exception as e:
            self._log("ERROR", f"Failed get indexed items: {e}")
        return items

    def get_current_config(self) -> Dict[str, Any]:
        """Returns current configuration state (sync)."""
        self._current_config["debug_mode"] = self.debug
        if self.llm_search and self.llm_search.model:
            try:
                info = self.llm_search.model.model_info
                self._current_config["model_id"] = info.model_id
                self._current_config["model_engine"] = info.model_engine
                self._current_config["context_length"] = info.context_length
                self._current_config["provider"] = "N/A"
                self._current_config["quantization"] = "N/A"
                if isinstance(self.llm_search.model, TeapotONNXLLM):
                    self._current_config["provider"] = getattr(
                        self.llm_search.model, "_provider", "N/A"
                    )
                    parts = info.model_id.split("-")
                    if len(parts) >= 3:
                        q = parts[-1]
                        self._current_config["quantization"] = (
                            q
                            if q
                            in ["fp32", "fp16", "int8", "q4", "q4f16", "bnb4", "uint8"]
                            else "unknown"
                        )
                    else:
                        self._current_config["quantization"] = "unknown"
            except Exception as e:
                self._log("WARN", f"Could not update config from LLMSearch: {e}")
        elif not self.llm_search:
            self._current_config.update(
                {
                    "model_id": "N/A (Setup Required)",
                    "model_engine": "N/A",
                    "provider": "N/A",
                    "quantization": "N/A",
                    "context_length": 0,
                }
            )
        if self.llm_search:
            self._current_config["max_results"] = getattr(
                self.llm_search, "max_results", self._current_config["max_results"]
            )
        return self._current_config.copy()

    def apply_settings(self, settings: Dict[str, Any]):
        """Applies settings (sync)."""
        restart_needed = False
        config_changed = False
        new_debug = settings.get("debug_mode", self.debug)
        if new_debug != self.debug:
            self.debug = new_debug
            log_level = logging.DEBUG if self.debug else logging.INFO
            base_logger = logging.getLogger("llamasearch")
            base_logger.setLevel(log_level)
            for handler in base_logger.handlers:
                # --- Corrected check using imported variable ---
                if isinstance(
                    handler,
                    (
                        logging.FileHandler,
                        QtLogHandler if _qt_logging_available else type(None),
                    ),
                ):
                    handler.setLevel(log_level)
                elif isinstance(handler, logging.StreamHandler):
                    handler.setLevel(logging.INFO)
            logger.setLevel(log_level)  # Update our specific instance
            self._log("INFO", f"Debug mode set to: {self.debug}")
            if self.llm_search:
                self.llm_search.debug = self.debug
                self.llm_search.verbose = self.debug
            config_changed = True

        new_max_results = settings.get(
            "max_results", self._current_config["max_results"]
        )
        if (
            isinstance(new_max_results, int)
            and new_max_results > 0
            and new_max_results != self._current_config["max_results"]
        ):
            self._current_config["max_results"] = new_max_results
            # --- Corrected: Add nested check for vectordb ---
            if self.llm_search:
                self.llm_search.max_results = new_max_results
                if self.llm_search.vectordb:
                    self.llm_search.vectordb.max_results = new_max_results
            self._log("INFO", f"Max search results set to: {new_max_results}")
            config_changed = True
        elif not isinstance(new_max_results, int) or new_max_results <= 0:
            self._log("WARN", f"Invalid Max Results: {new_max_results}.")

        msg, lvl = "", ""
        if restart_needed:
            msg, lvl = "Settings applied. Restart required for some changes.", "warning"
        elif config_changed:
            msg, lvl = "Settings applied successfully.", "success"
        else:
            msg, lvl = "No setting changes detected.", "info"
        self.signals.settings_applied.emit(msg, lvl)
        return restart_needed

    def close(self):
        """Cleans up resources."""
        self._log("INFO", "LlamaSearchApp closing...")
        self.executor.shutdown(wait=True)
        if self.llm_search:
            try:
                self.llm_search.close()
                self._log("INFO", "LLMSearch closed.")
            except Exception as e:
                self._log("ERROR", f"Error closing LLMSearch: {e}")
            self.llm_search = None
        self._log("INFO", "LlamaSearchApp closed.")
