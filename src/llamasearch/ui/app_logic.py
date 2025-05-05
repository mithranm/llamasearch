# src/llamasearch/ui/app_logic.py (Corrected)

import asyncio
import logging
import logging.handlers
import time
import threading
from concurrent.futures import ThreadPoolExecutor, CancelledError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # 'Any' is used for Qt signals with .emit()

from PySide6.QtCore import QObject, Signal as pyqtSignal, QTimer

from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.core.llmsearch import LLMSearch
from llamasearch.core.teapot import TeapotONNXLLM
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

logger = setup_logging("llamasearch.ui.app_logic")


class AppLogicSignals(QObject):
    """Holds signals emitted by the backend logic."""

    status_updated = pyqtSignal(str, str)
    search_completed = pyqtSignal(str, bool)
    crawl_index_completed = pyqtSignal(str, bool)
    manual_index_completed = pyqtSignal(str, bool)
    removal_completed = pyqtSignal(str, bool)
    refresh_needed = pyqtSignal()
    settings_applied = pyqtSignal(str, str)
    actions_should_reenable = pyqtSignal()
    _internal_task_completed = pyqtSignal(object, object, bool, pyqtSignal)


class LlamaSearchApp:
    """Backend logic handler for LlamaSearch GUI. Runs tasks in threads."""

    def __init__(self, executor: ThreadPoolExecutor, debug: bool = False):
        """Initialize the application logic."""
        logger.info("Initializing LlamaSearchApp backend...")
        self.debug = debug
        self.data_paths = data_manager.get_data_paths()
        self.llm_search: Optional[LLMSearch] = None
        self.signals = AppLogicSignals()
        self._shutdown_event = threading.Event()
        self._thread_pool = executor
        self._active_crawler: Optional[Crawl4AICrawler] = None
        self._current_config = self._get_default_config()
        logger.info(f"LlamaSearchApp initializing. Data paths: {self.data_paths}")
        if not self._initialize_llm_search():
            logger.error("LlamaSearchApp init failed. Backend non-functional.")
            QTimer.singleShot(
                100,
                lambda: self.signals.status_updated.emit(
                    "Backend initialization failed. Run setup.", "error"
                ),
            )
        else:
            logger.info("LlamaSearchApp backend ready.")
            QTimer.singleShot(150, self.signals.refresh_needed.emit)
        self.signals._internal_task_completed.connect(self._final_gui_callback)

    def _get_default_config(self) -> Dict[str, Any]:
        """Returns the default configuration state."""
        return {
            "model_id": "N/A",
            "model_engine": "N/A",
            "context_length": 0,
            "max_results": 3,
            "debug_mode": self.debug,
            "provider": "N/A",
            "quantization": "N/A",
        }

    def _initialize_llm_search(self) -> bool:
        """Initializes LLMSearch synchronously. Returns True on success."""
        if self.llm_search:
            logger.info("Closing existing LLMSearch instance...")
            try:
                self.llm_search.close()
            except Exception as e:
                logger.error(f"Error closing LLMSearch: {e}", exc_info=self.debug)
            self.llm_search = None
        index_dir = Path(self.data_paths["index"])
        logger.info(f"Attempting to initialize LLMSearch in: {index_dir}")
        try:
            self.llm_search = LLMSearch(
                storage_dir=index_dir,
                shutdown_event=self._shutdown_event,
                debug=self.debug,
                verbose=self.debug,
                max_results=self._current_config.get("max_results", 3),
            )
            if self.llm_search and self.llm_search.model:
                self._update_config_from_llm()
                logger.info(
                    f"LLMSearch initialized: {self._current_config.get('model_id')}"
                )
                return True
            else:
                logger.error("LLMSearch initialized, but LLM component failed.")
                if self.llm_search:
                    self.llm_search.close()
                self.llm_search = None
                return False
        except (ModelNotFoundError, SetupError) as e:
            logger.error(f"Model setup required: {e}. Run 'llamasearch-setup'.")
            self.llm_search = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing LLMSearch: {e}", exc_info=True)
            self.llm_search = None
            return False

    def _update_config_from_llm(self):
        """Safely updates the internal config state from the LLMSearch instance."""
        if not self.llm_search or not self.llm_search.model:
            self._current_config = self._get_default_config()
            self._current_config["model_id"] = "N/A (Setup Required or Load Failed)"
            return
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
                quant_options = {"fp32", "fp16", "int8", "q4", "q4f16", "bnb4", "uint8"}
                self._current_config["quantization"] = (
                    parts[-1]
                    if len(parts) >= 3 and parts[-1] in quant_options
                    else "unknown"
                )
            elif hasattr(self.llm_search, "llm_device_type"):
                self._current_config["provider"] = (
                    getattr(self.llm_search, "llm_device_type", "cpu").upper()
                    + " (Inferred)"
                )
            self._current_config["max_results"] = getattr(
                self.llm_search,
                "max_results",
                self._current_config.get("max_results", 3),
            )
        except Exception as e:
            logger.warning(f"Could not update config from LLMSearch model info: {e}")
            self._current_config["model_id"] += " (Info Error)"

    def _run_in_background(self, task_func, *args, completion_signal):
        """Submits function to thread pool, handles completion/errors."""
        if self._shutdown_event.is_set():
            logger.warning("Shutdown in progress, ignoring new task submission.")
            if hasattr(completion_signal, "emit") and callable(completion_signal.emit):
                QTimer.singleShot(
                    0,
                    lambda: completion_signal.emit(
                        "Shutdown in progress, task cancelled.", False
                    ),
                )
            QTimer.singleShot(0, self.signals.actions_should_reenable.emit)
            return
        try:
            logger.debug(f"Submitting task {task_func.__name__} to thread pool.")
            future = self._thread_pool.submit(task_func, *args)
            logger.debug(f"Task submitted. Future: {future}. Attaching done callback.")

            def _intermediate_callback(f):
                logger.debug(
                    f"Intermediate callback running for task completion (Future: {f})."
                )
                result = None
                exception = None
                cancelled = False
                try:
                    if f.cancelled():
                        cancelled = True
                        logger.debug("Future was cancelled.")
                    else:
                        exception = f.exception()
                        if exception:
                            logger.debug(
                                f"Future completed with exception: {exception}"
                            )
                        else:
                            result = f.result()
                            logger.debug(
                                f"Future completed with result: {type(result)}"
                            )
                except CancelledError:
                    cancelled = True
                    logger.debug("Future was cancelled (caught CancelledError).")
                except Exception as e:
                    logger.error(
                        f"Error retrieving future result/exception: {e}", exc_info=True
                    )
                    exception = e
                logger.debug("Scheduling actions re-enable from intermediate callback.")
                QTimer.singleShot(0, self.signals.actions_should_reenable.emit)
                logger.debug(
                    f"Scheduling final GUI callback. Result type: {type(result)}, Exception type: {type(exception)}, Cancelled: {cancelled}"
                )
                self.signals._internal_task_completed.emit(
                    result, exception, cancelled, completion_signal
                )

            future.add_done_callback(_intermediate_callback)
            logger.debug(f"Done callback attached to future {future}.")
        except Exception as e:
            logger.error(f"Failed to submit task: {e}", exc_info=True)
            if hasattr(completion_signal, "emit") and callable(completion_signal.emit):
                QTimer.singleShot(
                    0,
                    lambda err=e: completion_signal.emit(
                        f"Task Submission Error: {err}", False
                    ),
                )
            QTimer.singleShot(0, self.signals.actions_should_reenable.emit)

    def _final_gui_callback(
        self,
        result: Optional[Any],
        exception: Optional[Exception],
        cancelled: bool,
        completion_signal: Any,  # Accepts any Qt signal object; .emit is checked at runtime

    ):
        """Handles the final result/exception in the GUI thread after background task."""
        logger.debug(
            f">>> _final_gui_callback EXECUTING <<< Result type: {type(result)}, Exception type: {type(exception)}, Cancelled: {cancelled}"
        )
        can_emit = hasattr(completion_signal, "emit") and callable(
            completion_signal.emit
        )
        if not can_emit:
            logger.error(
                "Cannot emit completion signal: Invalid signal object provided."
            )
            return
        try:
            if cancelled:
                logger.info("Task was cancelled branch taken.")
                completion_signal.emit("Task cancelled during execution.", False)
                return
            if exception:
                logger.info("Task had exception branch taken.")
                if not self._shutdown_event.is_set():
                    logger.error(
                        f"Exception in background task: {exception}", exc_info=False
                    )
                    completion_signal.emit(f"Task Error: {exception}", False)
                else:
                    logger.warning(
                        f"Task ended with exception during shutdown: {exception}"
                    )
                    completion_signal.emit("Task interrupted by shutdown.", False)
                return
            logger.debug(f"Result before check: {result!r}")
            if result is not None:
                is_expected_tuple = (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and isinstance(result[1], bool)
                )
                logger.debug(
                    f"Is result the expected tuple structure? {is_expected_tuple}"
                )
                if is_expected_tuple:
                    result_message, success = result
                    result_message_str = str(result_message)
                    logger.debug(
                        f"Emitting completion signal: {getattr(completion_signal, 'signal', 'N/A')} with success={success}, message='{result_message_str[:100]}...' "
                    )
                    completion_signal.emit(result_message_str, success)
                else:
                    err_msg = f"Background task returned unexpected result type/structure: {type(result)}. Value: {result!r}"
                    logger.error(err_msg)
                    completion_signal.emit(
                        f"Task completed with unexpected result: {str(result)[:100]}",
                        False,
                    )
            else:
                err_msg = "Background task returned None without exception."
                logger.error(err_msg)
                completion_signal.emit(err_msg, False)
        except Exception as callback_exc:
            logger.error(
                f"Error processing task result or emitting signal in GUI callback: {callback_exc}",
                exc_info=True,
            )
            try:
                if can_emit:
                    completion_signal.emit(f"GUI Callback Error: {callback_exc}", False)
            except Exception as emit_err:
                logger.error(f"Failed even to emit error signal: {emit_err}")

    def submit_search(self, query: str):
        if self._shutdown_event.is_set():
            self.signals.search_completed.emit("Search cancelled: Shutdown.", False)
            return
        if not self.llm_search:
            self.signals.search_completed.emit(
                "Search failed: Backend not ready.", False
            )
            return
        if not query:
            self.signals.search_completed.emit("Please enter a query.", False)
            return
        logger.info(f"Submitting search: '{query[:50]}...'")
        self.signals.status_updated.emit(f"Searching '{query[:30]}...'", "info")
        self.signals.actions_should_reenable.emit()
        QTimer.singleShot(
            0,
            lambda: self._run_in_background(
                self._execute_search_task,
                query,
                completion_signal=self.signals.search_completed,
            ),
        )

    def submit_crawl_and_index(
        self,
        root_urls: List[str],
        target_links: int,
        max_depth: int,
        keywords: Optional[List[str]],
    ):
        if self._shutdown_event.is_set():
            self.signals.crawl_index_completed.emit("Task cancelled: Shutdown.", False)
            return
        if not self.llm_search:
            self.signals.crawl_index_completed.emit(
                "Task failed: Backend not ready.", False
            )
            return
        logger.info(f"Submitting crawl & index task for {len(root_urls)} URLs...")
        self.signals.status_updated.emit(
            f"Starting crawl & index for {len(root_urls)} URL(s)...", "info"
        )
        self.signals.actions_should_reenable.emit()
        QTimer.singleShot(
            0,
            lambda: self._run_in_background(
                self._execute_crawl_and_index_task,
                root_urls,
                target_links,
                max_depth,
                keywords,
                completion_signal=self.signals.crawl_index_completed,
            ),
        )

    def submit_manual_index(self, path_str: str):
        if self._shutdown_event.is_set():
            self.signals.manual_index_completed.emit(
                "Indexing cancelled: Shutdown.", False
            )
            return
        if not self.llm_search:
            self.signals.manual_index_completed.emit(
                "Indexing failed: Backend not ready.", False
            )
            return
        source_path = Path(path_str)
        if not source_path.exists():
            self.signals.manual_index_completed.emit(
                f"Error: Path does not exist: {path_str}", False
            )
            return
        logger.info(f"Submitting manual index task for: {source_path}")
        self.signals.status_updated.emit(f"Indexing '{source_path.name}'...", "info")
        self.signals.actions_should_reenable.emit()
        QTimer.singleShot(
            0,
            lambda: self._run_in_background(
                self._execute_manual_index_task,
                path_str,
                completion_signal=self.signals.manual_index_completed,
            ),
        )

    def submit_removal(self, source_path_to_remove: str):
        if self._shutdown_event.is_set():
            self.signals.removal_completed.emit("Removal cancelled: Shutdown.", False)
            return
        if not self.llm_search:
            self.signals.removal_completed.emit(
                "Error: Cannot remove, Backend not ready.", False
            )
            return
        if not isinstance(source_path_to_remove, str) or not source_path_to_remove:
            self.signals.removal_completed.emit("Error: Invalid source path.", False)
            return
        logger.info(f"Submitting removal task for source path: {source_path_to_remove}")
        try:
            display_name = Path(source_path_to_remove).name
        except Exception:
            display_name = source_path_to_remove[:40] + "..."
        self.signals.status_updated.emit(f"Removing '{display_name}'...", "info")
        self.signals.actions_should_reenable.emit()
        QTimer.singleShot(
            0,
            lambda: self._run_in_background(
                self._execute_removal_task,
                source_path_to_remove,
                completion_signal=self.signals.removal_completed,
            ),
        )

    def _execute_search_task(self, query: str) -> Tuple[str, bool]:
        """Executes the search query in the background. Ensures tuple return."""
        result_message = "Search failed unexpectedly."
        success = False
        try:
            logger.debug(f"Executing search task for: '{query[:50]}...'")
            if self._shutdown_event.is_set():
                return "Search cancelled (shutdown)", False
            if not self.llm_search:
                return "Search Error: LLMSearch instance not available.", False
            start_time = time.time()
            results = self.llm_search.llm_query(query, debug_mode=self.debug)
            duration = time.time() - start_time
            if self._shutdown_event.is_set():
                return "Search interrupted after generation (shutdown)", False
            logger.info(f"Search task completed in {duration:.2f} seconds.")
            result_message = results.get("formatted_response", "No response generated.")
            success = True
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(
                    f"Search task failed unexpectedly: {e}", exc_info=self.debug
                )
                result_message = f"Search Error: {e}"
                success = False
            else:
                logger.warning(f"Search failed during shutdown process: {e}")
                result_message = f"Search stopped due to shutdown: {e}"
                success = False
        return result_message, success

    def _execute_crawl_and_index_task(
        self,
        root_urls: List[str],
        target_links: int,
        max_depth: int,
        keywords: Optional[List[str]],
    ) -> Tuple[str, bool]:
        """Executes crawling and subsequent indexing. Uses Whoosh BM25."""
        logger.debug("Executing crawl & index task...")
        crawl_successful, index_successful = False, False
        crawl_duration, index_duration = 0.0, 0.0
        total_added_chunks = 0
        total_start_time = time.time()
        crawl_dir_base = Path(self.data_paths["crawl_data"])
        raw_output_dir = crawl_dir_base / "raw"
        crawl_start_time = 0
        loop: Optional[asyncio.AbstractEventLoop] = None
        policy = asyncio.get_event_loop_policy()
        final_message = "Task initialization failed."
        overall_success = False

        try:
            # Crawl Phase
            try:
                crawl_start_time = time.time()
                if self._shutdown_event.is_set():
                    raise InterruptedError("Shutdown before crawl.")
                logger.info(f"Starting crawl phase. Output: {crawl_dir_base}")
                self._active_crawler = Crawl4AICrawler(
                    root_urls=root_urls,
                    base_crawl_dir=crawl_dir_base,
                    target_links=target_links,
                    max_depth=max_depth,
                    relevance_keywords=keywords,
                    headless=True,
                    user_agent="LlamaSearchBot/1.0",
                    shutdown_event=self._shutdown_event,
                    verbose_logging=self.debug,
                )
                loop = policy.new_event_loop()
                asyncio.set_event_loop(loop)
                collected_urls = loop.run_until_complete(
                    self._active_crawler.run_crawl()
                )
                if self._active_crawler:
                    logger.debug("Closing crawler resources...")
                    loop.run_until_complete(self._active_crawler.close())
                    logger.debug("Crawler resources closed.")
                crawl_duration = time.time() - crawl_start_time
                if self._shutdown_event.is_set():
                    raise InterruptedError("Shutdown during crawl.")
                logger.info(
                    f"Crawl phase OK ({crawl_duration:.2f}s). Collected {len(collected_urls)} URLs."
                )
                crawl_successful = True
            except InterruptedError as e:
                if crawl_start_time > 0:
                    crawl_duration = time.time() - crawl_start_time
                logger.warning(f"Crawl interrupted ({crawl_duration:.2f}s): {e}")
                crawl_successful = False
            except Exception as crawl_exc:
                if crawl_start_time > 0:
                    crawl_duration = time.time() - crawl_start_time
                logger.error(
                    f"Crawl FAILED ({crawl_duration:.2f}s): {crawl_exc}",
                    exc_info=self.debug,
                )
                final_message = f"Crawl failed: {crawl_exc}"
                crawl_successful = False
            finally:
                self._active_crawler = None
                if loop is not None:
                    try:
                        if (
                            loop.is_running()
                        ):  # Check if running before trying to cancel
                            tasks = asyncio.all_tasks(loop=loop)
                            if tasks:
                                logger.debug(
                                    f"Cancelling {len(tasks)} remaining asyncio tasks..."
                                )
                                for task in tasks:
                                    task.cancel()
                                gather_future = asyncio.gather(
                                    *tasks, return_exceptions=True
                                )  # Gather results/exceptions
                                # Briefly run loop to allow cancellations to process
                                loop.run_until_complete(asyncio.sleep(0.1))
                                logger.debug(
                                    f"Gather results after cancel: {gather_future.result() if gather_future.done() else 'Not Done'}"
                                )
                        if not loop.is_closed():
                            loop.close()
                            logger.debug(f"Closed asyncio event loop {id(loop)}.")
                        try:  # Reset policy loop if needed
                            if policy.get_event_loop() is loop:
                                policy.set_event_loop(None)
                        except RuntimeError:
                            pass  # Ignore if no current loop set
                    except Exception as loop_close_err:
                        logger.error(
                            f"Error cleaning asyncio loop: {loop_close_err}",
                            exc_info=True,
                        )

            # Indexing Phase
            if crawl_successful:
                indexing_start_time = time.time()
                processed_files_count = 0
                try:
                    if self._shutdown_event.is_set():
                        raise InterruptedError("Shutdown before indexing.")
                    if not self.llm_search:
                        raise Exception(
                            "LLMSearch instance not available for indexing."
                        )
                    logger.info(
                        f"Starting indexing phase for crawled content in: {raw_output_dir}..."
                    )
                    if raw_output_dir.is_dir():
                        files_to_index = list(raw_output_dir.glob("*.md"))
                        logger.info(
                            f"Found {len(files_to_index)} markdown files to index."
                        )
                        for md_file in files_to_index:
                            if self._shutdown_event.is_set():
                                raise InterruptedError("Shutdown during indexing loop.")
                            logger.debug(f"Indexing file: {md_file.name}")
                            added, _ = self.llm_search.add_source(
                                str(md_file)
                            )  # Ignore rebuild flag
                            if added > 0:
                                total_added_chunks += added
                                processed_files_count += 1
                    else:
                        logger.warning(
                            f"Crawl 'raw' directory not found: {raw_output_dir}. No indexing performed."
                        )
                    index_duration = time.time() - indexing_start_time
                    logger.info(
                        f"File processing phase completed ({index_duration:.2f}s). Indexed {processed_files_count} files. Total chunks added: {total_added_chunks}."
                    )
                    # No final BM25 build needed with Whoosh
                    if self._shutdown_event.is_set():
                        raise InterruptedError("Shutdown after indexing phase.")
                    index_successful = True
                except InterruptedError as e:
                    if indexing_start_time > 0:
                        index_duration = time.time() - indexing_start_time
                    logger.warning(f"Indexing interrupted ({index_duration:.2f}s): {e}")
                    index_successful = False
                except Exception as index_exc:
                    if indexing_start_time > 0:
                        index_duration = time.time() - indexing_start_time
                    logger.error(
                        f"Indexing FAILED ({index_duration:.2f}s): {index_exc}",
                        exc_info=self.debug,
                    )
                    index_successful = False

        except Exception as outer_exc:
            logger.error(
                f"Unexpected error during crawl/index task execution: {outer_exc}",
                exc_info=True,
            )
            final_message = f"Task failed with unexpected error: {outer_exc}"
            overall_success = False

        # Final Reporting
        total_duration = time.time() - total_start_time
        shutdown_occurred = self._shutdown_event.is_set()
        overall_success = (
            crawl_successful and index_successful and not shutdown_occurred
        )
        crawl_status_msg = "Crawl skipped."
        if crawl_start_time > 0:
            crawl_status_msg = f"Crawl {'OK' if crawl_successful else ('INTERRUPTED' if shutdown_occurred else 'FAILED')} ({crawl_duration:.1f}s)."
        index_status_msg = "Index skipped."
        if crawl_successful:
            if index_duration > 0 or not index_successful:
                index_status_msg = f"Index {'OK' if index_successful else ('INTERRUPTED' if shutdown_occurred else 'FAILED')} ({index_duration:.1f}s, {total_added_chunks} chunks)."
        if overall_success or (
            "failed" not in final_message.lower()
            and "error" not in final_message.lower()
        ):
            final_message = f"Finished ({total_duration:.1f}s). {crawl_status_msg} {index_status_msg}".strip()
        logger.info(final_message)
        if overall_success and total_added_chunks > 0:
            QTimer.singleShot(0, self.signals.refresh_needed.emit)
        return final_message, overall_success

    def _execute_manual_index_task(self, path_str: str) -> Tuple[str, bool]:
        """Executes manual indexing. Uses Whoosh BM25."""
        task_successful = False
        final_message = "Task did not complete."
        start_time = time.time()
        source_path = Path(path_str)
        total_added_chunks = 0
        overall_success = False

        try:
            logger.debug(f"Executing manual index task for: {path_str}")
            if self._shutdown_event.is_set():
                raise InterruptedError("Shutdown before manual index.")
            if not self.llm_search:
                raise Exception("LLMSearch instance not available.")
            logger.info(f"Manually indexing source: {source_path.name}...")
            total_added_chunks, _ = self.llm_search.add_source(
                path_str
            )  # Ignore rebuild flag
            if self._shutdown_event.is_set():
                raise InterruptedError("Shutdown after manual index processing.")
            logger.info(
                f"Manual index processing complete for '{source_path.name}'. Added {total_added_chunks} chunks."
            )
            task_successful = True
        except InterruptedError as e:
            logger.warning(f"Manual index interrupted: {e}")
            task_successful = False
        except Exception as e:
            logger.error(
                f"Manual index FAILED for '{source_path.name}': {e}",
                exc_info=self.debug,
            )
            task_successful = False

        duration = time.time() - start_time
        shutdown_occurred = self._shutdown_event.is_set()
        overall_success = task_successful and not shutdown_occurred
        if shutdown_occurred:
            final_message = f"Indexing '{source_path.name}' interrupted ({duration:.1f}s). Added {total_added_chunks} chunks before stop."
        elif task_successful:
            final_message = f"Indexing '{source_path.name}' OK ({duration:.1f}s). Added {total_added_chunks} new chunks."
            if total_added_chunks > 0:
                QTimer.singleShot(0, self.signals.refresh_needed.emit)
        else:
            final_message = f"Indexing '{source_path.name}' FAILED ({duration:.1f}s){' (Interrupted)' if shutdown_occurred else ''}. See logs."
        logger.info(final_message)
        return final_message, overall_success

    def _execute_removal_task(self, source_path_to_remove: str) -> Tuple[str, bool]:
        """Executes removal of a source. Uses Whoosh BM25."""
        task_successful = False
        final_message = "Task did not complete."
        removal_occurred = False
        overall_success = False
        display_name = Path(source_path_to_remove).name

        try:
            logger.debug(f"Executing removal task for: {source_path_to_remove}")
            if self._shutdown_event.is_set():
                raise InterruptedError("Shutdown before removal.")
            if not self.llm_search:
                raise Exception("LLMSearch instance not available.")
            removal_occurred, _ = self.llm_search.remove_source(
                source_path_to_remove
            )  # Ignore rebuild flag
            if self._shutdown_event.is_set():
                raise InterruptedError("Shutdown after removal processing.")
            task_successful = True
        except InterruptedError as e:
            logger.warning(f"Removal interrupted: {e}")
            task_successful = False
        except Exception as e:
            logger.error(
                f"Removal FAILED for '{display_name}': {e}", exc_info=self.debug
            )
            task_successful = False

        shutdown_occurred = self._shutdown_event.is_set()
        overall_success = task_successful and not shutdown_occurred
        if shutdown_occurred:
            final_message = f"Removal of '{display_name}' interrupted."
        elif task_successful:
            if removal_occurred:
                final_message = f"Successfully removed source: {display_name}"
                QTimer.singleShot(0, self.signals.refresh_needed.emit)
            else:
                final_message = f"Source not found or already removed: {display_name}"
        else:
            final_message = f"Error removing '{display_name}'{' (Interrupted)' if shutdown_occurred else ''}. See logs."
        logger.info(final_message)
        return final_message, overall_success

    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """Retrieves indexed source information from LLMSearch (sync)."""
        if not self.llm_search:
            logger.warning("Cannot get indexed sources: LLMSearch not ready.")
            return []
        try:
            return self.llm_search.get_indexed_sources()
        except Exception as e:
            logger.error(f"Failed to get indexed sources: {e}", exc_info=self.debug)
            return []

    def get_current_config(self) -> Dict[str, Any]:
        """Returns current configuration state (sync)."""
        self._update_config_from_llm()
        self._current_config["debug_mode"] = self.debug
        return self._current_config.copy()

    def apply_settings(self, settings: Dict[str, Any]):
        """Applies settings (sync). Emits signal on completion."""
        restart_needed = False
        config_changed = False
        new_debug = settings.get("debug_mode", self.debug)
        if new_debug != self.debug:
            self.debug = new_debug
            log_level = logging.DEBUG if self.debug else logging.INFO
            root_logger = logging.getLogger("llamasearch")
            root_logger.setLevel(log_level)
            for handler in root_logger.handlers:
                if isinstance(
                    handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)
                ):
                    handler.setLevel(logging.DEBUG)  # File always DEBUG
                elif isinstance(handler, logging.StreamHandler):
                    handler.setLevel(log_level)  # Console follows flag
                try:
                    from llamasearch.ui.qt_logging import QtLogHandler

                    if QtLogHandler and isinstance(handler, QtLogHandler):
                        handler.setLevel(log_level)  # Qt follows flag
                except ImportError:
                    pass
            logger.info(f"Logging level set to: {'DEBUG' if self.debug else 'INFO'}")
            if self.llm_search:
                self.llm_search.debug = self.debug
                self.llm_search.verbose = self.debug
            config_changed = True
        new_max_results = settings.get(
            "max_results", self._current_config.get("max_results", 3)
        )
        try:
            n_res = int(new_max_results)
            current_max = self._current_config.get("max_results", 3)
            if n_res > 0 and n_res != current_max:
                self._current_config["max_results"] = n_res
                if self.llm_search:
                    self.llm_search.max_results = n_res
                logger.info(f"Max search results set to: {n_res}")
                config_changed = True
            elif n_res <= 0:
                logger.warning(
                    f"Invalid Max Results value ignored: {n_res}. Must be > 0."
                )
        except (ValueError, TypeError):
            logger.warning(f"Invalid Max Results type ignored: {new_max_results}.")
        msg, lvl = ("No changes applied.", "info")
        if restart_needed:
            msg, lvl = "Settings applied. Backend restart might be needed.", "warning"
        elif config_changed:
            msg, lvl = "Settings applied successfully.", "success"
        self.signals.settings_applied.emit(msg, lvl)

    def close(self):
        """Cleans up resources, signals shutdown, waits briefly for tasks."""
        logger.info(
            "Closing LlamaSearchApp backend resources (executor shutdown handled externally)..."
        )
        self._shutdown_event.set()
        if self._active_crawler:
            logger.debug("Requesting crawler abort...")
            self._active_crawler.abort()
        if self.llm_search:
            logger.debug("Closing LLMSearch instance...")
            self.llm_search.close()
        logger.info("LlamaSearchApp backend resources closed.")
