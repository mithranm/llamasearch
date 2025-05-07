# src/llamasearch/ui/app_logic.py

import asyncio
import logging
import logging.handlers
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import necessary Qt components
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.core.search_engine import LLMSearch  # Corrected import
from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

# Use module-level logger
logger = setup_logging("llamasearch.ui.app_logic")

class AppLogicSignals(QObject):
    """Holds signals emitted by the backend logic."""
    # Signal definitions are single statements per line
    status_updated = Signal(str, str)
    search_completed = Signal(str, bool)
    crawl_index_completed = Signal(str, bool)
    manual_index_completed = Signal(str, bool)
    removal_completed = Signal(str, bool)
    refresh_needed = Signal()
    settings_applied = Signal(str, str)
    actions_should_reenable = Signal()
    # Internal signal for task completion handling (parameter types help clarity)
    _internal_task_completed = Signal(object, object, bool, Signal)

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
        # Initialize config using the method
        self._current_config = self._get_default_config()

        logger.info(f"LlamaSearchApp initializing. Data paths: {self.data_paths}")

        init_success = self._initialize_llm_search()
        if not init_success:
            logger.error("LlamaSearchApp init failed. Backend non-functional.")
            # Use QTimer to ensure signal emission happens in the GUI event loop
            QTimer.singleShot(
                100,
                lambda: self.signals.status_updated.emit( # type: ignore [attr-defined]
                    "Backend initialization failed. Run setup.", "error"
                ),
            )
        else:
            logger.info("LlamaSearchApp backend ready.")
            # Refresh UI after a short delay to load existing sources
            QTimer.singleShot(150, self.signals.refresh_needed.emit) # type: ignore [attr-defined]

        # Connect the internal signal to its handler slot
        # Use attribute access for connect
        self.signals._internal_task_completed.connect(self._final_gui_callback)

    def _get_default_config(self) -> Dict[str, Any]:
        """Returns the default configuration state (excluding provider/quantization)."""
        # Single return statement
        return {
            "model_id": "N/A",
            "model_engine": "N/A",
            "context_length": 0,
            "max_results": 3,
            "debug_mode": self.debug,
        }

    def _initialize_llm_search(self) -> bool:
        """Initializes LLMSearch synchronously. Returns True on success."""
        if self.llm_search:
            logger.info("Closing existing LLMSearch instance...")
            try:
                self.llm_search.close()
            except Exception as e:
                logger.error(f"Error closing LLMSearch: {e}", exc_info=self.debug)
            finally:
                 self.llm_search = None # Ensure it's None even if close fails

        index_dir_path_str = self.data_paths.get("index")
        if not index_dir_path_str:
            logger.error("Index directory path not configured.")
            return False # Cannot initialize without index path
        index_dir = Path(index_dir_path_str)

        logger.info(f"Attempting to initialize LLMSearch in: {index_dir}")
        try:
            self.llm_search = LLMSearch(
                storage_dir=index_dir,
                shutdown_event=self._shutdown_event,
                debug=self.debug,
                verbose=self.debug,
                max_results=self._current_config.get("max_results", 3),
            )
            # Check if both instance and its model were initialized
            if self.llm_search and self.llm_search.model:
                self._update_config_from_llm()
                logger.info(
                    f"LLMSearch initialized: {self._current_config.get('model_id')}"
                )
                return True
            else:
                logger.error("LLMSearch initialization succeeded, but LLM component failed.")
                if self.llm_search:
                    # Clean up partially initialized instance
                    self.llm_search.close()
                self.llm_search = None
                return False
        except (ModelNotFoundError, SetupError) as e:
            logger.error(f"Model setup required: {e}. Run 'llamasearch-setup'.")
            self.llm_search = None # Ensure None on failure
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing LLMSearch: {e}", exc_info=True)
            self.llm_search = None # Ensure None on failure
            return False

    def _update_config_from_llm(self):
        """Safely updates the internal config state from the LLMSearch instance."""
        if not self.llm_search or not self.llm_search.model:
            # Reset to defaults if LLM isn't available
            self._current_config = self._get_default_config()
            self._current_config["model_id"] = "N/A (Setup Required or Load Failed)"
            return

        try:
            info = self.llm_search.model.model_info
            self._current_config["model_id"] = info.model_id
            self._current_config["model_engine"] = info.model_engine
            self._current_config["context_length"] = info.context_length
            # Removed provider and quantization

            # Update max_results from the LLMSearch instance
            self._current_config["max_results"] = getattr(
                self.llm_search, "max_results", self._current_config.get("max_results", 3)
            )
        except Exception as e:
            logger.warning(f"Could not update config from LLMSearch model info: {e}")
            # Indicate error in the UI if possible
            current_id = self._current_config.get('model_id', 'N/A')
            self._current_config["model_id"] = f"{current_id} (Info Error)"

    def _run_in_background(self, task_func, *args, completion_signal):
        """Submits function to thread pool, handles completion/errors via signals."""
        if self._shutdown_event.is_set():
            logger.warning("Shutdown in progress, ignoring new task submission.")
            # Try to emit the completion signal immediately indicating cancellation
            if hasattr(completion_signal, "emit"):
                QTimer.singleShot(
                    0,
                    lambda: completion_signal.emit( # type: ignore [attr-defined]
                        "Task not started: Shutdown in progress.", False
                    ),
                )
            # Ensure UI gets re-enabled if it was disabled before this check
            QTimer.singleShot(0, self.signals.actions_should_reenable.emit) # type: ignore [attr-defined]
            return

        try:
            logger.debug(f"Submitting task {task_func.__name__} to thread pool.")
            future = self._thread_pool.submit(task_func, *args)
            logger.debug(f"Task submitted. Future: {future}. Attaching done callback.")

            # Nested function to handle future completion
            def _intermediate_callback(f):
                logger.debug(f"Intermediate callback running for Future: {f}.")
                result = None
                exception = None
                cancelled = False
                try:
                    if f.cancelled():
                        cancelled = True
                        logger.debug("Future was cancelled.")
                    else:
                        # Check for exception first
                        exception = f.exception()
                        if exception:
                            logger.debug(f"Future completed with exception: {type(exception).__name__}")
                        else:
                            # Get result only if no exception
                            result = f.result()
                            logger.debug(f"Future completed with result type: {type(result)}")
                except CancelledError:
                    # Catch potential CancelledError during result/exception access
                    cancelled = True
                    logger.debug("Future was cancelled (caught CancelledError).")
                except Exception as e:
                    # Catch other errors during result/exception retrieval
                    logger.error(f"Error retrieving future result/exception: {e}", exc_info=True)
                    exception = e # Report this error

                # --- MODIFIED: Removed actions_should_reenable signal emission here ---

                # Emit the internal signal to trigger the final GUI callback
                logger.debug("Emitting internal task completed signal for final GUI update.")
                # Use attribute access for emit
                self.signals._internal_task_completed.emit(
                    result, exception, cancelled, completion_signal
                )

            # Attach the callback to the future object
            future.add_done_callback(_intermediate_callback)
            logger.debug(f"Done callback attached to future {future}.")

        except Exception as e:
            # Handle errors during task submission itself
            logger.error(f"Failed to submit task to thread pool: {e}", exc_info=True)
            if hasattr(completion_signal, "emit"):
                QTimer.singleShot(
                    0,
                    lambda err=e: completion_signal.emit(f"Task Submission Error: {err}", False), # type: ignore [attr-defined]
                )
            # Ensure UI is re-enabled if submission fails
            QTimer.singleShot(0, self.signals.actions_should_reenable.emit) # type: ignore [attr-defined]

    # Slot connected to _internal_task_completed signal
    @Slot(object, object, bool, Signal) # Add Slot decorator
    def _final_gui_callback(
        self,
        result: Optional[Any],
        exception: Optional[Exception],
        cancelled: bool,
        completion_signal: Signal, # Use specific type hint
    ):
        """Handles the final result/exception in the GUI thread after background task."""
        logger.debug(f">>> GUI Callback: Result={type(result)}, Exc={type(exception)}, Cancel={cancelled}")

        # Check if the provided signal is valid before attempting to emit
        can_emit = hasattr(completion_signal, "emit")
        if not can_emit:
            logger.error("Cannot emit completion signal: Invalid signal object provided.")
            return

        try:
            if cancelled:
                logger.info("Task was cancelled branch taken in final callback.")
                completion_signal.emit("Task cancelled during execution.", False) # type: ignore [attr-defined]
                return # Stop processing if cancelled

            if exception:
                logger.info("Task had exception branch taken in final callback.")
                # Check if shutdown happened *during* the task execution
                if not self._shutdown_event.is_set():
                    # Log the full error if not shutting down
                    logger.error(f"Exception in background task: {exception}", exc_info=False) # Avoid huge tracebacks in UI logs
                    completion_signal.emit(f"Task Error: {exception}", False) # type: ignore [attr-defined]
                else:
                    # Log less severely if shutdown was in progress
                    logger.warning(f"Task ended with exception during shutdown: {exception}")
                    completion_signal.emit("Task interrupted by shutdown.", False) # type: ignore [attr-defined]
                return # Stop processing if there was an exception

            # Process successful result
            logger.debug(f"Processing successful result: {result!r}")
            # Expect specific tuple structure: (message, success_bool)
            if result is not None: # Added check for None before tuple check
                is_expected_tuple = (
                    isinstance(result, tuple) and
                    len(result) == 2 and
                    isinstance(result[1], bool)
                )

                if is_expected_tuple:
                    result_message, success = result # Now safer after None check
                    # Ensure message is string for emission
                    result_message_str = str(result_message)
                    logger.debug(f"Emitting completion signal: Success={success}, Msg='{result_message_str[:100]}...'")
                    completion_signal.emit(result_message_str, success) # type: ignore [attr-defined]
                else:
                    # Handle unexpected result structure
                    err_msg = f"Background task returned unexpected result type/structure: {type(result)}. Value: {result!r}"
                    logger.error(err_msg)
                    # Emit failure signal with info about the unexpected result
                    completion_signal.emit(f"Task completed with unexpected result: {str(result)[:100]}", False) # type: ignore [attr-defined]
            else: # Handle case where result is None and not cancelled/exception
                 logger.warning("Task completed with None result and no exception/cancellation.")
                 # Decide how to signal completion in this case, e.g., emit failure or specific message
                 completion_signal.emit("Task completed without result.", False) # type: ignore [attr-defined]

            # --- MODIFIED: Moved actions_should_reenable emit here ---
            # Always schedule re-enabling actions after processing result/exception
            logger.debug("Scheduling actions re-enable from final GUI callback.")
            QTimer.singleShot(0, self.signals.actions_should_reenable.emit) # type: ignore [attr-defined]
            # --- END MODIFICATION ---

        except Exception as callback_exc:
            # Catch errors within this callback itself
            logger.error(f"Error processing task result or emitting signal in GUI callback: {callback_exc}", exc_info=True)
            # Try to emit an error signal about the callback failure
            try:
                if can_emit:
                    # Use attribute access for emit
                    completion_signal.emit(f"GUI Callback Error: {callback_exc}", False) # type: ignore [attr-defined]
            except Exception as emit_err:
                # Last resort logging if even error signal emission fails
                logger.critical(f"Failed even to emit error signal after callback failure: {emit_err}")
            # --- MODIFIED: Try to re-enable actions even if callback fails ---
            logger.debug("Attempting actions re-enable after callback exception.")
            QTimer.singleShot(0, self.signals.actions_should_reenable.emit) # type: ignore [attr-defined]
            # --- END MODIFICATION ---

    # --- Public Methods to Submit Tasks ---

    def submit_search(self, query: str):
        """Submits a search task to the background."""
        if self._shutdown_event.is_set():
            # Use attribute access for emit
            self.signals.search_completed.emit("Search cancelled: Shutdown active.", False) # type: ignore [attr-defined]
            return
        if not self.llm_search:
            # Use attribute access for emit
            self.signals.search_completed.emit("Search failed: Backend not ready.", False) # type: ignore [attr-defined]
            return
        if not query:
            # Use attribute access for emit
            self.signals.search_completed.emit("Please enter a query.", False) # type: ignore [attr-defined]
            return

        logger.info(f"Submitting search: '{query[:50]}...'")
        # Use attribute access for emit
        self.signals.status_updated.emit(f"Searching '{query[:30]}...'", "info") # type: ignore [attr-defined]
        # Note: UI disabling should happen in the view *before* calling submit
        # The re-enabling happens automatically via signals after task completion.

        # Use QTimer to defer the background submission slightly, ensuring UI updates first
        QTimer.singleShot(
            0, # Delay of 0 ms, runs in next event loop cycle
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
        """Submits a crawl and index task to the background."""
        if self._shutdown_event.is_set():
            # Use attribute access for emit
            self.signals.crawl_index_completed.emit("Task cancelled: Shutdown active.", False) # type: ignore [attr-defined]
            return
        if not self.llm_search:
            # Use attribute access for emit
            self.signals.crawl_index_completed.emit("Task failed: Backend not ready.", False) # type: ignore [attr-defined]
            return

        logger.info(f"Submitting crawl & index task for {len(root_urls)} URLs...")
        # Use attribute access for emit
        self.signals.status_updated.emit( # type: ignore [attr-defined]
            f"Starting crawl & index for {len(root_urls)} URL(s)...", "info"
        )
        # UI disabling happens in the view

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
        """Submits a manual indexing task to the background."""
        if self._shutdown_event.is_set():
            # Use attribute access for emit
            self.signals.manual_index_completed.emit("Indexing cancelled: Shutdown active.", False) # type: ignore [attr-defined]
            return
        if not self.llm_search:
            # Use attribute access for emit
            self.signals.manual_index_completed.emit("Indexing failed: Backend not ready.", False) # type: ignore [attr-defined]
            return

        source_path = Path(path_str)
        if not source_path.exists():
            # Use attribute access for emit
            self.signals.manual_index_completed.emit(f"Error: Path does not exist: {path_str}", False) # type: ignore [attr-defined]
            return

        logger.info(f"Submitting manual index task for: {source_path}")
        # Use attribute access for emit
        self.signals.status_updated.emit(f"Indexing '{source_path.name}'...", "info") # type: ignore [attr-defined]
        # UI disabling happens in the view

        QTimer.singleShot(
            0,
            lambda: self._run_in_background(
                self._execute_manual_index_task,
                path_str,
                completion_signal=self.signals.manual_index_completed,
            ),
        )

    def submit_removal(self, source_identifier_to_remove: str):
        """Submits a source removal task to the background."""
        if self._shutdown_event.is_set():
            # Use attribute access for emit
            self.signals.removal_completed.emit("Removal cancelled: Shutdown active.", False) # type: ignore [attr-defined]
            return
        if not self.llm_search:
            # Use attribute access for emit
            self.signals.removal_completed.emit("Error: Cannot remove, Backend not ready.", False) # type: ignore [attr-defined]
            return
        if not isinstance(source_identifier_to_remove, str) or not source_identifier_to_remove:
            # Use attribute access for emit
            self.signals.removal_completed.emit("Error: Invalid source identifier.", False) # type: ignore [attr-defined]
            return

        logger.info(f"Submitting removal task for source identifier: {source_identifier_to_remove}")
        try:
            # Create a display name for status update
            if source_identifier_to_remove.startswith("http"):
                display_name = source_identifier_to_remove
            else:
                display_name = Path(source_identifier_to_remove).name
            display_name = (display_name[:40] + "...") if len(display_name) > 40 else display_name
        except Exception:
            display_name = source_identifier_to_remove[:40] + "..."

        # Use attribute access for emit
        self.signals.status_updated.emit(f"Removing '{display_name}'...", "info") # type: ignore [attr-defined]
        # UI disabling happens in the view

        QTimer.singleShot(
            0,
            lambda: self._run_in_background(
                self._execute_removal_task,
                source_identifier_to_remove,
                completion_signal=self.signals.removal_completed,
            ),
        )

    # --- Private Task Execution Methods ---
    # These run in the background thread pool

    def _execute_search_task(self, query: str) -> Tuple[str, bool]:
        """Executes the search query. Returns (result_message, success_bool)."""
        result_message = "Search failed unexpectedly."
        success = False
        try:
            logger.debug(f"Executing search task for: '{query[:50]}...'")
            if self._shutdown_event.is_set():
                # Task checks for shutdown signal
                return "Search cancelled (shutdown detected)", False
            if not self.llm_search:
                return "Search Error: LLMSearch instance not available.", False

            start_time = time.time()
            results = self.llm_search.llm_query(query, debug_mode=self.debug)
            duration = time.time() - start_time

            if self._shutdown_event.is_set():
                # Check again after potentially long operation
                return "Search interrupted after processing (shutdown detected)", False

            logger.info(f"Search task completed in {duration:.2f} seconds.")
            # Expect formatted HTML response from llm_query
            result_message = results.get("formatted_response", "No response generated.")
            success = True # Assume success if no exception occurred
        except Exception as e:
            # Check if shutdown happened *during* the exception
            if not self._shutdown_event.is_set():
                logger.error(f"Search task failed unexpectedly: {e}", exc_info=self.debug)
                result_message = f"Search Error: {e}"
                success = False
            else:
                # Error likely due to shutdown interruption
                logger.warning(f"Search failed during shutdown process: {e}")
                result_message = f"Search stopped due to shutdown: {e}"
                success = False
        # Always return the expected tuple structure
        return result_message, success

    def _execute_crawl_and_index_task(
        self,
        root_urls: List[str],
        target_links: int,
        max_depth: int,
        keywords: Optional[List[str]],
    ) -> Tuple[str, bool]:
        """Executes crawling and subsequent indexing. Returns (status_message, success_bool)."""
        logger.debug("Executing crawl & index task...")
        crawl_successful, index_successful = False, False
        crawl_duration, index_duration = 0.0, 0.0
        total_added_chunks = 0
        crawl_start_time = 0.0
        indexing_start_time = 0.0
        loop: Optional[asyncio.AbstractEventLoop] = None
        policy = asyncio.get_event_loop_policy()
        final_message = "Task initialization failed."
        overall_success = False
        total_start_time = time.time() # Track total time
        collected_urls : Optional[List[str]] = None # Initialize collected_urls

        try:
            # --- Crawl Phase ---
            try:
                crawl_start_time = time.time()
                if self._shutdown_event.is_set():
                    raise InterruptedError("Shutdown before crawl start.")

                crawl_dir_base_path_str = self.data_paths.get("crawl_data")
                if not crawl_dir_base_path_str:
                     raise SetupError("Crawl data directory path not configured.")
                crawl_dir_base = Path(crawl_dir_base_path_str)
                logger.info(f"Starting crawl phase. Output: {crawl_dir_base}")

                # Create crawler instance
                self._active_crawler = Crawl4AICrawler(
                    root_urls=root_urls,
                    base_crawl_dir=crawl_dir_base,
                    target_links=target_links,
                    max_depth=max_depth,
                    relevance_keywords=keywords,
                    headless=True, # Typically True for backend tasks
                    user_agent="LlamaSearchBot/1.0", # Identify the bot
                    shutdown_event=self._shutdown_event, # Pass shutdown event
                    verbose_logging=self.debug,
                )

                # Manage asyncio loop for the crawler
                loop = policy.new_event_loop()
                asyncio.set_event_loop(loop)
                # Store the result of run_crawl
                collected_urls = loop.run_until_complete(self._active_crawler.run_crawl())

                # Close crawler resources gracefully
                if self._active_crawler:
                    logger.debug("Closing crawler resources...")
                    loop.run_until_complete(self._active_crawler.close())
                    logger.debug("Crawler resources closed.")

                crawl_duration = time.time() - crawl_start_time

                # Check for shutdown *after* crawl finishes/errors
                if self._shutdown_event.is_set():
                    raise InterruptedError("Shutdown detected during/after crawl.")

                # Check if collected_urls is valid before using its length
                url_count = len(collected_urls) if collected_urls is not None else 0
                logger.info(f"Crawl phase OK ({crawl_duration:.1f}s). Collected {url_count} URLs.")
                crawl_successful = True

            except InterruptedError as e:
                if crawl_start_time > 0:
                    crawl_duration = time.time() - crawl_start_time
                logger.warning(f"Crawl interrupted ({crawl_duration:.1f}s): {e}")
                crawl_successful = False # Mark as not successful if interrupted
                # Don't set final_message here, let reporting handle interruption message
            except Exception as crawl_exc:
                if crawl_start_time > 0:
                    crawl_duration = time.time() - crawl_start_time
                logger.error(f"Crawl FAILED ({crawl_duration:.1f}s): {crawl_exc}", exc_info=self.debug)
                final_message = f"Crawl failed: {crawl_exc}" # Set specific error
                crawl_successful = False
            finally:
                # Ensure crawler instance is cleared and loop is closed
                self._active_crawler = None
                if loop is not None:
                    try:
                        if loop.is_running():
                            # Cancel remaining tasks in the loop if any
                            tasks = asyncio.all_tasks(loop=loop)
                            # Add check if tasks is not None before iterating
                            if tasks:
                                logger.debug(f"Cancelling {len(tasks)} remaining asyncio tasks...")
                                for task in tasks:
                                    task.cancel()
                                # Give cancellations a moment to process
                                loop.run_until_complete(asyncio.sleep(0.1))
                        # Close the loop if it's not already closed
                        if not loop.is_closed():
                            loop.close()
                            logger.debug(f"Closed asyncio event loop {id(loop)}.")
                        # Reset the event loop policy if this was the loop set
                        current_policy_loop = None
                        try:
                             current_policy_loop = policy.get_event_loop()
                        except RuntimeError: # Can happen if no loop is set
                             pass
                        if current_policy_loop is loop:
                            policy.set_event_loop(None)
                    except RuntimeError: # Loop might already be closed or not set
                         pass
                    except Exception as loop_close_err:
                        logger.error(f"Error cleaning asyncio loop: {loop_close_err}", exc_info=True)

            # --- Indexing Phase (only if crawl was successful and URLs collected) ---
            indexing_start_time = 0.0 # Reset before phase
            if crawl_successful and collected_urls is not None:
                # --- MODIFIED: Reload reverse lookup ---
                if self.llm_search and hasattr(self.llm_search, '_load_reverse_lookup'):
                    try:
                        logger.info("Refreshing reverse lookup cache in LLMSearch instance...")
                        self.llm_search._load_reverse_lookup()
                    except Exception as lookup_reload_err:
                        logger.error(f"Failed to reload reverse lookup before indexing: {lookup_reload_err}", exc_info=self.debug)
                # --- END MODIFICATION ---

                try:
                    indexing_start_time = time.time()
                    if self._shutdown_event.is_set():
                        raise InterruptedError("Shutdown before indexing start.")
                    if not self.llm_search:
                        raise RuntimeError("LLMSearch instance not available for indexing.")

                    crawl_dir_base_path_str = self.data_paths.get("crawl_data")
                    if not crawl_dir_base_path_str:
                        raise SetupError("Crawl data dir path not configured.")
                    crawl_dir_base = Path(crawl_dir_base_path_str)
                    raw_output_dir = crawl_dir_base.joinpath("raw")
                    processed_files_count = 0
                    logger.info(f"Starting indexing phase for crawled content in: {raw_output_dir}...")

                    if raw_output_dir.is_dir():
                        files_to_index = list(raw_output_dir.glob("*.md"))
                        logger.info(f"Found {len(files_to_index)} markdown files to index.")
                        for md_file in files_to_index:
                            if self._shutdown_event.is_set():
                                raise InterruptedError("Shutdown during indexing loop.")

                            logger.debug(f"Indexing file: {md_file.name}")
                            # Pass internal_call=True for files from crawl directory
                            added_count, blocked = self.llm_search.add_source(str(md_file), internal_call=True)
                            if blocked:
                                 logger.warning(f"Indexing blocked for internal call? {md_file.name}") # Should not happen
                            elif added_count > 0:
                                total_added_chunks += added_count
                            # Count file as processed regardless of chunks added (e.g., if unchanged)
                            processed_files_count += 1
                    else:
                        logger.warning(f"Crawl 'raw' directory not found: {raw_output_dir}. No indexing performed.")

                    index_duration = time.time() - indexing_start_time
                    logger.info(f"File processing phase completed ({index_duration:.1f}s). Indexed {processed_files_count} files. Total chunks added: {total_added_chunks}.")

                    if self._shutdown_event.is_set():
                        raise InterruptedError("Shutdown detected after indexing phase.")
                    index_successful = True

                except InterruptedError as e:
                    if indexing_start_time > 0:
                        index_duration = time.time() - indexing_start_time
                    logger.warning(f"Indexing interrupted ({index_duration:.1f}s): {e}")
                    index_successful = False # Mark as unsuccessful
                except Exception as index_exc:
                    if indexing_start_time > 0:
                        index_duration = time.time() - indexing_start_time
                    logger.error(f"Indexing FAILED ({index_duration:.1f}s): {index_exc}", exc_info=self.debug)
                    final_message = f"Indexing failed: {index_exc}" # Set specific error
                    index_successful = False
            elif crawl_successful and collected_urls is None:
                 # Handle case where crawl succeeded but returned None (shouldn't happen ideally)
                 logger.warning("Crawl reported success but returned no URL list. Skipping indexing.")
                 index_successful = False # Treat as if indexing couldn't happen


        except Exception as outer_exc:
            # Catch errors outside the specific phases (e.g., before crawl starts)
            logger.error(f"Unexpected error during crawl/index task execution: {outer_exc}", exc_info=True)
            final_message = f"Task failed with unexpected error: {outer_exc}"
            overall_success = False # Ensure overall failure

        # --- Final Reporting ---
        total_duration = time.time() - total_start_time
        shutdown_occurred = self._shutdown_event.is_set()
        # Determine overall success: Crawl OK AND Index OK (or Index skipped because Crawl failed) AND no shutdown interruption
        index_phase_ok = (crawl_successful and index_successful) or not crawl_successful
        overall_success = crawl_successful and index_phase_ok and not shutdown_occurred

        # Build status message parts
        crawl_status_msg = "Crawl skipped."
        if crawl_start_time > 0: # If crawl phase was attempted
             if crawl_successful:
                 crawl_status_msg = f"Crawl OK ({crawl_duration:.1f}s)."
             elif shutdown_occurred:
                 crawl_status_msg = f"Crawl INTERRUPTED ({crawl_duration:.1f}s)."
             else:
                 crawl_status_msg = f"Crawl FAILED ({crawl_duration:.1f}s)."

        index_status_msg = "Index skipped."
        if crawl_successful: # Only report index status if crawl happened
            if indexing_start_time > 0: # If index phase was attempted
                if index_successful:
                    index_status_msg = f"Index OK ({index_duration:.1f}s, {total_added_chunks} chunks added)."
                elif shutdown_occurred:
                    index_status_msg = f"Index INTERRUPTED ({index_duration:.1f}s)."
                else:
                    index_status_msg = f"Index FAILED ({index_duration:.1f}s)."
            elif shutdown_occurred: # If shutdown happened before indexing could start
                index_status_msg = "Index INTERRUPTED (before start)."
            else: # If crawl succeeded but index phase didn't start (e.g., error before loop, or collected_urls was None)
                index_status_msg = "Index SKIPPED or FAILED (before start)."


        # Construct final message based on overall status
        if overall_success:
            final_message = f"Finished ({total_duration:.1f}s). {crawl_status_msg} {index_status_msg}".strip()
        elif shutdown_occurred:
            # Use the status parts to show where interruption occurred
            final_message = f"Task INTERRUPTED ({total_duration:.1f}s). {crawl_status_msg} {index_status_msg}".strip()
        elif "failed" not in final_message.lower() and "error" not in final_message.lower():
            # If no specific error message was set, use status parts for failure summary
            final_message = f"Task FAILED ({total_duration:.1f}s). {crawl_status_msg} {index_status_msg}".strip()
        # else: Keep the specific error message set during exception handling

        logger.info(final_message)
        # Trigger UI refresh only if successful and chunks were added
        if overall_success and total_added_chunks > 0:
            # Ensure signal emission happens in GUI thread
            QTimer.singleShot(0, self.signals.refresh_needed.emit) # type: ignore [attr-defined]

        return final_message, overall_success

    def _execute_manual_index_task(self, path_str: str) -> Tuple[str, bool]:
        """Executes manual indexing. Returns (status_message, success_bool)."""
        task_successful = False
        final_message = "Task did not complete."
        total_added_chunks = 0
        overall_success = False
        start_time = time.time()
        source_path = Path(path_str) # For logging/messages

        try:
            logger.debug(f"Executing manual index task for: {path_str}")
            if self._shutdown_event.is_set():
                raise InterruptedError("Shutdown before manual index start.")
            if not self.llm_search:
                raise RuntimeError("LLMSearch instance not available.")

            logger.info(f"Manually indexing source: {source_path.name}...")
            # Call add_source with default internal_call=False
            # It returns (chunks_added, was_blocked)
            added_count, was_blocked = self.llm_search.add_source(path_str)

            if self._shutdown_event.is_set():
                # Check for shutdown immediately after the potentially long operation
                raise InterruptedError("Shutdown detected after manual index processing.")

            if was_blocked:
                final_message = f"Indexing skipped for '{source_path.name}': Cannot manually index from managed crawl directory."
                task_successful = False # Blocked is not a success in this context
                logger.warning(final_message)
            elif added_count >= 0: # 0 chunks added is still success if not blocked (e.g., unchanged file)
                total_added_chunks = added_count
                logger.info(f"Manual index processing complete for '{source_path.name}'. Added {total_added_chunks} chunks.")
                task_successful = True
            # else: Should not happen, add_source returns >= 0 or raises

        except InterruptedError as e:
            logger.warning(f"Manual index interrupted: {e}")
            task_successful = False
            final_message = f"Indexing '{source_path.name}' interrupted." # Set specific message
        except Exception as e:
            logger.error(f"Manual index FAILED for '{source_path.name}': {e}", exc_info=self.debug)
            task_successful = False # Ensure false on exception
            final_message = f"Indexing '{source_path.name}' FAILED. Error: {e}" # Set specific message

        # --- Final Reporting ---
        shutdown_occurred = self._shutdown_event.is_set()
        # Overall success means task logic completed AND wasn't interrupted by shutdown
        overall_success = task_successful and not shutdown_occurred

        # Set final message only if not already set by specific errors/interruptions
        if task_successful and not shutdown_occurred:
            if total_added_chunks > 0:
                final_message = f"Indexing '{source_path.name}' OK ({time.time() - start_time:.1f}s). Added {total_added_chunks} new chunks."
                # Refresh UI only if chunks were actually added
                QTimer.singleShot(0, self.signals.refresh_needed.emit) # type: ignore [attr-defined]
            else:
                # File was processed but no chunks added (likely unchanged or empty)
                final_message = f"Indexing '{source_path.name}' OK ({time.time() - start_time:.1f}s). No new chunks added (file unchanged or empty)."
        # else: Keep the message set during exception or interruption handling

        logger.info(final_message)
        return final_message, overall_success

    def _execute_removal_task(self, source_identifier_to_remove: str) -> Tuple[str, bool]:
        """Executes removal of a source. Returns (status_message, success_bool)."""
        task_successful = False
        final_message = "Task did not complete."
        removal_occurred = False # Track if remove_source actually deleted something
        overall_success = False
        # Create display name early for messages
        try:
             if source_identifier_to_remove.startswith("http"):
                 display_name = source_identifier_to_remove
             else:
                 display_name = Path(source_identifier_to_remove).name
             display_name = (display_name[:40] + "...") if len(display_name) > 40 else display_name
        except Exception:
             display_name = source_identifier_to_remove[:40] + "..."

        try:
            logger.debug(f"Executing removal task for identifier: {source_identifier_to_remove}")
            if self._shutdown_event.is_set():
                raise InterruptedError("Shutdown before removal start.")
            if not self.llm_search:
                raise RuntimeError("LLMSearch instance not available.")

            # remove_source returns (removed_bool, blocked_bool) - ignore blocked here
            removal_occurred, _ = self.llm_search.remove_source(source_identifier_to_remove)

            if self._shutdown_event.is_set():
                # Check again after potentially long operation
                raise InterruptedError("Shutdown detected after removal processing.")

            task_successful = True # If no exception, consider the task logic successful

        except InterruptedError as e:
            logger.warning(f"Removal interrupted: {e}")
            task_successful = False
            final_message = f"Removal of '{display_name}' interrupted." # Set specific message
        except Exception as e:
            logger.error(f"Removal FAILED for '{display_name}': {e}", exc_info=self.debug)
            task_successful = False # Ensure false on exception
            final_message = f"Error removing '{display_name}'. See logs." # Set specific message

        # --- Final Reporting ---
        shutdown_occurred = self._shutdown_event.is_set()
        # Overall success means task logic completed AND wasn't interrupted by shutdown
        overall_success = task_successful and not shutdown_occurred

        # Set final message only if not already set by exception/interruption
        if task_successful and not shutdown_occurred:
            if removal_occurred:
                final_message = f"Successfully removed source: {display_name}"
                # Refresh UI only if removal actually happened
                QTimer.singleShot(0, self.signals.refresh_needed.emit) # type: ignore [attr-defined]
            else:
                # remove_source returned False, meaning source wasn't found
                final_message = f"Source not found or already removed: {display_name}"
        # else: Keep the message set during exception or interruption handling

        logger.info(final_message)
        return final_message, overall_success

    # --- Synchronous Data/Config Retrieval Methods ---

    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """Retrieves indexed source information from LLMSearch (synchronously)."""
        if not self.llm_search:
            logger.warning("Cannot get indexed sources: LLMSearch not ready.")
            return []
        try:
            # This call is expected to be relatively quick
            return self.llm_search.get_indexed_sources()
        except Exception as e:
            logger.error(f"Failed to get indexed sources: {e}", exc_info=self.debug)
            return []

    def get_current_config(self) -> Dict[str, Any]:
        """Returns current configuration state (synchronously)."""
        # Ensure config reflects the current state of the loaded LLM
        self._update_config_from_llm()
        # Add the app's debug state to the returned config
        self._current_config["debug_mode"] = self.debug
        # Return a copy to prevent external modification
        return self._current_config.copy()

    # --- Settings Application ---

    def apply_settings(self, settings: Dict[str, Any]):
        """Applies settings changes (synchronously). Emits signal on completion."""
        config_changed = False

        # --- Apply Debug Logging Setting ---
        new_debug = settings.get("debug_mode", self.debug)
        if new_debug != self.debug:
            self.debug = new_debug
            log_level = logging.DEBUG if self.debug else logging.INFO
            # Get the root logger configured by utils.setup_logging
            root_logger = logging.getLogger("llamasearch")
            root_logger.setLevel(log_level) # Set root level first

            # Adjust levels of existing handlers
            for handler in root_logger.handlers:
                if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
                    # File handler always captures DEBUG, regardless of app debug setting
                    handler.setLevel(logging.DEBUG)
                elif isinstance(handler, logging.StreamHandler):
                    # Console handler reflects the current debug setting
                    handler.setLevel(log_level)
                else:
                    # Other handlers (like Qt) reflect the current debug setting
                    handler.setLevel(log_level)

                # Specific adjustment for QtLogHandler if it exists
                try:
                    from llamasearch.ui.qt_logging import QtLogHandler
                    if QtLogHandler and isinstance(handler, QtLogHandler):
                        handler.setLevel(log_level)
                except ImportError:
                    # Qt logging not available
                    pass

            logger.info(f"Logging level set to: {'DEBUG' if self.debug else 'INFO'}")
            # Update debug flag in LLMSearch instance if it exists
            if self.llm_search:
                self.llm_search.debug = self.debug
                self.llm_search.verbose = self.debug # Keep verbose in sync
            config_changed = True

        # --- Apply Max Search Results Setting ---
        # Ensure we handle potential string input from UI correctly
        new_max_results_val = settings.get("max_results", self._current_config.get("max_results", 3))
        try:
            new_max_results = int(new_max_results_val) # Convert to int
            current_max = self._current_config.get("max_results", 3)
            # Check if value is positive and different from current
            if new_max_results > 0 and new_max_results != current_max:
                self._current_config["max_results"] = new_max_results
                # Update the backend instance if it exists
                if self.llm_search:
                    self.llm_search.max_results = new_max_results
                logger.info(f"Max search results set to: {new_max_results}")
                config_changed = True
            elif new_max_results <= 0:
                logger.warning(f"Invalid Max Results value ignored: {new_max_results}. Must be > 0.")
        except (ValueError, TypeError):
            # Catch errors if conversion to int fails
            logger.warning(f"Invalid Max Results type ignored: {new_max_results_val}.")

        # --- Emit Completion Signal ---
        # Determine message based on whether changes were made
        if config_changed:
            msg = "Settings applied successfully."
            lvl = "success"
        else:
            msg = "No changes applied."
            lvl = "info"

        # Let the SettingsView know the outcome
        # Use attribute access for emit
        self.signals.settings_applied.emit(msg, lvl) # type: ignore [attr-defined]

    # --- Cleanup ---
    def close(self):
        """Cleans up resources, signals shutdown."""
        logger.info("Closing LlamaSearchApp backend resources...")
        # Signal background tasks to stop
        self._shutdown_event.set()

        # Request crawler abort if active
        if self._active_crawler:
            logger.debug("Requesting crawler abort...")
            try:
                self._active_crawler.abort()
            except Exception as e:
                 logger.error(f"Error aborting crawler: {e}")

        # Close the main LLMSearch instance
        if self.llm_search:
            logger.debug("Closing LLMSearch instance...")
            try:
                self.llm_search.close()
            except Exception as e:
                 logger.error(f"Error closing LLMSearch: {e}")

        # Note: Executor shutdown is handled by MainWindow.closeEvent
        logger.info("LlamaSearchApp backend resources closed.")