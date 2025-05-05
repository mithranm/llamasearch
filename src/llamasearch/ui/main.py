# src/llamasearch/ui/main.py

#!/usr/bin/env python3
import sys
import signal
from concurrent.futures import ThreadPoolExecutor
import asyncio # Import asyncio
import logging # Import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget,
                               QVBoxLayout, QWidget, QMessageBox)

# Ensure the import path is correct relative to the project structure
from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.ui.components import header_component
from llamasearch.ui.views.search_view import SearchAndIndexView
from llamasearch.ui.views.settings_view import SettingsView
from llamasearch.ui.views.terminal_view import TerminalView
from llamasearch.utils import setup_logging

# Setup logging early, potentially attaching Qt handler
# Use the correct logger name
logger = setup_logging("llamasearch.ui.main", use_qt_handler=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LlamaSearch")
        self._is_closing = False # Flag to prevent double close operations

        # Initialize backend first
        # Use a reasonable number of workers, 1 might be too few if multiple tasks run
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="LlamaSearchWorker")
        # Determine if debug logging is enabled (e.g., from command line args if passed down)
        # For simplicity here, default to False. In a real app, parse args.
        is_debug = '--debug' in sys.argv
        self.backend = LlamaSearchApp(executor=self._executor, debug=is_debug)

        central = QWidget()
        self.main_layout = QVBoxLayout(central)

        # Add Header
        app_header = header_component(self.backend.data_paths)
        self.main_layout.addWidget(app_header)

        # Add Tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Search & Index Tab
        sa_tab = SearchAndIndexView(self.backend)
        self.tabs.addTab(sa_tab, "Search & Index")

        # Settings Tab
        st_tab = SettingsView(self.backend)
        self.tabs.addTab(st_tab, "Settings")

        # Terminal/Log View Tab
        term_tab = TerminalView(self.backend)
        self.tabs.addTab(term_tab, "Logs")

        central.setLayout(self.main_layout)
        self.setCentralWidget(central)
        self.resize(1200, 800)
        logger.info("MainWindow UI Initialized.")

    def closeEvent(self, event):
        """Ensure backend resources are released on close."""
        if self._is_closing:
             logger.debug("Close event ignored: Already closing.")
             event.ignore() # Already closing
             return

        self._is_closing = True
        logger.info("Close event received, initiating shutdown sequence...")

        # 1. Signal backend to stop accepting new tasks and clean up
        logger.info("Signaling backend shutdown...")
        try:
            # Backend close should signal _shutdown_event and close LLMSearch etc.
            self.backend.close()
            logger.info("Backend close() method called.")
        except Exception as be:
             logger.error(f"Error during backend close: {be}", exc_info=True)

        # 2. Shutdown the ThreadPoolExecutor
        # cancel_futures=True will attempt to cancel pending tasks.
        # wait=True ensures we wait for running tasks to finish (or be cancelled).
        logger.info("Shutting down thread pool executor (waiting for tasks)...")
        # Graceful Shutdown: Give tasks a chance to finish cleanly after shutdown signal
        # Shutdown without waiting indefinitely. Adjust timeout as needed.
        # Note: cancel_futures=True is crucial here for tasks that respect cancellation.
        try:
            self._executor.shutdown(wait=True, cancel_futures=True) # Wait for completion/cancellation
            logger.info("Executor shutdown complete.")
        except Exception as exec_shutdown_err:
             logger.error(f"Error during executor shutdown: {exec_shutdown_err}", exc_info=True)


        # 3. Clean up any stray asyncio tasks (if crawler was running)
        # This is a safety measure; ideally, the crawler task manages its own loop cleanup.
        logger.debug("Checking for stray asyncio tasks/loops...")
        try:
             all_loops = []
             # Heuristic to find potentially running loops (less reliable)
             if sys.version_info >= (3, 7):
                 try:
                     tasks = asyncio.all_tasks()
                     if tasks:
                          # Get unique loops associated with pending tasks
                          loops = {t.get_loop() for t in tasks if not t.done()}
                          all_loops.extend(list(loops))
                 except RuntimeError: # Can happen if no loop is set
                      logger.debug("RuntimeError getting asyncio tasks (no loop likely set).")
                      pass

             if not all_loops: # Fallback if no tasks found / older python
                  # Check default loop (might not be the right one)
                   try:
                        loop = asyncio.get_event_loop_policy().get_event_loop()
                        # Check if it's running *and* not already closed
                        if loop.is_running() and not loop.is_closed():
                             all_loops.append(loop)
                   except RuntimeError:
                        logger.debug("RuntimeError getting default event loop (no loop likely set).")
                        pass # No default loop running

             if all_loops:
                 logger.warning(f"Found {len(all_loops)} potentially running asyncio loops during final shutdown. Attempting cleanup.")
                 for loop in all_loops:
                     if loop.is_running() and not loop.is_closed():
                         logger.debug(f"Stopping loop {id(loop)}...")
                         # Stop the loop from the thread it's running in if possible,
                         # otherwise use threadsafe version. Stopping immediately might be okay here.
                         try:
                            loop.call_soon_threadsafe(loop.stop)
                            # Optionally, run briefly to process the stop? Risky.
                            # loop.call_soon_threadsafe(lambda: loop.run_until_complete(asyncio.sleep(0.01)))
                         except Exception as stop_err:
                              logger.error(f"Error stopping loop {id(loop)}: {stop_err}")
                         # Closing the loop here is risky if it wasn't created here.
                         # Rely on process exit for final cleanup if stop doesn't fully work.
             else:
                  logger.info("No active asyncio loops detected during final shutdown.")

        except Exception as ae:
            logger.error(f"Error during final asyncio cleanup check: {ae}", exc_info=True)


        logger.info("Shutdown sequence complete. Accepting close event.")
        event.accept()
        # Force exit if it still hangs after cleanup (use as last resort - increases risk of data corruption)
        # QTimer.singleShot(5000, lambda: sys.exit(0))


# --- SIGINT Handler Function ---
_app_instance = None # Global ref to QApplication
_main_window = None # Global ref to MainWindow

def _handle_sigint(*_):
    """Handle Ctrl+C by initiating the main window close sequence."""
    logger.info("SIGINT received, requesting application close...")
    if _main_window:
        # Trigger the closeEvent method from the main thread
        logger.debug("Calling MainWindow.close() via QTimer...")
        QTimer.singleShot(0, _main_window.close)
    elif _app_instance:
         # Fallback if main window ref is lost
         logger.warning("Main window reference not found, attempting direct app quit.")
         QTimer.singleShot(0, _app_instance.quit)
    else:
         logger.error("Neither main window nor app instance found for SIGINT handling.")

def main():
    global _app_instance, _main_window
    # Optional: Set High DPI scaling attributes
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    # Set application details (optional but good practice)
    QApplication.setApplicationName("LlamaSearch")
    QApplication.setOrganizationName("LlamaSearchOrg") # Replace if applicable
    QApplication.setApplicationVersion(__import__("llamasearch").__version__)

    # --- Setup Logging ---
    # Determine debug level early, e.g., from sys.argv
    debug_mode = '--debug' in sys.argv
    log_level = logging.DEBUG if debug_mode else logging.INFO
    # Setup root logger first, getting the specific ui.main logger
    logger = setup_logging("llamasearch.ui.main", level=log_level, use_qt_handler=True)
    logger.info(f"--- Starting LlamaSearch GUI (Version: {QApplication.applicationVersion()}) ---")
    logger.info(f"Log Level set to: {'DEBUG' if debug_mode else 'INFO'}")
    # --- End Logging Setup ---

    _app_instance = QApplication(sys.argv) # Store app instance
    try:
        _main_window = MainWindow()             # Store main window instance
        _main_window.show()
    except Exception as init_err:
        logger.critical(f"Failed to initialize MainWindow: {init_err}", exc_info=True)
        QMessageBox.critical(_app_instance.activeWindow(), "Initialization Error", f"Failed to start LlamaSearch GUI:\n{init_err}\n\nSee logs for details.") #type: ignore
        sys.exit(1)


    # Set up SIGINT handler to call our custom handler
    # This might need platform-specific handling if signal doesn't work well with Qt loop
    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        logger.debug("SIGINT handler registered.")
        # Need timer to ensure Python interpreter stays awake to receive signal
        timer = QTimer()
        timer.start(500) # Check every 500 ms
        timer.timeout.connect(lambda: None) # Required dummy slot
    except Exception as sig_err:
         logger.error(f"Could not register SIGINT handler: {sig_err}. Ctrl+C might not work gracefully.")


    exit_code = _app_instance.exec()
    logger.info(f"Qt application finished with exit code {exit_code}.")

    # Ensure executor is definitely shut down if exec() returns before closeEvent completes fully
    if hasattr(_main_window, '_executor') and _main_window._executor and not _main_window._executor._shutdown:
         logger.warning("Executor was not shut down before application exit. Forcing quick shutdown.")
         _main_window._executor.shutdown(wait=False, cancel_futures=True) # Quick non-waiting shutdown

    logger.info("--- LlamaSearch GUI Exiting ---")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()