#!/usr/bin/env python3
import sys
import signal

from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget,
                               QVBoxLayout, QWidget)

# Ensure the import path is correct relative to the project structure
from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.ui.components import header_component
from llamasearch.ui.views.search_view import SearchAndIndexView
from llamasearch.ui.views.settings_view import SettingsView
from llamasearch.ui.views.terminal_view import TerminalView
from llamasearch.utils import setup_logging

# Setup logging early, potentially attaching Qt handler
logger = setup_logging("llamasearch.gui_main", use_qt_handler=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LlamaSearch")

        # Initialize backend first
        # <<< Removed requires_gpu argument >>>
        self.backend = LlamaSearchApp(debug=False) # Set debug via args/env if needed

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

    def closeEvent(self, event):
        """Ensure backend resources are released on close."""
        logger.info("Close event received, shutting down backend...")
        self.backend.close() # Call backend close method
        logger.info("Backend shutdown sequence initiated.")
        super().closeEvent(event)
        # Explicitly exit after cleanup if needed, though Qt usually handles this
        # sys.exit(0)


# --- SIGINT Handler Function ---
_app_instance = None # Global ref to QApplication

def _handle_sigint(*_):
    """Handle Ctrl+C by quitting the Qt application."""
    logger.info("SIGINT received, requesting application quit.")
    if _app_instance:
        # Use QTimer to ensure quit happens in the GUI thread
        QTimer.singleShot(0, _app_instance.quit)

def main():
    global _app_instance
    # Optional: Set High DPI scaling attributes
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    _app_instance = QApplication(sys.argv) # Store app instance
    win = MainWindow()
    win.show()

    # Set up SIGINT handler
    signal.signal(signal.SIGINT, _handle_sigint)
    # Need timer to ensure Python interpreter stays awake to receive signal
    timer = QTimer()
    timer.start(500) # Check every 500 ms
    timer.timeout.connect(lambda: None) # Required dummy slot

    exit_code = _app_instance.exec()
    logger.info(f"Qt application finished with exit code {exit_code}.")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()