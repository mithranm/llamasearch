#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
# Ensure the import path is correct relative to the project structure
from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.ui.views.settings_view import SettingsView # Import specific class now
from llamasearch.ui.views.search_view import SearchAndIndexView
from llamasearch.ui.views.terminal_view import TerminalView
from llamasearch.ui.components import header_component
# --- Import setup_logging ---
from llamasearch.utils import setup_logging

# --- Setup logging early, before creating backend ---
# This ensures the QtLogHandler is attached if available
logger = setup_logging("llamasearch.gui_main", use_qt_handler=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LlamaSearch")

        # Initialize backend first
        # Read debug from args/env var if needed here? For now, False.
        # Backend will use the already configured logger
        self.backend = LlamaSearchApp(requires_gpu=False, debug=False)

        central = QWidget()
        self.main_layout = QVBoxLayout(central)

        # --- Add Header ---
        app_header = header_component(self.backend.data_paths)
        self.main_layout.addWidget(app_header)

        # --- Add Tabs ---
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Add search & index tab
        sa_tab = SearchAndIndexView(self.backend)
        self.tabs.addTab(sa_tab, "Search & Index")

        # Add settings tab
        st_tab = SettingsView(self.backend) # Use class directly
        self.tabs.addTab(st_tab, "Settings")

        # --- Add Terminal/Log View Tab ---
        term_tab = TerminalView(self.backend)
        self.tabs.addTab(term_tab, "Logs")

        central.setLayout(self.main_layout)
        self.setCentralWidget(central)
        self.resize(1200, 800)

    def closeEvent(self, event):
        """Ensure backend resources are released on close."""
        logger.info("Close event received, shutting down backend...")
        self.backend.close()
        logger.info("Backend shutdown complete.")
        super().closeEvent(event)

def main():
    # Optional: Set High DPI scaling attributes
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

# Guard for entry point execution
if __name__ == "__main__":
    # Logging is set up at the top level now
    main()