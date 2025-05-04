#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.ui.views.settings_view import settings_view
from llamasearch.ui.views.search_view import SearchAndIndexView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LlamaSearch")

        self.backend = LlamaSearchApp(requires_gpu=False, debug=False)
        central = QWidget()
        lay = QVBoxLayout(central)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        # Add settings tab
        st_tab = settings_view(self.backend)
        self.tabs.addTab(st_tab, "Settings")

        # Add search & index tab
        sa_tab = SearchAndIndexView(self.backend)
        self.tabs.addTab(sa_tab, "Search & Index")

        central.setLayout(lay)
        self.setCentralWidget(central)
        self.resize(1200,800)

    def closeEvent(self, event):
        self.backend.close()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
