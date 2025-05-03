#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import QObject, Signal, Slot

from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.ui.views.crawl_view import CrawlView
from llamasearch.ui.views.search_view import SearchView
from llamasearch.ui.views.settings_view import settings_view
from llamasearch.ui.views.terminal_view import TerminalView
from llamasearch.ui.components import header_component

# Create a signal class for communication
class AppSignals(QObject):
    crawl_started = Signal()
    crawl_finished = Signal()
    search_started = Signal()
    search_finished = Signal()

def main():
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle("LlamaSearch")
    
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)
    
    # Add header at the top
    header = header_component()
    layout.addWidget(header)
    
    # Create tab widget and add tabs
    tabs = QTabWidget()
    layout.addWidget(tabs)
    
    # Create signals object
    signals = AppSignals()
    
    # Create backend instance with signals
    backend = LlamaSearchApp(use_cpu=False, debug=False, signals=signals)
    
    # Create views
    crawl_tab = CrawlView(backend)
    search_tab = SearchView(backend)
    settings_tab = settings_view(backend)
    terminal_tab = TerminalView(backend)
    
    # Add tabs
    tabs.addTab(crawl_tab, "Crawl Website")
    tabs.addTab(search_tab, "Search Content")
    tabs.addTab(settings_tab, "Settings")
    tabs.addTab(terminal_tab, "Terminal")
    
    # Store previous tab index
    previous_tab_index = 0
    
    # Define slots for tab switching
    @Slot()
    def on_crawl_started():
        nonlocal previous_tab_index
        previous_tab_index = tabs.currentIndex()
        tabs.setCurrentIndex(3)  # Switch to Terminal tab (index 3)
    
    @Slot()
    def on_crawl_finished():
        nonlocal previous_tab_index
        tabs.setCurrentIndex(previous_tab_index)  # Switch back to previous tab
    
    @Slot()
    def on_search_started():
        nonlocal previous_tab_index
        previous_tab_index = tabs.currentIndex()
        tabs.setCurrentIndex(3)  # Switch to Terminal tab (index 3)
    
    @Slot()
    def on_search_finished():
        nonlocal previous_tab_index
        tabs.setCurrentIndex(previous_tab_index)  # Switch back to previous tab
    
    # Connect signals to slots
    signals.crawl_started.connect(on_crawl_started)
    signals.crawl_finished.connect(on_crawl_finished)
    signals.search_started.connect(on_search_started)
    signals.search_finished.connect(on_search_finished)
    
    main_window.setCentralWidget(central_widget)
    main_window.resize(1000, 700)
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()