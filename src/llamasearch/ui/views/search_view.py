# src/llamasearch/ui/views/search_view.py

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QSpinBox,
    QFormLayout,
    QHeaderView,
    QMessageBox,
)
from PySide6.QtCore import (
    Qt,
    # QMetaObject, # No longer needed for these calls
    Slot,
    QTimer, # <-- Import QTimer
)
# from typing import cast, Any # No longer needed for these calls

from pathlib import Path
import logging
import shlex

logger = logging.getLogger(__name__)


class SearchAndIndexView(QWidget):
    """GUI View for Search, Crawl/Index, Manual Indexing, and Source Management."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()
        # Connect backend signals to UI slots
        self.backend.signals.status_updated.connect(self._set_status)
        self.backend.signals.search_completed.connect(self._on_search_complete)
        self.backend.signals.crawl_index_completed.connect(
            self._on_crawl_index_complete
        )
        self.backend.signals.manual_index_completed.connect(
            self._on_manual_index_complete
        )
        self.backend.signals.removal_completed.connect(self._on_removal_complete)
        self.backend.signals.refresh_needed.connect(self.update_data_display)
        QTimer.singleShot(100, self.update_data_display) # Keep initial load timer

    def init_ui(self):
        # --- No changes needed in init_ui itself ---
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        # Search Section
        search_group_layout = QVBoxLayout()
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("<b>Question:</b>"))
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question...")
        query_layout.addWidget(self.query_input)
        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        query_layout.addWidget(self.search_btn)
        search_group_layout.addLayout(query_layout)
        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        self.search_results.setPlaceholderText("Search results...")
        search_group_layout.addWidget(self.search_results, 1)
        main_layout.addLayout(search_group_layout)
        # Crawling Section
        crawl_group_layout = QVBoxLayout()
        crawl_group_layout.addWidget(
            QLabel("<b>Crawl & Index Websites (Keyword Priority):</b>")
        )
        self.urls_text = QTextEdit()
        self.urls_text.setPlaceholderText("Enter root URLs...")
        self.urls_text.setFixedHeight(60)
        crawl_group_layout.addWidget(self.urls_text)
        crawl_params_layout = QHBoxLayout()
        form = QFormLayout()
        form.setHorizontalSpacing(20)
        self.target_links_spin = QSpinBox()
        self.target_links_spin.setRange(1, 1000)
        self.target_links_spin.setValue(15)
        self.target_links_spin.setToolTip("Max pages per URL.")
        form.addRow("Max Pages:", self.target_links_spin)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(0, 10)
        self.depth_spin.setValue(1)
        self.depth_spin.setToolTip("Max crawl depth")
        form.addRow("Max Depth:", self.depth_spin)
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Optional: guide api ...")
        self.keyword_input.setToolTip("Keywords for priority.")
        form.addRow("Relevance Keywords:", self.keyword_input)
        crawl_params_layout.addLayout(form)
        crawl_params_layout.addStretch()
        self.crawl_and_index_btn = QPushButton("Start Crawl & Index")
        self.crawl_and_index_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        crawl_params_layout.addWidget(
            self.crawl_and_index_btn, 0, Qt.AlignmentFlag.AlignBottom
        )
        crawl_group_layout.addLayout(crawl_params_layout)
        main_layout.addLayout(crawl_group_layout)
        # Manual Indexing Section
        local_index_layout = QHBoxLayout()
        local_index_layout.addWidget(QLabel("<b>Manually Index Local Content:</b>"))
        self.index_file_btn = QPushButton("Index File...")
        self.index_dir_btn = QPushButton("Index Directory...")
        self.index_file_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        self.index_dir_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        local_index_layout.addWidget(self.index_file_btn)
        local_index_layout.addWidget(self.index_dir_btn)
        local_index_layout.addStretch()
        main_layout.addLayout(local_index_layout)
        # Indexed Sources Table
        main_layout.addWidget(QLabel("<b>Indexed Sources:</b>"))
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(
            ["Source ID / Path", "Display Name / URL", "Actions"]
        )
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.data_table.setAlternatingRowColors(True)
        main_layout.addWidget(self.data_table, 2)
        # Status Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-style: italic; color: #555;")
        main_layout.addWidget(self.status_label)
        # Connect Signals
        self.search_btn.clicked.connect(self.do_search)
        self.query_input.returnPressed.connect(self.do_search)
        self.crawl_and_index_btn.clicked.connect(self.do_crawl_and_index)
        self.index_file_btn.clicked.connect(self.do_index_file)
        self.index_dir_btn.clicked.connect(self.do_index_dir)


    @Slot(str, str)
    def _set_status(self, message: str, level: str = "info"):
        """Updates the status label using QTimer.singleShot."""
        # Use lambda to capture current values and call methods in GUI thread
        QTimer.singleShot(0, lambda: self.status_label.setText(f"Status: {message}"))

        color = {"error": "red", "warning": "orange", "success": "green"}.get(
            level, "#555"
        )
        style = f"font-style: italic; color: {color};"
        QTimer.singleShot(0, lambda: self.status_label.setStyleSheet(style))


    @Slot()
    def do_search(self):
        """Initiates a search operation."""
        query = self.query_input.text().strip()
        if not query:
            self._set_status("Enter a query.", level="warning")
            self.search_results.setPlainText("Please enter a query.")
            return
        self._set_status(f"Searching '{query[:30]}...'")
        self._disable_actions(True)
        self.search_results.setPlainText("Submitting search...")
        self.backend.submit_search(query)

    @Slot(str, bool)
    def _on_search_complete(self, result_message: str, success: bool):
        """Handles completion of the search task."""
        # This slot is called directly via signal, already in GUI thread
        self.search_results.setPlainText(result_message)
        self._set_status(
            "Search complete." if success else "Search failed.",
            level="success" if success else "error",
        )
        self._disable_actions(False)

    @Slot()
    def do_crawl_and_index(self):
        """Initiates a crawl and index operation."""
        lines = self.urls_text.toPlainText().splitlines()
        root_urls = [
            ln.strip() for ln in lines if ln.strip().startswith(("http://", "https://"))
        ]
        if not root_urls:
            self._set_status("No valid URLs provided.", level="warning")
            return
        tlinks = self.target_links_spin.value()
        md = self.depth_spin.value()
        keyword_text = self.keyword_input.text().strip()
        keywords_list = None
        if keyword_text:
            try:
                keywords_list = shlex.split(keyword_text)
            except ValueError:
                self._set_status("Error parsing keywords.", level="warning")
                keywords_list = None
        self._set_status(f"Starting crawl & index for {len(root_urls)} URL(s)...")
        self._disable_actions(True) # Disable actions before starting background task
        self.backend.submit_crawl_and_index(root_urls, tlinks, md, keywords_list)

    @Slot(str, bool)
    def _on_crawl_index_complete(self, result_message: str, success: bool):
        """Handles completion of the crawl/index task."""
        # This slot is called directly via signal, already in GUI thread
        self._set_status(result_message, level="success" if success else "error")
        self._disable_actions(False) # Re-enable actions after task completion

    def _get_start_dir(self) -> str:
        start_dir = str(Path.home())
        try:
            crawl_path = self.backend.data_paths.get("crawl_data")
            index_path = self.backend.data_paths.get("index")
            base_path = self.backend.data_paths.get("base")
            if crawl_path and Path(crawl_path).exists():
                return str(Path(crawl_path))
            if index_path and Path(index_path).exists():
                return str(Path(index_path))
            if base_path and Path(base_path).exists():
                return str(Path(base_path))
        except Exception as e:
            logger.warning(f"Error getting start dir: {e}")
        return start_dir

    @Slot()
    def do_index_file(self):
        start_dir = self._get_start_dir()
        fp, _ = QFileDialog.getOpenFileName(self, "Select File", start_dir)
        if fp:
            self._set_status(f"Indexing file '{Path(fp).name}'...")
            self._disable_actions(True)
            self.backend.submit_manual_index(fp)

    @Slot()
    def do_index_dir(self):
        start_dir = self._get_start_dir()
        dp = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if dp:
            self._set_status(f"Indexing directory '{Path(dp).name}'...")
            self._disable_actions(True)
            self.backend.submit_manual_index(dp)

    @Slot(str, bool)
    def _on_manual_index_complete(self, result_message: str, success: bool):
        # This slot is called directly via signal, already in GUI thread
        self._set_status(result_message, level="success" if success else "error")
        self._disable_actions(False)

    @Slot()
    def remove_item(self, source_id: str):
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove source:\n'{source_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_status(f"Removing source '{source_id[:30]}...'")
            self._disable_actions(True)
            self.backend.submit_removal(source_id)
        else:
            self._set_status("Removal cancelled.", level="info")

    @Slot(str, bool)
    def _on_removal_complete(self, result_message: str, success: bool):
        # This slot is called directly via signal, already in GUI thread
        self._set_status(result_message, level="success" if success else "error")
        self._disable_actions(False)

    def _disable_actions(self, disable: bool):
        """Enable/disable UI elements during tasks using QTimer.singleShot."""
        widgets_to_disable = [
            self.search_btn,
            self.crawl_and_index_btn,
            self.index_file_btn,
            self.index_dir_btn,
            self.data_table,
            self.urls_text,
            self.keyword_input,
            self.target_links_spin,
            self.depth_spin,
            self.query_input,
        ]
        for widget in widgets_to_disable:
            # Use QTimer.singleShot to queue the setDisabled call
            QTimer.singleShot(0, lambda w=widget, d=disable: w.setDisabled(d))

    @Slot()
    def update_data_display(self):
        """Updates the indexed sources table."""
        # This method is called via signal or QTimer, assumed to be in GUI thread.
        # Direct UI updates are safe here.
        try:
            items = self.backend.get_crawl_data_items()
            self.data_table.setRowCount(0) # Clear previous items
            self.data_table.setRowCount(len(items))
            for i, item_data in enumerate(items):
                source_id = item_data.get("hash", "N/A")
                display_name = item_data.get("url", "N/A")
                id_item = QTableWidgetItem(source_id)
                id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                id_item.setToolTip(source_id)
                name_item = QTableWidgetItem(display_name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                name_item.setToolTip(display_name)
                self.data_table.setItem(i, 0, id_item)
                self.data_table.setItem(i, 1, name_item)
                remove_btn = QPushButton("Remove")
                remove_btn.setStyleSheet("QPushButton { padding: 2px 5px; }")
                remove_btn.setToolTip(f"Remove source: {source_id}")
                # Disconnect previous connections to avoid duplicates
                try:
                    # Attempt to disconnect all signals from the button's clicked signal
                    remove_btn.clicked.disconnect()
                except (TypeError, RuntimeError):
                     # If no connections exist, disconnect() raises TypeError or RuntimeError
                     pass # Ignore if no connection exists

                # Connect with lambda capturing the correct source_id
                remove_btn.clicked.connect(
                    lambda checked=False, sid=source_id: self.remove_item(sid)
                )
                self.data_table.setCellWidget(i, 2, remove_btn)
            logger.debug(f"Updated indexed sources table with {len(items)} items.")
        except Exception as e:
            logger.error(f"Failed to update data display table: {e}", exc_info=True)
            # Use _set_status which is thread-safe (though likely not needed here)
            self._set_status("Error updating indexed sources list.", level="error")