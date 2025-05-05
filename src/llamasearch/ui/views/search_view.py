# src/llamasearch/ui/views/search_view.py

import logging
import shlex
import time
from pathlib import Path

# --- Import QDesktopServices and QUrl ---
from PySide6.QtCore import QTimer, Qt, Slot

# --- End Import ---
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class SearchAndIndexView(QWidget):
    """GUI View for Search, Crawl/Index, Manual Indexing, and Source Management."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self._actions_disabled = False
        self.init_ui()
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
        self.backend.signals.actions_should_reenable.connect(
            lambda: self._disable_actions(False)
        )
        QTimer.singleShot(100, self.update_data_display)

    def init_ui(self):
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
        # QTextEdit does not support setOpenExternalLinks; links will be clickable if setHtml is used and QTextEdit is read-only.
        search_group_layout.addWidget(self.search_results, 1)
        main_layout.addLayout(search_group_layout)

        # Crawling Section
        crawl_group_layout = QVBoxLayout()
        crawl_group_layout.addWidget(
            QLabel("<b>Crawl & Index Websites (Keyword Priority):</b>")
        )
        self.urls_text = QTextEdit()
        self.urls_text.setPlaceholderText(
            "Enter root URLs (https://example.com), one per line..."
        )
        self.urls_text.setFixedHeight(60)
        crawl_group_layout.addWidget(self.urls_text)
        crawl_params_layout = QHBoxLayout()
        form = QFormLayout()
        form.setHorizontalSpacing(20)
        self.target_links_spin = QSpinBox()
        self.target_links_spin.setRange(1, 1000)
        self.target_links_spin.setValue(5)
        self.target_links_spin.setToolTip("Max unique pages per root URL.")
        form.addRow("Max Pages:", self.target_links_spin)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(0, 10)
        self.depth_spin.setValue(1)
        self.depth_spin.setToolTip("Max crawl depth relative to root URL.")
        form.addRow("Max Depth:", self.depth_spin)
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText(
            "Optional space-separated: guide api tutorial..."
        )
        self.keyword_input.setToolTip(
            "Keywords to prioritize links containing these terms."
        )
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
        # --- Adjust Columns for URL ---
        self.data_table.setColumnCount(5)  # Added column for URL
        self.data_table.setHorizontalHeaderLabels(
            ["Source URL / Path", "Filename", "Chunks", "Modified", "Actions"]
        )
        # --- End Adjust Columns ---
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.data_table.verticalHeader().setVisible(False)
        # --- Adjust Column Resizing ---
        self.data_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )  # URL/Path
        self.data_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )  # Filename
        self.data_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )  # Chunks
        self.data_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )  # Modified
        self.data_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )  # Actions
        # --- End Adjust Resizing ---
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
        logger.debug(f"Setting status (Lvl: {level}): {message}")
        QTimer.singleShot(0, lambda: self.status_label.setText(f"Status: {message}"))
        color = {"error": "red", "warning": "orange", "success": "green"}.get(
            level, "#555"
        )
        style = f"font-style: italic; color: {color};"
        QTimer.singleShot(0, lambda: self.status_label.setStyleSheet(style))

    @Slot()
    def do_search(self):
        if self._actions_disabled:
            logger.warning("Search action ignored: Actions currently disabled.")
            return
        query = self.query_input.text().strip()
        if not query:
            self._set_status("Enter a query.", level="warning")
            self.search_results.setPlainText("Please enter a query.")
            return
        self._set_status(f"Searching '{query[:30]}...'")
        self._disable_actions(True)
        self.search_results.setHtml(
            "<i>Submitting search...</i>"
        )  # Use setHtml for potential links later
        self.backend.submit_search(query)

    @Slot(str, bool)
    def _on_search_complete(self, result_message: str, success: bool):
        logger.debug(f"_on_search_complete called. Success: {success}")
        display_message = result_message
        if success:
            # If backend provides HTML, use setHtml, otherwise setPlainText
            if "<" in result_message and ">" in result_message:  # Basic HTML check
                self.search_results.setHtml(display_message)
            else:
                self.search_results.setPlainText(
                    display_message
                )  # Fallback for plain text
            status_msg = "Search complete."
        else:
            # Display errors as plain text
            if not result_message or result_message.isspace():
                display_message = "Search failed. See logs."
            self.search_results.setPlainText(display_message)
            status_msg = "Search failed."

        self._set_status(status_msg, level="success" if success else "error")
        # Re-enabling handled by signal

    @Slot(str, bool)
    def _on_crawl_index_complete(self, result_message: str, success: bool):
        logger.debug(f"Slot _on_crawl_index_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        if success:
            self.update_data_display()
        # Re-enabling handled by signal

    def _get_start_dir(self) -> str:
        start_dir = str(Path.home())
        try:
            index_path = self.backend.data_paths.get("index")
            base_path = self.backend.data_paths.get("base")
            if index_path and Path(index_path).exists():
                return str(Path(index_path))
            if base_path and Path(base_path).exists():
                return str(Path(base_path))
        except Exception as e:
            logger.warning(f"Error getting start dir: {e}")
        return start_dir

    @Slot()
    def do_index_file(self):
        if self._actions_disabled:
            logger.warning("Index File action ignored.")
            return
        start_dir = self._get_start_dir()
        # --- Adjusted filter to include more common document types ---
        file_filter = (
            "Supported Files (*.md *.markdown *.txt *.html *.htm);;All Files (*)"
        )
        # --- End Filter Adjustment ---
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select File to Index", start_dir, file_filter
        )
        if fp:
            self._set_status(f"Indexing file '{Path(fp).name}'...")
            self._disable_actions(True)
            self.backend.submit_manual_index(fp)

    @Slot()
    def do_index_dir(self):
        if self._actions_disabled:
            logger.warning("Index Directory action ignored.")
            return
        start_dir = self._get_start_dir()
        dp = QFileDialog.getExistingDirectory(
            self, "Select Directory to Index", start_dir
        )
        if dp:
            self._set_status(f"Indexing directory '{Path(dp).name}'...")
            self._disable_actions(True)
            self.backend.submit_manual_index(dp)

    @Slot(str, bool)
    def _on_manual_index_complete(self, result_message: str, success: bool):
        logger.debug(f"Slot _on_manual_index_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        if success:
            self.update_data_display()
        # Re-enabling handled by signal

    @Slot()
    def remove_item(self, source_id: str):  # Use source_id (URL or path)
        if self._actions_disabled:
            logger.warning("Remove item action ignored.")
            return
        try:
            display_name = (
                Path(source_id).name if not source_id.startswith("http") else source_id
            )
        except Exception:
            display_name = source_id[:40] + "..."

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove all chunks for:\n\n'{display_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_status(f"Removing source '{display_name}'...")
            self._disable_actions(True)
            self.backend.submit_removal(source_id)  # Pass the ID (URL or path)
        else:
            self._set_status("Removal cancelled.", level="info")

    @Slot(str, bool)
    def _on_removal_complete(self, result_message: str, success: bool):
        logger.debug(f"Slot _on_removal_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        if success:
            self.update_data_display()
        # Re-enabling handled by signal

    def _disable_actions(self, disable: bool):
        if self._actions_disabled == disable:
            return
        self._actions_disabled = disable
        logger.debug(f"Setting actions disabled state to: {disable}")
        widgets_to_toggle = [
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
        for widget in widgets_to_toggle:
            widget.setDisabled(disable)
        self._set_table_buttons_enabled(not disable)

    def _set_table_buttons_enabled(self, enabled: bool):
        if not hasattr(self, "data_table"):
            return
        # --- Update column index for actions ---
        action_col_index = 4  # Actions are now in column 4
        # --- End Update ---
        for row in range(self.data_table.rowCount()):
            widget = self.data_table.cellWidget(row, action_col_index)
            if isinstance(widget, QPushButton):
                widget.setEnabled(enabled)

    @Slot()
    def update_data_display(self):
        logger.debug("Updating data display table...")
        try:
            items = self.backend.get_indexed_sources()
            self.data_table.setRowCount(0)
            self.data_table.setRowCount(len(items))
            # --- Define action column index ---
            action_col_index = 4
            # --- End Define ---
            for i, item_data in enumerate(items):
                source_path = item_data.get("source_path", "N/A")
                original_url = item_data.get("original_url")  # Get the URL
                filename = item_data.get("filename", "N/A")
                chunk_count = item_data.get("chunk_count", "N/A")
                mtime_val = item_data.get("mtime")
                mtime_str = (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_val))
                    if mtime_val is not None
                    else "N/A"
                )

                # --- Determine primary display and tooltip ---
                display_source = original_url if original_url else source_path
                tooltip = (
                    f"URL: {original_url}\nLocal Path: {source_path}\nModified: {mtime_str}"
                    if original_url
                    else f"Path: {source_path}\nModified: {mtime_str}"
                )
                source_item_id = (
                    original_url if original_url else source_path
                )  # ID for removal action
                # --- End Determine ---

                # --- Create table items ---
                source_item = QTableWidgetItem(display_source)
                source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                source_item.setToolTip(tooltip)

                name_item = QTableWidgetItem(filename)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                name_item.setToolTip(f"Filename: {filename}")

                chunk_item = QTableWidgetItem(str(chunk_count))
                chunk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                chunk_item.setFlags(chunk_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                chunk_item.setToolTip(f"{chunk_count} chunks indexed.")

                mtime_item = QTableWidgetItem(mtime_str)
                mtime_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                mtime_item.setFlags(mtime_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                mtime_item.setToolTip(f"Last Modified: {mtime_str}")
                # --- End Create ---

                # --- Set table items ---
                self.data_table.setItem(i, 0, source_item)
                self.data_table.setItem(i, 1, name_item)
                self.data_table.setItem(i, 2, chunk_item)
                self.data_table.setItem(i, 3, mtime_item)  # Add mtime item
                # --- End Set ---

                remove_btn = QPushButton("Remove")
                remove_btn.setStyleSheet("QPushButton { padding: 2px 5px; }")
                remove_btn.setToolTip(f"Remove all chunks for source: {display_source}")
                remove_btn.setEnabled(not self._actions_disabled)

                try:
                    remove_btn.clicked.disconnect()
                except (TypeError, RuntimeError):
                    pass
                # --- Pass source_item_id (URL or path) to remove_item ---
                remove_btn.clicked.connect(
                    lambda checked=False, sp=source_item_id: self.remove_item(sp)
                )
                # --- End Pass ---
                self.data_table.setCellWidget(i, action_col_index, remove_btn)

            # self.data_table.resizeColumnsToContents() # Maybe adjust manually or keep stretch
            logger.debug(f"Updated indexed sources table with {len(items)} items.")
        except Exception as e:
            logger.error(f"Failed to update data display table: {e}", exc_info=True)
            self._set_status("Error updating indexed sources list.", level="error")

    @Slot()
    def do_crawl_and_index(self):
        if self.backend is None:
            logger.error("Backend not initialized.")
            return
        if self._actions_disabled:
            logger.warning("Crawl and Index action ignored.")
            return

        urls_text_content = self.urls_text.toPlainText().strip()
        target_urls = [
            url.strip() for url in urls_text_content.splitlines() if url.strip()
        ]
        crawl_depth = self.depth_spin.value()
        target_links = self.target_links_spin.value()
        keywords_str = self.keyword_input.text().strip()

        if not target_urls:
            QMessageBox.warning(
                self, "Input Error", "Please enter at least one target URL."
            )
            return

        keywords = []
        if keywords_str:
            try:
                keywords = [
                    kw.strip() for kw in shlex.split(keywords_str) if kw.strip()
                ]
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", f"Error parsing keywords: {e}")
                return

        logger.info(
            f"Starting crawl and index: URLs={target_urls}, Depth={crawl_depth}, Target Pages={target_links}, Keywords={keywords}"
        )
        self._set_status(
            f"Starting crawl: {len(target_urls)} URLs, Depth={crawl_depth}, Keywords={keywords}"
        )
        self._disable_actions(True)
        # Use UI values for target_links and max_depth
        self.backend.submit_crawl_and_index(
            root_urls=target_urls,
            target_links=target_links,
            max_depth=crawl_depth,
            keywords=keywords,
        )
