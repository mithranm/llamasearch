# src/llamasearch/ui/views/search_view.py

import logging
import shlex
import time
from pathlib import Path

# --- Remove unused QUrl and QDesktopServices ---
from PySide6.QtCore import QTimer, Qt, Slot

# from PySide6.QtGui import QDesktopServices # Removed
# --- End Remove ---
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
        QTimer.singleShot(150, self.update_data_display)

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
        self.search_results.setReadOnly(True)  # Ensure it's read-only for links to work
        self.search_results.setPlaceholderText("Search results...")
        # --- Removed incorrect setOpenExternalLinks call ---
        # self.search_results.setOpenExternalLinks(True) # REMOVED - AttributeError
        # --- End Removal ---
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
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels(
            ["Source URL / Path", "Filename", "Chunks", "Modified", "Actions"]
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
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
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
        self.search_results.setHtml("<i>Submitting search...</i>")
        self.backend.submit_search(query)

    @Slot(str, bool)
    def _on_search_complete(self, result_message: str, success: bool):
        logger.debug(f"_on_search_complete called. Success: {success}")
        display_message = result_message
        if success:
            # Assume backend provides HTML, set it. Links should work if read-only.
            self.search_results.setHtml(display_message)
            status_msg = "Search complete."
        else:
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
            crawl_path = self.backend.data_paths.get("crawl_data")
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
        if self._actions_disabled:
            logger.warning("Index File action ignored: Actions disabled.")
            return
        start_dir = self._get_start_dir()
        file_filter = (
            "Supported Files (*.md *.markdown *.txt *.html *.htm);;All Files (*)"
        )
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
            logger.warning("Index Directory action ignored: Actions disabled.")
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
    def remove_item(self, source_identifier: str):
        if self._actions_disabled:
            logger.warning("Remove item action ignored: Actions disabled.")
            return
        if not isinstance(source_identifier, str) or not source_identifier:
            logger.error(
                f"Invalid source identifier received for removal: {source_identifier}"
            )
            self._set_status(
                "Error: Invalid source identifier for removal.", level="error"
            )
            return

        try:
            display_name = source_identifier
            if not source_identifier.startswith("http"):
                try:
                    display_name = Path(source_identifier).name
                except Exception:
                    pass
            display_name = (
                (display_name[:70] + "...") if len(display_name) > 70 else display_name
            )
        except Exception:
            display_name = str(source_identifier)[:70] + "..."

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove all indexed content for:\n\n'{display_name}'?\n\n(This cannot be undone)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_status(f"Removing source '{display_name}'...")
            self._disable_actions(True)
            self.backend.submit_removal(source_identifier)
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
        """Enables or disables buttons within the data table."""
        if not hasattr(self, "data_table"):
            return
        action_col_index = 4
        for row in range(self.data_table.rowCount()):
            cell_widget = self.data_table.cellWidget(row, action_col_index)
            if isinstance(cell_widget, QWidget):
                button = cell_widget.findChild(QPushButton)
                if button:
                    button.setEnabled(enabled)
                else:
                    cell_widget.setEnabled(enabled)

    @Slot()
    def update_data_display(self):
        """Updates the table with indexed sources, prioritizing URL display."""
        logger.debug("Updating data display table...")
        self._disable_actions(True)
        try:
            items = self.backend.get_indexed_sources()
            self.data_table.setRowCount(0)
            self.data_table.setRowCount(len(items))
            action_col_index = 4

            for i, item_data in enumerate(items):
                source_path = item_data.get("source_path", "N/A")
                original_url = item_data.get("original_url")
                filename = item_data.get("filename", "N/A")
                chunk_count = item_data.get("chunk_count", "N/A")
                mtime_val = item_data.get("mtime")
                mtime_str = (
                    time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_val))
                    if mtime_val is not None
                    else "N/A"
                )

                display_source = "N/A"
                source_identifier_for_action = None
                base_tooltip = f"Modified: {mtime_str}"
                tooltip_text = base_tooltip

                if isinstance(original_url, str) and original_url.strip():
                    display_source = original_url.strip()
                    source_identifier_for_action = display_source
                    tooltip_text = f"URL: {display_source}\nLocal Path: {source_path}\n{base_tooltip}"
                elif isinstance(source_path, str) and source_path != "N/A":
                    display_source = source_path
                    source_identifier_for_action = source_path
                    tooltip_text = f"Path: {display_source}\n{base_tooltip}"
                else:
                    display_source = item_data.get("identifier", "(Unknown Source)")
                    source_identifier_for_action = display_source
                    tooltip_text = f"Identifier: {display_source}\n{base_tooltip}"

                source_item = QTableWidgetItem(display_source)
                source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                source_item.setToolTip(tooltip_text)

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

                self.data_table.setItem(i, 0, source_item)
                self.data_table.setItem(i, 1, name_item)
                self.data_table.setItem(i, 2, chunk_item)
                self.data_table.setItem(i, 3, mtime_item)

                remove_btn = QPushButton("Remove")
                remove_btn.setStyleSheet("QPushButton { padding: 2px 5px; }")
                remove_btn.setToolTip(f"Remove all chunks for source: {display_source}")
                remove_btn.setEnabled(not self._actions_disabled)

                if source_identifier_for_action:
                    try:
                        remove_btn.clicked.disconnect()
                    except (TypeError, RuntimeError):
                        pass
                    remove_btn.clicked.connect(
                        lambda checked=False,
                        sid=source_identifier_for_action: self.remove_item(sid)
                    )
                else:
                    remove_btn.setDisabled(True)
                    remove_btn.setToolTip("Cannot remove: Missing source identifier.")

                container_widget = QWidget()
                btn_layout = QHBoxLayout(container_widget)
                btn_layout.addWidget(remove_btn)
                btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                btn_layout.setContentsMargins(0, 0, 0, 0)
                container_widget.setLayout(btn_layout)
                self.data_table.setCellWidget(i, action_col_index, container_widget)

            logger.debug(f"Updated indexed sources table with {len(items)} items.")
        except Exception as e:
            logger.error(f"Failed to update data display table: {e}", exc_info=True)
            self._set_status("Error updating indexed sources list.", level="error")
        finally:
            self._disable_actions(False)

    @Slot()
    def do_crawl_and_index(self):
        if self.backend is None:
            logger.error("Backend not initialized.")
            return
        if self._actions_disabled:
            logger.warning("Crawl and Index action ignored: Actions disabled.")
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

        invalid_urls = [
            url
            for url in target_urls
            if not (url.startswith("http://") or url.startswith("https://"))
        ]
        if invalid_urls:
            QMessageBox.warning(
                self,
                "Input Error",
                "Invalid URL format (must start with http:// or https://):\n- "
                + "\n- ".join(invalid_urls),
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
            f"Starting crawl: {len(target_urls)} URLs, Depth={crawl_depth}..."
        )
        self._disable_actions(True)
        self.backend.submit_crawl_and_index(
            root_urls=target_urls,
            target_links=target_links,
            max_depth=crawl_depth,
            keywords=keywords,
        )
