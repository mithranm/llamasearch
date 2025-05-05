# src/llamasearch/ui/views/search_view.py

import logging
import shlex
import time # Import time for modification time formatting
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtCore import (
    Qt, Slot)
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTextEdit,
                               QVBoxLayout, QWidget)

# --- Use the specific logger for this module ---
logger = logging.getLogger(__name__)


class SearchAndIndexView(QWidget):
    """GUI View for Search, Crawl/Index, Manual Indexing, and Source Management."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self._actions_disabled = False # Internal flag for UI state
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
        # --- Connect the new signal to re-enable actions ---
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
        search_group_layout.addWidget(self.search_results, 1)
        main_layout.addLayout(search_group_layout)
        # Crawling Section
        crawl_group_layout = QVBoxLayout()
        crawl_group_layout.addWidget(
            QLabel("<b>Crawl & Index Websites (Keyword Priority):</b>")
        )
        self.urls_text = QTextEdit()
        self.urls_text.setPlaceholderText("Enter root URLs (https://example.com), one per line...")
        self.urls_text.setFixedHeight(60)
        crawl_group_layout.addWidget(self.urls_text)
        crawl_params_layout = QHBoxLayout()
        form = QFormLayout()
        form.setHorizontalSpacing(20)
        self.target_links_spin = QSpinBox()
        self.target_links_spin.setRange(1, 1000)
        self.target_links_spin.setValue(15)
        self.target_links_spin.setToolTip("Max unique pages per root URL.")
        form.addRow("Max Pages:", self.target_links_spin)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(0, 10)
        self.depth_spin.setValue(1)
        self.depth_spin.setToolTip("Max crawl depth relative to root URL.")
        form.addRow("Max Depth:", self.depth_spin)
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Optional space-separated: guide api tutorial...")
        self.keyword_input.setToolTip("Keywords to prioritize links containing these terms.")
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
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(
            ["Source Path", "Filename", "Chunks", "Actions"]
        )
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch # Path
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents # Filename
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents # Chunks
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents # Actions
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
        logger.debug(f"Setting status (Lvl: {level}): {message}")
        QTimer.singleShot(0, lambda: self.status_label.setText(f"Status: {message}"))
        color = {"error": "red", "warning": "orange", "success": "green"}.get(
            level, "#555"
        )
        style = f"font-style: italic; color: {color};"
        QTimer.singleShot(0, lambda: self.status_label.setStyleSheet(style))


    @Slot()
    def do_search(self):
        """Initiates a search operation."""
        if self._actions_disabled:
             logger.warning("Search action ignored: Actions currently disabled.")
             return
        query = self.query_input.text().strip()
        if not query:
            self._set_status("Enter a query.", level="warning")
            self.search_results.setPlainText("Please enter a query.")
            return
        self._set_status(f"Searching '{query[:30]}...'")
        self._disable_actions(True) # Disable actions
        self.search_results.setPlainText("Submitting search...")
        self.backend.submit_search(query) # Backend now handles re-enabling via signal

    @Slot(str, bool)
    def _on_search_complete(self, result_message: str, success: bool):
        logger.debug(f"_on_search_complete called. Success: {success}, Message: '{result_message[:100]}...'" ) # Log entry
        """Handles completion signal for the search task."""
        logger.debug(f"Slot _on_search_complete triggered. Success: {success}")

        # --- Add Check for Empty/Whitespace Response ---
        display_message = result_message
        if success and (not result_message or result_message.isspace()):
            display_message = "(No specific answer generated by the LLM)"
            logger.warning("Search succeeded but LLM response was empty/whitespace.")
        elif not success and (not result_message or result_message.isspace()):
            # If it failed and the message is empty, provide a generic failure message
            display_message = "Search failed. See logs for details."
        # --- End Check ---

        self.search_results.setPlainText(display_message) # Use display_message
        self._set_status(
            "Search complete." if success else "Search failed.",
            level="success" if success else "error",
        )
        # UI re-enabling is now handled by the actions_should_reenable signal

    @Slot(str, bool)
    def _on_crawl_index_complete(self, result_message: str, success: bool):
        """Handles completion signal for the crawl/index task."""
        logger.debug(f"Slot _on_crawl_index_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        # UI re-enabling is now handled by the actions_should_reenable signal
        if success:
            self.update_data_display()

    def _get_start_dir(self) -> str:
        start_dir = str(Path.home())
        try:
            index_path = self.backend.data_paths.get("index")
            if index_path and Path(index_path).exists():
                return str(Path(index_path))
            base_path = self.backend.data_paths.get("base")
            if base_path and Path(base_path).exists():
                return str(Path(base_path))
        except Exception as e:
            logger.warning(f"Error getting start dir: {e}")
        return start_dir

    @Slot()
    def do_index_file(self):
        if self._actions_disabled:
             logger.warning("Index File action ignored: Actions currently disabled.")
             return
        start_dir = self._get_start_dir()
        file_filter = "Supported Files (*.md *.markdown *.txt *.html *.htm *.pdf *.docx *.doc *.odt *.rtf *.epub);;All Files (*)"
        fp, _ = QFileDialog.getOpenFileName(self, "Select File to Index", start_dir, file_filter)
        if fp:
            self._set_status(f"Indexing file '{Path(fp).name}'...")
            self._disable_actions(True) # Disable actions
            self.backend.submit_manual_index(fp) # Backend handles re-enabling

    @Slot()
    def do_index_dir(self):
        if self._actions_disabled:
             logger.warning("Index Directory action ignored: Actions currently disabled.")
             return
        start_dir = self._get_start_dir()
        dp = QFileDialog.getExistingDirectory(self, "Select Directory to Index", start_dir)
        if dp:
            self._set_status(f"Indexing directory '{Path(dp).name}'...")
            self._disable_actions(True) # Disable actions
            self.backend.submit_manual_index(dp) # Backend handles re-enabling

    @Slot(str, bool)
    def _on_manual_index_complete(self, result_message: str, success: bool):
        """Handles completion signal for the manual index task."""
        logger.debug(f"Slot _on_manual_index_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        # UI re-enabling is now handled by the actions_should_reenable signal
        if success:
            self.update_data_display()

    @Slot()
    def remove_item(self, source_path_to_remove: str):
        """ Triggered by clicking the 'Remove' button in the table. """
        if self._actions_disabled:
             logger.warning("Remove item action ignored: Actions currently disabled.")
             return
        try:
             display_name = Path(source_path_to_remove).name
        except Exception:
             display_name = source_path_to_remove[:40] + "..."

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove all indexed chunks for:\n\n'{display_name}'\n({source_path_to_remove})?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_status(f"Removing source '{display_name}'...")
            self._disable_actions(True) # Disable actions
            self.backend.submit_removal(source_path_to_remove) # Backend handles re-enabling
        else:
            self._set_status("Removal cancelled.", level="info")

    @Slot(str, bool)
    def _on_removal_complete(self, result_message: str, success: bool):
        """Handles completion signal for the removal task."""
        logger.debug(f"Slot _on_removal_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        # UI re-enabling is now handled by the actions_should_reenable signal
        if success:
            self.update_data_display()

    @Slot() # This slot is now connected to backend.signals.actions_should_reenable
    def _reenable_ui_actions(self):
         """Slot specifically for re-enabling UI elements."""
         self._disable_actions(False)

    # Now takes the disable state as an argument
    def _disable_actions(self, disable: bool):
        """Enable/disable UI elements. Should be called via QTimer or signal."""
        if self._actions_disabled == disable:
            # Avoid redundant calls if state is already correct
            # logger.debug(f"Actions already in desired state (disabled={disable}). Skipping.")
            return

        self._actions_disabled = disable
        logger.debug(f"Setting actions disabled state to: {disable}")

        widgets_to_toggle = [
            self.search_btn,
            self.crawl_and_index_btn,
            self.index_file_btn,
            self.index_dir_btn,
            self.data_table, # Toggle table interaction
            self.urls_text,
            self.keyword_input,
            self.target_links_spin,
            self.depth_spin,
            self.query_input,
        ]
        for widget in widgets_to_toggle:
            # Direct call is okay IF this method is guaranteed to run in the GUI thread
            # which it is now, since it's called by do_search etc. or the reenable signal.
            widget.setDisabled(disable)

        # Toggle remove buttons within the table
        self._set_table_buttons_enabled(not disable)


    def _set_table_buttons_enabled(self, enabled: bool):
        """Iterates through table rows and enables/disables the 'Remove' button."""
        # This method might be called from _disable_actions, ensure it runs in GUI thread.
        # Since _disable_actions is now called directly or via signal, this should be safe.
        # logger.debug(f"Setting table remove buttons enabled state to: {enabled}")
        if not hasattr(self, 'data_table'): # Check if table exists (init safety)
             return
        for row in range(self.data_table.rowCount()):
            widget = self.data_table.cellWidget(row, 3) # Assuming button is in column 3
            if isinstance(widget, QPushButton):
                widget.setEnabled(enabled)

    @Slot()
    def update_data_display(self):
        """Updates the indexed sources table."""
        # This method is called via signal or QTimer, assumed to be in GUI thread.
        logger.debug("Updating data display table...")
        try:
            items = self.backend.get_indexed_sources()
            self.data_table.setRowCount(0) # Clear previous items
            self.data_table.setRowCount(len(items))
            for i, item_data in enumerate(items):
                source_path = item_data.get("source_path", "N/A")
                display_name = item_data.get("filename", "N/A")
                chunk_count = item_data.get("chunk_count", "N/A")
                mtime_val = item_data.get("mtime")
                mtime_str = (time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_val))
                             if mtime_val is not None else "N/A")

                path_item = QTableWidgetItem(source_path)
                path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                path_item.setToolTip(f"Full Path: {source_path}\nLast Modified: {mtime_str}")

                name_item = QTableWidgetItem(display_name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                name_item.setToolTip(f"Filename: {display_name}\nLast Modified: {mtime_str}")

                chunk_item = QTableWidgetItem(str(chunk_count))
                chunk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                chunk_item.setFlags(chunk_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                chunk_item.setToolTip(f"{chunk_count} chunks indexed for this source.")

                self.data_table.setItem(i, 0, path_item)
                self.data_table.setItem(i, 1, name_item)
                self.data_table.setItem(i, 2, chunk_item)

                remove_btn = QPushButton("Remove")
                remove_btn.setStyleSheet("QPushButton { padding: 2px 5px; }")
                remove_btn.setToolTip(f"Remove all chunks for source: {source_path}")
                remove_btn.setEnabled(not self._actions_disabled) # Set initial state

                # Disconnect previous connections to avoid duplicates
                try:
                    remove_btn.clicked.disconnect()
                except (TypeError, RuntimeError):
                     pass

                remove_btn.clicked.connect(
                    lambda checked=False, sp=source_path: self.remove_item(sp)
                )
                self.data_table.setCellWidget(i, 3, remove_btn)

            self.data_table.resizeColumnsToContents()
            logger.debug(f"Updated indexed sources table with {len(items)} items.")
        except Exception as e:
            logger.error(f"Failed to update data display table: {e}", exc_info=True)
            self._set_status("Error updating indexed sources list.", level="error")

    @Slot()
    def do_crawl_and_index(self):
        """Handles the 'Crawl and Index' button click."""
        if self.backend is None:
            logger.error("Backend not initialized.")
            return

        if self._actions_disabled:
            logger.warning("Crawl and Index action ignored: Actions currently disabled.")
            return

        # Get crawl parameters from input fields
        target_url = self.urls_text.toPlainText().strip()
        crawl_depth_str = self.depth_spin.text().strip()
        additional_links_str = self.keyword_input.text().strip()

        if not target_url:
            QMessageBox.warning(self, "Input Error", "Please enter a target URL to crawl.")
            return

        try:
            crawl_depth = int(crawl_depth_str) if crawl_depth_str else 1 # Default depth
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid crawl depth. Please enter a number.")
            return

        # Parse additional links (split by newline or space, filter empty)
        additional_links = []
        if additional_links_str:
            # Use shlex to handle potential spaces within quoted URLs (though unlikely needed here)
            try:
                # Split lines first, then split each line by space/shlex rules
                raw_links = [link for line in additional_links_str.splitlines() for link in shlex.split(line) if link]
                additional_links = [link.strip() for link in raw_links if link.strip()] # Clean up whitespace
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", f"Error parsing additional links: {e}")
                return

        logger.info(f"Starting crawl and index: URL={target_url}, Depth={crawl_depth}, Additional={additional_links}")
        self._set_status(f"Starting crawl and index: URL={target_url}, Depth={crawl_depth}, Additional={additional_links}")
        self._disable_actions(True) # Disable actions
        self.backend.submit_crawl_and_index(target_url, crawl_depth, additional_links) # Backend handles re-enabling