# src/llamasearch/ui/views/search_view.py

import logging
import shlex
import time
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Slot
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
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

# Use module-level logger
logger = logging.getLogger(__name__)


class SearchAndIndexView(QWidget):
    """GUI View for Search, Crawl/Index, Manual Indexing, and Source Management."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self._actions_disabled = False  # Track UI disabled state
        self.init_ui()
        self.connect_signals()
        # Use QTimer to delay initial data load slightly after UI setup
        QTimer.singleShot(150, self.update_data_display)

    def connect_signals(self):
        """Connect signals from the backend to UI slots."""
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

        # Connect button clicks to actions
        self.search_btn.clicked.connect(self.do_search)
        self.query_input.returnPressed.connect(self.do_search)
        self.crawl_and_index_btn.clicked.connect(self.do_crawl_and_index)
        self.index_file_btn.clicked.connect(self.do_index_file)
        self.index_dir_btn.clicked.connect(self.do_index_dir)

    def init_ui(self):
        """Initialize the user interface components."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # --- Search Section ---
        search_group_layout = QVBoxLayout()

        query_layout = QHBoxLayout()
        query_label = QLabel("<b>Question:</b>")
        query_layout.addWidget(query_label)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question...")
        query_layout.addWidget(self.query_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        query_layout.addWidget(self.search_btn)

        search_group_layout.addLayout(query_layout)

        self.search_results = QTextBrowser()
        self.search_results.setOpenExternalLinks(True)
        self.search_results.setReadOnly(True)
        self.search_results.setPlaceholderText("Search results...")
        # Give results vertical space priority
        search_group_layout.addWidget(self.search_results, 1)

        main_layout.addLayout(search_group_layout)

        # --- Crawling Section ---
        crawl_group_layout = QVBoxLayout()
        crawl_label = QLabel("<b>Crawl & Index Websites (Keyword Priority):</b>")
        crawl_group_layout.addWidget(crawl_label)

        self.urls_text = QTextEdit()
        self.urls_text.setPlaceholderText(
            "Enter root URLs (https://example.com), one per line..."
        )
        self.urls_text.setFixedHeight(60) # Keep URLs box relatively small
        crawl_group_layout.addWidget(self.urls_text)

        crawl_params_layout = QHBoxLayout()
        crawl_form = QFormLayout()
        crawl_form.setHorizontalSpacing(20)

        self.target_links_spin = QSpinBox()
        self.target_links_spin.setRange(1, 1000)
        self.target_links_spin.setValue(5)
        self.target_links_spin.setToolTip("Max unique pages per root URL.")
        crawl_form.addRow("Max Pages:", self.target_links_spin)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(0, 10)
        self.depth_spin.setValue(1)
        self.depth_spin.setToolTip("Max crawl depth relative to root URL.")
        crawl_form.addRow("Max Depth:", self.depth_spin)

        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText(
            "Optional space-separated: guide api tutorial..."
        )
        self.keyword_input.setToolTip(
            "Keywords to prioritize links containing these terms."
        )
        crawl_form.addRow("Relevance Keywords:", self.keyword_input)

        crawl_params_layout.addLayout(crawl_form)
        crawl_params_layout.addStretch()

        self.crawl_and_index_btn = QPushButton("Start Crawl & Index")
        self.crawl_and_index_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        # Align button to bottom within its horizontal layout space
        crawl_params_layout.addWidget(
            self.crawl_and_index_btn, 0, Qt.AlignmentFlag.AlignBottom
        )

        crawl_group_layout.addLayout(crawl_params_layout)
        main_layout.addLayout(crawl_group_layout)

        # --- Manual Indexing Section ---
        local_index_layout = QHBoxLayout()
        local_index_label = QLabel("<b>Manually Index Local Content:</b>")
        local_index_layout.addWidget(local_index_label)

        self.index_file_btn = QPushButton("Index File...")
        self.index_file_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        local_index_layout.addWidget(self.index_file_btn)

        self.index_dir_btn = QPushButton("Index Directory...")
        self.index_dir_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        local_index_layout.addWidget(self.index_dir_btn)

        local_index_layout.addStretch()
        main_layout.addLayout(local_index_layout)

        # --- Indexed Sources Table ---
        table_label = QLabel("<b>Indexed Sources:</b>")
        main_layout.addWidget(table_label)

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

        # Adjust column resize modes for better layout
        header = self.data_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        self.data_table.setAlternatingRowColors(True)
        # Give table vertical space priority
        main_layout.addWidget(self.data_table, 2)

        # --- Status Label ---
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-style: italic; color: #555;")
        main_layout.addWidget(self.status_label)

    # --- Action Disabling ---
    def _disable_actions(self, disable: bool):
        """Enable/disable interactive UI elements during background tasks."""
        if self._actions_disabled == disable:
            # Avoid redundant calls if state hasn't changed
            return
        self._actions_disabled = disable
        logger.debug(f"Setting actions disabled state to: {disable}")

        widgets_to_toggle = [
            self.search_btn,
            self.query_input,
            self.crawl_and_index_btn,
            self.urls_text,
            self.target_links_spin,
            self.depth_spin,
            self.keyword_input,
            self.index_file_btn,
            self.index_dir_btn,
            self.data_table, # Disable the whole table during actions
        ]
        for widget in widgets_to_toggle:
            widget.setDisabled(disable)

        # Specifically handle buttons *inside* the table if table itself is enabled
        if not disable:
            self._set_table_buttons_enabled(True)
        # If disabling actions, table buttons are implicitly disabled by table being disabled

    def _set_table_buttons_enabled(self, enabled: bool):
        """Enables or disables 'Remove' buttons within the data table."""
        if not hasattr(self, "data_table"):
            return
        action_col_index = 4 # Assuming actions are in the 5th column (index 4)
        for row in range(self.data_table.rowCount()):
            cell_widget = self.data_table.cellWidget(row, action_col_index)
            if isinstance(cell_widget, QWidget):
                # Find the button within the cell widget's layout more robustly
                buttons = cell_widget.findChildren(QPushButton)
                for button in buttons:
                    if button.text() == "Remove": # Check button text if needed
                        button.setEnabled(enabled)

    # --- Slots for Backend Signals ---
    @Slot(str, str)
    def _set_status(self, message: str, level: str = "info"):
        """Updates the status label with message and color."""
        logger.debug(f"Setting status (Lvl: {level}): {message}")
        # Ensure updates happen in GUI thread
        QTimer.singleShot(0, lambda: self.status_label.setText(f"Status: {message}"))
        color_map = {
            "error": "red",
            "warning": "orange",
            "success": "green",
            "info": "#555", # Default info color
        }
        color = color_map.get(level, "#555") # Default to info color
        style = f"font-style: italic; color: {color};"
        QTimer.singleShot(0, lambda: self.status_label.setStyleSheet(style))

    @Slot(str, bool)
    def _on_search_complete(self, result_message: str, success: bool):
        """Handles search completion: displays results and updates status."""
        logger.debug(f"Slot _on_search_complete triggered. Success: {success}")
        if success:
            self.search_results.setHtml(result_message)
            self._set_status("Search complete.", level="success")
        else:
            # Display error in results area too for visibility
            self.search_results.setHtml(f"<font color='red'>{result_message}</font>")
            self._set_status(f"Search Error: {result_message}", level="error")

        # --- Re-enable actions regardless of success/failure ---
        self._disable_actions(False)

    @Slot(str, bool)
    def _on_crawl_index_complete(self, result_message: str, success: bool):
        """Handles crawl/index completion: updates status and refreshes data."""
        logger.debug(f"Slot _on_crawl_index_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        if success:
            # Refresh table on success to show newly indexed sources
            self.update_data_display()
        # Re-enabling is handled by the `actions_should_reenable` signal

    @Slot(str, bool)
    def _on_manual_index_complete(self, result_message: str, success: bool):
        """Handles manual index completion: updates status and refreshes data."""
        logger.debug(f"Slot _on_manual_index_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        # Refresh only if chunks were actually added (indicated in message)
        if success and "added" in result_message.lower() and "0 chunks" not in result_message.lower():
            self.update_data_display()
        # Re-enabling is handled by the `actions_should_reenable` signal

    @Slot(str, bool)
    def _on_removal_complete(self, result_message: str, success: bool):
        """Handles source removal completion: updates status and refreshes data."""
        logger.debug(f"Slot _on_removal_complete triggered. Success: {success}")
        self._set_status(result_message, level="success" if success else "error")
        # Refresh only if removal actually occurred (indicated in message)
        if success and "removed" in result_message.lower():
            self.update_data_display()
        # Re-enabling is handled by the `actions_should_reenable` signal

    # --- UI Action Methods ---
    @Slot()
    def do_search(self):
        """Initiates a search query."""
        if self._actions_disabled:
            return
        query = self.query_input.text().strip()
        if not query:
            self._set_status("Please enter a query.", level="warning")
            self.search_results.setPlainText("Please enter a query.")
            return

        self._set_status(f"Searching '{query[:30]}...'")
        self._disable_actions(True) # Disable UI immediately
        self.search_results.setHtml("<i>Submitting search...</i>") # Provide feedback
        self.backend.submit_search(query)

    @Slot()
    def do_crawl_and_index(self):
        """Initiates crawling and indexing websites."""
        if self._actions_disabled:
            return

        urls_text_content = self.urls_text.toPlainText().strip()
        target_urls = [
            url.strip() for url in urls_text_content.splitlines() if url.strip()
        ]
        crawl_depth = self.depth_spin.value()
        target_links = self.target_links_spin.value()
        keywords_str = self.keyword_input.text().strip()

        if not target_urls:
            QMessageBox.warning(self, "Input Error", "Please enter at least one target URL.")
            return

        invalid_urls = [
            url for url in target_urls if not (url.startswith("http://") or url.startswith("https://"))
        ]
        if invalid_urls:
            QMessageBox.warning(
                self,
                "Input Error",
                "Invalid URL format (must start with http:// or https://):\n- " + "\n- ".join(invalid_urls),
            )
            return

        keywords = []
        if keywords_str:
            try:
                # Use shlex to handle potential quotes in keywords if needed
                keywords = [kw.strip() for kw in shlex.split(keywords_str) if kw.strip()]
            except ValueError as e:
                QMessageBox.warning(self, "Input Error", f"Error parsing keywords: {e}")
                return

        logger.info(
            f"Starting crawl and index: URLs={target_urls}, Depth={crawl_depth}, Target Pages={target_links}, Keywords={keywords}"
        )
        self._set_status(f"Starting crawl: {len(target_urls)} URLs, Depth={crawl_depth}...")
        self._disable_actions(True) # Disable UI
        self.backend.submit_crawl_and_index(
            root_urls=target_urls,
            target_links=target_links,
            max_depth=crawl_depth,
            keywords=keywords,
        )

    def _get_start_dir(self) -> str:
        """Determines a sensible default directory for file dialogs."""
        start_dir = str(Path.home())
        try:
            paths = self.backend.data_paths
            crawl_path = paths.get("crawl_data")
            index_path = paths.get("index")
            base_path = paths.get("base")
            # Prioritize existing directories
            if crawl_path and Path(crawl_path).is_dir():
                return str(Path(crawl_path))
            if index_path and Path(index_path).is_dir():
                return str(Path(index_path))
            if base_path and Path(base_path).is_dir():
                return str(Path(base_path))
        except Exception as e:
            logger.warning(f"Error getting start dir: {e}")
        return start_dir

    @Slot()
    def do_index_file(self):
        """Opens file dialog to select a file for manual indexing."""
        if self._actions_disabled:
            return
        start_dir = self._get_start_dir()
        file_filter = "Supported Files (*.md *.markdown *.txt *.html *.htm);;All Files (*)"
        # Use self as parent for the dialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File to Index", start_dir, file_filter)
        if file_path: # Check if a file was selected
            self._set_status(f"Indexing file '{Path(file_path).name}'...")
            self._disable_actions(True) # Disable UI
            self.backend.submit_manual_index(file_path)

    @Slot()
    def do_index_dir(self):
        """Opens directory dialog to select a directory for manual indexing."""
        if self._actions_disabled:
            return
        start_dir = self._get_start_dir()
        # Use self as parent for the dialog
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory to Index", start_dir)
        if dir_path: # Check if a directory was selected
            self._set_status(f"Indexing directory '{Path(dir_path).name}'...")
            self._disable_actions(True) # Disable UI
            self.backend.submit_manual_index(dir_path)

    @Slot()
    def remove_item(self, source_identifier: str):
        """Initiates removal of an indexed source after confirmation."""
        if self._actions_disabled:
            return

        if not isinstance(source_identifier, str) or not source_identifier:
            logger.error(f"Invalid source identifier received for removal: {source_identifier}")
            self._set_status("Error: Invalid source identifier for removal.", level="error")
            return

        # Create user-friendly display name for confirmation dialog
        display_name = source_identifier
        is_url = source_identifier.startswith("http://") or source_identifier.startswith("https://")
        try:
            # Try to get filename for local paths
            if not is_url:
                display_name = Path(source_identifier).name
            # Truncate long names/URLs for display
            display_name = (display_name[:70] + "...") if len(display_name) > 70 else display_name
        except Exception:
            # Ignore errors in generating display name, use original identifier
            pass

        reply = QMessageBox.question(
            self, # Parent widget
            "Confirm Removal",
            f"Remove all indexed content for:\n\n'{display_name}'?\n\n(This cannot be undone)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, # Buttons
            QMessageBox.StandardButton.No, # Default button
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._set_status(f"Removing source '{display_name}'...")
            self._disable_actions(True) # Disable UI
            # Pass the actual identifier (URL or full path) to the backend
            self.backend.submit_removal(source_identifier)
        else:
            self._set_status("Removal cancelled.", level="info")

    # --- Data Display Update ---
    @Slot()
    def update_data_display(self):
        """Updates the table with indexed sources, prioritizing URL display."""
        logger.debug("Updating data display table...")
        self.data_table.setDisabled(True) # Disable during update
        try:
            # Fetch data synchronously from backend
            items = self.backend.get_indexed_sources()
            self.data_table.setRowCount(0) # Clear existing rows
            self.data_table.setRowCount(len(items))
            action_col_index = 4 # Column index for the "Actions"

            for row_index, item_data in enumerate(items):
                # --- Extract data and determine primary display/action ID ---
                original_url = item_data.get("original_url")
                source_path = item_data.get("source_path", "N/A")
                filename = item_data.get("filename", "N/A")
                chunk_count = item_data.get("chunk_count", "N/A")
                mtime_val = item_data.get("mtime")
                mtime_str = "N/A"
                if mtime_val is not None:
                    try:
                        # Format time, handle potential invalid timestamp
                        mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime_val))
                    except (OSError, ValueError):
                         mtime_str = "(Invalid Date)"

                # Use the flag from backend aggregation logic
                is_url_source = item_data.get("is_url_source", False)

                display_source_text = "N/A"
                source_identifier_for_action = None # The ID passed to remove_item
                tooltip_base = f"Filename: {filename}\nChunks: {chunk_count}\nModified: {mtime_str}"

                if is_url_source and isinstance(original_url, str) and original_url:
                    display_source_text = original_url
                    source_identifier_for_action = original_url
                    tooltip_text = f"URL: {original_url}\nLocal Path: {source_path}\n{tooltip_base}"
                elif isinstance(source_path, str) and source_path != "N/A":
                    display_source_text = source_path
                    source_identifier_for_action = source_path
                    tooltip_text = f"Path: {source_path}\n{tooltip_base}"
                else: # Fallback if both missing or invalid
                    display_source_text = item_data.get("identifier", "(Unknown Source)")
                    source_identifier_for_action = display_source_text # Use best guess for action ID
                    tooltip_text = f"Identifier: {display_source_text}\n{tooltip_base}"
                # --- End identifier logic ---

                # --- Create Table Items ---
                # Source URL / Path Item
                source_item = QTableWidgetItem(display_source_text)
                source_item.setFlags(source_item.flags() & ~Qt.ItemFlag.ItemIsEditable) # Read-only
                source_item.setToolTip(tooltip_text) # Set detailed tooltip

                # Filename Item
                name_item = QTableWidgetItem(filename)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                name_item.setToolTip(f"Filename: {filename}") # Simple tooltip

                # Chunk Count Item
                chunk_item = QTableWidgetItem(str(chunk_count))
                chunk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                chunk_item.setFlags(chunk_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                chunk_item.setToolTip(f"{chunk_count} chunks indexed.")

                # Modified Time Item
                mtime_item = QTableWidgetItem(mtime_str)
                mtime_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                mtime_item.setFlags(mtime_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                mtime_item.setToolTip(f"Last Modified: {mtime_str}")

                # --- Set Items in Table Row ---
                self.data_table.setItem(row_index, 0, source_item)
                self.data_table.setItem(row_index, 1, name_item)
                self.data_table.setItem(row_index, 2, chunk_item)
                self.data_table.setItem(row_index, 3, mtime_item)

                # --- Action Button ---
                remove_btn = QPushButton("Remove")
                remove_btn.setStyleSheet("QPushButton { padding: 2px 5px; }")
                remove_btn.setToolTip(f"Remove source: {display_source_text}")
                # Button enabled state depends on whether actions are globally disabled
                remove_btn.setEnabled(not self._actions_disabled)

                if source_identifier_for_action:
                    # Use lambda to capture the correct identifier for THIS row's button
                    # The `sid=source_identifier_for_action` part is crucial
                    remove_btn.clicked.connect(
                        lambda checked=False, sid=source_identifier_for_action: self.remove_item(sid)
                    )
                else:
                    # Disable button if no valid identifier found for action
                    remove_btn.setDisabled(True)
                    remove_btn.setToolTip("Cannot remove: Missing source identifier.")

                # Embed button in a layout within a widget for centering in the cell
                button_container_widget = QWidget()
                button_layout = QHBoxLayout(button_container_widget)
                button_layout.addWidget(remove_btn)
                button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                button_layout.setContentsMargins(0, 0, 0, 0) # Remove margins
                button_container_widget.setLayout(button_layout)
                self.data_table.setCellWidget(row_index, action_col_index, button_container_widget)

            logger.debug(f"Updated indexed sources table with {len(items)} items.")
        except Exception as e:
            logger.error(f"Failed to update data display table: {e}", exc_info=True)
            self._set_status("Error updating indexed sources list.", level="error")
        finally:
            # Re-enable table interaction only if actions are not generally disabled
            self.data_table.setDisabled(self._actions_disabled)