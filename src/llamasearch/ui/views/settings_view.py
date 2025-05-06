# src/llamasearch/ui/views/settings_view.py
import logging
from pathlib import Path
from typing import Dict

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QScrollArea,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

logger = logging.getLogger(__name__)

class SettingsView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # Status label for feedback (initially hidden)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_label.setVisible(False)

        # Dictionary to store path labels for easy update (initialized before init_ui)
        self.path_labels: Dict[str, QLabel] = {}

        self.init_ui()
        self.load_settings()

        # Connect signal from backend to update status
        # Use attribute access for connect
        self.app.signals.settings_applied.connect(self.show_status_message)

    def init_ui(self):
        """Initialize the UI components."""
        tabs = QTabWidget()
        general_tab = self._create_general_tab()
        system_tab = self._create_system_tab()

        tabs.addTab(general_tab, "General")
        tabs.addTab(system_tab, "System")

        self.main_layout.addWidget(tabs)
        # Add status label at the bottom
        self.main_layout.addWidget(self.status_label)

    @Slot(str, str)
    def show_status_message(self, message: str, level: str = "success", duration: int = 3500):
        """
        Display a status message at the bottom of the settings view.
        Assumes it's called in the GUI thread via a signal connection.
        """
        self.status_label.setText(message)

        color_map = {
            "error": "red",
            "warning": "orange",
            "success": "green",
            "info": "#333", # Darker color for info
        }
        color = color_map.get(level, "green") # Default to success color
        style = f"color: {color}; font-weight: bold;"
        self.status_label.setStyleSheet(style)

        self.status_label.setVisible(True)

        # Use QTimer to hide the message after a duration
        QTimer.singleShot(duration, lambda: self.status_label.setVisible(False))

    def _create_general_tab(self):
        """Creates the 'General' settings tab content."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Active Model Information Group ---
        group_model = QGroupBox("Active Model Information")
        model_layout = QFormLayout(group_model)

        # Initialize labels
        self.model_id_label = QLabel("N/A")
        self.model_engine_label = QLabel("N/A")
        self.model_context_label = QLabel("N/A")

        # Add rows to layout
        model_layout.addRow("Model ID:", self.model_id_label)
        model_layout.addRow("Engine:", self.model_engine_label)
        model_layout.addRow("Context Length:", self.model_context_label)

        # Add setup button
        setup_button = QPushButton("Run Model Setup...")
        setup_button.setToolTip("Show setup instructions (requires restart)")
        # Use attribute access for connect
        setup_button.clicked.connect(self._run_model_setup)
        model_layout.addRow(setup_button) # Add button without label

        # --- Search & Generation Parameters Group ---
        group_params = QGroupBox("Search & Generation Parameters")
        params_layout = QFormLayout(group_params)

        # Max Results Spinner
        self.results_spinner = QSpinBox()
        self.results_spinner.setRange(1, 20) # Set reasonable range
        self.results_spinner.setToolTip("Maximum number of retrieved document chunks used for context.")
        params_layout.addRow("Max Retrieved Chunks:", self.results_spinner)

        # Debug Checkbox
        self.debug_checkbox = QCheckBox("Enable Debug Logging")
        self.debug_checkbox.setToolTip("Show detailed logs in the Logs tab and console.")
        params_layout.addRow("", self.debug_checkbox) # Add checkbox without label

        # Apply Button
        self.apply_button = QPushButton("Apply General Settings")
        # Use attribute access for connect
        self.apply_button.clicked.connect(self._apply_general_settings)

        # Add groups and button to the main tab layout
        layout.addWidget(group_model)
        layout.addWidget(group_params)
        layout.addStretch() # Push elements upwards
        layout.addWidget(self.apply_button)

        # Set the layout for the tab widget
        tab.setLayout(layout)
        return tab

    def _create_system_tab(self):
        """Creates the 'System' settings tab content."""
        # Use QScrollArea for potentially long path lists
        tab = QScrollArea()
        tab.setWidgetResizable(True) # Allow inner widget to resize

        content_widget = QWidget() # Widget to hold the actual content
        layout = QVBoxLayout(content_widget)

        # --- Data Storage Paths Group ---
        paths_group = QGroupBox("Data Storage Paths")
        paths_layout = QFormLayout(paths_group)

        # Get paths from backend
        paths = self.app.data_paths

        # Clear existing labels before creating new ones (if method is called multiple times)
        self.path_labels.clear()

        # Create label and button for each path
        for key, default_path_str in paths.items():
            # Create label with current path
            path_label = QLabel(str(default_path_str))
            path_label.setWordWrap(True) # Allow text wrapping
            # Allow users to select/copy the path text
            path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.path_labels[key] = path_label # Store label reference

            # Create 'Open' button for the directory
            button = QPushButton("Open")
            # Use lambda to capture the correct path for the button click
            # Ensure path_str is captured correctly in the lambda context
            current_path_for_button = str(default_path_str)
            button.clicked.connect(
                lambda checked=False, p=current_path_for_button: self._open_directory(p)
            )
            button.setToolTip(f"Open the '{key.replace('_', ' ').title()}' directory")

            # Horizontal layout for label + button
            row_layout = QHBoxLayout()
            row_layout.addWidget(path_label) # Add label first
            row_layout.addWidget(button) # Add button next to label

            # Add row to the form layout
            paths_layout.addRow(f"<b>{key.replace('_', ' ').title()}:</b>", row_layout)

        # Add the paths group to the main layout
        layout.addWidget(paths_group)
        layout.addStretch() # Push group upwards

        # Set the layout for the content widget
        content_widget.setLayout(layout)
        # Set the content widget inside the scroll area
        tab.setWidget(content_widget)
        return tab

    @Slot()
    def _open_directory(self, path_str: str):
        """Opens the specified directory path in the system's file explorer."""
        logger.debug(f"Attempting to open directory: {path_str}")
        path = Path(path_str)
        if path.exists() and path.is_dir():
            url = QUrl.fromLocalFile(str(path))
            success = QDesktopServices.openUrl(url)
            if not success:
                # Show error message using the status label method
                self.show_status_message(
                    f"Error: Could not open directory {path_str}", level="error"
                )
                logger.error(f"Failed QDesktopServices.openUrl for path: {path_str}")
        else:
            self.show_status_message(f"Error: Directory not found: {path_str}", level="error")
            logger.warning(f"Directory not found for opening: {path_str}")

    @Slot()
    def _run_model_setup(self):
        """Shows an informational message about running the setup script."""
        QMessageBox.information(
            self, # Parent widget
            "Model Setup",
            "To download or update models, please run:\n\n"
            "<code>llamasearch-setup</code>\n\n"
            "in your terminal.\n\nA restart of LlamaSearch might be required after setup.",
            QMessageBox.StandardButton.Ok, # Only show OK button
        )

    @Slot()
    def _apply_general_settings(self):
        """Applies settings from the 'General' tab synchronously."""
        logger.debug("Applying general settings...")
        settings_to_apply = {
            "max_results": self.results_spinner.value(),
            "debug_mode": self.debug_checkbox.isChecked(),
        }
        try:
            # Call backend directly. Backend emits 'settings_applied' signal.
            self.app.apply_settings(settings_to_apply)
            # Status message is handled by the connected slot show_status_message
        except Exception as e:
            # Show error directly only if the apply_settings call itself fails
            self.show_status_message(f"Error applying settings: {e}", level="error")
            logger.error(f"Apply settings error: {e}", exc_info=True)

    def load_settings(self):
        """Loads current configuration from the backend into UI elements."""
        logger.debug("Loading settings into UI.")
        # This runs in the GUI thread during init or refresh, direct access is fine
        config = self.app.get_current_config()

        # Update labels in the 'General' tab
        self.model_id_label.setText(config.get("model_id", "N/A"))
        self.model_engine_label.setText(config.get("model_engine", "N/A"))
        self.model_context_label.setText(str(config.get("context_length", "N/A")))
        # Removed provider/quantization labels

        # Update controls in the 'General' tab
        self.results_spinner.setValue(config.get("max_results", 3))
        self.debug_checkbox.setChecked(config.get("debug_mode", False))

        # Update paths in the 'System' tab
        paths = self.app.data_paths
        for key, label_widget in self.path_labels.items():
            path_text = str(paths.get(key, "N/A"))
            label_widget.setText(path_text)
            # Find the associated button to update its connection path
            # Get the parent layout of the label
            row_layout_widget = label_widget.parentWidget()
            button = None
            if row_layout_widget:
                 # Find the QPushButton within that layout/widget
                 button = row_layout_widget.findChild(QPushButton)

            if button:
                # Disconnect old lambda and connect new one to ensure path is up-to-date
                try:
                    # Disconnect all slots connected to clicked signal
                    button.clicked.disconnect()
                except (TypeError, RuntimeError): # Handles case where no slots are connected
                    pass
                # Reconnect with the potentially updated path
                # Ensure path_text is captured correctly for this specific button
                button.clicked.connect(lambda checked=False, p=path_text: self._open_directory(p))
            else:
                 logger.warning(f"Could not find button associated with path label for key: {key}")