# src/llamasearch/ui/views/settings_view.py
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QTabWidget,
    QScrollArea,
    QMessageBox,
)
from PySide6.QtCore import (
    Qt,
    QTimer, # <-- Import QTimer
    Slot,
    # QMetaObject, # No longer needed for these calls
    QUrl,
)
from PySide6.QtGui import QDesktopServices

# from typing import cast, Any # No longer needed for these calls
from typing import Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SettingsView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_label.setVisible(False)
        self.init_ui()
        self.load_settings()
        # Connect signal directly to the slot
        self.app.signals.settings_applied.connect(self.show_status_message)

    def init_ui(self):
        # --- No changes needed in init_ui itself ---
        tabs = QTabWidget()
        general_tab = self._create_general_tab()
        system_tab = self._create_system_tab()
        tabs.addTab(general_tab, "General")
        tabs.addTab(system_tab, "System")
        self.main_layout.addWidget(tabs)
        self.main_layout.addWidget(self.status_label)

    @Slot(str, str)
    def show_status_message(self, message, level="success", duration=3500):
        """Display a status message. Assumes called in GUI thread via signal."""
        # Direct update since called via signal
        self.status_label.setText(message)

        color = {"error": "red", "warning": "orange"}.get(level, "green")
        style = f"color: {color}; font-weight: bold;"
        self.status_label.setStyleSheet(style)

        self.status_label.setVisible(True)

        # Hide after delay using QTimer
        QTimer.singleShot(duration, lambda: self.status_label.setVisible(False))


    def _create_general_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group_model = QGroupBox("Active Model Information")
        model_layout = QFormLayout(group_model)
        self.model_id_label = QLabel("N/A")
        self.model_engine_label = QLabel("N/A")
        self.model_provider_label = QLabel("N/A")
        self.model_quant_label = QLabel("N/A")
        self.model_context_label = QLabel("N/A")
        model_layout.addRow("Model ID:", self.model_id_label)
        model_layout.addRow("Engine:", self.model_engine_label)
        model_layout.addRow("ONNX Provider:", self.model_provider_label)
        model_layout.addRow("Quantization:", self.model_quant_label)
        model_layout.addRow("Context Length:", self.model_context_label)
        setup_button = QPushButton("Run Model Setup...")
        setup_button.setToolTip("Show setup instructions")
        setup_button.clicked.connect(self._run_model_setup)
        model_layout.addRow(setup_button)
        group_params = QGroupBox("Search & Generation Parameters")
        params_layout = QFormLayout(group_params)
        self.results_spinner = QSpinBox()
        self.results_spinner.setRange(1, 20)
        self.results_spinner.setToolTip("Max chunks for context.")
        params_layout.addRow("Max Retrieved Chunks:", self.results_spinner)
        self.debug_checkbox = QCheckBox("Enable Debug Logging")
        self.debug_checkbox.setToolTip("Show detailed logs.")
        params_layout.addRow("", self.debug_checkbox)
        self.apply_button = QPushButton("Apply General Settings")
        self.apply_button.clicked.connect(self._apply_general_settings)
        layout.addWidget(group_model)
        layout.addWidget(group_params)
        layout.addStretch()
        layout.addWidget(self.apply_button)
        return tab

    def _create_system_tab(self):
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        paths_group = QGroupBox("Data Storage Paths")
        paths_layout = QFormLayout(paths_group)
        self.path_labels: Dict[str, QLabel] = {}
        paths = self.app.data_paths
        for key, default_path in paths.items():
            path_label = QLabel(str(default_path))
            path_label.setWordWrap(True)
            path_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            row_layout = QHBoxLayout()
            row_layout.addWidget(path_label)
            button = QPushButton("Open")
            button.clicked.connect(
                lambda checked=False, p=str(default_path): self._open_directory(p)
            )
            button.setToolTip(f"Open {key} directory")
            row_layout.addWidget(button)
            paths_layout.addRow(f"<b>{key.replace('_', ' ').title()}:</b>", row_layout)
            self.path_labels[key] = path_label
        layout.addWidget(paths_group)
        layout.addStretch()
        content.setLayout(layout)
        tab.setWidget(content)
        return tab

    @Slot()
    def _open_directory(self, path_str: str):
        path = Path(path_str)
        if path.exists() and path.is_dir():
            url = QUrl.fromLocalFile(str(path))
            if not QDesktopServices.openUrl(url):
                # Use show_status_message since this is a direct UI action result
                self.show_status_message(
                    f"Error: Could not open {path_str}", level="error"
                )
                logger.error(f"Failed openUrl: {path_str}")
        else:
            self.show_status_message(f"Error: Not found: {path_str}", level="error")
            logger.warning(f"Dir not found: {path_str}")

    @Slot()
    def _run_model_setup(self):
        QMessageBox.information(
            self,
            "Model Setup",
            "Run:\n\n<code>llamasearch-setup</code>\n\nin your terminal. Restart required.",
            QMessageBox.StandardButton.Ok,
        )

    @Slot()
    def _apply_general_settings(self):
        """Applies settings synchronously and lets backend emit status."""
        settings_to_apply = {
            "max_results": self.results_spinner.value(),
            "debug_mode": self.debug_checkbox.isChecked(),
        }
        try:
            # Call backend directly, which will emit settings_applied signal
            self.app.apply_settings(settings_to_apply)
        except Exception as e:
            # Show error directly if the call itself fails
            self.show_status_message(f"Error applying settings: {e}", level="error")
            logger.error(f"Apply settings error: {e}", exc_info=True)

    def load_settings(self):
        """Loads current config into UI elements."""
        # This runs in the GUI thread during init, direct access is fine
        config = self.app.get_current_config()
        self.model_id_label.setText(config.get("model_id", "N/A"))
        self.model_engine_label.setText(config.get("model_engine", "N/A"))
        self.model_provider_label.setText(config.get("provider", "N/A"))
        self.model_quant_label.setText(config.get("quantization", "N/A"))
        self.model_context_label.setText(str(config.get("context_length", "N/A")))
        self.results_spinner.setValue(config.get("max_results", 3))
        self.debug_checkbox.setChecked(config.get("debug_mode", False))
        paths = self.app.data_paths
        for key, label in self.path_labels.items():
            label.setText(str(paths.get(key, "N/A")))