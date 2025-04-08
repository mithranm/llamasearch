# src/llamasearch/ui/views/settings_view.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, 
                              QLineEdit, QComboBox, QGroupBox, QPushButton, QFileDialog,
                              QSpinBox, QFormLayout, QCheckBox, QTabWidget, QScrollArea,
                              )
from PySide6.QtCore import Qt, QTimer
import os
from pathlib import Path

class SettingsView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        
        # Main layout for the widget
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)
        
        # Create a status label that will be used for notifications
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_label.setVisible(False)  # Initially hidden
        
        self.init_ui()
    
    def init_ui(self):
        # Create tabs for organizing settings
        tabs = QTabWidget()
        
        # Create tab widgets
        models_tab = self._create_models_tab()
        system_tab = self._create_system_tab()
        
        # Add tabs
        tabs.addTab(models_tab, "Models")
        tabs.addTab(system_tab, "System")
        
        # Add tabs to main layout
        self.main_layout.addWidget(tabs)
        
        # Add status label at the bottom of the main layout
        self.main_layout.addWidget(self.status_label)
    
    def show_status_message(self, message, duration=3000):
        """Display a status message for the specified duration (in milliseconds)"""
        self.status_label.setText(message)
        self.status_label.setVisible(True)
        
        # Hide the message after the specified duration
        QTimer.singleShot(duration, lambda: self.status_label.setVisible(False))
    
    def _create_models_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model selection group
        group_model = QGroupBox("LLM Model Selection")
        form_layout = QFormLayout(group_model)
        
        # Engine selection
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["llamacpp", "hf"])
        current_engine = self.app.get_model_config()["model_engine"]
        self.engine_combo.setCurrentText(current_engine)
        form_layout.addRow("Model Engine:", self.engine_combo)
        
        # Create a group for GGUF local models
        group_gguf = QGroupBox("GGUF Local Models")
        gguf_layout = QVBoxLayout(group_gguf)
        
        # Scan for local GGUF models
        self.gguf_combo = QComboBox()
        self.gguf_combo.setMinimumWidth(300)
        
        # Add option for custom path
        self.gguf_combo.addItem("-- Custom Path --")
        
        # Add available local models
        available_models = self.app.get_available_models()
        for model in available_models.get("local_models", []):
            self.gguf_combo.addItem(model["name"], userData=model["path"])
        
        # Select current model if it's in the list
        current_model = self.app.get_model_config()["model_name"]
        model_path = self.app.get_model_config()["custom_model_path"]
        
        # Initialize field for custom path
        self.custom_model_path = QLineEdit(model_path)
        
        # Connect signals
        self.gguf_combo.currentIndexChanged.connect(self._on_gguf_model_changed)
        
        # Add browse button for custom path
        browse_layout = QHBoxLayout()
        browse_layout.addWidget(self.custom_model_path)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_model_file)
        browse_layout.addWidget(browse_button)
        
        gguf_layout.addWidget(QLabel("Select GGUF Model:"))
        gguf_layout.addWidget(self.gguf_combo)
        gguf_layout.addWidget(QLabel("Custom Path:"))
        gguf_layout.addLayout(browse_layout)
        
        # Create a group for Hugging Face models
        group_hf = QGroupBox("Hugging Face Models")
        hf_layout = QVBoxLayout(group_hf)
        
        # Add HF model selector
        self.hf_combo = QComboBox()
        self.hf_combo.setMinimumWidth(300)
        self.hf_combo.setEditable(True)
        
        # Add suggested models
        for model in available_models.get("huggingface_suggestions", []):
            self.hf_combo.addItem(model)
        
        # Add current model if not in list
        if current_engine == "hf" and current_model not in available_models.get("huggingface_suggestions", []):
            self.hf_combo.addItem(current_model)
            self.hf_combo.setCurrentText(current_model)
        
        hf_layout.addWidget(QLabel("Select Hugging Face Model:"))
        hf_layout.addWidget(self.hf_combo)
        
        # Add download instructions
        help_text = (
            "Add custom GGUF models to: " + str(self.app.data_paths["models"]) + "\n" +
            "Hugging Face models will be downloaded automatically.\n" +
            "For large models, ensure you have enough RAM and disk space."
        )
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        
        # Generate model parameters
        group_params = QGroupBox("Generation Parameters")
        params_layout = QFormLayout(group_params)
        
        # Temperature
        self.temp_spinner = QSpinBox()
        self.temp_spinner.setRange(0, 100)
        self.temp_spinner.setValue(70)  # Default 0.7
        self.temp_spinner.setSuffix(" %")
        params_layout.addRow("Temperature:", self.temp_spinner)
        
        # Context length
        self.context_spinner = QSpinBox()
        self.context_spinner.setRange(1024, 16384)
        self.context_spinner.setSingleStep(1024)
        self.context_spinner.setValue(4096)
        params_layout.addRow("Context Length:", self.context_spinner)
        
        # Max results
        self.results_spinner = QSpinBox()
        self.results_spinner.setRange(1, 10)
        self.results_spinner.setValue(3)
        params_layout.addRow("Max Search Results:", self.results_spinner)
        
        # Debug mode
        self.debug_checkbox = QCheckBox("Enable Debug Mode")
        self.debug_checkbox.setChecked(self.app.debug)
        params_layout.addRow("", self.debug_checkbox)
        
        # Apply button
        self.apply_button = QPushButton("Apply Model Settings")
        self.apply_button.clicked.connect(self._apply_model_settings)
        
        # Add all components to the layout
        layout.addWidget(group_model)
        layout.addWidget(group_gguf)
        layout.addWidget(group_hf)
        layout.addWidget(group_params)
        layout.addWidget(help_label)
        layout.addWidget(self.apply_button)
        
        # Initial UI state based on engine
        self._update_ui_for_engine(current_engine)
        
        # Connect engine change
        self.engine_combo.currentTextChanged.connect(self._update_ui_for_engine)
        
        return tab
    
    def _on_gguf_model_changed(self, index):
        if index == 0:  # Custom path option
            self.custom_model_path.setEnabled(True)
        else:
            model_path = self.gguf_combo.currentData()
            if model_path:
                self.custom_model_path.setText(model_path)
                self.custom_model_path.setEnabled(False)
    
    def _browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", 
            str(self.app.data_paths["models"]), 
            "GGUF Models (*.gguf *.bin);;All Files (*)"
        )
        if file_path:
            self.custom_model_path.setText(file_path)
            # Set combo box to custom option
            self.gguf_combo.setCurrentIndex(0)
    
    def _update_ui_for_engine(self, engine):
        if engine == "llamacpp":
            self.gguf_combo.setEnabled(True)
            self.custom_model_path.setEnabled(self.gguf_combo.currentIndex() == 0)
            self.hf_combo.setEnabled(False)
        else:  # hf
            self.gguf_combo.setEnabled(False)
            self.custom_model_path.setEnabled(False)
            self.hf_combo.setEnabled(True)
    
    def _apply_model_settings(self):
        engine = self.engine_combo.currentText()
        model_name = ""
        custom_path = ""
        
        if engine == "llamacpp":
            if self.gguf_combo.currentIndex() == 0:  # Custom path
                # Use the filename as model_name
                custom_path = self.custom_model_path.text()
                model_name = os.path.basename(custom_path)
            else:
                model_name = self.gguf_combo.currentText()
                # Store the full path in custom_path
                custom_path = self.gguf_combo.currentData() or ""
        else:  # hf
            model_name = self.hf_combo.currentText()
            custom_path = ""
        
        # Apply model settings
        self.app.set_model_config(model_name, engine, custom_path)
        
        # Apply other settings
        self.app.debug = self.debug_checkbox.isChecked()
        
        # Show success message using our dedicated method
        self.show_status_message("Settings applied successfully!")
    
    def _create_system_tab(self):
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Force CPU option
        cpu_group = QGroupBox("Processing Options")
        cpu_layout = QVBoxLayout(cpu_group)
        
        self.cpu_checkbox = QCheckBox("Force CPU Usage (disable GPU)")
        self.cpu_checkbox.setChecked(self.app.use_cpu)
        cpu_layout.addWidget(self.cpu_checkbox)
        
        # Apply CPU setting
        cpu_button = QPushButton("Apply CPU Setting")
        cpu_button.clicked.connect(self._apply_cpu_setting)
        cpu_layout.addWidget(cpu_button)
        
        # Data paths
        paths_group = QGroupBox("Data Paths")
        paths_layout = QVBoxLayout(paths_group)
        
        paths = self.app.data_paths
        paths_text = ""
        for k, v in paths.items():
            paths_text += f"<b>{k}</b>: {v}<br>"
        
        paths_label = QLabel(paths_text)
        paths_label.setTextFormat(Qt.TextFormat.RichText)
        paths_label.setWordWrap(True)
        paths_layout.addWidget(paths_label)
        
        # Add to layout
        layout.addWidget(cpu_group)
        layout.addWidget(paths_group)
        layout.addStretch()
        
        tab.setWidget(content)
        return tab
    
    def _apply_cpu_setting(self):
        self.app.use_cpu = self.cpu_checkbox.isChecked()
        
        # Force reload of model if needed
        if self.app.llm_instance:
            self.app.llm_instance.unload_model()
            self.app.llm_instance = None
        
        # Show success message using our dedicated method
        self.show_status_message("CPU setting applied. Model will reload with new setting.")

def settings_view(app):
    """Create and return a SettingsView instance"""
    return SettingsView(app)