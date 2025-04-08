# src/llamasearch/ui/views/settings_view.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QRadioButton, QHBoxLayout, QLineEdit

def settings_view(app):
    widget = QWidget()
    layout = QVBoxLayout(widget)
    
    layout.addWidget(QLabel("Settings"))
    layout.addWidget(QLabel("## Data Paths"))
    paths = app.data_paths
    path_info = "\n".join([f"**{k}**: {v}" for k, v in paths.items()])
    layout.addWidget(QLabel(path_info))
    
    layout.addWidget(QLabel("## LLM Selection"))
    radio_widget = QWidget()
    radio_layout = QHBoxLayout(radio_widget)
    standard_radio = QRadioButton("Standard LLM")
    cpu_radio = QRadioButton("CPU-optimized LLM")
    standard_radio.setChecked(True)
    radio_layout.addWidget(standard_radio)
    radio_layout.addWidget(cpu_radio)
    layout.addWidget(radio_widget)
    
    status_label = QLineEdit("Using Standard LLM")
    status_label.setReadOnly(True)
    layout.addWidget(status_label)
    
    def update_llm_type():
        selection = "CPU-optimized LLM" if cpu_radio.isChecked() else "Standard LLM"
        app.use_cpu = (selection == "CPU-optimized LLM")
        app.llm_instance = None  # reset LLM instance
        status_label.setText(f"Using {selection}")
    
    standard_radio.toggled.connect(update_llm_type)
    cpu_radio.toggled.connect(update_llm_type)
    
    return widget
