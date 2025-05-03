# src/llamasearch/ui/components.py
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PySide6.QtGui import QPixmap
from pathlib import Path

def header_component():
    """
    Creates a header widget with a square logo and title.
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)
    
    # Get path to logo (adjust path as needed)
    logo_path = Path("public/llamasearch.png")
    logo_label = QLabel()
    if logo_path.exists():
        pixmap = QPixmap(str(logo_path))
        pixmap = pixmap.scaled(100, 100)
        logo_label.setPixmap(pixmap)
    else:
        logo_label.setText("No Logo")
    
    title_label = QLabel("LlamaSearch")
    title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-left: 10px;")
    
    layout.addWidget(logo_label)
    layout.addWidget(title_label)
    return widget
