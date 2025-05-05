# src/llamasearch/ui/views/terminal_view.py

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QTextCharFormat
from PySide6.QtWidgets import (QLabel, QPushButton, QTextEdit, QVBoxLayout,
                               QWidget)

from ..qt_logging import qt_log_emitter


class TerminalView(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()
        # Connect the global log emitter signal to our slot
        # QueuedConnection ensures the slot runs in the GUI thread
        qt_log_emitter.log_message.connect(
            self.append_log_message, Qt.ConnectionType.QueuedConnection
        )

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("<b>Logs</b>")
        layout.addWidget(title)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        font = QFont("Courier New", 10)  # Monospace font
        self.log_view.setFont(font)
        layout.addWidget(self.log_view)

        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.log_view.clear)
        layout.addWidget(clear_btn)

        self.setLayout(layout)

    # Slot receives log messages from the QtLogHandler signal
    @Slot(str, str)
    def append_log_message(self, level_name: str, message: str):
        """Appends a log message to the QTextEdit, potentially with color."""
        # Assured to run in GUI thread by QueuedConnection

        color_map = {
            "DEBUG": QColor("gray"),
            "INFO": QColor("black"),
            "WARNING": QColor("darkOrange"),
            "ERROR": QColor("red"),
            "CRITICAL": QColor("darkRed"),
        }
        color = color_map.get(level_name, QColor("black"))

        text_format = QTextCharFormat()
        text_format.setForeground(color)
        if level_name in ["ERROR", "CRITICAL"]:
            text_format.setFontWeight(QFont.Weight.Bold)

        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        # Insert newline only if needed
        current_text = self.log_view.toPlainText()
        if current_text and not current_text.endswith("\n"):
            cursor.insertText("\n")

        cursor.mergeCharFormat(text_format)
        cursor.insertText(message)
        cursor.setCharFormat(QTextCharFormat())  # Reset format

        self.log_view.ensureCursorVisible()  # Auto-scroll
