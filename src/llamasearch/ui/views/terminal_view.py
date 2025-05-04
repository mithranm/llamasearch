# src/llamasearch/ui/views/terminal_view.py

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from PySide6.QtCore import QTimer, Slot

class TerminalView(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("Terminal Logs")
        layout.addWidget(title)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        refresh_btn = QPushButton("Refresh Logs")
        refresh_btn.clicked.connect(self.refresh_logs)
        layout.addWidget(refresh_btn)

        self.setLayout(layout)

        # Timer to automatically refresh logs every 3 seconds.
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_logs)
        self.timer.start(3000)

    @Slot()
    def refresh_logs(self):
        logs = self.backend.get_live_logs()
        if logs:
            self.log_view.setPlainText(logs[0][1])
