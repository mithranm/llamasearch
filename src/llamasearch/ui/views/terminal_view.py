# src/llamasearch/ui/views/terminal_view.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor

class TerminalView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.init_ui()
        # Set up timer to refresh logs regularly
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_logs)
        self.timer.start(1000)  # Update every second
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create terminal display (read-only text edit with monospace font)
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("font-family: Consolas, monospace; background-color: #242424; color: #f0f0f0;")
        self.terminal.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)  # Disable line wrapping for terminal-like display
        layout.addWidget(self.terminal)
        
        # Button row
        button_row = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Terminal")
        self.clear_btn.clicked.connect(self.clear_terminal)
        self.auto_scroll = QPushButton("Auto-scroll")
        self.auto_scroll.setCheckable(True)
        self.auto_scroll.setChecked(True)
        button_row.addWidget(self.clear_btn)
        button_row.addWidget(self.auto_scroll)
        layout.addLayout(button_row)
    
    def update_logs(self):
        """Update the terminal with latest logs"""
        logs = self.app.get_live_logs()
        if logs and logs[0] and len(logs[0]) > 1:
            current_text = self.terminal.toPlainText()
            new_text = logs[0][1]  # logs format is [["system", "<all logs concatenated>"]]
            
            # Only update if there are new logs
            if new_text != current_text:
                self.terminal.setPlainText(new_text)
                
                # Autoscroll if enabled
                if self.auto_scroll.isChecked():
                    self.terminal.moveCursor(QTextCursor.MoveOperation.End)
    
    def clear_terminal(self):
        """Clear the terminal display"""
        self.app.all_logs.clear()
        self.terminal.clear()