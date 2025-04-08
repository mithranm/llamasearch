# src/llamasearch/ui/views/search_view.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QHBoxLayout
from PySide6.QtCore import Qt, Slot, QTimer

class SearchView(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()
        
        # Set up timer to poll for search results
        self.result_timer = QTimer(self)
        self.result_timer.timeout.connect(self.check_search_results)
        self.result_timer.setInterval(500)  # Check every 500ms
        
        # Flag to track if a search is in progress
        self.search_in_progress = False
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Search Your Indexed Content")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px 0;")
        layout.addWidget(title)
        
        # Search input area
        input_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("What would you like to know about your indexed content?")
        self.search_btn = QPushButton("Search")
        self.search_btn.setMinimumWidth(100)
        input_layout.addWidget(self.query_input)
        input_layout.addWidget(self.search_btn)
        layout.addLayout(input_layout)
        
        # Connect enter key in line edit to search
        self.query_input.returnPressed.connect(self.do_search)
        
        # Results area
        results_label = QLabel("Results:")
        layout.addWidget(results_label)
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setMinimumHeight(400)
        layout.addWidget(self.results_display)
        
        # Connect button to search function
        self.search_btn.clicked.connect(self.do_search)
    
    def do_search(self):
        """Initiate a search operation"""
        query = self.query_input.text().strip()
        if not query:
            self.results_display.setHtml("Please enter a search query.")
            return
        
        # Mark search as in progress
        self.search_in_progress = True
        
        # Disable search button while processing
        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching...")
        
        # Start the search (this will emit signals to switch to terminal tab)
        initial_result = self.backend.search_content(query)
        self.results_display.setHtml(initial_result)
        
        # Start polling for results
        self.result_timer.start()
    
    @Slot()
    def check_search_results(self):
        """Check if search results are ready and update the display"""
        if not self.search_in_progress:
            return
        
        # Get the latest result
        current_result = self.backend.get_search_result()
        
        # Check if search is still in progress
        if current_result != "Searching, please wait...":
            # Search is complete, update UI
            self.results_display.setHtml(current_result)
            self.search_btn.setEnabled(True)
            self.search_btn.setText("Search")
            self.search_in_progress = False
            self.result_timer.stop()
        else:
            # Still searching, keep polling
            self.results_display.setHtml("Searching, please wait...")