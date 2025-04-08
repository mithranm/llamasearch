# src/llamasearch/ui/views/search_view.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QHBoxLayout, QScrollArea, QFrame, QDialog)
from PySide6.QtCore import Qt, Slot, QTimer, Signal
from PySide6.QtGui import QFont

class SearchResultCard(QFrame):
    """Widget representing a single search result card"""
    clicked = Signal(str, str)  # Signal emits source and content when clicked
    
    def __init__(self, index, source, content, score=None):
        super().__init__()
        self.index = index
        self.source = source
        self.content = content
        self.score = score
        self.init_ui()
        
    def init_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(1)
        self.setMidLineWidth(0)
        
        # Set styles
        self.setStyleSheet("""
            SearchResultCard {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin: 8px;
                padding: 10px;
            }
            SearchResultCard:hover {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
            }
            QPushButton {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QLabel[isLink="true"] {
                color: #0066cc;
                text-decoration: underline;
            }
            QLabel[isLink="true"]:hover {
                color: #004499;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Source header with clickable link
        source_layout = QHBoxLayout()
        
        # Check if source is a URL
        if self.source.startswith(('http://', 'https://')):
            source_label = QLabel(f"<strong>Source {self.index}:</strong> <a href='{self.source}'>{self.source}</a>")
            source_label.setTextFormat(Qt.TextFormat.RichText)
            source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            source_label.setOpenExternalLinks(True)
        else:
            source_label = QLabel(f"<strong>Source {self.index}:</strong> {self.source}")
            source_label.setTextFormat(Qt.TextFormat.RichText)
            
        source_label.setWordWrap(True)
        source_layout.addWidget(source_label)
        layout.addLayout(source_layout)
        
        # Score if available
        if self.score is not None:
            score_label = QLabel(f"Relevance Score: {self.score:.2f}")
            score_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addWidget(score_label)
        
        # Preview of the content (limited to 300 chars)
        content_preview = self.content
        if len(content_preview) > 300:
            content_preview = content_preview[:297] + "..."
        
        content_label = QLabel(content_preview)
        content_label.setWordWrap(True)
        content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(content_label)
        
        # Action buttons
        button_layout = QHBoxLayout()
        view_more = QPushButton("View Full Content")
        view_more.setMaximumWidth(150)
        view_more.clicked.connect(self.emit_clicked)
        button_layout.addStretch()
        button_layout.addWidget(view_more)
        layout.addLayout(button_layout)
        
    def mousePressEvent(self, event):
        # Emit the signal when the card is clicked
        super().mousePressEvent(event)
        
    def emit_clicked(self):
        self.clicked.emit(self.source, self.content)
        
class ResultDetailDialog(QDialog):
    """Modal dialog showing the full content of a search result"""
    def __init__(self, source, content, parent=None):
        super().__init__(parent)
        self.source = source
        self.content = content
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"Search Result Detail")
        self.resize(700, 500)
        
        layout = QVBoxLayout(self)
        
        # Source header with clickable link if it's a URL
        if self.source.startswith(('http://', 'https://')):
            source_label = QLabel(f"<strong>Source:</strong> <a href='{self.source}'>{self.source}</a>")
            source_label.setTextFormat(Qt.TextFormat.RichText)
            source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            source_label.setOpenExternalLinks(True)
        else:
            source_label = QLabel(f"<strong>Source:</strong> {self.source}")
            source_label.setTextFormat(Qt.TextFormat.RichText)
            
        source_label.setWordWrap(True)
        layout.addWidget(source_label)
        
        # Full content in a text edit widget
        content_edit = QTextEdit()
        content_edit.setReadOnly(True)
        content_edit.setPlainText(self.content)
        content_edit.setFont(QFont("Monospace", 10))
        layout.addWidget(content_edit)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

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
        
        # Store the raw search result
        self.raw_search_result = None
    
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
        
        # AI Overview section
        overview_frame = QFrame()
        overview_frame.setFrameShape(QFrame.Shape.StyledPanel)
        overview_frame.setFrameShadow(QFrame.Shadow.Sunken)
        overview_frame.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 5px; border: 1px solid #ddd;")
        
        overview_layout = QVBoxLayout(overview_frame)
        overview_title = QLabel("AI Overview")
        overview_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        overview_layout.addWidget(overview_title)
        
        self.overview_text = QTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setMaximumHeight(200)
        self.overview_text.setStyleSheet("background-color: white; border: 1px solid #ddd;")
        overview_layout.addWidget(self.overview_text)
        
        layout.addWidget(overview_frame)
        
        # Search Results section
        results_title = QLabel("Search Results")
        results_title.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 15px;")
        layout.addWidget(results_title)
        
        # Create a scroll area for search results
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container widget for results
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll_area.setWidget(self.results_container)
        layout.addWidget(scroll_area)
        layout.setStretchFactor(scroll_area, 1)  # Give the scroll area more space
        
        # Status label at the bottom
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Connect button to search function
        self.search_btn.clicked.connect(self.do_search)
    
    def do_search(self):
        """Initiate a search operation"""
        query = self.query_input.text().strip()
        if not query:
            self.status_label.setText("Please enter a search query.")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Mark search as in progress
        self.search_in_progress = True
        
        # Disable search button while processing
        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching...")
        self.status_label.setText("Searching, please wait...")
        
        # Start the search (this will emit signals to switch to terminal tab)
        initial_result = self.backend.search_content(query)
        self.overview_text.setPlainText("Searching, please wait...")
        
        # Start polling for results
        self.result_timer.start()
    
    def clear_results(self):
        """Clear all search results from the UI"""
        # Clear the overview
        self.overview_text.clear()
        
        # Clear all result cards
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def show_result_detail(self, source, content):
        """Show the detail dialog for a search result"""
        dialog = ResultDetailDialog(source, content, self)
        dialog.exec()
    
    def parse_and_display_results(self, result_text):
        """Parse the result text and display it in the UI"""
        # Store the raw result for debugging
        self.raw_search_result = result_text
        
        # Parse the sections
        if "## AI Summary" in result_text and "## Retrieved Chunks" in result_text:
            # Extract AI Summary section
            ai_summary_start = result_text.find("## AI Summary") + len("## AI Summary")
            ai_summary_end = result_text.find("## Retrieved Chunks")
            ai_summary = result_text[ai_summary_start:ai_summary_end].strip()
            
            # Display the AI summary
            self.overview_text.setPlainText(ai_summary)
            
            # Extract Retrieved Chunks section
            chunks_section = result_text[ai_summary_end + len("## Retrieved Chunks"):].strip()
            
            # Process each chunk
            chunk_parts = chunks_section.split("Chunk ")
            chunks = []
            
            for part in chunk_parts:
                if not part.strip():
                    continue
                
                # Extract chunk details
                try:
                    # Find source
                    source_start = part.find("Source: ") + len("Source: ")
                    source_end = part.find("\n", source_start)
                    source = part[source_start:source_end].strip()
                    
                    # Find content
                    content_start = part.find("Content: ") + len("Content: ")
                    content = part[content_start:].strip()
                    
                    # Try to extract score if available
                    score = None
                    if "Score: " in part:
                        score_start = part.find("Score: ") + len("Score: ")
                        score_end = part.find("\n", score_start)
                        try:
                            score = float(part[score_start:score_end].strip())
                        except ValueError:
                            pass
                    
                    chunks.append({"source": source, "content": content, "score": score})
                except Exception as e:
                    print(f"Error parsing chunk: {e}")
            
            # Display chunks as cards
            for i, chunk in enumerate(chunks):
                card = SearchResultCard(i+1, chunk["source"], chunk["content"], chunk.get("score"))
                card.clicked.connect(self.show_result_detail)
                self.results_layout.addWidget(card)
                
            # Update status
            self.status_label.setText(f"Found {len(chunks)} results.")
        else:
            # Just show the entire result if it doesn't match expected format
            self.overview_text.setPlainText(result_text)
            self.status_label.setText("Search complete.")
    
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
            self.parse_and_display_results(current_result)
            self.search_btn.setEnabled(True)
            self.search_btn.setText("Search")
            self.search_in_progress = False
            self.result_timer.stop()
        else:
            # Still searching, keep polling
            self.status_label.setText("Searching, please wait...")