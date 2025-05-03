# src/llamasearch/ui/views/crawl_view.py
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSlider, 
                              QRadioButton, QCheckBox, QPushButton, QTextEdit, QFormLayout, 
                              QTabWidget, QGroupBox, QSplitter, QFrame)
from PySide6.QtCore import Qt, QTimer

class CrawlView(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()
        
        # Set up timer to refresh stats regularly
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats_display)
        self.stats_timer.start(5000)  # Update every 5 seconds
        
        # Initial stats update
        self.update_stats_display()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Create a splitter to divide the screen horizontally
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left side - Crawl controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        splitter.addWidget(left_widget)
        
        # Right side - Stats display
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)
        
        # ===== LEFT SIDE - CRAWL CONTROLS =====
        # Crawl parameters using a form layout
        form = QFormLayout()
        self.url_input = QLineEdit("https://example.com")
        form.addRow("Website URL:", self.url_input)
        
        self.target_links_slider = QSlider(Qt.Orientation.Horizontal)
        self.target_links_slider.setMinimum(5)
        self.target_links_slider.setMaximum(100)
        self.target_links_slider.setValue(15)
        self.target_links_value = QLabel("15")
        self.target_links_slider.valueChanged.connect(lambda v: self.target_links_value.setText(str(v)))
        
        target_links_layout = QHBoxLayout()
        target_links_layout.addWidget(self.target_links_slider)
        target_links_layout.addWidget(self.target_links_value)
        form.addRow("Max Links to Collect:", target_links_layout)
        
        self.max_depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_depth_slider.setMinimum(1)
        self.max_depth_slider.setMaximum(5)
        self.max_depth_slider.setValue(2)
        self.max_depth_value = QLabel("2")
        self.max_depth_slider.valueChanged.connect(lambda v: self.max_depth_value.setText(str(v)))
        
        max_depth_layout = QHBoxLayout()
        max_depth_layout.addWidget(self.max_depth_slider)
        max_depth_layout.addWidget(self.max_depth_value)
        form.addRow("Max Crawl Depth:", max_depth_layout)
        
        # API type radio buttons
        self.api_radio_public = QRadioButton("Jina API (Public)")
        self.api_radio_private = QRadioButton("Mithran API (Private)")
        self.api_radio_public.setChecked(True)
        api_layout = QHBoxLayout()
        api_layout.addWidget(self.api_radio_public)
        api_layout.addWidget(self.api_radio_private)
        form.addRow("API Type:", api_layout)
        
        # Mithran options (initially hidden unless private API is selected)
        self.key_id_label = QLabel("API Key ID:")
        self.key_id_input = QLineEdit("your-key-id")
        self.private_key_label = QLabel("RSA Private Key Path:")
        self.private_key_input = QLineEdit("~/.ssh/id_rsa")
        
        # Initially hide Mithran options
        self.key_id_label.setVisible(False)
        self.key_id_input.setVisible(False)
        self.private_key_label.setVisible(False)
        self.private_key_input.setVisible(False)
        
        form.addRow(self.key_id_label, self.key_id_input)
        form.addRow(self.private_key_label, self.private_key_input)
        
        # Toggle mithran options based on API type
        self.api_radio_private.toggled.connect(self.toggle_mithran_options)
        
        # Checkbox for CPU-optimized LLM
        self.cpu_checkbox = QCheckBox("Use CPU-optimized LLM")
        form.addRow(self.cpu_checkbox)
        
        left_layout.addLayout(form)
        
        # Create a row of action buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Crawling")
        self.check_status_btn = QPushButton("Check Status")
        self.clear_crawl_btn = QPushButton("Clear Crawl Data")
        self.clear_index_btn = QPushButton("Clear Index")
        self.show_logs_btn = QPushButton("Show Logs")
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.check_status_btn)
        btn_row.addWidget(self.clear_crawl_btn)
        btn_row.addWidget(self.clear_index_btn)
        btn_row.addWidget(self.show_logs_btn)
        left_layout.addLayout(btn_row)
        
        # Status display label
        self.status_label = QLabel("Ready to crawl. Enter a URL and click 'Start Crawling'.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        # Logs display (read-only)
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMaximumHeight(150)  # Limit height to prevent taking up too much space
        left_layout.addWidget(self.logs_text)
        
        # ===== RIGHT SIDE - STATS DISPLAY =====
        # Stats display title
        right_layout.addWidget(QLabel("<h3>System Status and Stats</h3>"))
        
        # Index stats
        index_group = QGroupBox("Index Statistics")
        index_layout = QVBoxLayout(index_group)
        self.index_stats_text = QTextEdit()
        self.index_stats_text.setReadOnly(True)
        self.index_stats_text.setMaximumHeight(150)
        index_layout.addWidget(self.index_stats_text)
        right_layout.addWidget(index_group)
        
        # Crawl data stats
        crawl_group = QGroupBox("Crawl Data")
        crawl_layout = QVBoxLayout(crawl_group)
        self.crawl_stats_text = QTextEdit()
        self.crawl_stats_text.setReadOnly(True)
        crawl_layout.addWidget(self.crawl_stats_text)
        right_layout.addWidget(crawl_group)
        
        # Set splitter sizes (left:right ratio of 60:40)
        splitter.setSizes([600, 400])
        
        # Connect buttons to callback methods
        self.start_btn.clicked.connect(self.start_crawl)
        self.check_status_btn.clicked.connect(self.check_status)
        self.clear_crawl_btn.clicked.connect(self.clear_crawl)
        self.clear_index_btn.clicked.connect(self.clear_index)
        self.show_logs_btn.clicked.connect(self.show_logs)
    
    def toggle_mithran_options(self, checked):
        """Show/hide Mithran API options based on radio button selection"""
        self.key_id_label.setVisible(checked)
        self.key_id_input.setVisible(checked)
        self.private_key_label.setVisible(checked)
        self.private_key_input.setVisible(checked)
    
    def update_stats_display(self):
        """Update the stats display with the latest information"""
        try:
            # Update index stats
            index_stats = self.backend.get_index_stats()
            index_text = (
                f"Documents: {index_stats['doc_count']}\n"
                f"Chunks: {index_stats['chunk_count']}\n"
                f"Files: {index_stats['file_count']}\n"
                f"Size: {index_stats['size_mb']} MB\n"
            )
            self.index_stats_text.setPlainText(index_text)
            
            # Update crawl data stats
            crawl_stats = self.backend.get_crawl_data_stats()
            lookup = self.backend.get_crawl_data_lookup()
            
            crawl_text = (
                f"Files: {crawl_stats['file_count']}\n"
                f"Size: {crawl_stats['size_mb']} MB\n\n"
                f"Crawled URLs:\n"
            )
            
            # Add reverse lookup info (convert hashes to URLs)
            if lookup:
                for hash_val, url in lookup.items():
                    crawl_text += f"• {url}\n"
            else:
                crawl_text += "No crawl data available yet."
                
            self.crawl_stats_text.setPlainText(crawl_text)
        except Exception as e:
            self.logs_text.append(f"Error updating stats: {str(e)}")
    
    def start_crawl(self):
        url = self.url_input.text()
        target_links = self.target_links_slider.value()
        max_depth = self.max_depth_slider.value()
        api_type = "mithran" if self.api_radio_private.isChecked() else "jina"
        key_id = self.key_id_input.text() if api_type == "mithran" else None
        private_key = self.private_key_input.text() if api_type == "mithran" else None
        # Update backend CPU flag based on checkbox
        self.backend.use_cpu = self.cpu_checkbox.isChecked()
        result = self.backend.crawl_website(url, target_links, max_depth, api_type, key_id, private_key)
        self.status_label.setText(result)
        # Update stats immediately
        self.update_stats_display()
    
    def check_status(self):
        status = self.backend.check_crawl_status()
        self.status_label.setText(status)
        # Update stats
        self.update_stats_display()
    
    def clear_crawl(self):
        status = self.backend.clear_crawl_data()
        self.status_label.setText(status)
        # Update stats
        self.update_stats_display()
    
    def clear_index(self):
        status = self.backend.clear_index_data()
        self.status_label.setText(status)
        # Update stats
        self.update_stats_display()
    
    def show_logs(self):
        logs = self.backend.get_live_logs()
        if logs:
            self.logs_text.setPlainText(logs[0][1])