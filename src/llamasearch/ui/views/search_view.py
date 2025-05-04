# src/llamasearch/ui/views/search_and_index_view.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QTextEdit, QTableWidget, QTableWidgetItem, QAbstractItemView, QFileDialog,
    QSpinBox, QFormLayout, QRadioButton
)
from PySide6.QtCore import QTimer

class SearchAndIndexView(QWidget):
    """
    A single tab that includes:
     - A query box + search button
     - A multi-line for root URLs
     - config for target links, max depth, phrase, API
     - A button to start multi-crawl
     - A button to pick local files/dirs for indexing
     - A table listing what's in the crawl_data / index, with "Remove" buttons
    """

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()
        self.update_data_display()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Top row: user query
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("Question:"))
        self.query_input = QLineEdit()
        query_layout.addWidget(self.query_input)
        self.search_btn = QPushButton("Search")
        query_layout.addWidget(self.search_btn)
        main_layout.addLayout(query_layout)

        # search results
        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        main_layout.addWidget(self.search_results)

        # Next row: multi-line root URLs
        self.urls_text = QTextEdit()
        self.urls_text.setPlaceholderText("Enter root URLs for crawling, one per line.\nExample:\nhttps://example.com\nhttps://another.org")
        main_layout.addWidget(self.urls_text)

        # form for target links, depth, phrase, API
        form = QFormLayout()
        self.target_links_spin = QSpinBox()
        self.target_links_spin.setRange(1,100)
        self.target_links_spin.setValue(10)
        form.addRow("Max Links:", self.target_links_spin)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1,5)
        self.depth_spin.setValue(2)
        form.addRow("Max Depth:", self.depth_spin)

        self.phrase_input = QLineEdit()
        form.addRow("Phrase:", self.phrase_input)

        # API radio
        self.api_radio_jina = QRadioButton("Jina")
        self.api_radio_mithran = QRadioButton("Mithran")
        self.api_radio_jina.setChecked(True)
        api_hlay = QHBoxLayout()
        api_hlay.addWidget(self.api_radio_jina)
        api_hlay.addWidget(self.api_radio_mithran)
        form.addRow("API Type:", api_hlay)

        main_layout.addLayout(form)

        # multi-crawl button
        self.multicrawl_btn = QPushButton("Start Multi-Crawl")
        main_layout.addWidget(self.multicrawl_btn)

        # local file indexing
        self.index_file_btn = QPushButton("Index Local File")
        self.index_dir_btn = QPushButton("Index Local Directory")
        loc_layout = QHBoxLayout()
        loc_layout.addWidget(self.index_file_btn)
        loc_layout.addWidget(self.index_dir_btn)
        main_layout.addLayout(loc_layout)

        # status
        self.status_label = QLabel("Status: idle.")
        main_layout.addWidget(self.status_label)

        # data table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["Hash","URL","Actions"])
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        main_layout.addWidget(self.data_table)

        # logs
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMaximumHeight(120)
        main_layout.addWidget(self.logs_text)

        # connect signals
        self.search_btn.clicked.connect(self.do_search)
        self.multicrawl_btn.clicked.connect(self.do_multi_crawl)
        self.index_file_btn.clicked.connect(self.do_index_file)
        self.index_dir_btn.clicked.connect(self.do_index_dir)

        # periodic logs + data refresh
        self.timer = QTimer()
        self.timer.timeout.connect(self.periodic_refresh)
        self.timer.start(3000)

    def do_search(self):
        query = self.query_input.text().strip()
        if not query:
            self.search_results.setPlainText("Please enter a query.")
            return
        res = self.backend.do_search(query)
        self.search_results.setPlainText(res)
        self.update_data_display()

    def do_multi_crawl(self):
        lines = self.urls_text.toPlainText().splitlines()
        lines = [ln.strip() for ln in lines if ln.strip()]
        if not lines:
            self.status_label.setText("No root URLs provided.")
            return
        tlinks = self.target_links_spin.value()
        md = self.depth_spin.value()
        phr = self.phrase_input.text().strip()
        api_t = "mithran" if self.api_radio_mithran.isChecked() else "jina"
        msg = self.backend.multi_crawl(lines, tlinks, md, api_t, phr)
        self.status_label.setText(msg)

    def do_index_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select File to Index", str(self.backend.data_paths["base"]))
        if fp:
            msg = self.backend.index_local_path(fp)
            self.status_label.setText(msg)
            self.update_data_display()

    def do_index_dir(self):
        dp = QFileDialog.getExistingDirectory(self, "Select Directory to Index", str(self.backend.data_paths["base"]))
        if dp:
            msg = self.backend.index_local_path(dp)
            self.status_label.setText(msg)
            self.update_data_display()

    def update_data_display(self):
        items = self.backend.get_crawl_data_items()
        self.data_table.setRowCount(len(items))
        for i, it in enumerate(items):
            hval = it["hash"]
            url = it["url"]
            self.data_table.setItem(i, 0, QTableWidgetItem(hval))
            self.data_table.setItem(i, 1, QTableWidgetItem(url))
            # actions
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda checked, hv=hval: self.remove_item(hv))
            self.data_table.setCellWidget(i, 2, remove_btn)

    def remove_item(self, hash_val: str):
        msg = self.backend.remove_item_from_index(hash_val)
        self.status_label.setText(msg)
        self.update_data_display()

    def periodic_refresh(self):
        logs = self.backend.get_live_logs()
        if logs:
            self.logs_text.setPlainText(logs[0][1])