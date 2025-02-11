# Software Architecture - File Specifications

## 1. Overview
This project is designed to scrape, extract, and process data from a website given from the user input in the chat interface. The software consists of three main files:

- **`crawler.py`** - Performs a filtered search through a website, retrieving additional associated links up to a depth of 3 before stopping. Calls the extractor via Jita.
- **`extractor.py`** - Processes data by extracting relevant information from the retrieved links and converts it into a structured format.
- **`utils.py`** - Contains helper functions used by both `crawler.py` and `extractor.py` to support search and extraction operations.

---

## 2. File Descriptions

### **`crawler.py`**
- **Purpose:** Implements a web crawler that recursively retrieves links.
- **Functionality:**
  - Fetches web page information from user input.
  - Extracts and filters associated links.
  - Limits recursion depth to prevent excessive link gathering.
  - Calls `extractor.py` via Jita for data extraction from links.
- **Inputs:** Seed URL.
- **Outputs:** A collection of filtered links.

### **`extractor.py`**
- **Purpose:** Extracts and processes data from web pages.
- **Functionality:**
  - Processes links provided by `crawler.py`.
  - Parses webpage content.
  - Searches for and extracts relevant information.
- **Inputs:** URLs from `crawler.py`.
- **Outputs:** Extracted data.

### **`utils.py`**
- **Purpose:** Provides helper functions for `crawler.py` and `extractor.py`.
- **Functionality:**
  - Functions for handling HTTP requests.
  - Link validation and filtering.
  - HTML parsing utilities.
  - Logging and debugging tools.

---
## 3. Dependencies
- Python 3.9+
---

## 4. Usage
To run the crawler:
```sh
python3 crawler.py
