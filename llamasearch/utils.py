import os
import json
import logging
from datetime import datetime
from pathlib import Path

def find_project_root():
    """Finds the root of the project by looking for `pyproject.toml`."""
    current_dir = os.path.abspath(os.path.dirname(__file__))

    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "setup.py")):
            return current_dir

        current_dir = os.path.dirname(current_dir)

    raise RuntimeError(
        "Could not find project root. Please check your project structure."
    )

def setup_logging(name, level=logging.INFO):
    """
    Set up logging to both console and file.
    
    Args:
        name: Logger name, typically __name__ from the calling module
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logs directory in project root if it doesn't exist
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler with a dynamic filename based on date
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"{name.split('.')[-1]}_{today}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Create console handler for critical and error logs only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_query(query, context_chunks, response, debug_info=None):
    """
    Log query, retrieved chunks, and generated response to a JSON file.
    
    Args:
        query: User query string
        context_chunks: List of chunks retrieved from vector DB
        response: Generated response string
        debug_info: Optional debug information dictionary
    """
    project_root = find_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"query_{timestamp}.json")
    
    # Prepare the log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "context_chunks": context_chunks,
        "response": response
    }
    
    # Add debug info if available
    if debug_info:
        log_data["debug_info"] = debug_info
    
    # Write to file
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    return log_file