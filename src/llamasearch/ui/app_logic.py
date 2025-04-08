# src/llamasearch/ui/app_logic.py
import threading
import queue
import shutil
import atexit
import signal
import sys
import json
from typing import Dict, Any, Optional, Tuple, List
import logging
import time

from llamasearch.setup_utils import get_data_paths
from llamasearch.core.crawler import smart_crawl, clear_crawl_data_directory
from llamasearch.core.llm import LLMSearch  # unified RAG + LLM code

# A custom logging handler that writes messages to a queue.
class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

class LlamaSearchApp:
    """
    Main application class for the LlamaSearch UI.
    """
    def __init__(self, use_cpu: bool = False, debug: bool = False, signals=None):
        self.use_cpu = use_cpu
        self.debug = debug
        self.signals = signals  # Store signals object
        self.data_paths = get_data_paths()
        self.llm_instance = None
        self.current_crawl_thread = None
        self.crawl_status = "Ready to crawl"
        self.log_queue = queue.Queue()
        self.all_logs: List[str] = []  # Accumulate log messages
        self.setup_logging()
        self.start_log_queue_processor()
        # Register signal handlers only in the main thread.
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Store model configuration
        self.model_name = "qwen2.5-1.5b-instruct-q4_k_m"  # Default model
        self.model_engine = "llamacpp"  # Default engine (llamacpp or hf)
        self.custom_model_path = ""
    
    def setup_logging(self) -> None:
        # Configure root logger first to capture all logs
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicates
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
        
        # Add our queue handler to the root logger
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(queue_handler)
        
        # Now set up the local logger as before
        self.logger = logging.getLogger("llamasearch.ui")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Also log to file
        file_handler = logging.FileHandler(self.data_paths["logs"] / "ui.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
    
    def start_log_queue_processor(self) -> None:
        """Start a background thread to drain the log queue into a list."""
        def process_queue() -> None:
            while True:
                try:
                    msg = self.log_queue.get(timeout=1)
                    self.all_logs.append(msg)
                except queue.Empty:
                    pass
        t = threading.Thread(target=process_queue, daemon=True)
        t.start()
    
    def signal_handler(self, sig, frame) -> None:
        self.logger.info(f"Received signal {sig}, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self) -> None:
        if self.llm_instance is not None:
            self.logger.info("Cleaning up LLM resources...")
            self.llm_instance.unload_model()
            self.llm_instance = None
    
    def get_llm(self) -> LLMSearch:
        if self.llm_instance is not None:
            return self.llm_instance
        self.logger.info(f"Initializing LLMSearch with model_engine='{self.model_engine}' and model_name='{self.model_name}'")
        self.llm_instance = LLMSearch(
            storage_dir=self.data_paths["index"],
            models_dir=self.data_paths["models"],
            model_name=self.model_name,
            model_engine=self.model_engine,
            custom_model_path=self.custom_model_path,
            force_cpu=self.use_cpu,
            debug=self.debug
        )
        return self.llm_instance

    def set_model_config(self, model_name: str, model_engine: str, custom_model_path: str = "") -> bool:
        """
        Update model configuration. Returns True if configuration changed.
        """
        changed = False
        if model_name != self.model_name or model_engine != self.model_engine or custom_model_path != self.custom_model_path:
            changed = True
            
        self.model_name = model_name
        self.model_engine = model_engine
        self.custom_model_path = custom_model_path
        
        if changed and self.llm_instance is not None:
            # Unload existing model to apply new configuration
            self.llm_instance.unload_model()
            self.llm_instance = None
            self.logger.info(f"Model configuration changed to {model_engine}:{model_name}")
            
        return changed

    def get_model_config(self) -> Dict[str, str]:
        """
        Get current model configuration
        """
        return {
            "model_name": self.model_name,
            "model_engine": self.model_engine,
            "custom_model_path": self.custom_model_path
        }

    def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models
        """
        try:
            llm = self.get_llm()
            return llm.get_available_models()
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return {"local_models": [], "huggingface_suggestions": []}

    def get_live_logs(self) -> List[List[str]]:
        """
        Return the accumulated logs as a list suitable for display.
        Format: [["system", "<all logs concatenated>"]]
        """
        return [["system", "\n".join(self.all_logs)]]

    def get_index_stats(self) -> Dict[str, Any]:
        index_dir = self.data_paths["index"]
        if not index_dir.exists():
            return {"doc_count": 0, "chunk_count": 0, "size_mb": 0}
        file_count = sum(1 for _ in index_dir.glob('**/*') if _.is_file())
        total_size = sum(f.stat().st_size for f in index_dir.glob('**/*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        doc_count = 0
        chunk_count = 0
        llm = self.get_llm()
        if hasattr(llm, 'vectordb') and hasattr(llm.vectordb, 'document_metadata'):
            doc_count = len(set(m.get('source', '') for m in llm.vectordb.document_metadata))
            chunk_count = len(llm.vectordb.document_metadata)
        return {
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "file_count": file_count,
            "size_mb": round(size_mb, 2)
        }
    
    def get_crawl_data_stats(self) -> Dict[str, Any]:
        raw_dir = self.data_paths["crawl_data"] / "raw"
        file_count = sum(1 for f in raw_dir.glob('**/*') if f.is_file())
        total_size = sum(f.stat().st_size for f in raw_dir.glob('**/*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        return {"file_count": file_count, "size_mb": round(size_mb, 2)}
    
    def get_crawl_data_lookup(self) -> Dict[str, Any]:
        lookup_path = self.data_paths["crawl_data"] / "reverse_lookup.json"
        try:
            with open(lookup_path, "r", encoding="utf-8") as f:
                lookup = json.load(f)
            return lookup
        except Exception as e:
            self.logger.error(f"Error reading reverse lookup table: {e}")
            return {}
    
    def get_crawl_data_info(self) -> Tuple[str, str]:
        stats = self.get_crawl_data_stats()
        lookup = self.get_crawl_data_lookup()
        stats_markdown = f"""
### Crawl Data Statistics:
- Files in raw data: {stats['file_count']}
- Total size: {stats['size_mb']} MB
"""
        if lookup:
            lookup_markdown = "### Reverse Lookup Table:\n" + "\n".join(
                [f"- **{k}**: {v}" for k, v in lookup.items()]
            )
        else:
            lookup_markdown = "### Reverse Lookup Table:\nNo crawl data available."
        return stats_markdown, lookup_markdown

    def clear_crawl_data(self) -> str:
        try:
            self.logger.info("Clearing crawl data...")
            clear_crawl_data_directory()
            self.logger.info("Crawl data cleared successfully.")
            return "Crawl data cleared successfully."
        except Exception as e:
            self.logger.error(f"Error clearing crawl data: {e}")
            return f"Error clearing crawl data: {e}"
    
    def clear_index_data(self) -> str:
        try:
            self.logger.info("Clearing index data...")
            index_dir = self.data_paths["index"]
            shutil.rmtree(index_dir, ignore_errors=True)
            index_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Index data cleared successfully.")
            return "Index data cleared successfully."
        except Exception as e:
            self.logger.error(f"Error clearing index data: {e}")
            return f"Error clearing index data: {e}"
    
    def crawl_website(
        self,
        url: str,
        target_links: int = 10,
        max_depth: int = 2,
        api_type: str = "jina",
        key_id: Optional[str] = None,
        private_key_path: Optional[str] = None
    ) -> str:
        if not url:
            return "Error: Please provide a valid URL."
        self.crawl_status = f"Starting crawl of {url}..."
        self.logger.info(f"Starting crawl of {url} (target_links={target_links}, depth={max_depth}, api={api_type})")
        
        # Emit crawl_started signal if signals are available
        if self.signals:
            self.signals.crawl_started.emit()
        
        def crawl_thread() -> None:
            try:
                self.crawl_status = (
                    f"Crawling {url}... (Target: {target_links} links, Depth: {max_depth}, API: {api_type})"
                )
                links = smart_crawl(
                    start_url=url,
                    target_links=target_links,
                    max_depth=max_depth,
                    api_type=api_type,
                    private_key_path=private_key_path,
                    key_id=key_id
                )
                if links:
                    self.logger.info(f"Crawl complete. {len(links)} docs added to raw data.")
                    llm = self.get_llm()
                    raw_dir = self.data_paths["crawl_data"] / "raw"
                    num_chunks = llm.add_documents_from_directory(str(raw_dir))
                    stats = self.get_index_stats()
                    self.crawl_status = (
                        f"✅ Crawl complete for {url}\n\n"
                        f"• Added {len(links)} new pages to the database\n"
                        f"• Indexed {num_chunks} chunks from these pages\n\n"
                        f"Current index:\n"
                        f"  {stats['doc_count']} docs\n"
                        f"  {stats['chunk_count']} chunks\n"
                        f"  {stats['size_mb']} MB data\n\n"
                        "You can now search or crawl more sites."
                    )
                else:
                    self.crawl_status = "⚠️ Crawl failed: No links were collected."
                    self.logger.warning("Crawl failed: No links were collected.")
                
                # Emit crawl_finished signal if signals are available
                if self.signals:
                    self.signals.crawl_finished.emit()
                    
            except Exception as e:
                self.crawl_status = f"❌ Crawl error: {str(e)}"
                self.logger.error(f"Crawl error: {str(e)}")
                
                # Emit crawl_finished signal if signals are available
                if self.signals:
                    self.signals.crawl_finished.emit()
        
        t = threading.Thread(target=crawl_thread, daemon=True)
        t.start()
        self.current_crawl_thread = t
        return "Crawl started in the background. See logs for updates."
    
    def check_crawl_status(self) -> str:
        return self.crawl_status
    
    def search_content(self, query: str) -> str:
        if not query.strip():
            return "Please enter a search query."
        
        # Emit search_started signal if signals are available
        if self.signals:
            self.signals.search_started.emit()
        
        try:
            llm = self.get_llm()
            index_dir = self.data_paths["index"]
            if not index_dir.exists() or not any(index_dir.iterdir()):
                # Emit search_finished signal if no content found
                if self.signals:
                    self.signals.search_finished.emit()
                return "No indexed content found. Please crawl a website first."
            
            self.logger.info(f"Searching for: {query}")
            
            def search_thread():
                try:
                    result = llm.llm_query(query, debug_mode=self.debug)
                    
                    # Get the response - prefer formatted_response if available
                    if "formatted_response" in result:
                        response_text = result.get("formatted_response", "No response generated.")
                    else:
                        # Fall back to original format
                        response_text = (
                            "## AI Summary\n"
                            f"{result.get('response', 'No response generated.')}\n\n"
                            "## Retrieved Chunks\n"
                            f"{result.get('retrieved_display', '')}"
                        )
                    
                    # Update the results in the main thread
                    self.search_result = response_text
                    
                    # Emit search_finished signal after search completes
                    if self.signals:
                        self.signals.search_finished.emit()
                        
                except Exception as e:
                    self.logger.error(f"Error during search thread: {e}")
                    self.search_result = f"Error during search: {str(e)}"
                    
                    # Emit search_finished signal if error occurs
                    if self.signals:
                        self.signals.search_finished.emit()
            
            # Store initial value
            self.search_result = "Searching, please wait..."
            
            # Start search in background thread
            t = threading.Thread(target=search_thread, daemon=True)
            t.start()
            
            # Return immediately with initial message
            return self.search_result
            
        except Exception as e:
            self.logger.error(f"Error setting up search: {e}")
            
            # Emit search_finished signal if error occurs
            if self.signals:
                self.signals.search_finished.emit()
                
            return f"Error during search: {str(e)}"
            
    def get_search_result(self) -> str:
        """Get the latest search result"""
        if hasattr(self, 'search_result'):
            return self.search_result
        return "No search has been performed yet."