# src/llamasearch/ui/app_logic.py

import logging
import sys
import signal
import queue
import threading
from typing import List, Optional, Dict, Any
from pathlib import Path

from killeraiagent.models import LLM, ModelInfo, LlamaCppCLI, HuggingFaceLLM, NullModel
from llamasearch.core.llmsearch import LLMSearch
from llamasearch.core.concurrent_crawler import ConcurrentAsyncCrawler
from llamasearch.utils import setup_logging, get_llamasearch_dir

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
    Main backend logic for the minimal UI. 
    - Manage LLM
    - Manage concurrency BFS crawler
    - Manage LLMSearch instance
    - Provide methods for local file indexing, removing items, etc.
    """
    def __init__(self, requires_gpu=False, debug=False):
        self.requires_gpu = requires_gpu
        self.debug = debug
        self.log_queue = queue.Queue()
        self.all_logs: List[str] = []
        self.signals = None
        self.data_path = get_llamasearch_dir()
        self.start_log_queue_processor()
        self.setup_logging()
        self.model_info = NullModel()
        self.llm = None
        self.llm_search: Optional[LLMSearch] = None
        self.logger.info("LlamaSearchApp init complete.")
        self._init_model()

    def _init_model(self):
        if not self.model_info.model_path.exists():
            self.logger.error(f"Model file not found => {self.model_info.model_path}")
        else:
            self.llm = self._construct_llm(self.model_info, self.requires_gpu)

    def _construct_llm(self, info: ModelInfo, requires_gpu: bool) -> LLM:
        if info.model_engine == "llamacpp":
            return LlamaCppCLI(info, requires_gpu=requires_gpu)
        elif info.model_engine == "hf":
            return HuggingFaceLLM(info, requires_gpu=requires_gpu)
        else:
            self.logger.error(f"Unsupported engine => {info.model_engine}")
            return NullModel()

    def start_log_queue_processor(self):
        def process_queue():
            while True:
                try:
                    msg = self.log_queue.get(timeout=1)
                    self.all_logs.append(msg)
                except queue.Empty:
                    pass
        t = threading.Thread(target=process_queue, daemon=True)
        t.start()

    def setup_logging(self):
        self.logger = setup_logging(__name__)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        qh = QueueHandler(self.log_queue)
        root_logger.addHandler(qh)

    def close(self):
        if self.llm_search:
            self.llm_search.close()
            self.llm_search = None
        if self.llm:
            try:
                self.llm.unload()
            except:
                pass
            self.llm = None

    def __del__(self):
        self.close()

    def signal_handler(self, sig, frame):
        self.logger.info(f"Received signal => {sig}")
        self.close()
        sys.exit(0)

    def get_live_logs(self) -> List[List[str]]:
        return [["system", "\n".join(self.all_logs)]]

    def set_model_config(self, model_id: str, engine: str, custom_path: str) -> bool:
        changed = (
            model_id != self.model_info.model_id or 
            engine != self.model_info.model_engine or 
            custom_path != str(self.model_info.model_path)
        )
        if changed:
            self.logger.info(f"Model config changed => {engine}:{model_id}")
            self.close()
            from pathlib import Path
            self.model_info = ModelInfo(
                model_id=model_id,
                model_engine=engine,
                model_path=Path(custom_path) if custom_path else Path("")
            )
            if self.model_info.model_path and self.model_info.model_path.exists():
                self.llm = self._construct_llm(self.model_info, self.requires_gpu)
                self.logger.info("New model loaded.")
            else:
                self.logger.error("New model path not found.")
                self.llm = None
        return changed

    def get_model_config(self) -> Dict[str,str]:
        return {
            "model_id": self.model_info.model_id,
            "model_engine": self.model_info.model_engine,
            "custom_model_path": str(self.model_info.model_path)
        }

    def get_available_models(self) -> Dict[str,Any]:
        ret = {
            "llamacpp": [
                {"name": f.name, "path": str(f.resolve())}
                for f in Path(self.data_path / "models").glob("*.gguf")
            ],
            "hf": ["google/gemma-3-1b-it"]
        }
        return ret

    def get_llm_search(self) -> LLMSearch:
        if self.llm_search is None:
            if not self.llm:
                raise RuntimeError("No valid LLM.")
            self.llm_search = LLMSearch(
                model=self.llm,
                storage_dir=self.data_path / "index",
                models_dir=self.data_path / "models",
                debug=self.debug
            )
        return self.llm_search

    def do_search(self, query: str) -> str:
        """
        Perform a RAG search with the existing index. 
        """
        if not self.llm:
            return "No model loaded."
        if not query.strip():
            return "Enter a query."

        index_dir = self.data_path / "index"
        if not index_dir.exists() or not any(index_dir.iterdir()):
            return "No index found. Please index data or crawl first."

        try:
            llms = self.get_llm_search()
            result = llms.llm_query(query, debug_mode=self.debug)
            if "formatted_response" in result:
                return result["formatted_response"]
            else:
                return (
                    "## AI Summary\n" + result.get("response","") + 
                    "\n\n## Retrieved Chunks\n" + result.get("retrieved_display","")
                )
        except Exception as e:
            self.logger.error(f"Search error => {e}")
            return f"Error => {e}"

    def index_local_path(self, path_str: str) -> str:
        """
        Index a local file or directory. 
        We'll call llm_search.add_documents_from_directory for directories,
        or else for a single file's parent. 
        """
        from pathlib import Path
        p = Path(path_str)
        if not p.exists():
            return f"Path not found => {path_str}"
        llms = self.get_llm_search()
        if p.is_dir():
            new_chunks = llms.add_documents_from_directory(p)
            return f"Indexed directory => {path_str}, added {new_chunks} chunk(s)."
        else:
            # single file => just pass parent
            new_chunks = llms.add_documents_from_directory(p.parent)
            return f"Indexed file => {path_str}, added {new_chunks} chunk(s)."

    def multi_crawl(self,
                    root_urls: List[str],
                    target_links: int=10,
                    max_depth:int=2,
                    api_type:str="jina",
                    phrase: Optional[str]=None,
                    key_id: Optional[str]=None,
                    private_key: Optional[str]=None) -> str:
        """
        Perform concurrent BFS of multiple root URLs in a background thread. 
        We'll index each fetched page as we go.
        """
        if not root_urls:
            return "No root URLs provided."

        # Spawn background thread so as not to block the UI
        def crawl_thread():
            self.logger.info(f"[multi_crawl] => root={root_urls}, links={target_links}, depth={max_depth}, phrase={phrase}")
            # We'll define a callback that is invoked whenever we fetch a page
            # We'll store it as .md in raw and call add_document on it
            from .concurrent_crawler import ConcurrentAsyncCrawler
            import asyncio
            import os
            from datetime import datetime
            from pathlib import Path

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # We'll store saved .md in self.data_path["crawl_data"]/raw
            # We'll define a function that saves content, calls add_document
            raw_dir = self.data_path / "crawl_data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            llms = self.get_llm_search()

            async def on_fetch_page(url:str, content:str):
                # create hashed filename
                import hashlib, json
                h = hashlib.sha256(url.encode()).hexdigest()
                mdfile = raw_dir / f"{h}.md"
                # Add metadata
                meta = {
                    "source": url,
                    "extracted_at": datetime.now().isoformat()
                }
                content_to_write = f"<!--\nMETADATA: {json.dumps(meta, indent=2)}\n-->\n\n{content}"
                mdfile.write_text(content_to_write, encoding="utf-8")

                # Now index
                llms.add_document(mdfile)

            crawler = ConcurrentAsyncCrawler(
                root_urls=root_urls,
                target_links=target_links,
                max_depth=max_depth,
                api_type=api_type,
                phrase=phrase,
                key_id=key_id,
                private_key=private_key,
                concurrency=5,
                on_fetch_page=on_fetch_page
            )
            found = loop.run_until_complete(crawler.run_crawl())
            loop.close()
            self.logger.info(f"[multi_crawl] => done. Found {len(found)} links.")
        
        t = threading.Thread(target=crawl_thread, daemon=True)
        t.start()
        return "Multi-crawl started in background."

    def clear_crawl_data(self) -> str:
        try:
            clear_crawl_data_directory()
            return "Crawl data cleared."
        except Exception as e:
            self.logger.error(f"Err clearing => {e}")
            return f"Error => {e}"

    def clear_index_data(self) -> str:
        idx_dir = self.data_path / "index"
        import shutil
        try:
            shutil.rmtree(idx_dir, ignore_errors=True)
            idx_dir.mkdir(parents=True, exist_ok=True)
            if self.llm_search:
                self.llm_search.close()
                self.llm_search = None
            return "Index data cleared."
        except Exception as e:
            self.logger.error(f"Err => {e}")
            return f"Error => {e}"

    def remove_item_from_index(self, hash_val: str) -> str:
        """
        Suppose each doc is hashed. We remove that doc from vectordb if found.
        We'll also remove from raw data if user wants. 
        For a minimal approach, let's just remove from vectordb. 
        """
        # We'll do it by searching the reverse lookup table or 
        # searching the metadata for a matching file. 
        # We'll do a simpler approach => search doc with "source" containing the file path that ends with hash_val 
        # In real code, you'd store the mapping from hash->source in a dictionary and do vectordb._remove_document 
        return "Item removal not fully implemented in this minimal example"

    def get_crawl_data_items(self) -> List[Dict[str, str]]:
        """
        Return a list of items in the crawl_data raw folder, so we can show them in the UI.
        E.g. [ {hash: "...", url: "...", path: "..."} ]
        We rely on the reverse_lookup.json to map hash->url
        """
        import json
        rev_path = self.data_path / "crawl_data" / "reverse_lookup.json"
        items = []
        if rev_path.exists():
            d = json.loads(rev_path.read_text(encoding="utf-8"))
            for hval, url in d.items():
                items.append({"hash": hval, "url": url, "path": str(self.data_path / "crawl_data" / "raw" / f"{hval}.md")})
        return items
