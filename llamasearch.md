# Snapshot

## Filesystem Tree

```
llamasearch/
├── src/
│   └── llamasearch/
│       ├── core/
│       │   ├── __init__.py
│       │   ├── bm25.py
│       │   ├── chunker.py
│       │   ├── crawler.py
│       │   ├── embedder.py
│       │   ├── llmsearch.py
│       │   ├── teapot.py
│       │   └── vectordb.py
│       ├── ui/
│       │   ├── views/
│       │   │   ├── search_view.py
│       │   │   ├── settings_view.py
│       │   │   └── terminal_view.py
│       │   ├── __init__.py
│       │   ├── app_logic.py
│       │   ├── components.py
│       │   ├── main.py
│       │   └── qt_logging.py
│       ├── __init__.py
│       ├── __main__.py
│       ├── data_manager.py
│       ├── exceptions.py
│       ├── hardware.py
│       ├── protocols.py
│       ├── setup.py
│       └── utils.py
└── tests/
    └── __init__.py
```

## File Contents

Files are ordered alphabetically by path.

### File: src\llamasearch\__init__.py

```python
"""LlamaSearch - RAG-based search application."""

__version__ = "0.1.0"
```

---
### File: src\llamasearch\__main__.py

```python
#!/usr/bin/env python3
"""
__main__.py - Consolidated entry point for LlamaSearch (runtime).

Handles GUI, MCP, and CLI modes for crawling, indexing, searching, etc.
Requires models to be present. Run `llamasearch-setup` first if needed.
"""

import argparse
import sys
from pathlib import Path
import asyncio
import tarfile
import logging
import json

# Import GUI, MCP, and app logic modules:
from llamasearch.utils import setup_logging

# --- Use the revised Crawl4AICrawler ---
from llamasearch.core.crawler import Crawl4AICrawler

from llamasearch.core.llmsearch import LLMSearch
from llamasearch.exceptions import ModelNotFoundError
from llamasearch.data_manager import data_manager

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LlamaSearch: Crawl, index, and search documents. Requires setup (`llamasearch-setup`)."
    )
    parser.add_argument(
        "--mode", choices=["gui", "mcp", "cli"], default="gui", help="Run mode"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(
        dest="command", help="CLI subcommands (if mode==cli)"
    )

    # --- Updated CRAWL PARSER ---
    crawl_parser = subparsers.add_parser(
        "crawl", help="Crawl website using Crawl4AI (custom priority)"
    )
    crawl_parser.add_argument("--url", type=str, required=True, help="Root URL")
    crawl_parser.add_argument("--target-links", type=int, default=15, help="Max unique pages to save")
    crawl_parser.add_argument("--max-depth", type=int, default=2, help="Max crawl depth")
    # Removed --phrase and API args
    # Added optional keywords arg for relevance scoring
    crawl_parser.add_argument(
        "--keywords", type=str, nargs='*', help="Optional keywords to guide crawl relevance (space-separated)"
    )
    crawl_parser.add_argument(
        "--output-dir", type=str, help="Base directory for crawl data (defaults to settings 'crawl_data')"
    )
    # --- END CRAWL PARSER MODIFICATION ---

    # Build-index subcommand (no changes needed here)
    build_index_parser = subparsers.add_parser(
        "build-index", help="Build index from crawl data" # Updated help slightly
    )
    # --- MODIFIED build-index input: Point to base crawl dir ---
    build_index_parser.add_argument(
        "--crawl-data-dir",
        type=str,
        help="Path to crawl data directory containing 'raw' subdir and 'reverse_lookup.json' (defaults to settings 'crawl_data')",
    )
    # --- END MODIFICATION ---
    build_index_parser.add_argument(
        "--index-dir", type=str, help="Store index data here (defaults to settings)"
    )

    # Search subcommand (no changes needed here)
    search_parser = subparsers.add_parser("search", help="Query the index")
    search_parser.add_argument("--query", type=str, required=True, help="Search query")
    search_parser.add_argument(
        "--generate", action="store_true", default=True, help="Generate AI response"
    )
    search_parser.add_argument(
        "--index-dir", type=str, help="Index data directory (defaults to settings)"
    )

    # Export-index subcommand (no changes needed here)
    export_index_parser = subparsers.add_parser(
        "export-index", help="Export index as tar.gz"
    )
    export_index_parser.add_argument(
        "--output", type=str, required=True, help="Output .tar.gz filename"
    )
    export_index_parser.add_argument(
        "--index-dir", type=str, help="Index directory to export (defaults to settings)"
    )

    # Set storage directory subcommand (no changes needed here)
    set_parser = subparsers.add_parser("set", help="Set a storage directory")
    set_parser.add_argument(
        "--key",
        type=str,
        required=True,
        choices=["crawl_data", "index", "models", "logs"],
        help="Directory key",
    )
    set_parser.add_argument("--path", type=str, required=True, help="New path")

    args = parser.parse_args()

    data_paths = data_manager.get_data_paths()

    # Determine crawl data base directory
    crawl_dir_base_path_str = data_paths["crawl_data"]
    if args.command == "crawl" and args.output_dir:
        crawl_dir_base_path_str = args.output_dir
    elif args.command == "build-index" and args.crawl_data_dir:
         crawl_dir_base_path_str = args.crawl_data_dir # Use crawl_data_dir for build-index source
    crawl_dir_base = Path(crawl_dir_base_path_str)

    # Determine index base directory
    index_dir_base_path_str = data_paths["index"]
    if args.command in ["build-index", "search", "export-index"] and args.index_dir:
        index_dir_base_path_str = args.index_dir
    index_dir_base = Path(index_dir_base_path_str)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
        logger.debug(
            f"Crawl Dir Base: {crawl_dir_base}, Index Dir Base: {index_dir_base}"
        )

    # --- Mode Handling ---
    if args.mode == "gui":
        try:
            from llamasearch.ui.main import main as gui_main

            logger.info("Starting GUI mode...")
            gui_main()
        except ModelNotFoundError as e:
            logger.error(f"GUI Error: {e}")
            logger.error("Please run 'llamasearch-setup' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to start GUI: {e}", exc_info=args.debug)
            sys.exit(1)

    elif args.mode == "mcp":
        logger.warning("MCP mode implementation pending.")
        sys.exit(1)

    elif args.mode == "cli":
        llmsearch_instance = None
        try:
            if args.command == "crawl":
                try:
                    # --- Use Revised Crawl4AICrawler ---
                    logger.info("Starting crawl with Crawl4AI (custom priority)...")
                    crawler = Crawl4AICrawler(
                        root_urls=[args.url],
                        base_crawl_dir=crawl_dir_base,
                        target_links=args.target_links,
                        max_depth=args.max_depth,
                        relevance_keywords=args.keywords # Pass optional keywords
                    )
                    # run_crawl is now async
                    collected_urls = asyncio.run(crawler.run_crawl())
                    # --- REMOVED ARCHIVE EXPORT ---
                    # archive_path = crawler.export_archive() # This method no longer exists
                    logger.info(f"Crawl completed. Collected {len(collected_urls)} pages.")
                    # --- End Crawl4AICrawler usage ---
                except Exception as e:
                    logger.error(f"Crawl error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command in ["build-index", "search"]:
                # Logic remains the same, but depends on the crawl output format
                # LLMSearch initialization handles model checks.
                logger.info(f"Initializing LLMSearch for '{args.command}'...")
                llmsearch_instance = LLMSearch(
                    storage_dir=index_dir_base, debug=args.debug, verbose=args.debug
                )
                logger.info("LLMSearch initialized.")

                if args.command == "build-index":
                    # --- MODIFIED: Build index directly from crawl data dir ---
                    target_dir = crawl_dir_base / "raw" # Index the 'raw' subdir
                    reverse_lookup_file = crawl_dir_base / "reverse_lookup.json"

                    if not target_dir.is_dir():
                         logger.error(f"Crawl 'raw' directory not found: {target_dir}. Did the crawl run?")
                         sys.exit(1)
                    if not reverse_lookup_file.is_file():
                         logger.error(f"Crawl 'reverse_lookup.json' not found in: {crawl_dir_base}")
                         # Optional: Could proceed without it, but source info will be missing
                         sys.exit(1)
                    # --- END MODIFICATION ---

                    logger.info(f"Building index from: {target_dir}")
                    # Ensure add_documents can handle the .md files + reverse_lookup.json structure
                    # We assume LLMSearch/VectorDB is adapted or can handle this structure
                    # TODO: Potentially pass reverse_lookup_file path to add_documents_from_directory if needed
                    added = llmsearch_instance.add_documents_from_directory(target_dir)
                    logger.info(f"Index build finished. Added {added} new chunks.")
                    # No temp dir cleanup needed now

                elif args.command == "search":
                    if not args.generate:
                        logger.warning(
                            "CLI search currently always generates response."
                        )
                    res = llmsearch_instance.llm_query(
                        args.query, debug_mode=args.debug
                    )
                    print("\n--- AI Answer ---")
                    print(res.get("response", "N/A."))
                    if args.debug:
                        print("\n--- Retrieved Context ---")
                        # Use the raw context if available for CLI debug
                        retrieved_context_str = res.get("retrieved_context", "N/A.")
                        # Simple heuristic to check if it's the formatted string or raw chunks
                        if isinstance(retrieved_context_str, str) and retrieved_context_str.startswith("--- Chunk"):
                             print(retrieved_context_str)
                        elif "debug_info" in res and "vector_results" in res["debug_info"]:
                             # If raw context isn't directly in result, reconstruct from debug
                             print("Raw Chunks (from debug_info):")
                             docs = res["debug_info"]["vector_results"].get("documents", [])
                             metas = res["debug_info"]["vector_results"].get("metadatas", [])
                             scores = res["debug_info"]["vector_results"].get("scores", [])
                             for i, doc in enumerate(docs):
                                  meta_str = json.dumps(metas[i]) if i < len(metas) else "{}"
                                  score_str = f"{scores[i]:.4f}" if i < len(scores) else "N/A"
                                  print(f"--- Chunk {i+1} | Score: {score_str} | Meta: {meta_str} ---")
                                  print(doc)
                                  print("-"*(15+len(str(i+1)))) # Separator
                        else:
                             print("Raw retrieved context details not available in debug info.")

                        print("\n--- Debug Info ---")
                        print(json.dumps(res.get("debug_info", {}), indent=2))


            elif args.command == "export-index":
                # No changes needed
                output_path = Path(args.output).resolve()
                if not str(output_path).endswith(".tar.gz"):
                    logger.error("Output must be .tar.gz")
                    sys.exit(1)
                if output_path.exists():
                    logger.error(f"Output exists: {output_path}")
                    sys.exit(1)
                if not index_dir_base.is_dir():
                    logger.error(f"Index dir not found: {index_dir_base}")
                    sys.exit(1)
                try:
                    logger.info(
                        f"Exporting index from {index_dir_base} to {output_path}..."
                    )
                    with tarfile.open(output_path, "w:gz") as tar:
                        # Use arcname to place index contents directly in archive root or specific dir
                        tar.add(str(index_dir_base), arcname=index_dir_base.name)
                    logger.info(f"Index exported to {output_path}")
                except Exception as e:
                    logger.error(f"Export error: {e}", exc_info=args.debug)
                    sys.exit(1)

            elif args.command == "set":
                # No changes needed
                try:
                    data_manager.set_data_path(args.key, args.path)
                    logger.info(f"Set '{args.key}' path to '{args.path}'.")
                except Exception as e:
                    logger.error(f"Set path error: {e}", exc_info=args.debug)
                    sys.exit(1)
            else:
                logger.error("No valid CLI subcommand provided.")
                # Show help for the main parser if no command is given
                parser.print_help()
                sys.exit(1)

        except ModelNotFoundError as e:
            logger.error(f"CLI Error: {e}")
            logger.error("Please run 'llamasearch-setup' first.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected CLI error: {e}", exc_info=args.debug)
            sys.exit(1)
        finally:
            if llmsearch_instance:
                llmsearch_instance.close()
                logger.debug("LLMSearch closed.")

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    try:  # Graceful exit setup
        if not sys.platform.startswith("darwin"):
            import signal

            signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
            signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    except ImportError:
        logger.warning("Signal handling unavailable.")
    except Exception as e:
        logger.warning(f"Signal handler setup error: {e}")
    main()
```

---
### File: src\llamasearch\core\__init__.py

```python

```

---
### File: src\llamasearch\core\bm25.py

```python
# src/llamasearch/core/bm25.py

import os
import json
from typing import Dict, Any, Optional, List, Set

import spacy

# ADDED: Import is_package and custom exception
import spacy.util
from llamasearch.exceptions import ModelNotFoundError

import numpy as np
from rank_bm25 import BM25Okapi

from llamasearch.utils import setup_logging

logger = setup_logging(__name__)


# --- MODIFIED FUNCTION ---
def load_nlp_model() -> spacy.language.Language:
    """
    Loads a spaCy English model after checking if it's installed.
    Attempts 'trf' first, falls back to 'sm'. Raises ModelNotFoundError if unavailable.
    """
    primary_model = "en_core_web_trf"
    fallback_model = "en_core_web_sm"

    # Check primary model
    logger.debug(f"Checking for SpaCy model: {primary_model}")
    if spacy.util.is_package(primary_model):
        try:
            logger.info(f"Loading SpaCy model '{primary_model}'.")
            nlp = spacy.load(primary_model)
            return nlp
        except Exception as e:
            logger.warning(
                f"Found SpaCy model '{primary_model}' but failed to load: {e}. Trying fallback."
            )
    else:
        logger.warning(f"SpaCy model '{primary_model}' not found.")

    # Check fallback model
    logger.debug(f"Checking for SpaCy model: {fallback_model}")
    if spacy.util.is_package(fallback_model):
        try:
            logger.info(f"Loading fallback SpaCy model '{fallback_model}'.")
            nlp = spacy.load(fallback_model)
            return nlp
        except Exception as e:
            logger.error(
                f"Found fallback SpaCy model '{fallback_model}' but failed to load: {e}."
            )
            raise ModelNotFoundError(
                f"SpaCy models '{primary_model}' or '{fallback_model}' could not be loaded. "
                f"Please run 'llamasearch-setup'. Error: {e}"
            ) from e
    else:
        logger.error(
            f"Neither SpaCy model '{primary_model}' nor '{fallback_model}' found."
        )
        raise ModelNotFoundError(
            f"Required SpaCy models ('{primary_model}' or '{fallback_model}') not found. "
            f"Please run 'llamasearch-setup'."
        )


def improved_tokenizer(text: str, nlp: spacy.language.Language) -> List[str]:
    """
    Tokenizes text using spaCy.
    Recognized proper nouns (PROPN) and entities are preserved in their original case.
    All other tokens are lowercased.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.pos_ == "PROPN" or token.ent_type_:
            tokens.append(token.text)
        else:
            tokens.append(token.text.lower())
    return tokens


class BM25Retriever:
    """
    BM25Retriever builds an index over documents and uses BM25Okapi for keyword retrieval.
    Uses spaCy for tokenization, checking for model availability first.
    Supports explicit index building and removal.
    """

    def __init__(self, storage_dir: str, boost_proper_nouns: bool = True) -> None:
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.boost_proper_nouns = boost_proper_nouns
        self.metadata_path = os.path.join(self.storage_dir, "meta.json")

        # --- Load NLP model (now performs check) ---
        try:
            self.nlp = load_nlp_model()
        except ModelNotFoundError as e:
            logger.error(f"Failed to initialize BM25Retriever: {e}")
            raise  # Re-raise to prevent object creation without a model

        self.bm25: Optional[BM25Okapi] = None
        self.term_indices: Dict[str, Set[int]] = {}
        self._index_needs_rebuild = True
        self._load_or_init_index()
        if self._index_needs_rebuild and self.tokenized_corpus:
            self.build_index()

    def _load_or_init_index(self) -> None:
        """Loads existing metadata or initializes empty lists."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.doc_ids = data.get("doc_ids", [])
                self.tokenized_corpus = data.get("tokenized_corpus", [])
                if len(self.documents) != len(self.doc_ids) or len(
                    self.documents
                ) != len(self.tokenized_corpus):
                    logger.error(
                        "Mismatch in loaded BM25 data lengths. Resetting index."
                    )
                    self.documents, self.doc_ids, self.tokenized_corpus = [], [], []
                    self._index_needs_rebuild = True
                elif self.tokenized_corpus:
                    self._index_needs_rebuild = True
                    logger.info(
                        f"Loaded {len(self.documents)} BM25 documents. Index requires rebuild."
                    )
                else:
                    self._index_needs_rebuild = False
                    logger.info("Loaded empty BM25 metadata.")
            except Exception as e:
                logger.error(f"Error loading BM25 metadata: {e}. Resetting index.")
                self.documents, self.doc_ids, self.tokenized_corpus = [], [], []
                self._index_needs_rebuild = True
        else:
            logger.info(
                f"No BM25 metadata found at {self.metadata_path}. Starting new index."
            )
            self.documents, self.doc_ids, self.tokenized_corpus = [], [], []
            self._index_needs_rebuild = False

    def build_index(self) -> None:
        """Explicitly builds/rebuilds the BM25Okapi index and term indices."""
        if not self.tokenized_corpus:
            logger.info("BM25: No documents to index.")
            self.bm25 = None
            self.term_indices = {}
            self._index_needs_rebuild = False
            return

        logger.info(
            f"BM25: Building index for {len(self.tokenized_corpus)} documents..."
        )
        try:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._build_term_indices()
            self._index_needs_rebuild = False
            logger.info("BM25: Index built successfully.")
        except Exception as e:
            logger.error(f"BM25: Failed to build index: {e}", exc_info=True)
            self.bm25 = None
            self._index_needs_rebuild = True

    def is_index_built(self) -> bool:
        """Checks if the BM25 index is built and ready for queries."""
        return self.bm25 is not None and not self._index_needs_rebuild

    def _build_term_indices(self) -> None:
        """Builds a term-to-document index (mapping term to set of doc indices)."""
        self.term_indices = {}
        for idx, tokens in enumerate(self.tokenized_corpus):
            for token in tokens:
                if token not in self.term_indices:
                    self.term_indices[token] = set()
                self.term_indices[token].add(idx)

    def save(self) -> None:
        """Saves the index metadata (documents, doc_ids, tokenized_corpus)."""
        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "tokenized_corpus": self.tokenized_corpus,
        }
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        temp_path = os.path.join(self.storage_dir, "temp_bm25_meta.json")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.metadata_path)
            logger.debug("BM25 metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving BM25 metadata: {e}")
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def add_document(self, text: str, doc_id: str) -> None:
        """Adds a document's text and ID. Does NOT rebuild the index."""
        if doc_id in self.doc_ids:
            logger.warning(
                f"BM25: Document ID '{doc_id}' already exists. Skipping add."
            )
            return

        tokens = improved_tokenizer(text, self.nlp)  # Uses self.nlp checked in __init__
        self.documents.append(text)
        self.doc_ids.append(doc_id)
        self.tokenized_corpus.append(tokens)
        self._index_needs_rebuild = True
        self.save()

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document by ID. Does NOT rebuild the index. Returns True if removed."""
        try:
            idx = self.doc_ids.index(doc_id)
            self.documents.pop(idx)
            self.doc_ids.pop(idx)
            self.tokenized_corpus.pop(idx)
            self._index_needs_rebuild = True
            self.save()
            logger.info(
                f"BM25: Successfully marked document '{doc_id}' for removal. Rebuild index."
            )
            return True
        except ValueError:
            logger.warning(f"BM25: Document ID '{doc_id}' not found for removal.")
            return False
        except Exception as e:
            logger.error(f"BM25: Error removing document '{doc_id}': {e}")
            return False

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Performs a BM25 query. Requires the index to be built."""
        if not self.is_index_built():
            logger.warning("BM25 index not built or needs rebuild. Attempting build...")
            self.build_index()
            if not self.is_index_built():
                logger.error("BM25 index build failed. Returning empty results.")
                return {"query": query_text, "ids": [], "scores": [], "documents": []}
            logger.info("BM25 index was rebuilt before querying.")

        bm25_instance = self.bm25
        assert bm25_instance is not None, (
            "BM25 index should be initialized at this point"
        )

        query_tokens = improved_tokenizer(query_text, self.nlp)  # Uses self.nlp
        try:
            scores = bm25_instance.get_scores(query_tokens)
        except ValueError:
            logger.warning(
                f"BM25 query '{query_text[:50]}...' contains no terms found in index."
            )
            return {"query": query_text, "ids": [], "scores": [], "documents": []}
        except Exception as e:
            logger.error(f"BM25 get_scores error: {e}", exc_info=True)
            return {"query": query_text, "ids": [], "scores": [], "documents": []}

        if self.boost_proper_nouns:
            doc = self.nlp(query_text)  # Uses self.nlp
            proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
            if proper_nouns:
                logger.debug(f"BM25 Boosting query with proper nouns: {proper_nouns}")
                for token in proper_nouns:
                    doc_indices = self.term_indices.get(token, set())
                    for idx in doc_indices:
                        if idx < len(scores):
                            scores[idx] *= 2.0

        actual_n = min(n_results, len(self.documents))
        if actual_n == 0:
            return {"query": query_text, "ids": [], "scores": [], "documents": []}

        if actual_n < len(scores):
            top_indices_unsorted = np.argpartition(scores, -actual_n)[-actual_n:]
            top_indices = top_indices_unsorted[
                np.argsort(scores[top_indices_unsorted])
            ][::-1]
        else:
            top_indices = np.argsort(scores)[::-1]

        final_indices = [idx for idx in top_indices if scores[idx] > 0.0][:n_results]
        top_ids = [self.doc_ids[i] for i in final_indices]
        top_docs = [self.documents[i] for i in final_indices]
        top_scores = [float(scores[i]) for i in final_indices]

        return {
            "query": query_text,
            "ids": top_ids,
            "scores": top_scores,
            "documents": top_docs,
        }
```

---
### File: src\llamasearch\core\chunker.py

```python
# src/killeraiagent/utils/markdown_chunker.py

import re
import logging
from pathlib import Path
from typing import List, Dict, Generator, Optional, Any
import markdown
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

class MarkdownChunker:
    """
    Chunks Markdown text based on structural elements (headings, paragraphs, code blocks)
    and size constraints, suitable for RAG document preparation.
    """
    def __init__(
        self,
        min_chunk_size: int = 150,   # Smaller min size might be okay for doc chunks
        max_chunk_size: int = 1000,  # Allow larger chunks for code blocks etc.
        overlap_percent: float = 0.1, # Overlap based on percentage of max_chunk_size
        combine_under_min_size: bool = True, # Try to combine small consecutive sections
    ):
        if not (0 <= overlap_percent < 1):
            raise ValueError("overlap_percent must be between 0 and 1 (exclusive of 1)")
        if min_chunk_size <= 0 or max_chunk_size <= 0:
            raise ValueError("Chunk sizes must be positive")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size cannot exceed max_chunk_size")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = int(max_chunk_size * overlap_percent)
        self.combine_under_min_size = combine_under_min_size

        # Tags indicating potential section breaks or distinct content blocks
        self.section_break_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'pre', 'table']
        self.content_tags = ['p', 'li', 'code'] # Tags with primary text content

    def _clean_text(self, text: str) -> str:
        """ Basic whitespace normalization. """
        text = re.sub(r'\s+', ' ', text) # Normalize whitespace
        return text.strip()

    def _extract_sections(self, markdown_text: str) -> List[str]:
        """ Extracts text sections based on HTML structure after Markdown conversion. """
        sections = []
        current_section = []

        try:
            # Use 'extra' for tables, fenced code, etc.
            html = markdown.markdown(markdown_text, extensions=['extra', 'fenced_code', 'tables'])
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:
            logger.warning(f"Failed to parse Markdown to HTML: {e}. Falling back to paragraph splitting.")
            # Fallback: split by paragraphs if HTML parsing fails
            paragraphs = re.split(r'\n{2,}', markdown_text)
            sections = [self._clean_text(p) for p in paragraphs if self._clean_text(p)]
            # Skip the HTML processing loop if fallback occurred
            soup = None # Indicate fallback

        # Iterate through all direct children of the body if HTML parsing succeeded
        if soup:
            for element in soup.find_all(True, recursive=False):
                # Ensure it's a Tag object before accessing .name
                if not isinstance(element, Tag):
                    continue

                tag_name = element.name.lower()
                text = self._clean_text(element.get_text())

                if not text:
                    continue
                # Duplicate check removed below

                # If it's a section break tag, finalize the current section (if any)
                # and start a new section with this element.
                if tag_name in self.section_break_tags:
                    if current_section:
                        sections.append(" ".join(current_section))
                        current_section = []
                    sections.append(text) # Add the break tag's content as its own section

                # If it's a primary content tag, add it to the current section
                elif tag_name in self.content_tags or tag_name == 'div': # Include divs as they might contain mixed content
                    current_section.append(text)

                # Handle other tags (like ul, ol directly under body) - extract their text
                elif not current_section and text: # If buffer is empty, start with this text
                    current_section.append(text)

        # Add any remaining content in the buffer
        if current_section:
            sections.append(" ".join(current_section))

        # Combine small consecutive sections if enabled
        if self.combine_under_min_size:
            combined_sections = []
            buffer = []
            buffer_len = 0
            for sec in sections:
                sec_len = len(sec)
                # If buffer is empty OR adding section doesn't exceed max AND buffer is currently too small
                if not buffer or \
                   (buffer_len + sec_len + 1 < self.max_chunk_size and buffer_len < self.min_chunk_size):
                    buffer.append(sec)
                    buffer_len += sec_len + (1 if buffer else 0)
                else:
                    # Yield the buffer (if long enough) and start new buffer
                    if buffer_len >= self.min_chunk_size:
                        combined_sections.append(" ".join(buffer))
                    elif buffer: # If buffer too small but not empty, add it as is
                        combined_sections.append(" ".join(buffer))

                    buffer = [sec]
                    buffer_len = sec_len
            # Add last buffer
            if buffer_len >= self.min_chunk_size:
                 combined_sections.append(" ".join(buffer))
            elif buffer: # Add even if small
                 combined_sections.append(" ".join(buffer))
            sections = combined_sections


        return [sec for sec in sections if sec] # Filter out any empty strings


    def chunk_document(self, markdown_text: str, source: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Chunks a Markdown document into sections, then splits large sections.
        """
        if not markdown_text:
            return

        sections = self._extract_sections(markdown_text)
        chunk_id_counter = 0

        for section in sections:
            section_len = len(section)

            # If the section is within the desired size range
            if section_len <= self.max_chunk_size and section_len >= self.min_chunk_size:
                yield {
                    "chunk": section,
                    "metadata": {"source": source or "unknown", "chunk_id": chunk_id_counter},
                    "embedding_text": section
                }
                chunk_id_counter += 1
            # If the section is smaller than the minimum chunk size (and wasn't combined)
            elif section_len < self.min_chunk_size:
                 # Yield small sections as they are - combining happened earlier if enabled
                 yield {
                     "chunk": section,
                     "metadata": {"source": source or "unknown", "chunk_id": chunk_id_counter},
                     "embedding_text": section
                 }
                 chunk_id_counter += 1
            # If the section is larger than the maximum chunk size, split it
            else:
                start = 0
                while start < section_len:
                    end = min(start + self.max_chunk_size, section_len)

                    # Try to find a natural break point (like sentence end) near the end
                    # Look backwards from 'end' for '.', '!', '?', '\n'
                    best_break = end
                    possible_breaks = [m.start() + 1 for m in re.finditer(r'[.!?\n]\s+', section[max(start, end - self.overlap_size * 2):end])]
                    if possible_breaks:
                        natural_break = max(possible_breaks) # Furthest break point within window
                        if natural_break > start + self.min_chunk_size: # Ensure chunk isn't too small
                            best_break = natural_break

                    chunk_text = section[start:best_break].strip()

                    if len(chunk_text) >= self.min_chunk_size or start == 0: # Ensure first chunk is yielded even if small
                         yield {
                             "chunk": chunk_text,
                             "metadata": {"source": source or "unknown", "chunk_id": chunk_id_counter},
                             "embedding_text": chunk_text
                         }
                         chunk_id_counter += 1
                    elif chunk_text: # If we created a tiny chunk due to splitting logic, log it
                        logger.debug(f"Skipping very small chunk ({len(chunk_text)} chars) created during split of large section in {source}")


                    # Determine next start position with overlap
                    # Ensure overlap doesn't make us go backwards significantly
                    next_start = max(start + self.min_chunk_size, best_break - self.overlap_size)
                    if next_start >= section_len: # Prevent infinite loop if overlap pushes us past the end
                        break
                    start = next_start


    def process_file(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """ Reads a Markdown file and yields chunks. """
        path = Path(file_path)
        if not path.is_file() or path.suffix.lower() != ".md":
            logger.warning(f"Skipping non-markdown file: {file_path}")
            return

        try:
            markdown_text = path.read_text(encoding="utf-8")
            yield from self.chunk_document(markdown_text, source=str(path))
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
```

---
### File: src\llamasearch\core\crawler.py

```python
#!/usr/bin/env python3
"""
crawler.py – Asynchronous crawler using a custom priority queue logic.

Crawls web content based on keyword relevance in scraped markdown and link URLs,
saving files to a central 'raw' directory and maintaining a global 'reverse_lookup.json'.
Includes timeouts and robust queue handling.
"""

import asyncio
import time
import re
import logging
import os
from urllib.parse import urlparse, urljoin, unquote
from pathlib import Path
import json
import hashlib
import inspect
from typing import List, Dict, Tuple, Optional, Set

# --- Crawl4AI Imports ---
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    CrawlResult,
)
from crawl4ai.models import CrawlResultContainer
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

# Use standard logger, setup is handled by llamasearch.utils
logger = logging.getLogger(__name__)

# --- Default Keywords for Scoring ---
DEFAULT_RELEVANCE_KEYWORDS = [
    "documentation",
    "guide",
    "tutorial",
    "api",
    "reference",
    "manual",
    "developer",
    "usage",
    "examples",
    "concepts",
    "getting started",
]

# --- Constants ---
DEFAULT_PAGE_TIMEOUT_MS = 30000  # 30 seconds page load timeout
DEFAULT_FETCH_TIMEOUT_S = (
    45  # 45 seconds overall fetch timeout including potential redirects
)

# --- Helper Functions ---


def sanitize_string(s: str, max_length: int = 40) -> str:
    """Convert to a filesystem-friendly string."""
    s = unquote(s)
    s = re.sub(r"^[a-zA-Z]+://", "", s)
    s = re.sub(r"https?://(www\.)?", "", s)
    s = re.sub(r'[/:\\?*"<>| ]+', "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if len(s) > max_length:
        s = s[:max_length]
    return s if s else "default"


async def fetch_single(
    crawler: AsyncWebCrawler,
    url: str,
    cfg: CrawlerRunConfig,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_S,
) -> Optional[CrawlResult]:
    """
    Helper to fetch a single URL with an overall timeout.
    Handles documented return types: CrawlResultContainer, AsyncGenerator.
    """
    try:
        # Apply overall timeout to the arun call
        raw = await asyncio.wait_for(
            crawler.arun(url=url, config=cfg), timeout=timeout_seconds
        )
        logger.debug(f"Type returned by arun for {url}: {type(raw)}")

        # 1. Container (stream=False - Expected in v0.6.x)
        if isinstance(raw, CrawlResultContainer):
            if raw._results and isinstance(raw[0], CrawlResult):
                logger.debug(
                    f"Handling CrawlResultContainer for {url}, extracting first result."
                )
                return raw[0]
            else:
                logger.warning(
                    f"CrawlResultContainer for {url} was empty or contained unexpected type."
                )
                return None

        # 2. Direct result (very rare in v0.6.x for stream=False)
        elif isinstance(raw, CrawlResult):
            logger.warning(
                f"arun directly returned CrawlResult for {url} (unexpected for stream=False)."
            )
            return raw

        # 3. Async generator (Expected for stream=True, which we aren't using)
        elif inspect.isasyncgen(raw):
            logger.warning(
                f"arun returned an async generator for {url} unexpectedly (stream=False). Consuming first item."
            )
            try:
                async for first_result in raw:
                    # Should only yield one in this scenario if it happens
                    if isinstance(first_result, CrawlResult):
                        return first_result
                    else:
                        logger.warning(
                            f"Generator for {url} yielded non-CrawlResult: {type(first_result)}"
                        )
                        return None
                logger.warning(f"Generator for {url} was empty.")
                return None
            except Exception as agen_ex:
                logger.error(
                    f"Error consuming unexpected generator for {url}: {agen_ex}",
                    exc_info=True,
                )
                return None
            finally:
                # Ensure generator is closed if it exists and has aclose
                if hasattr(raw, "aclose"):
                    await raw.aclose()

        # 4. Fallback / Unexpected
        else:
            logger.error(
                f"Unexpected return type from crawler.arun for {url}: {type(raw)}. Check Crawl4AI version/behavior."
            )
            return None

    except asyncio.TimeoutError:
        logger.error(
            f"Timeout ({timeout_seconds}s) occurred during fetch_single for {url}"
        )
        return None  # Treat timeout as failure
    except Exception as e:
        logger.error(f"Exception during fetch_single for {url}: {e}", exc_info=True)
        return None


class Crawl4AICrawler:
    """Asynchronous crawler prioritizing links based on keyword relevance."""

    def __init__(
        self,
        root_urls: List[str],
        base_crawl_dir: Path,
        target_links: int = 15,
        max_depth: int = 2,
        relevance_keywords: Optional[List[str]] = None,
        headless: bool = True,  # Option for debugging
        verbose_logging: bool = False,  # Option for debugging
    ):
        if not root_urls:
            raise ValueError("At least one root URL must be provided.")
        self.root_urls = [self.normalize_url(u) for u in root_urls]
        self.target_links = target_links
        self.max_crawl_level = max_depth + 1
        self.base_crawl_dir = base_crawl_dir
        self.relevance_keywords = (
            [kw.lower() for kw in relevance_keywords]
            if relevance_keywords
            else DEFAULT_RELEVANCE_KEYWORDS
        )
        logger.info(f"Using relevance keywords: {self.relevance_keywords}")

        self.raw_markdown_dir = self.base_crawl_dir / "raw"
        self.reverse_lookup_path = self.base_crawl_dir / "reverse_lookup.json"
        self.raw_markdown_dir.mkdir(parents=True, exist_ok=True)
        self._reverse_lookup: Dict[str, str] = {}
        self._load_reverse_lookup()

        logger.info(
            f"Initialized Crawl4AICrawler for root(s): {', '.join(self.root_urls)}"
        )
        logger.info(f"Output directory (markdown): {self.raw_markdown_dir}")
        logger.info(f"Reverse lookup file: {self.reverse_lookup_path}")
        logger.info(
            f"Target links: {self.target_links}, Max Depth (User): {max_depth} => Max Crawl Level: {self.max_crawl_level}"
        )

        # --- Browser/Strategy Config - incorporating debugging options ---
        self._browser_cfg = BrowserConfig(
            browser_type="chromium",
            headless=headless,  # Control headful/headless mode
            verbose=verbose_logging,  # Control verbose logs
            # Enable capture features for debugging hangs (if needed)
            # capture_console=True,
            # capture_network=True
        )
        self._strategy = AsyncPlaywrightCrawlerStrategy(config=self._browser_cfg)
        # --- Run Config - adding page timeout ---
        self._run_cfg = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            stream=False,
            page_timeout=DEFAULT_PAGE_TIMEOUT_MS,  # Add page load timeout
        )

    def normalize_url(self, url: str) -> str:
        """Ensure URL has schema, strip fragment, ensure trailing slash for root."""
        if not isinstance(url, str):
            logger.warning(
                f"Normalize received non-string URL: {type(url)}. Returning empty string."
            )
            return ""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        url, *_ = url.split("#", 1)
        parsed = urlparse(url)
        path = parsed.path if (parsed.path and parsed.path != "/") else "/"
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        if (not parsed.path or parsed.path == "/") and not normalized.endswith("/"):
            normalized += "/"
        return normalized

    def _generate_key(self, url: str) -> str:
        """Generates a unique key (hash) for a URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def _load_reverse_lookup(self):
        """Loads the global reverse lookup file if it exists."""
        if self.reverse_lookup_path.exists():
            try:
                with open(self.reverse_lookup_path, "r", encoding="utf-8") as f:
                    self._reverse_lookup = json.load(f)
                logger.info(
                    f"Loaded existing global reverse lookup ({len(self._reverse_lookup)} entries)."
                )
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON from {self.reverse_lookup_path}. Starting fresh."
                )
                self._reverse_lookup = {}
            except Exception as e:
                logger.error(f"Error loading reverse lookup: {e}. Starting fresh.")
                self._reverse_lookup = {}
        else:
            logger.info("No existing global reverse lookup found.")
            self._reverse_lookup = {}

    def _save_reverse_lookup(self):
        """Saves the updated global hash -> URL mapping to JSON."""
        temp_path = None
        try:
            temp_path = self.reverse_lookup_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._reverse_lookup, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.reverse_lookup_path)
            logger.info(f"Global reverse lookup saved to {self.reverse_lookup_path}")
        except Exception as e:
            logger.error(f"Error saving reverse lookup: {e}")
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _score_link_priority(self, link_url: str) -> float:
        """
        Calculates the priority score for adding a link URL to the queue.
        Scores based on keywords found in the URL path/query. Lower is better.
        """
        link_score_component = 0
        link_lower = link_url.lower()

        try:
            parsed_link = urlparse(link_lower)
            path_query = parsed_link.path + "?" + parsed_link.query
            for keyword in self.relevance_keywords:
                try:
                    link_score_component += len(
                        re.findall(r"\b" + re.escape(keyword) + r"\b", path_query)
                    )
                except re.error:
                    link_score_component += path_query.count(keyword)

        except Exception as e:
            logger.warning(f"Could not parse or score link URL {link_url}: {e}")
            return float("inf")

        link_priority_score = 1.0 / (link_score_component + 0.01)
        return max(link_priority_score, 1e-6)

    async def run_crawl(self) -> List[str]:
        """Performs the asynchronous crawl using a priority queue based on link URL relevance. Robust against hangs."""
        logger.info(
            f"Starting keyword-priority crawl. Output Dir: {self.raw_markdown_dir}"
        )
        collected_urls: List[str] = []
        failed_urls: List[str] = []
        queue: asyncio.PriorityQueue[Tuple[float, int, str]] = asyncio.PriorityQueue()
        visited: Set[str] = set()
        last_progress_time = time.time()
        STALL_TIMEOUT = 60  # seconds without progress before warning
        GLOBAL_TIMEOUT = 600  # max crawl duration in seconds (10 min)
        start_time = time.time()

        for url in self.root_urls:
            norm_url = self.normalize_url(url)
            if norm_url and norm_url not in visited:
                root_priority = self._score_link_priority(norm_url)
                await queue.put((root_priority, 1, norm_url))
                visited.add(norm_url)

        crawler = AsyncWebCrawler(crawler_strategy=self._strategy)
        collected_count = 0

        async def crawl_loop():
            nonlocal last_progress_time, collected_count
            while not queue.empty() and collected_count < self.target_links:
                logger.info(f"--- DIAG: Queue size: {queue.qsize()}, Collected: {collected_count}, Visited: {len(visited)} ---")
                priority, current_depth_level, current_url = await queue.get()
                logger.info(f"--- DIAG: Dequeued URL: {current_url} (Priority: {priority}, Depth: {current_depth_level}) ---")
                try:
                    logger.debug(
                        f"Processing Q (Pri:{priority:.3f}, D:{current_depth_level}): {current_url}"
                    )
                    if current_depth_level > self.max_crawl_level:
                        logger.debug(
                            f"Max crawl level ({self.max_crawl_level}) reached for {current_url} (level {current_depth_level}). Skipping fetch."
                        )
                        continue
                    logger.info(
                        f"Crawling [L:{current_depth_level}/{self.max_crawl_level} | C:{collected_count}/{self.target_links} | Q:{queue.qsize()}]: {current_url}"
                    )
                    result = await fetch_single(
                        crawler, current_url, self._run_cfg
                    )
                    if result is None or not getattr(result, 'success', True):
                        logger.warning(
                            f"Fetch failed or returned None for {current_url}. Skipping."
                        )
                        failed_urls.append(current_url)
                        continue
                    if getattr(result, 'markdown', None) is not None:
                        key = self._generate_key(current_url)
                        md_path = self.raw_markdown_dir / f"{key}.md"
                        try:
                            md_content = result.markdown
                            if not isinstance(md_content, str):
                                logger.error(f"Markdown content for {current_url} is not a string. Skipping.")
                                failed_urls.append(current_url)
                                continue
                            md_path.write_text(md_content, encoding="utf-8")
                            self._reverse_lookup[key] = current_url
                            if current_url not in collected_urls:
                                collected_urls.append(current_url)
                                collected_count += 1
                                last_progress_time = time.time()
                            logger.debug(
                                f"Saved MD: {md_path.name} ({len(md_content)} chars)"
                            )
                        except OSError as e:
                            logger.error(f"Failed write MD {md_path}: {e}")
                            failed_urls.append(current_url)
                            continue
                        next_depth_level = current_depth_level + 1
                        if next_depth_level <= self.max_crawl_level and getattr(result, 'links', None):
                            links_added_from_page = 0
                            internal_links_raw = (
                                result.links.get("internal", [])
                                if isinstance(result.links, dict)
                                else []
                            )
                            for link_item in internal_links_raw:
                                link_url_rel = None
                                if isinstance(link_item, dict):
                                    link_url_rel = link_item.get("href")
                                elif isinstance(link_item, str):
                                    link_url_rel = link_item
                                else:
                                    logger.warning(
                                        f"Skipping unexpected link item type: {type(link_item)} from {current_url}"
                                    )
                                    continue
                                if not link_url_rel or not isinstance(link_url_rel, str):
                                    logger.debug(
                                        f"Skipping link item with missing or invalid href: {link_item}"
                                    )
                                    continue
                                try:
                                    abs_link = urljoin(current_url, link_url_rel)
                                    norm_link = self.normalize_url(abs_link)
                                    if not norm_link:
                                        continue
                                    if (
                                        self.is_valid_content_url(norm_link)
                                        and norm_link not in visited
                                    ):
                                        visited.add(norm_link)
                                        link_priority = self._score_link_priority(norm_link)
                                        logger.info(f"--- DIAG: Enqueuing URL: {norm_link} (Priority: {link_priority}, Depth: {next_depth_level}) ---")
                                        await queue.put((link_priority, next_depth_level, norm_link))
                                        links_added_from_page += 1
                                except Exception as link_err:
                                    logger.debug(
                                        f"Err processing link '{link_url_rel}' from {current_url}: {link_err}"
                                    )
                            if links_added_from_page > 0:
                                logger.debug(
                                    f"Added {links_added_from_page} links from {current_url} to queue (depth {next_depth_level}) with calculated priorities."
                                )
                    else:
                        logger.warning(
                            f"No markdown content retrieved from {current_url}"
                        )
                        failed_urls.append(current_url)
                except asyncio.CancelledError:
                    logger.info("Crawl task cancelled.")
                    break
                except Exception as e:
                    logger.error(
                        f"Error processing item {current_url}: {e}", exc_info=True
                    )
                    failed_urls.append(current_url)
                finally:
                    queue.task_done()
                # Stall detection
                if time.time() - last_progress_time > STALL_TIMEOUT:
                    logger.warning(f"Crawl appears stalled: no progress for {STALL_TIMEOUT}s. Queue size: {queue.qsize()}.")
                    # Dump queue contents for debugging
                    try:
                        queue_contents = []
                        qsize = queue.qsize()
                        for _ in range(qsize):
                            item = await queue.get()
                            queue_contents.append(item)
                            await queue.put(item)  # put it back
                        logger.warning(f"--- DIAG: Current queue contents (up to {len(queue_contents)}): {[item[2] for item in queue_contents]}")
                    except Exception as diag_exc:
                        logger.error(f"Error dumping queue contents: {diag_exc}")
                    last_progress_time = time.time()  # reset to avoid repeated warnings

            # After loop exits, log why
            if queue.empty():
                logger.info("--- DIAG: Crawl loop exited because queue is empty.")
            if collected_count >= self.target_links:
                logger.info("--- DIAG: Crawl loop exited because collected_count reached target.")
            if not queue.empty() and collected_count < self.target_links:
                logger.warning(f"--- DIAG: Crawl loop exited with non-empty queue and collected_count < target. Queue size: {queue.qsize()}.")
                # Dump remaining queue contents
                try:
                    queue_contents = []
                    qsize = queue.qsize()
                    for _ in range(qsize):
                        item = await queue.get()
                        queue_contents.append(item)
                        await queue.put(item)
                    logger.warning(f"--- DIAG: Remaining queue contents (up to {len(queue_contents)}): {[item[2] for item in queue_contents]}")
                except Exception as diag_exc:
                    logger.error(f"Error dumping remaining queue contents: {diag_exc}")

        try:
            # Run crawl loop with global timeout
            logger.info("--- DIAG: Entering asyncio.wait_for(crawl_loop)... ---")
            await asyncio.wait_for(crawl_loop(), timeout=GLOBAL_TIMEOUT)
            logger.info("--- DIAG: Exited asyncio.wait_for(crawl_loop) successfully. ---")
        except asyncio.TimeoutError:
            logger.error(
                f"Global crawl timeout ({GLOBAL_TIMEOUT}s) reached. Terminating crawl early."
            )
        except Exception as e:
            logger.error(f"Unexpected error during crawl: {e}")
            failed_urls.append(f"GENERAL_ERROR: {e}")
        finally:
            logger.info("--- DIAG: Entering finally block of run_crawl... ---")
            end_time = time.time()
            duration = end_time - start_time
            # Log summary
            if failed_urls:
                logger.warning(f"Crawl skipped/failed for {len(failed_urls)} URLs:")
                for failure in failed_urls[:10]: # Log first 10
                    logger.warning(f"  - {failure}")

            logger.info("--- DIAG: Calling _save_reverse_lookup()... ---")
            self._save_reverse_lookup()
            logger.info("--- DIAG: Finished _save_reverse_lookup(). ---")

            logger.info(
                f"Crawl finished in {duration:.2f}s. "
                f"Collected: {len(collected_urls)}, "
                f"Visited: {len(visited)}, Queue Left: {queue.qsize() if 'queue' in locals() else 'N/A'}"
            )

            logger.info("--- DIAG: Preparing to return from run_crawl. ---")

        logger.info("--- DIAG: End of run_crawl function. Returning results. ---")
        return list(self._reverse_lookup.keys())

    def is_valid_content_url(self, url: str) -> bool:
        """Basic filtering for URLs likely not containing primary content."""
        if not isinstance(url, str) or not url:
            return False
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ["http", "https"]:
                return False
            path_low = parsed.path.lower()
            ignore_patterns = [
                r"/cdn-cgi/",
                r"/__",
                r"/wp-json/",
                r"/wp-admin/",
                r"/wp-content/",
                r"/wp-includes/",
                r"/api/",
                r"/v[1-9]/api/",
                r"/rest/",
                r"/feed/",
                r"/rss/",
                r"/atom/",
                r"/xmlrpc\.php",
                r"/assets/",
                r"/static/",
                r"/media/",
                r"/images/",
                r"/css/",
                r"/js/",
                r"/fonts/",
                r"/email-protection",
                r"/cdn-cgi/l/email-protection",
                r"/ajax/",
                r"/json/",
                r"/login",
                r"/signup",
                r"/register",
                r"/auth/",
                r"/account",
                r"/user",
                r"/profile",
                r"/admin",
                r"/search",
                r"/find",
                r"/cart",
                r"/checkout",
                r"/order",
                r"/tag/",
                r"/category/",
                r"/author/",
                r"tel:",
                r"mailto:",
                r"javascript:",
            ]
            if any(re.search(p, path_low) for p in ignore_patterns):
                return False
            path_part_for_ext = parsed.path.split("?")[0].lower()
            ignore_extensions = r"\.(jpg|jpeg|png|gif|bmp|tif|tiff|mp3|mp4|avi|mov|wmv|webp|svg|css|js|json|xml|ico|woff|woff2|ttf|eot|otf|pdf|zip|tar|gz|rar|7z|exe|dmg|iso|ppt|pptx|doc|docx|xls|xlsx|csv|txt|rtf)$"
            if re.search(ignore_extensions, path_part_for_ext):
                return False

            root_domains = {
                urlparse(root).netloc
                for root in self.root_urls
                if urlparse(root).netloc
            }
            current_domain = parsed.netloc
            if not current_domain:
                return False
            if current_domain not in root_domains:
                is_subdomain_or_match = False
                for root_domain in root_domains:
                    if root_domain and (
                        current_domain == root_domain
                        or current_domain.endswith("." + root_domain)
                    ):
                        is_subdomain_or_match = True
                        break
                if not is_subdomain_or_match:
                    return False
            return True
        except Exception as e:
            logger.debug(f"Error validating URL '{url}': {e}")
            return False
```

---
### File: src\llamasearch\core\embedder.py

```python
"""
Enhanced embedder optimized for CPU with multi-threading.
Includes checks for model existence before loading.
"""

import numpy as np
import torch
import gc
import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from multiprocessing.synchronize import Lock as SyncLock
from pydantic import BaseModel, Field, validator
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from llamasearch.exceptions import ModelNotFoundError
from llamasearch.data_manager import data_manager  # To get models path

# Import hardware detection (assuming it provides CPU info)
from llamasearch.hardware import detect_hardware_info, HardwareInfo
from llamasearch.utils import setup_logging

logger = setup_logging(__name__)  # Get logger using utility

DEFAULT_MODEL_NAME = "teapotai/teapotembedding"
DEVICE_TYPE = "cpu" # Hardcoded for CPU-only


class EmbedderConfig(BaseModel):
    """Configuration for the embedder optimized for CPU."""

    model_name: str = DEFAULT_MODEL_NAME
    device: str = Field(default=DEVICE_TYPE) # Always CPU
    max_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=8, gt=0)
    auto_optimize: bool = True
    num_workers: int = Field(default=1, ge=1)
    instruction: str = "Given a passage, represent its content for retrieval."
    threads_per_worker: int = Field(default=1, ge=1)
    # use_half_precision removed, not relevant for CPU

    @validator("num_workers")
    def validate_workers(cls, v, values):
        """Ensure a reasonable number of workers based on CPU cores."""
        is_auto_optimizing = values.get("auto_optimize", True)
        cpu_limit = max(1, (os.cpu_count() or 4) // 2) # Default reasonable limit
        limit = v

        if v > cpu_limit:
            limit = cpu_limit
            if is_auto_optimizing:
                logger.debug(f"Auto-limiting CPU workers from {v} to {limit}.")
            else:
                logger.warning(
                    f"Provided CPU workers ({v}) exceeds reasonable limit ({limit}), capping."
                )

        final_workers = max(1, limit)
        if final_workers != v:
            logger.info(
                f"Adjusted num_workers from {v} to {final_workers} for CPU."
            )
        return final_workers

    @classmethod
    def from_hardware(cls, model_name: str = DEFAULT_MODEL_NAME) -> "EmbedderConfig":
        """Create an optimized CPU configuration based on detected hardware."""
        hw: HardwareInfo = detect_hardware_info()
        config_data = {
            "model_name": model_name,
            "auto_optimize": True,
            "device": DEVICE_TYPE # Always CPU
            }

        logger.info("Configuring for CPU based on detected hardware.")
        # Use physical cores for worker calculation
        physical_cores = hw.cpu.physical_cores if hw.cpu.physical_cores else (os.cpu_count() or 2) # Fallback if detection fails
        
        # Aim for roughly half the physical cores as workers, min 1, max 8
        cpu_workers = min(8, max(1, physical_cores // 2))
        config_data["num_workers"] = cpu_workers

        # Distribute remaining cores as threads per worker
        threads_per = min(
            4, # Limit threads per worker to avoid excessive context switching
            max(
                1,
                physical_cores // cpu_workers if cpu_workers > 0 else physical_cores,
            ),
        )
        config_data["threads_per_worker"] = threads_per
        logger.info(
            f"Auto CPU config: {cpu_workers} workers, {threads_per} threads per worker."
        )

        # Set batch size based on available RAM
        if hw.memory.available_gb > 30:
            config_data["batch_size"] = 32
        elif hw.memory.available_gb > 15:
            config_data["batch_size"] = 16
        elif hw.memory.available_gb > 7:
            config_data["batch_size"] = 8
        else:
            config_data["batch_size"] = 4 # Lower batch size for low RAM
        logger.info(
            f"Available RAM: {hw.memory.available_gb:.1f} GB. Auto batch size: {config_data['batch_size']}."
        )

        final_config = cls(**config_data)
        logger.info(
            f"Auto-optimized EmbedderConfig: device={final_config.device}, batch_size={final_config.batch_size}, "
            f"workers={final_config.num_workers}, threads_per_worker={final_config.threads_per_worker}"
        )
        return final_config


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create an instruction template for the E5 instruct model."""
    if task_description and not task_description.endswith((".", "!", "?", ":")):
        task_description += ":"
    return f"Instruct: {task_description}\nQuery: {query}"


class EnhancedEmbedder:
    """
    Enhanced embedder optimized for CPU with multi-threading.
    Includes checks for model existence before loading.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        # device parameter removed - always CPU
        max_length: int = 512,
        batch_size: int = 0,
        auto_optimize: bool = True,
        num_workers: int = 0,
        instruction: str = "",
    ):
        config_data = {"model_name": model_name, "auto_optimize": auto_optimize}

        if auto_optimize:
            logger.info("Auto-optimizing embedder configuration based on hardware (CPU)...")
            base_config = EmbedderConfig.from_hardware(model_name)
            config_data.update(base_config.dict())
            # Ensure device is CPU even if base_config somehow missed it
            config_data["device"] = DEVICE_TYPE
        else:
             logger.info("Using provided embedder configuration (auto_optimize=False).")
             config_data["device"] = DEVICE_TYPE # Force CPU
             config_data["batch_size"] = batch_size or 8
             config_data["num_workers"] = num_workers or 1
             config_data["instruction"] = instruction or EmbedderConfig().instruction
             # Calculate threads_per_worker if not auto-optimized
             hw = detect_hardware_info()
             physical_cores = hw.cpu.physical_cores if hw.cpu.physical_cores else (os.cpu_count() or 2)
             config_data["threads_per_worker"] = config_data.get(
                 "threads_per_worker", # Keep if provided
                 min(
                     4,
                     max(
                         1,
                         physical_cores // config_data["num_workers"]
                         if config_data["num_workers"] > 0
                         else physical_cores,
                     ),
                 ),
             )


        # Override auto-config with specific user inputs if provided
        if batch_size > 0:
            config_data["batch_size"] = batch_size
        if num_workers > 0:
            config_data["num_workers"] = num_workers
        if max_length != 512:
            config_data["max_length"] = max_length
        if instruction:
            config_data["instruction"] = instruction
        else:
            # Ensure instruction is set even if not provided and not auto-optimizing
            config_data["instruction"] = config_data.get(
                "instruction", EmbedderConfig().instruction
            )

        # Final config object creation
        self.config = EmbedderConfig(**config_data)

        # Ensure final config reflects CPU only settings
        if self.config.device != DEVICE_TYPE:
             logger.warning(f"Overriding configured device '{self.config.device}' to '{DEVICE_TYPE}'.")
             self.config.device = DEVICE_TYPE


        try:
            self.models_dir = Path(data_manager.get_data_paths()["models"])
            logger.info(f"Embedder using models directory: {self.models_dir}")
        except Exception as e:
            logger.error(
                f"Failed to get models directory from data_manager: {e}. Using default relative path.",
                exc_info=True,
            )
            self.models_dir = Path(".") / ".llamasearch" / "models"
            self.models_dir.mkdir(parents=True, exist_ok=True)

        # Rough check if max_length is feasible before loading the full model
        try:
            temp_model_check = SentenceTransformer(
                self.config.model_name, cache_folder=str(self.models_dir)
            )
            # Check if setting max_seq_length works (might raise error if model doesn't support it well)
            temp_model_check.max_seq_length = self.config.max_length
            logger.debug(f"Model seems compatible with max_length {self.config.max_length}")
            del temp_model_check
        except Exception as e:
            logger.warning(
                f"Could not verify max_seq_length ({self.config.max_length}) on temp model check: {e}. Using model default if needed."
            )
            # Don't fail here, let the main loading handle it


        # Models dictionary will hold only one model for CPU
        self.model: Optional[SentenceTransformer] = None
        self.model_lock: Optional[SyncLock] = None
        self._lock_initialized = False

        logger.info(
            f"Final Embedder Config: model={self.config.model_name}, device={self.config.device}, "
            f"batch_size={self.config.batch_size}, workers={self.config.num_workers}, "
            f"threads_per_worker={self.config.threads_per_worker}"
        )

        self._load_model()  # Load the single CPU model

    def _initialize_lock(self):
        """Initialize lock safely for multiprocessing/threading."""
        if not self._lock_initialized and self.config.num_workers > 1:
            try:
                # Use lock for multi-worker CPU case for thread safety
                logger.debug("Initializing multiprocessing lock for embedder worker.")
                # Use spawn context if available for better isolation, otherwise default
                try:
                    mp_context = torch.multiprocessing.get_context("spawn")
                except Exception:
                    logger.warning("Spawn context not available, using default multiprocessing context for lock.")
                    import multiprocessing as mp # Fallback
                    mp_context = mp.get_context()

                self.model_lock = mp_context.Lock()
                self._lock_initialized = True
                logger.debug("Lock initialized.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize multiprocessing lock: {e}. Proceeding without lock (potential race conditions).",
                    exc_info=True,
                )
                self.model_lock = None
                self._lock_initialized = True # Mark as initialized even if failed
        elif self.config.num_workers <= 1:
            self.model_lock = None # No lock needed for single worker
            self._lock_initialized = True

    def _load_model(self):
        """Load the embedding model onto the CPU after checking existence."""
        model_name = self.config.model_name
        logger.info(f"Preparing to load embedder model: {model_name} onto CPU")

        # --- CHECK MODEL EXISTENCE ---
        try:
            logger.debug(
                f"Checking for model '{model_name}' in cache: {self.models_dir}"
            )
            snapshot_download(
                repo_id=model_name,
                cache_dir=self.models_dir,
                local_files_only=True,  # Check cache only
            )
            logger.info(f"Model '{model_name}' found locally in cache.")
        except (EntryNotFoundError, LocalEntryNotFoundError):
            logger.error(
                f"Embedder model '{model_name}' not found in cache directory: {self.models_dir}"
            )
            raise ModelNotFoundError(
                f"Embedder model '{model_name}' not found locally. "
                f"Please run 'llamasearch-setup' or ensure the model exists at the specified path."
            )
        except Exception as e:
            logger.error(
                f"Error checking model cache for '{model_name}': {e}", exc_info=True
            )
            raise ModelNotFoundError(
                f"Error accessing embedder model '{model_name}' cache. "
                f"Check permissions or run 'llamasearch-setup'. Error: {e}"
            )

        # --- PROCEED WITH LOADING (if check passed) ---
        try:
            # Set torch threads based on workers and threads_per_worker
            total_threads = self.config.num_workers * self.config.threads_per_worker
            logical_cores = os.cpu_count() or 1
            effective_threads = max(1, min(total_threads, logical_cores))

            # Only set threads if > 0, avoid setting 0 threads
            if effective_threads > 0 :
                current_threads = torch.get_num_threads()
                # Only change if necessary
                if current_threads != effective_threads:
                    torch.set_num_threads(effective_threads)
                    logger.info(
                        f"Set torch global CPU threads from {current_threads} to: {torch.get_num_threads()}"
                    )
                else:
                     logger.info(f"Torch global CPU threads already set to {current_threads}.")

            logger.info(
                f"Loading embedding model '{model_name}' onto device '{DEVICE_TYPE}'"
            )
            model = SentenceTransformer(
                model_name,
                device=DEVICE_TYPE,
                cache_folder=str(self.models_dir),  # Explicitly specify cache folder
            )
            model.max_seq_length = self.config.max_length
            logger.debug(
                f"Model {model_name} on {DEVICE_TYPE} max_seq_length set to {model.max_seq_length}"
            )

            # No FP16 conversion for CPU
            self.model = model

            # Initialize lock after model loading attempt
            self._initialize_lock()

        except ModelNotFoundError:
            raise  # Re-raise if check failed earlier
        except Exception as e:
            logger.error(
                f"Error loading embedding model '{model_name}' from cache: {e}",
                exc_info=True,
            )
            self.model = None
            self.model_lock = None
            raise RuntimeError(
                f"Failed to load embedder model '{model_name}' from cache: {e}"
            ) from e

    def _get_model_and_lock(self) -> tuple:
        """Return the single CPU model instance and its lock."""
        self._initialize_lock() # Ensure lock is initialized if needed
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")
        return self.model, self.model_lock

    def _embed_batch_task(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts using the single CPU model (task for executor)."""
        if not texts:
            return np.array([], dtype=np.float32)
        try:
            model, lock = self._get_model_and_lock()
            input_texts = [
                get_detailed_instruct(self.config.instruction, text) for text in texts
            ]
            embeddings = None
            acquired_lock = False

            # Acquire lock only if it exists (i.e., num_workers > 1)
            if lock:
                acquired_lock = lock.acquire(timeout=15) # Timeout for lock acquisition
                if not acquired_lock:
                    # If lock fails, log warning and return empty array for this batch
                    logger.warning(
                        f"Worker failed to acquire lock for batch of size {len(texts)}, skipping."
                    )
                    return np.array([], dtype=np.float32) # Return empty, indicates failure for this batch

            try:
                # Process the batch within the lock (if acquired)
                effective_batch_size = min(len(input_texts), self.config.batch_size)
                embeddings = model.encode(
                    input_texts,
                    batch_size=effective_batch_size,
                    show_progress_bar=False, # Progress handled by outer loop
                    convert_to_numpy=True,
                    normalize_embeddings=True, # Assuming normalization is desired
                )
                # Ensure float32 type
                if embeddings is not None and embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
            except Exception as e:
                logger.error(
                    f"Error during model.encode on CPU for batch size {len(texts)}: {e}",
                    exc_info=True,
                )
                # Return empty on error, indicating failure for this batch
                return np.array([], dtype=np.float32)
            finally:
                # Release lock if it was acquired
                if acquired_lock and lock:
                    lock.release()

            # Return results, handle None case
            return (
                embeddings if embeddings is not None else np.array([], dtype=np.float32)
            )
        except Exception as e:
            # Catch errors in getting model/lock or other unexpected issues
            logger.error(
                f"Error processing batch (size {len(texts)}) in worker task: {e}",
                exc_info=True,
            )
            return np.array([], dtype=np.float32) # Return empty on major task error


    def embed_strings(
        self, strings: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of strings using CPU workers."""
        if self.model is None:
             raise RuntimeError("Cannot embed strings, model is not loaded.")
        if not strings:
            return np.array([], dtype=np.float32)

        # --- String Truncation ---
        model_max_len = self.config.max_length
        try:
            # Use actual model max length if available
            model_max_len = self.model.max_seq_length or model_max_len
        except Exception:
            pass # Use config value if model property access fails
        
        # Estimate max characters (heuristic, depends on tokenization)
        # Using 6 chars/token as a generous upper bound estimate
        max_input_chars = model_max_len * 6
        truncated_strings = []
        num_truncated = 0
        for s in strings:
            if len(s) > max_input_chars:
                truncated_strings.append(s[:max_input_chars])
                num_truncated += 1
            else:
                truncated_strings.append(s)

        if num_truncated > 0:
            logger.warning(
                f"Truncated {num_truncated} strings longer than estimated {max_input_chars} chars "
                f"(based on model max_length {model_max_len}). Adjust max_length if needed."
            )

        # --- Embedding Process ---
        total = len(truncated_strings)
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers

        progress_bar = tqdm(
            total=total,
            desc="Generating embeddings (CPU)",
            unit="text",
            disable=not show_progress or total <= batch_size, # Disable for small jobs or if requested
        )
        
        all_embeddings_list = [] # Store results from futures

        # Use ThreadPoolExecutor for parallel batch processing on CPU
        # It handles the case num_workers=1 correctly (runs sequentially)
        logger.debug(f"Running embedding with {num_workers} worker(s).")
        executor_factory = ThreadPoolExecutor # Standard choice for I/O bound or GIL-releasing tasks like ST encode

        with executor_factory(max_workers=num_workers) as executor:
            futures = []
            # Submit batches to the executor
            for i in range(0, total, batch_size):
                batch = truncated_strings[i : min(i + batch_size, total)]
                if batch: # Ensure non-empty batch
                    futures.append(
                        executor.submit(self._embed_batch_task, batch)
                    )
            
            # Collect results as they complete
            for fut in as_completed(futures):
                try:
                    result_emb = fut.result()
                    # Check if the result is valid before appending
                    if result_emb is not None and isinstance(result_emb, np.ndarray) and result_emb.size > 0:
                        all_embeddings_list.append(result_emb)
                        progress_bar.update(result_emb.shape[0]) # Update by number of embeddings processed
                    elif result_emb is not None and isinstance(result_emb, np.ndarray) and result_emb.size == 0:
                        # Log if a worker explicitly returned an empty array (e.g., due to lock failure)
                        logger.debug("Worker task returned an empty result array.")
                    else:
                         logger.warning(
                            f"Embedding worker thread returned unexpected result type: {type(result_emb)}"
                        )
                except Exception as e:
                    # Catch errors from the future itself (e.g., task raised exception not caught internally)
                    logger.error(
                        f"Error retrieving result from embedding worker future: {e}",
                        exc_info=True,
                    )
                
                # Periodic garbage collection during long processes
                if len(all_embeddings_list) % 20 == 0: # Every 20 completed batches
                     self._try_gc()


        progress_bar.close()

        # --- Combine Results ---
        if not all_embeddings_list:
            logger.warning("Embedding process yielded no valid results.")
            emb_dim = self.get_embedding_dimension()
            # Return empty array with correct shape if possible
            return np.empty((0, emb_dim) if emb_dim else 0, dtype=np.float32)

        # Initialize valid_embeddings *before* the try block
        valid_embeddings = []
        try:
            # Filter out any remaining Nones or empty arrays just in case
            valid_embeddings = [
                emb
                for emb in all_embeddings_list
                if isinstance(emb, np.ndarray) and emb.size > 0
            ]
            if not valid_embeddings:
                logger.error("No valid numpy arrays found in embedding results after filtering.")
                emb_dim = self.get_embedding_dimension()
                return np.empty((0, emb_dim) if emb_dim else 0, dtype=np.float32)

            # Combine the valid embeddings
            combined = np.vstack(valid_embeddings)
            # Ensure contiguous array with float32 type
            result = np.ascontiguousarray(combined, dtype=np.float32)

        except ValueError as e:
            # Handle potential shape mismatches during vstack
            logger.error(
                f"Error during vstack of embeddings (shape mismatch?): {e}",
                exc_info=True,
            )
            shapes = [emb.shape for emb in valid_embeddings] # Use the filtered list
            logger.debug(f"Shapes of collected embeddings: {shapes}")
            emb_dim = self.get_embedding_dimension()
            return np.empty((0, emb_dim) if emb_dim else 0, dtype=np.float32) # Return empty on error

        self._try_gc() # Final GC
        if result.shape[0] != total:
            logger.warning(f"Expected {total} embeddings, but got {result.shape[0]}. Some batches might have failed.")
        else:
             logger.info(
                f"Generated {result.shape[0]} embeddings with dimension {result.shape[1]}."
            )
        return result

    def _try_gc(self):
        """Try to run garbage collection."""
        gc.collect()
        # No CUDA cache to empty

    def embed_string(self, text: str) -> Optional[np.ndarray]:
        """Embed a single string using the CPU model."""
        if self.model is None:
             logger.error("Cannot embed string, model is not loaded.")
             return None
        if not text:
            logger.warning("Attempted to embed an empty string.")
            return None # Return None for empty input

        # --- String Truncation (same logic as embed_strings) ---
        model_max_len = self.config.max_length
        try:
            model_max_len = self.model.max_seq_length or model_max_len
        except Exception:
            pass
        max_input_chars = model_max_len * 6
        if len(text) > max_input_chars:
            original_len = len(text)
            text = text[:max_input_chars]
            logger.debug(f"Truncated single string from {original_len} to {max_input_chars} chars.")

        # --- Embedding ---
        try:
            # Get model and lock (lock is likely None for single string, but use the helper)
            model, lock = self._get_model_and_lock()
            input_text = get_detailed_instruct(self.config.instruction, text)

            embedding = None
            acquired_lock = False
            if lock: # Check if lock exists (might if called concurrently)
                acquired_lock = lock.acquire(timeout=5) # Short timeout for single embed
                if not acquired_lock:
                     logger.warning("Failed lock for single string embed, returning None.")
                     return None

            try:
                 # Encode the single string
                 embedding = model.encode(
                     input_text,
                     show_progress_bar=False,
                     convert_to_numpy=True,
                     normalize_embeddings=True,
                 )
            except Exception as e:
                logger.error(f"Error during single string encode: {e}", exc_info=True)
                return None # Return None on encoding error
            finally:
                 if acquired_lock and lock:
                     lock.release()

            # Process the result
            if embedding is not None:
                # Ensure correct dtype
                return (
                    embedding.astype(np.float32)
                    if embedding.dtype != np.float32
                    else embedding
                )
            else:
                # Handle case where model returns None
                logger.warning(f"Model returned None embedding for single string: '{text[:50]}...'")
                return None
        except Exception as e:
            # Catch errors in getting model/lock or other issues
            logger.error(f"Error embedding single string: {e}", exc_info=True)
            return None

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns the embedding dimension of the loaded CPU model."""
        if self.model is None:
            logger.warning("Cannot get embedding dimension, no model loaded.")
            return None
        try:
            # Directly access the model's method
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return None

    def embed_batch(self, strings: List[str]) -> np.ndarray:
        """Legacy alias for embed_strings."""
        logger.debug("embed_batch called, redirecting to embed_strings.")
        return self.embed_strings(strings)

    def similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """Calculate cosine similarity between two sets of normalized embeddings."""
        if not isinstance(embeddings1, np.ndarray) or not isinstance(
            embeddings2, np.ndarray
        ):
            raise TypeError("Inputs must be numpy arrays.")
        
        # Handle empty inputs gracefully
        if embeddings1.size == 0 or embeddings2.size == 0:
             # Return empty array with correct dimensions if possible
             # If one is (0, D) and other is (N, D), result should be (0, N)
             # If one is (N, D) and other is (0, D), result should be (N, 0)
             # If both are empty, result is (0, 0)
            return np.empty(
                (embeddings1.shape[0], embeddings2.shape[0]), dtype=np.float32
            )
            
        # Reshape 1D arrays to 2D
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)
            
        # Dimension check
        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(
                f"Embedding dimensions must match: {embeddings1.shape[1]} != {embeddings2.shape[1]}"
            )
            
        # Calculate dot product (cosine similarity for normalized vectors)
        # Ensure float32 for potentially better performance/consistency
        sim = np.dot(embeddings1.astype(np.float32), embeddings2.astype(np.float32).T)
        
        # Clip values to handle potential floating point inaccuracies
        return np.clip(sim, -1.0, 1.0)

    def close(self):
        """Release resources and free memory."""
        logger.info("Closing CPU Embedder and releasing resources...")
        if self.model is not None:
             try:
                 # Explicitly delete the model object
                 del self.model
                 self.model = None
                 logger.debug("Embedder model deleted.")
             except Exception as e:
                 logger.warning(f"Error deleting model during close: {e}")
        
        # Clear lock reference
        self.model_lock = None
        self._lock_initialized = False
        
        # Run final garbage collection
        self._try_gc()
        logger.info("CPU Embedder closed.")
```

---
### File: src\llamasearch\core\llmsearch.py

```python
# src/llamasearch/core/llmsearch.py

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

from llamasearch.utils import setup_logging
from llamasearch.core.vectordb import VectorDB
from llamasearch.core.embedder import EnhancedEmbedder, DEFAULT_MODEL_NAME
from llamasearch.core.teapot import load_teapot_onnx_llm, TeapotONNXLLM
from llamasearch.protocols import LLM, ModelInfo  # Added ModelInfo import
from llamasearch.exceptions import ModelNotFoundError

logger = setup_logging(__name__)


class LLMSearch:
    """
    A RAG-based search class using the Teapot ONNX model.
    Manages LLM, Embedder, and VectorDB instances. Checks for model availability.
    """

    def __init__(
        self,
        storage_dir: Path,
        teapot_onnx_quant: str = "auto",
        teapot_provider: Optional[str] = None,
        teapot_provider_opts: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_results: int = 3,
        embedder_model: Optional[str] = None,
        embedder_batch_size: int = 32,
        embedder_device: Optional[str] = None,
        vectordb_similarity_threshold: float = 0.25,
        vectordb_max_chunk_size: int = 512,
        vectordb_chunk_overlap: int = 64,
        vectordb_min_chunk_size: int = 128,
        vectordb_collection_name: str = "default",
        max_workers: int = 1,
        debug: bool = False,
    ):
        self.verbose = verbose
        self.max_results = max_results
        self.debug = debug
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.model: Optional[LLM] = None
        self.embedder: Optional[EnhancedEmbedder] = None
        self.vectordb: Optional[VectorDB] = None
        self.context_length: int = 0  # Initialize context length
        self.llm_device_type: str = "cpu"  # Initialize device type

        # --- Initialize components with error handling for missing models ---
        try:
            # --- Load Teapot ONNX LLM ---
            self.logger.info("Initializing Teapot ONNX LLM for LLMSearch...")
            loaded_model: Optional[LLM] = load_teapot_onnx_llm(
                onnx_quantization=teapot_onnx_quant,
                preferred_provider=teapot_provider,
                preferred_options=teapot_provider_opts,
            )
            # Check if model loading was successful before assigning and accessing
            if loaded_model is None:
                # This case should ideally be caught by exceptions inside load_teapot...
                # but handle defensively.
                raise RuntimeError("load_teapot_onnx_llm returned None unexpectedly.")

            self.model = loaded_model  # Assign only after successful load

            # --- Access model attributes safely *after* assignment ---
            model_info: ModelInfo = self.model.model_info  # Get info object
            self.context_length = model_info.context_length
            if hasattr(self.model, "device"):
                self.llm_device_type = self.model.device.type
                self.logger.info(
                    f"LLMSearch using {model_info.model_id} on device {self.model.device}. Context: {self.context_length}"
                )
            else:
                self.llm_device_type = "cpu"
                self.logger.warning("LLM assuming CPU.")
                self.logger.info(
                    f"LLMSearch using {model_info.model_id}. Context: {self.context_length}"
                )

            # --- Initialize Embedder ---
            compute_device = embedder_device or (
                "cpu" if self.llm_device_type == "cpu" else "cuda"
            )
            self.logger.info(f"Configuring Embedder on device: {compute_device}")
            # EnhancedEmbedder.__init__ handles its own model checks
            self.embedder = EnhancedEmbedder(
                model_name=embedder_model or DEFAULT_MODEL_NAME,
                batch_size=embedder_batch_size,
                num_workers=max_workers,
            )

            # --- Initialize VectorDB ---
            self.logger.info(f"Initializing VectorDB (storage: {self.storage_dir})")
            # VectorDB.__init__ -> BM25Retriever.__init__ -> load_nlp_model checks spaCy
            self.vectordb = VectorDB(
                embedder=self.embedder,  # Pass the successfully initialized embedder
                storage_dir=self.storage_dir,
                collection_name=vectordb_collection_name,
                max_chunk_size=vectordb_max_chunk_size,
                chunk_overlap=vectordb_chunk_overlap,
                min_chunk_size=vectordb_min_chunk_size,
                embedder_batch_size=embedder_batch_size,
                similarity_threshold=vectordb_similarity_threshold,
                max_results=self.max_results,
                device=compute_device,
            )
            self.logger.info("LLMSearch components initialized successfully.")

        except ModelNotFoundError as e:
            self.logger.error(f"LLMSearch initialization failed: {e}")
            self.close()  # Attempt cleanup
            raise  # Re-raise the specific error
        except Exception as e:
            self.logger.error(
                f"Unexpected error during LLMSearch initialization: {e}", exc_info=True
            )
            self.close()  # Attempt cleanup
            raise RuntimeError(
                "LLMSearch failed to initialize due to an unexpected error."
            ) from e

    def add_document(self, file_path: Path) -> int:
        """Adds a source (file/dir) to VectorDB, delegating processing."""
        if not self.vectordb:
            self.logger.error("VectorDB is not initialized.")
            return 0
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"Source path not found: {file_path}")
            return 0

        self.logger.info(f"Requesting VectorDB to add source: {file_path}")
        try:
            added_count = self.vectordb.add_source(file_path)
            if added_count > 0:
                logger.info(
                    f"VectorDB added {added_count} chunks from: {file_path.name}"
                )
            else:
                logger.info(
                    f"VectorDB added 0 new chunks for source: {file_path.name}."
                )
            return added_count
        except Exception as e:
            logger.error(
                f"Failed add source {file_path} via VectorDB: {e}", exc_info=self.debug
            )
            return 0

    def add_documents_from_directory(
        self, directory_path: Path, recursive: bool = True
    ) -> int:
        """Adds all processable files from a directory using VectorDB."""
        if not self.vectordb:
            self.logger.error("VectorDB is not initialized.")
            return 0
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            self.logger.error(f"{directory_path} is not a valid directory.")
            return 0

        total_chunks_added, files_processed, process_count = 0, 0, 0
        files_failed = []
        glob_pattern = "**/*" if recursive else "*"
        logger.info(
            f"Scanning directory {'recursively' if recursive else ''}: {directory_path}"
        )

        for p in directory_path.glob(glob_pattern):
            if p.is_file():
                process_count += 1
                logger.debug(f"Requesting VectorDB to add source: {p}")
                try:
                    added = self.add_document(p)  # Use the single doc method
                    if added > 0:
                        total_chunks_added += added
                        files_processed += 1
                except Exception as e:
                    logger.error(
                        f"Error requesting add for {p.name}: {e}", exc_info=self.debug
                    )
                    files_failed.append(p.name)

        logger.info(
            f"Directory scan complete. Processed {process_count} files. VectorDB added {total_chunks_added} new chunks from {files_processed} files."
        )
        if files_failed:
            logger.warning(
                f"Errors requesting processing for {len(files_failed)} files: {', '.join(files_failed)}"
            )
        return total_chunks_added

    def _get_token_count(self, text: str) -> int:
        """Calculates token count using Teapot's tokenizer or estimation."""
        # Check if model and tokenizer exist before using them
        if (
            self.model
            and isinstance(self.model, TeapotONNXLLM)
            and hasattr(self.model, "_tokenizer")
            and self.model._tokenizer
        ):
            try:
                return len(self.model._tokenizer.encode(text, add_special_tokens=False))
            except Exception as e:
                logger.warning(
                    f"Could not use Teapot tokenizer for count: {e}. Estimating."
                )
        return max(1, len(text) // 4)  # Fallback estimate

    def llm_query(self, query_text: str, debug_mode: bool = False) -> Dict[str, Any]:
        """RAG-based retrieval + LLM generation using Teapot ONNX"""
        # Check initialization before proceeding
        if self.model is None or not hasattr(self.model, "generate"):
            return {
                "response": "Error: LLM not initialized.",
                "formatted_response": "Error: LLM not initialized.",
            }
        if self.vectordb is None:
            return {
                "response": "Error: VectorDB not initialized.",
                "formatted_response": "Error: VectorDB not initialized.",
            }

        debug_info: Dict[str, Any] = {}
        final_context, retrieved_display, query_time, gen_time = "", "", -1.0, -1.0

        try:  # Retrieve Context
            logger.debug("Performing vector search for query: '%s...'", query_text[:50])
            query_start_time = time.time()
            results = self.vectordb.vectordb_query(query_text, max_out=self.max_results)
            query_time = time.time() - query_start_time
            debug_info["vector_query_time"] = f"{query_time:.3f}s"
            debug_info["vector_results_count"] = len(results.get("documents", []))

            docs, metas, scores = (
                results.get("documents", []),
                results.get("metadatas", []),
                results.get("scores", []),
            )
            if not docs:
                final_context = "[No relevant context found in documents]"
                retrieved_display = "No relevant chunks retrieved."
                logger.warning("No relevant context found for query.")
            else:
                logger.info(f"Retrieved {len(docs)} chunks.")
                temp_context, temp_display = "", ""
                # Use self.context_length which was set safely in __init__
                prompt_base_len = self._get_token_count(
                    f"Context:\n\n\nQuery: {query_text}\n\nAnswer:"
                )
                available_context_tokens = (
                    self.context_length - prompt_base_len - 200
                )  # Safety margin
                logger.debug(
                    f"Context limit: {self.context_length}, Base prompt: {prompt_base_len}, Available: {available_context_tokens}"
                )

                for i, doc_text in enumerate(docs):
                    score = scores[i] if i < len(scores) else 0.0
                    source = metas[i].get("source", "N/A") if i < len(metas) else "N/A"
                    chunk_id_meta = (
                        metas[i].get("chunk_id", "N/A") if i < len(metas) else "N/A"
                    )
                    chunk_identifier = (
                        chunk_id_meta
                        if chunk_id_meta != "N/A"
                        else metas[i].get("original_chunk_index", f"docidx_{i}")
                    )

                    header = f"[Doc {i + 1} | Source: {os.path.basename(source)} | Score: {score:.2f}]\n"
                    doc_chunk = f"{header}{doc_text}\n\n"
                    display_chunk = f"--- Chunk {i + 1} (Score: {score:.2f}) ---\nSource: {source}\nChunk ID: {chunk_identifier}\n{doc_text}\n\n"

                    current_context_len = self._get_token_count(temp_context)
                    doc_chunk_len = self._get_token_count(doc_chunk)

                    if current_context_len + doc_chunk_len <= available_context_tokens:
                        temp_context += doc_chunk
                        temp_display += display_chunk
                    else:
                        logger.warning(
                            f"Stopping context inclusion at chunk {i + 1}/{len(docs)} due to limit."
                        )
                        debug_info["context_truncated_at_chunk"] = i + 1
                        break
                final_context, retrieved_display = (
                    temp_context.strip(),
                    temp_display.strip(),
                )
        except Exception as e:
            logger.error(f"Error during vector DB query: {e}", exc_info=self.debug)
            return {
                "response": f"Error during context retrieval: {e}",
                "formatted_response": f"Error: {e}",
            }

        # Construct Prompt
        system_instruction = "Answer the query using *only* the provided Context. If the answer isn't in the Context, say so."
        prompt = f"{system_instruction}\n\nContext:\n{final_context}\n\nQuery: {query_text}\n\nAnswer:"
        debug_info["final_prompt_len_chars"] = len(prompt)
        debug_info["final_prompt_len_tokens"] = self._get_token_count(prompt)
        if self.debug:
            logger.debug(f"--- LLM Prompt Start ---\n{prompt}\n--- LLM Prompt End ---")

        # Generate Response
        logger.info("Generating response with LLM...")
        gen_start = time.time()
        text_response, raw_llm_output = "Error: LLM generation failed.", None
        try:
            # self.model is guaranteed non-None here due to check at start of method
            text_response, raw_llm_output = self.model.generate(
                prompt=prompt,
                max_tokens=max(150, self.context_length // 3),
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.15,
                do_sample=True,
            )
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=self.debug)
            raw_llm_output = {"error": str(e)}
        gen_time = time.time() - gen_start
        logger.info(
            f"LLM generation took {gen_time:.2f}s. Len: {len(text_response)} chars."
        )
        debug_info["llm_generation_time"] = f"{gen_time:.3f}s"
        try:
            debug_info["raw_llm_output"] = (
                json.dumps(raw_llm_output) if self.debug else "Disabled"
            )
        except TypeError:
            debug_info["raw_llm_output"] = (
                str(raw_llm_output) if self.debug else "Disabled"
            )

        # Format and Return
        formatted_response = f"## AI Answer\n{text_response}\n\n## Retrieved Context\n{retrieved_display}"
        return {
            "response": text_response,
            "debug_info": debug_info if debug_mode else {},
            "retrieved_context": retrieved_display,
            "formatted_response": formatted_response,
            "query_time_seconds": query_time,
            "generation_time_seconds": gen_time,
        }

    def close(self) -> None:
        """Unload models and release resources."""
        # Use temporary variables to avoid accessing potentially non-existent attributes
        model_to_close = getattr(self, "model", None)
        embedder_to_close = getattr(self, "embedder", None)
        vectordb_to_close = getattr(self, "vectordb", None)

        self.logger.info("Closing LLMSearch and its components...")
        if model_to_close:
            try:
                if hasattr(model_to_close, "unload"):
                    model_to_close.unload()
                del self.model
                self.model = None
                logger.debug("LLM closed.")
            except Exception as e:
                logger.error(f"Error closing LLM: {e}", exc_info=self.debug)
        if embedder_to_close:
            try:
                if hasattr(embedder_to_close, "close"):
                    embedder_to_close.close()
                del self.embedder
                self.embedder = None
                logger.debug("Embedder closed.")
            except Exception as e:
                logger.error(f"Error closing Embedder: {e}", exc_info=self.debug)
        if vectordb_to_close:
            try:
                if hasattr(vectordb_to_close, "close"):
                    vectordb_to_close.close()
                del self.vectordb
                self.vectordb = None
                logger.debug("VectorDB closed.")
            except Exception as e:
                logger.error(f"Error closing VectorDB: {e}", exc_info=self.debug)
        self.logger.info("LLMSearch closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

---
### File: src\llamasearch\core\teapot.py

```python
# src/llamasearch/core/teapot.py

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import gc
import torch
import onnxruntime
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# --- Remove unused BatchEncoding ---
from transformers import AutoTokenizer, PreTrainedTokenizer  # , BatchEncoding

# --- Import Project Protocols & Utilities ---
from llamasearch.protocols import LLM, ModelInfo
from llamasearch.hardware import detect_hardware_info, HardwareInfo
from llamasearch.data_manager import data_manager
from llamasearch.utils import setup_logging
from llamasearch.exceptions import ModelNotFoundError, SetupError

logger = setup_logging(__name__)

# --- Constants ---
TEAPOT_REPO_ID = "teapotai/teapotllm"
ONNX_SUBFOLDER = "onnx"
REQUIRED_ONNX_BASENAMES = ["encoder_model", "decoder_model", "decoder_with_past_model"]

# --- ADDED Constant Definition ---
TEAPOT_BASE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",  # T5 Tokenizer specific
]
# --- END ADDED Constant Definition ---


# --- Concrete ModelInfo Implementation ---
class TeapotONNXModelInfo(ModelInfo):
    """Implementation of ModelInfo protocol for Teapot ONNX."""

    def __init__(self, model_id: str, quant_suffix: str, context_len: int):
        self._model_id_base = model_id
        self._quant_suffix = quant_suffix
        self._context_len = context_len

    @property
    def model_id(self) -> str:
        quant_str = self._quant_suffix.lstrip("_") if self._quant_suffix else "fp32"
        return f"{self._model_id_base}-onnx-{quant_str}"

    @property
    def model_engine(self) -> str:
        return "onnx_teapot"

    @property
    def description(self) -> str:
        quant_str = self._quant_suffix.lstrip("_") if self._quant_suffix else "fp32"
        return f"Teapot ONNX model ({quant_str} quantization)"

    @property
    def context_length(self) -> int:
        return self._context_len


# --- Helper functions ---
def _determine_onnx_provider(
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Determines the best available ONNX Runtime provider."""
    provider = preferred_provider or "CPUExecutionProvider"
    options = preferred_options if preferred_options is not None else {}
    available_providers = onnxruntime.get_available_providers()
    logger.debug(f"Available ONNX Runtime providers: {available_providers}")
    if preferred_provider:
        if preferred_provider not in available_providers:
            logger.warning(
                f"Preferred provider '{preferred_provider}' not available. Falling back to CPU."
            )
            provider, options = "CPUExecutionProvider", {}
        else:
            provider = preferred_provider
            logger.info(f"Using preferred ONNX provider: {provider}")
    else:
        if "CUDAExecutionProvider" in available_providers:
            provider, options["device_id"] = (
                "CUDAExecutionProvider",
                options.get("device_id", 0),
            )
        elif "ROCMExecutionProvider" in available_providers:
            provider, options["device_id"] = (
                "ROCMExecutionProvider",
                options.get("device_id", 0),
            )
        elif "CoreMLExecutionProvider" in available_providers:
            provider = "CoreMLExecutionProvider"
        else:
            provider = "CPUExecutionProvider"
        logger.info(f"Auto-selecting available ONNX provider: {provider}")
    return provider, options if options else None


def _select_onnx_quantization(
    hw: HardwareInfo,
    onnx_provider: str,
    onnx_provider_opts: Optional[Dict[str, Any]],
    preference: str,
) -> str:
    """Selects the most appropriate ONNX quantization suffix."""
    preference_map = {
        p: f"_{p}" for p in ["fp16", "int8", "q4", "q4f16", "bnb4", "uint8"]
    }
    preference_map["fp32"] = ""
    if preference != "auto":
        if preference in preference_map:
            logger.info(f"Using user-preferred ONNX quantization: {preference}")
            return preference_map[preference]
        else:
            logger.warning(
                f"Invalid quantization preference '{preference}'. Falling back to 'auto'."
            )
    logger.info("Performing automatic ONNX quantization selection...")
    req_fp32_gb, req_fp16_gb, req_int8_gb, req_q4_gb = 8.0, 5.0, 3.0, 2.0
    ram_gb, has_avx2 = hw.memory.available_gb, hw.cpu.supports_avx2
    logger.info(
        f"System Info - Available RAM: {ram_gb:.1f} GB, AVX2: {has_avx2}, Provider: {onnx_provider}"
    )
    is_gpu = (
        "CUDAExecutionProvider" in onnx_provider
        or "ROCMExecutionProvider" in onnx_provider
    )
    is_coreml = "CoreMLExecutionProvider" in onnx_provider
    selected_quant = "_bnb4"
    if is_gpu:
        if ram_gb >= req_fp16_gb + 5.0:
            selected_quant = "_fp16"
        elif ram_gb >= req_int8_gb + 2.0:
            selected_quant = "_int8"
        elif ram_gb >= req_q4_gb + 1.0:
            selected_quant = "_q4"
        else:
            selected_quant = "_bnb4"
        logger.info(f"GPU Selection based on System RAM: {selected_quant}")
    elif is_coreml:
        selected_quant = "_int8"
        logger.info(f"CoreML Selection: {selected_quant}")
    else:  # CPU Logic
        if ram_gb >= req_fp32_gb:
            selected_quant = ""
        elif ram_gb >= req_fp16_gb:
            selected_quant = "_fp16"
        elif ram_gb >= req_int8_gb:
            selected_quant = "_int8"
            if not has_avx2:
                logger.warning(
                    "Selecting INT8 on CPU without detected AVX2 support. Performance might be suboptimal."
                )
        elif ram_gb >= req_q4_gb:
            selected_quant = "_q4"
        else:
            selected_quant = "_bnb4"
        logger.info(f"CPU Selection based on RAM/AVX2: {selected_quant}")
    return selected_quant


# --- TeapotONNXLLM Wrapper Class ---
class TeapotONNXLLM(LLM):
    """Wraps the loaded Teapot ONNX model and tokenizer."""

    def __init__(
        self,
        model: Any,  # Keep Any type hint
        tokenizer: PreTrainedTokenizer,
        quant_suffix: str,
        provider: str,
        provider_options: Optional[Dict[str, Any]],
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._info = TeapotONNXModelInfo(
            TEAPOT_REPO_ID, quant_suffix, tokenizer.model_max_length or 512
        )
        self._is_loaded = True
        self._provider = provider
        self._provider_options = provider_options
        logger.info(
            f"Initialized {self.model_info.model_id} wrapper on device '{self.device}' with provider {self._provider}"
        )

    @property
    def model_info(self) -> ModelInfo:
        """Returns model metadata."""
        return self._info

    @property
    def device(self) -> torch.device:
        """Determines the effective device the model is running on."""
        # --- Add assertion ---
        assert self._is_loaded and self._model is not None, (
            "Model must be loaded to access device."
        )
        # --- End assertion ---
        if hasattr(self._model, "device") and isinstance(
            self._model.device, torch.device
        ):
            return self._model.device
        provider_name = self._provider
        if provider_name == "CUDAExecutionProvider":
            return torch.device("cuda")
        if provider_name == "ROCMExecutionProvider":
            return torch.device("rocm")
        if provider_name == "CoreMLExecutionProvider":
            return (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        return torch.device("cpu")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[str, Any]:
        """Generates text using the loaded ONNX model."""
        if not self._is_loaded:
            return "Error: Model not loaded.", {"error": "Model not loaded"}
        # --- Add assertions ---
        assert self._model is not None, "Model cannot be None during generation."
        assert self._tokenizer is not None, (
            "Tokenizer cannot be None during generation."
        )
        # --- End assertions ---
        try:
            target_device = self.device
            max_input_length = self.model_info.context_length - max_tokens
            if max_input_length <= 0:
                logger.warning(
                    f"max_tokens ({max_tokens}) exceeds context length ({self.model_info.context_length}). Truncating severely."
                )
                max_input_length = max(10, self.model_info.context_length // 2)

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_length,
                padding=False,
            ).to(target_device)

            gen_kwargs = {"max_new_tokens": max_tokens, **kwargs}
            # Sampling logic
            if temperature > 0.0 or (top_p is not None and top_p < 1.0):
                gen_kwargs["do_sample"] = True
                if temperature > 0.0:
                    gen_kwargs["temperature"] = temperature
                if top_p is not None and top_p < 1.0:
                    gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False
            if repeat_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = repeat_penalty

            input_ids = inputs.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("Tokenizer did not return a Tensor for input_ids")
            in_tokens = input_ids.shape[1]
            logger.debug(
                f"Generating ONNX. Input Tokens:{in_tokens}, Device:{target_device}, Kwargs:{gen_kwargs}"
            )

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            out_ids = outputs[0]
            result = self._tokenizer.decode(out_ids, skip_special_tokens=True)
            out_tokens = out_ids.shape[0] if isinstance(out_ids, torch.Tensor) else 0

            return result.strip(), {
                "output_token_count": out_tokens,
                "input_token_count": in_tokens,
            }
        except Exception as e:
            logger.error(f"ONNX generation error: {e}", exc_info=True)
            return f"Error during generation: {e}", {"error": str(e)}

    def load(self) -> bool:
        """Returns loading status (model is loaded on init)."""
        return self._is_loaded

    def unload(self) -> None:
        """Unloads the model and attempts garbage collection."""
        if not self._is_loaded:
            return
        logger.info(f"Unloading TeapotONNXLLM ({self.model_info.model_id})...")
        # Get device type *before* deleting the model
        dev_type = "cpu"  # Default
        try:
            if self._model is not None:
                dev_type = self.device.type  # Get device type if model exists
        except Exception:
            logger.warning("Could not determine device type before unloading.")

        try:
            if hasattr(self, "_model"):
                del self._model
            if hasattr(self, "_tokenizer"):
                del self._tokenizer
        except Exception as e:
            logger.error(f"Error deleting internal model/tokenizer references: {e}")

        # Set internal attributes to None
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        gc.collect()
        if dev_type == "cuda" and torch.cuda.is_available():
            logger.debug("Clearing CUDA cache after unloading TeapotONNXLLM.")
            torch.cuda.empty_cache()
        logger.info("TeapotONNXLLM unloaded.")


# --- Loader Function ---
def load_teapot_onnx_llm(
    onnx_quantization: str = "auto",
    preferred_provider: Optional[str] = None,
    preferred_options: Optional[Dict[str, Any]] = None,
) -> Optional[TeapotONNXLLM]:
    """Loads the Teapot ONNX model from the assembled 'active_teapot' directory."""
    logger.info(
        f"--- Initializing Teapot ONNX LLM (Quant Pref: {onnx_quantization}) ---"
    )
    onnx_model, tokenizer = None, None
    try:
        hw_info = detect_hardware_info()
        provider, options = _determine_onnx_provider(
            preferred_provider, preferred_options
        )
        quant_suffix = _select_onnx_quantization(
            hw_info, provider, options, onnx_quantization
        )
        logger.info(
            f"Loading with ONNX Provider: {provider}, Target Quant Suffix: '{quant_suffix}'"
        )

        paths = data_manager.get_data_paths()
        models_dir_str = paths.get("models")
        if not models_dir_str:
            raise SetupError(
                "Models directory path not found in data_manager settings."
            )
        model_cache_dir = Path(models_dir_str)
        active_model_dir = model_cache_dir / "active_teapot"

        logger.info(
            f"Attempting to load from active model directory: {active_model_dir}"
        )

        # Verification of active_model_dir content
        if not active_model_dir.is_dir():
            raise ModelNotFoundError(
                f"Active directory '{active_model_dir}' not found. Please run 'llamasearch-setup'."
            )

        # --- Corrected: Use TEAPOT_BASE_FILES constant ---
        required_files_rel = list(TEAPOT_BASE_FILES)  # Use the defined constant
        # --- End Correction ---
        onnx_sub = active_model_dir / ONNX_SUBFOLDER
        if not onnx_sub.is_dir():
            raise ModelNotFoundError(
                f"ONNX subfolder missing in {active_model_dir}. Please run 'llamasearch-setup'."
            )
        for basename in REQUIRED_ONNX_BASENAMES:
            onnx_fname = f"{basename}{quant_suffix}.onnx"
            required_files_rel.append(f"{ONNX_SUBFOLDER}/{onnx_fname}")

        missing_files = []
        for req_file_rel in required_files_rel:
            req_file_abs = active_model_dir / req_file_rel
            if not req_file_abs.exists():
                missing_files.append(req_file_rel)
        if missing_files:
            logger.error(
                f"Required model files missing in {active_model_dir}: {missing_files}"
            )
            raise ModelNotFoundError(
                f"Required files missing in '{active_model_dir}'. Please run 'llamasearch-setup'."
            )

        # Load from the verified 'active_teapot' directory
        logger.debug(f"Loading ONNX model components from {active_model_dir}...")
        onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            active_model_dir,
            export=False,
            provider=provider,
            provider_options=options,
            use_io_binding=(
                "CUDAExecutionProvider" in provider
                or "ROCMExecutionProvider" in provider
            ),
            library_name="transformers",
            local_files_only=True,
        )
        logger.debug(f"Loading tokenizer from {active_model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(
            active_model_dir, use_fast=True, local_files_only=True
        )

        logger.info(
            "Teapot ONNX model and tokenizer loaded successfully from active directory."
        )
        llm_instance = TeapotONNXLLM(
            onnx_model, tokenizer, quant_suffix, provider, options
        )
        return llm_instance

    except ModelNotFoundError:
        logger.error(
            "ModelNotFoundError occurred during LLM init. Setup might be needed."
        )
        raise
    except Exception as e:
        logger.error(
            f"Failed during Teapot ONNX LLM initialization: {e}", exc_info=True
        )
        if onnx_model is not None:
            del onnx_model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"Failed to load Teapot ONNX model ({e.__class__.__name__}): {e}"
        ) from e
```

---
### File: src\llamasearch\core\vectordb.py

```python
# src/llamasearch/core/vectordb.py

import os
import re
import json
import shutil
import subprocess
import numpy as np
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

from llamasearch.utils import setup_logging
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.chunker import MarkdownChunker
from llamasearch.core.bm25 import BM25Retriever

from sklearn.metrics.pairwise import cosine_similarity

logger = setup_logging(__name__)


class VectorDB:
    """
    VectorDB integrates:
      - File processing (conversion, chunking)
      - Vector-based similarity (via embeddings)
      - BM25-based keyword retrieval
    """

    def __init__(
        self,
        storage_dir: Path,
        collection_name: str,
        embedder: Optional[EnhancedEmbedder] = None,
        # Chunker settings used internally
        max_chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        # Embedder settings
        text_embedding_size: int = 768,  # Potentially redundant
        embedder_batch_size: int = 32,
        # Query settings
        similarity_threshold: float = 0.2,
        max_results: int = 3,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        # Other settings
        device: str = "cpu",
        enable_deduplication: bool = True,
        dedup_similarity_threshold: float = 0.8,
    ):
        self.collection_name = collection_name
        self.max_chunk_size = max_chunk_size
        self.text_embedding_size = (
            text_embedding_size  # Consider deriving from embedder if possible
        )
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.embedder_batch_size = embedder_batch_size
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.storage_dir = storage_dir
        self.device = device
        self.enable_deduplication = enable_deduplication
        self.dedup_similarity_threshold = dedup_similarity_threshold

        collection_dir = self.storage_dir / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir = collection_dir / "vector"
        self.bm25_dir = collection_dir / "bm25"
        self.vector_dir.mkdir(exist_ok=True)
        self.bm25_dir.mkdir(exist_ok=True)

        self.metadata_path = self.vector_dir / "meta.json"
        self.embeddings_path = self.vector_dir / "embeddings.npy"

        if embedder is None:
            logger.info(
                f"No embedder provided, creating EnhancedEmbedder with batch_size={self.embedder_batch_size}, device={self.device}"
            )
            self.embedder = EnhancedEmbedder(
                batch_size=self.embedder_batch_size, device=self.device
            )
        else:
            self.embedder = embedder

        logger.info("Initializing Markdown Chunker for VectorDB.")
        self.markdown_chunker = MarkdownChunker(
            max_chunk_size=self.max_chunk_size,
            min_chunk_size=self.min_chunk_size,
            overlap_percent=(self.chunk_overlap / self.max_chunk_size)
            if self.max_chunk_size > 0
            else 0.1,
            combine_under_min_size=True,
        )

        self.bm25 = BM25Retriever(storage_dir=str(self.bm25_dir))  # Pass string path

        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []
        self.processed_chunks: List[Set[str]] = []  # Tokens for deduplication

        self._load_metadata()

    def _process_file_to_markdown(self, file_path: Path) -> Path:
        """
        If docx/pdf => convert with pandoc. Return final path (md or original).
        Handles pandoc check and conversion errors.
        Returns the path to the (potentially temporary) markdown file or the original path.
        """
        ext = file_path.suffix.lower()
        supported_conversion_ext = [
            ".docx",
            ".doc",
            ".pdf",
            ".odt",
            ".rtf",
        ]  # Add more if pandoc supports them
        passthrough_ext = [".md", ".markdown", ".txt", ".html", ".htm"]

        if ext in passthrough_ext:
            return file_path
        elif ext in supported_conversion_ext:
            pandoc_path = shutil.which("pandoc")
            if not pandoc_path:
                logger.warning(
                    f"pandoc not found, cannot convert {file_path.name}. Using original (results may vary)."
                )
                return file_path

            # Use a temporary file within the vector store's directory
            temp_dir = self.vector_dir / "temp_conversions"
            temp_dir.mkdir(exist_ok=True)
            # Create a unique temp filename based on hash of original path?
            # For simplicity, using original name + .md.temp
            mdfile = temp_dir / (file_path.stem + ".md.temp")

            # Avoid re-converting if a valid temp file exists (e.g., from previous run)
            if mdfile.exists() and mdfile.stat().st_size > 10:
                logger.debug(f"Using existing temporary markdown file: {mdfile}")
                return mdfile
            elif mdfile.exists():  # Exists but is tiny/empty
                try:
                    mdfile.unlink()
                except OSError:
                    pass

            cmd = [pandoc_path, str(file_path), "-t", "markdown", "-o", str(mdfile)]
            logger.info(f"Converting {file_path.name} => {mdfile.name} using pandoc")
            try:
                res = subprocess.run(cmd, capture_output=True, check=False, timeout=120)
                if res.returncode != 0:
                    stderr_output = (
                        res.stderr.decode("utf-8", "ignore")
                        if res.stderr
                        else "No stderr."
                    )
                    logger.error(
                        f"Pandoc conversion error (Code {res.returncode}) for {file_path.name}: {stderr_output}"
                    )
                    if mdfile.exists():
                        mdfile.unlink()  # Clean up failed file
                    return file_path
                logger.info(f"Successfully converted {file_path.name} to Markdown.")
                return mdfile
            except FileNotFoundError:
                logger.error("Pandoc command failed. Is pandoc installed and in PATH?")
                return file_path
            except subprocess.TimeoutExpired:
                logger.error(f"Pandoc conversion timed out for {file_path.name}.")
                if mdfile.exists():
                    mdfile.unlink()
                return file_path
            except Exception as e:
                logger.error(f"Pandoc conversion exception for {file_path.name}: {e}")
                if mdfile.exists():
                    mdfile.unlink()
                return file_path
        else:
            logger.debug(
                f"File type {file_path.suffix} not configured for conversion, using original."
            )
            return file_path

    def add_source(self, source_path: Path) -> int:
        """
        Adds a single source file or directory recursively to the VectorDB.
        Handles file reading, conversion (if necessary), chunking, embedding,
        and storage. Skips already processed and unchanged files.
        """
        source_path = Path(source_path).resolve()  # Use resolved path
        total_chunks_added = 0

        if not source_path.exists():
            logger.error(f"Source path not found: {source_path}")
            self._remove_document(str(source_path))  # Use resolved path string
            return 0

        if source_path.is_file():
            source_id = str(source_path)  # Use resolved path string as ID
            if self.is_document_processed(source_id):
                logger.info(f"File already processed and unchanged: {source_path.name}")
                return 0

            processed_path = self._process_file_to_markdown(source_path)
            temp_file_used = processed_path.name.endswith(".md.temp")

            if not processed_path.is_file():
                logger.warning(
                    f"Skipping non-file path after conversion attempt: {processed_path}"
                )
                if temp_file_used:
                    try:
                        processed_path.unlink()
                    except OSError:
                        pass
                return 0

            try:
                logger.debug(f"Reading content from {processed_path}")
                content = processed_path.read_text(encoding="utf-8", errors="ignore")

                logger.debug(f"Chunking content from {processed_path.name}")
                # Pass resolved source_id to chunker metadata
                chunks = list(
                    self.markdown_chunker.chunk_document(content, source=source_id)
                )

                if not chunks:
                    logger.warning(
                        f"No chunks generated from {processed_path.name}. Skipping."
                    )
                    added_count = 0
                else:
                    logger.info(
                        f"Adding {len(chunks)} chunks from {source_path.name} (processed as {processed_path.name}) to VectorDB."
                    )
                    added_count = self.add_document_chunks(source_id, chunks)

                total_chunks_added += added_count

            except Exception as e:
                logger.error(
                    f"Failed to process and add file {source_path}: {e}", exc_info=True
                )
            finally:
                # Clean up temporary markdown file if it was created
                if temp_file_used and processed_path.exists():
                    try:
                        processed_path.unlink()
                        logger.debug(
                            f"Removed temporary markdown file: {processed_path}"
                        )
                    except OSError as e_unlink:
                        logger.warning(
                            f"Could not remove temporary file {processed_path}: {e_unlink}"
                        )

        elif source_path.is_dir():
            logger.info(f"Processing directory recursively: {source_path}")
            files_processed = 0
            files_failed = []
            supported_suffixes = [
                ".md",
                ".markdown",
                ".txt",
                ".html",
                ".htm",
                ".pdf",
                ".docx",
                ".doc",
                ".odt",
                ".rtf",
            ]
            for item in source_path.rglob("*"):
                if item.is_file() and item.suffix.lower() in supported_suffixes:
                    try:
                        added = self.add_source(item)  # Recursive call for each file
                        if added > 0:
                            total_chunks_added += added
                            files_processed += 1
                        elif not self.is_document_processed(str(item.resolve())):
                            files_failed.append(item.name)
                    except Exception as e:
                        logger.error(
                            f"Error processing file {item} within directory: {e}"
                        )
                        files_failed.append(item.name)
                elif item.is_file():
                    logger.debug(
                        f"Skipping unsupported file type in directory: {item.name}"
                    )
            logger.info(
                f"Directory scan complete for {source_path}. Added {total_chunks_added} new chunks from {files_processed} files."
            )
            if files_failed:
                logger.warning(
                    f"Failed to process or add chunks for {len(files_failed)} files in directory: {', '.join(files_failed)}"
                )
        else:
            logger.warning(
                f"Source path is neither a file nor a directory: {source_path}"
            )

        # Consolidate BM25 index build after processing all sources from a top-level call
        # This is tricky with recursion. A better approach might be a flag or separate build step.
        # For now, let add_document_chunks handle the rebuild, accepting inefficiency for directories.
        # if total_chunks_added > 0 and hasattr(self.bm25, '_index_needs_rebuild') and self.bm25._index_needs_rebuild:
        #      logger.info("Attempting BM25 index build after processing source(s).")
        #      self.bm25.build_index()

        return total_chunks_added

    def _load_metadata(self) -> bool:
        """Load document metadata and rebuild processed_chunks set."""
        if not self.metadata_path.exists():
            logger.info(f"No metadata file at {self.metadata_path}. Starting fresh.")
            return False
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.document_metadata = data.get("metadata", [])
            if len(self.documents) != len(self.document_metadata):
                logger.error("Mismatch in loaded docs vs metadata length. Resetting.")
                self.documents, self.document_metadata = [], []
                self.processed_chunks = []
                return False

            logger.info(f"Loaded {len(self.documents)} documents from metadata.")
            # Rebuild processed_chunks tokens on load for deduplication check
            self.processed_chunks = [self._get_tokens(doc) for doc in self.documents]
            logger.debug(
                f"Rebuilt {len(self.processed_chunks)} token sets for deduplication."
            )
            return True
        except Exception as e:
            logger.error(f"Error loading metadata: {e}. Resetting.")
            self.documents, self.document_metadata, self.processed_chunks = [], [], []
            return False

    def _save_metadata(self) -> None:
        """Save document text and metadata atomically."""
        logger.debug(f"Saving metadata for {len(self.documents)} docs.")
        temp_path = self.metadata_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"documents": self.documents, "metadata": self.document_metadata},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(temp_path, self.metadata_path)  # Atomic replace
            logger.debug("Metadata saved.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _get_tokens(self, text: str) -> Set[str]:
        """Simple tokenization (lowercase words) for Jaccard similarity."""
        return set(re.findall(r"\b\w+\b", text.lower()))

    def is_document_processed(self, source_id: str) -> bool:
        """
        Checks if a document (identified by source_id) is already processed
        and unchanged based on its mtime stored in metadata.
        If the file has changed or doesn't exist, removes old data associated with it.
        """
        source_path = Path(source_id)
        if not source_path.exists():
            logger.warning(
                f"Source file {source_id} not found. Removing any existing data."
            )
            self._remove_document(source_id)
            return False  # Can't be processed if it doesn't exist

        try:
            current_mtime = source_path.stat().st_mtime
        except OSError as e:
            logger.error(f"Could not get mtime for {source_id}: {e}. Assuming changed.")
            self._remove_document(source_id)
            return False  # Treat as changed if mtime fails

        needs_removal = False
        is_present_unchanged = False
        for i, md in enumerate(self.document_metadata):
            if md.get("source") == source_id:
                stored_mtime = md.get("mtime")
                if (
                    stored_mtime is not None
                    and abs(stored_mtime - current_mtime) < 1e-4
                ):
                    is_present_unchanged = True
                    break  # Found and unchanged, stop checking
                else:
                    # Found, but mtime differs or was missing
                    logger.info(
                        f"Detected change in {source_id}. Marked for removal and re-processing."
                    )
                    needs_removal = True
                    break  # Mark for removal

        if needs_removal:
            self._remove_document(source_id)
            return False  # Needs reprocessing
        elif is_present_unchanged:
            return True  # Already processed and unchanged
        else:
            return False  # Not found in metadata

    def _remove_document(self, source_id: str) -> None:
        """
        Removes all data (chunks, metadata, embeddings, BM25 entries)
        associated with the given source_id.
        """
        old_doc_count = len(self.documents)
        indices_to_remove = {
            i
            for i, m in enumerate(self.document_metadata)
            if m.get("source") == source_id
        }

        if not indices_to_remove:
            logger.debug(f"No existing chunks found for source {source_id} to remove.")
            return

        logger.info(
            f"Removing {len(indices_to_remove)} chunks associated with source: {source_id}"
        )

        # Filter out documents, metadata, and processed chunks
        new_docs = []
        new_meta = []
        new_processed = []
        original_indices_kept = []  # Store original indices of items we keep
        for i in range(old_doc_count):
            if i not in indices_to_remove:
                new_docs.append(self.documents[i])
                new_meta.append(self.document_metadata[i])
                new_processed.append(self.processed_chunks[i])
                original_indices_kept.append(i)

        num_removed = old_doc_count - len(new_docs)
        if num_removed > 0:
            self.documents = new_docs
            self.document_metadata = new_meta
            self.processed_chunks = new_processed

            # Update Embeddings
            if (
                self.embeddings_path.exists()
                and self.embeddings_path.stat().st_size > 0
            ):
                try:
                    all_embeddings = np.load(self.embeddings_path, allow_pickle=False)
                    if len(original_indices_kept) > all_embeddings.shape[0]:
                        logger.error(
                            f"Metadata inconsistency: trying to keep {len(original_indices_kept)} indices but only {all_embeddings.shape[0]} embeddings exist."
                        )
                        # Decide recovery strategy: maybe delete embeddings and force rebuild?
                        # For now, proceed cautiously, might lead to errors later.
                        valid_indices_kept = [
                            idx
                            for idx in original_indices_kept
                            if idx < all_embeddings.shape[0]
                        ]
                    else:
                        valid_indices_kept = original_indices_kept

                    if valid_indices_kept:
                        filtered_embeddings = all_embeddings[valid_indices_kept]
                    else:
                        # If no embeddings are kept, create an empty array with correct dimensions
                        dim = (
                            all_embeddings.shape[1]
                            if all_embeddings.ndim > 1
                            else self.text_embedding_size
                        )
                        filtered_embeddings = np.empty((0, dim), dtype=np.float32)

                    # Save updated embeddings atomically
                    temp_embed_path = self.embeddings_path.with_suffix(".npy.tmp")
                    np.save(temp_embed_path, filtered_embeddings, allow_pickle=False)
                    os.replace(temp_embed_path, self.embeddings_path)
                    logger.debug(
                        f"Updated embeddings file. New shape: {filtered_embeddings.shape}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error updating embeddings file after removal: {e}. State might be inconsistent.",
                        exc_info=True,
                    )
                    # Consider deleting embeddings file to force rebuild?
                    self.embeddings_path.unlink(
                        missing_ok=True
                    )  # Delete potentially corrupt file

            # Update BM25 - requires removing and rebuilding
            bm25_needs_rebuild = False
            doc_ids_to_remove = [f"doc_{i}" for i in indices_to_remove]
            for doc_id in doc_ids_to_remove:
                if self.bm25.remove_document(
                    doc_id
                ):  # remove_document now just updates lists
                    bm25_needs_rebuild = True

            # Save updated state immediately
            self._save_metadata()
            if bm25_needs_rebuild:
                logger.info("Rebuilding BM25 index after document removal.")
                self.bm25.build_index()

            logger.info(
                f"Successfully removed {num_removed} chunks for source {source_id}."
            )
        else:
            logger.debug(
                f"No chunks were actually removed for {source_id} (consistency check)."
            )

    def add_document_chunks(self, source_id: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Add pre-generated chunks for a single document source ID.
        Computes embeddings, updates BM25, handles deduplication if enabled.
        """
        if not chunks:
            logger.warning(f"No chunks provided for source_id: {source_id}")
            return 0

        original_file_path = Path(source_id)
        doc_mtime = (
            original_file_path.stat().st_mtime if original_file_path.exists() else 0
        )
        base_meta = {
            "source": source_id,
            "filename": original_file_path.name,
            "mtime": doc_mtime,
        }

        # Deduplication and Filtering
        filtered_chunks_data = []
        num_skipped_duplicates = 0
        temp_new_tokens = []  # Track tokens for intra-doc deduplication

        for i, ch_data in enumerate(chunks):
            text = ch_data.get("chunk", "")
            if not text:
                continue

            # Intra-document check first
            current_tokens = self._get_tokens(text)
            is_intra_doc_duplicate = False
            if self.enable_deduplication:
                for existing_tokens in temp_new_tokens:
                    intersection = len(current_tokens.intersection(existing_tokens))
                    union = len(current_tokens.union(existing_tokens))
                    if (
                        union > 0
                        and (intersection / union) > self.dedup_similarity_threshold
                    ):
                        is_intra_doc_duplicate = True
                        break

            # Then check against global index
            is_global_duplicate = (
                self._is_duplicate_chunk(text) if self.enable_deduplication else False
            )

            if is_intra_doc_duplicate or is_global_duplicate:
                # logger.debug(f"Skipping duplicate chunk {i} for {source_id} (Intra: {is_intra_doc_duplicate}, Global: {is_global_duplicate}): {text[:50]}...")
                num_skipped_duplicates += 1
                continue

            # If not duplicate, add tokens for intra-doc check and prepare data
            if self.enable_deduplication:
                temp_new_tokens.append(current_tokens)

            chunk_meta = base_meta.copy()
            # Merge metadata, giving priority to chunk-specific keys if they exist
            chunk_meta.update(ch_data.get("metadata", {}))
            # Ensure chunk_id from chunker is preserved if present
            if (
                "chunk_id" not in chunk_meta
                and ch_data.get("metadata", {}).get("chunk_id") is not None
            ):
                chunk_meta["chunk_id"] = ch_data["metadata"]["chunk_id"]
            # Add index within the *original* list of chunks for reference
            chunk_meta["original_chunk_index"] = i

            filtered_chunks_data.append((text, chunk_meta))

        if num_skipped_duplicates > 0:
            logger.debug(
                f"Skipped {num_skipped_duplicates} duplicate chunks for {source_id}."
            )

        if not filtered_chunks_data:
            logger.warning(
                f"No valid (non-empty, non-duplicate) chunks to add for {source_id}"
            )
            return 0

        # Batch Processing
        doc_start_idx = len(self.documents)
        batch_size = self.embedder_batch_size
        all_batches = [
            filtered_chunks_data[i : i + batch_size]
            for i in range(0, len(filtered_chunks_data), batch_size)
        ]

        temp_embed_path = self.embeddings_path.with_suffix(".npy.tmp")
        if temp_embed_path.exists():
            temp_embed_path.unlink()

        added_chunks_count = 0
        new_texts = []
        new_metas = []
        new_embeddings_list = []
        bm25_needs_rebuild = False

        try:
            for batch_idx, batch in enumerate(all_batches):
                texts_in_batch = [item[0] for item in batch]
                metadata_in_batch = [item[1] for item in batch]

                emb_array = self.embedder.embed_strings(
                    texts_in_batch, show_progress=False
                )  # Use embed_strings
                if emb_array is None or emb_array.size == 0:
                    logger.warning(
                        f"Embedder returned empty result for batch {batch_idx} of {source_id}. Skipping."
                    )
                    continue

                emb_array = np.array(emb_array, dtype=np.float32)
                new_embeddings_list.append(emb_array)
                new_texts.extend(texts_in_batch)
                new_metas.extend(metadata_in_batch)

                # Add tokens to global list for future deduplication checks
                if self.enable_deduplication:
                    for text_val in texts_in_batch:
                        self.processed_chunks.append(self._get_tokens(text_val))

                added_chunks_count += len(batch)
                # logger.debug(f"Processed batch {batch_idx+1}/{len(all_batches)} for {source_id}, added {len(batch)} chunks.")

            # Post-Batch Processing
            if added_chunks_count > 0 and new_embeddings_list:
                final_new_embeddings = np.vstack(new_embeddings_list)
                np.save(temp_embed_path, final_new_embeddings, allow_pickle=False)
                self._merge_embeddings(str(temp_embed_path))  # Pass path string

                self.documents.extend(new_texts)
                self.document_metadata.extend(new_metas)

                # Add to BM25
                for i in range(added_chunks_count):
                    final_doc_idx = doc_start_idx + i
                    doc_id = f"doc_{final_doc_idx}"  # Use index-based ID for BM25 consistency
                    try:
                        self.bm25.add_document(new_texts[i], doc_id)
                        bm25_needs_rebuild = True
                    except Exception as e:
                        logger.error(
                            f"Failed to add document {doc_id} to BM25 index: {e}"
                        )

                self._save_metadata()  # Save after updates

                if bm25_needs_rebuild:
                    logger.info(
                        f"Rebuilding BM25 index after adding {added_chunks_count} chunks for {source_id}."
                    )
                    self.bm25.build_index()

            if temp_embed_path.exists():
                temp_embed_path.unlink(missing_ok=True)

            logger.info(
                f"Successfully processed {added_chunks_count} chunks for source {source_id}"
            )
            return added_chunks_count

        except Exception as e:
            logger.error(
                f"Error during batch embedding/saving for {source_id}: {e}",
                exc_info=True,
            )
            if temp_embed_path.exists():
                temp_embed_path.unlink(missing_ok=True)
            # Rollback attempt
            try:
                if added_chunks_count > 0:
                    del self.documents[-added_chunks_count:]
                    del self.document_metadata[-added_chunks_count:]
                    if self.enable_deduplication:
                        # Rollback processed_chunks (might be imperfect if error was mid-batch)
                        del self.processed_chunks[-added_chunks_count:]
                    logger.warning(
                        f"Attempted rollback of {added_chunks_count} chunks for {source_id} after error."
                    )
                    # Force metadata save after rollback attempt
                    self._save_metadata()
            except Exception as rollback_e:
                logger.error(f"Error during rollback: {rollback_e}")
            return 0  # Indicate failure

    def _merge_embeddings(self, temp_path_str: str) -> None:
        """Merge new embeddings from temp file with existing embeddings file."""
        temp_path = Path(temp_path_str)
        logger.debug(
            f"Merging new embeddings from {temp_path} into {self.embeddings_path}"
        )
        if self.embeddings_path.exists() and self.embeddings_path.stat().st_size > 0:
            try:
                old_arr = np.load(self.embeddings_path, allow_pickle=False)
                new_arr = np.load(temp_path, allow_pickle=False)

                if old_arr.ndim == 1:
                    old_arr = old_arr.reshape(1, -1)  # Handle edge case
                if new_arr.ndim == 1:
                    new_arr = new_arr.reshape(1, -1)  # Handle edge case

                if (
                    old_arr.size > 0
                    and new_arr.size > 0
                    and old_arr.shape[1:] != new_arr.shape[1:]
                ):
                    raise ValueError(
                        f"Embedding dimension mismatch: Existing {old_arr.shape}, New {new_arr.shape}"
                    )
                elif old_arr.size == 0 and new_arr.size > 0:
                    merged = new_arr  # Existing was empty
                elif old_arr.size > 0 and new_arr.size == 0:
                    merged = old_arr  # New is empty
                elif old_arr.size > 0 and new_arr.size > 0:
                    merged = np.vstack([old_arr, new_arr])
                else:  # Both empty
                    merged = np.empty((0, self.text_embedding_size), dtype=np.float32)

                # Save merged array atomically
                final_temp_path = self.embeddings_path.with_suffix(".npy.tmp")
                np.save(final_temp_path, merged, allow_pickle=False)
                os.replace(final_temp_path, self.embeddings_path)
                logger.debug(f"Embeddings merged. New shape: {merged.shape}")
            except Exception as e:
                logger.error(f"Error merging embeddings: {e}", exc_info=True)
                final_temp_path = self.embeddings_path.with_suffix(".npy.tmp")
                if final_temp_path.exists():
                    final_temp_path.unlink(missing_ok=True)
                raise
        else:
            # No existing file or it's empty, just move the temporary file
            logger.debug(
                f"Creating new embeddings file at {self.embeddings_path} from {temp_path}"
            )
            shutil.move(str(temp_path), self.embeddings_path)

    def _is_duplicate_chunk(self, text: str) -> bool:
        """Deduplicate using Jaccard similarity over token sets."""
        if not self.enable_deduplication:
            return False
        chunk_tokens = self._get_tokens(text)
        if not chunk_tokens:
            return False

        # Check against existing chunks already loaded/added globally
        # Check the most recent N first for efficiency
        check_limit = min(len(self.processed_chunks), 500)  # Limit check scope
        indices_to_check = range(
            len(self.processed_chunks) - 1,
            len(self.processed_chunks) - 1 - check_limit,
            -1,
        )

        for i in indices_to_check:
            if i < 0:
                break  # Boundary check
            existing_tokens = self.processed_chunks[i]
            if not existing_tokens:
                continue

            intersection = len(chunk_tokens.intersection(existing_tokens))
            union = len(chunk_tokens.union(existing_tokens))
            if union > 0:
                sim = intersection / union
                if sim > self.dedup_similarity_threshold:
                    # logger.debug(f"Duplicate detected (Jaccard > {self.dedup_similarity_threshold:.2f})")
                    return True
        return False

    def _vector_search(self, query_text: str, n_ret: int) -> Dict[str, Any]:
        """Vector search using cosine similarity."""
        empty_res = {"query": query_text, "ids": [], "scores": [], "score_details": []}
        query_text = query_text.strip()
        if not query_text:
            return empty_res
        if (
            not self.embeddings_path.exists()
            or self.embeddings_path.stat().st_size == 0
        ):
            logger.warning(f"No embeddings found or empty at {self.embeddings_path}")
            return empty_res

        try:
            all_embeddings = np.load(self.embeddings_path, allow_pickle=False)
            if all_embeddings.size == 0:
                logger.warning("Embeddings file loaded but contains no data.")
                return empty_res
            # Ensure 2D array
            if all_embeddings.ndim == 1:
                expected_dim = self.text_embedding_size
                if all_embeddings.shape[0] == expected_dim:
                    all_embeddings = all_embeddings.reshape(1, -1)
                else:
                    logger.error(
                        f"Loaded 1D embedding array with unexpected shape {all_embeddings.shape}. Expected ({expected_dim},). Cannot reshape."
                    )
                    return empty_res

            q_emb = self.embedder.embed_string(query_text)
            if q_emb is None or q_emb.size == 0:
                logger.error("Failed to generate query embedding.")
                return empty_res
            q_emb_np = np.array(q_emb, dtype=np.float32).reshape(1, -1)

            if q_emb_np.shape[1] != all_embeddings.shape[1]:
                logger.error(
                    f"Query embedding dimension ({q_emb_np.shape[1]}) != stored dimension ({all_embeddings.shape[1]})"
                )
                return empty_res

            scores = cosine_similarity(q_emb_np, all_embeddings)[0]
            num_available = len(scores)
            actual_n_ret = min(n_ret, num_available)
            if actual_n_ret == 0:
                return empty_res

            # Use argpartition for efficiency if n_ret << num_available
            if actual_n_ret < num_available // 2 and num_available > 100:
                top_indices_unsorted = np.argpartition(scores, -actual_n_ret)[
                    -actual_n_ret:
                ]
                top_indices = top_indices_unsorted[
                    np.argsort(scores[top_indices_unsorted])
                ][::-1]
            else:
                top_indices = np.argsort(scores)[::-1][:actual_n_ret]

            # Filter by similarity threshold AFTER finding top N candidates
            final_indices = [
                i for i in top_indices if scores[i] >= self.similarity_threshold
            ]

            doc_ids = [f"doc_{i}" for i in final_indices]
            doc_scores = [float(scores[i]) for i in final_indices]
            details = [
                {"index": int(i), "cos_sim": float(scores[i])} for i in final_indices
            ]

            return {
                "query": query_text,
                "ids": doc_ids,
                "scores": doc_scores,
                "score_details": details,
            }
        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            return empty_res

    def _bm25_search(self, query_text: str, n_ret: int) -> Dict[str, Any]:
        """BM25 keyword search."""
        empty_res = {"query": query_text, "ids": [], "scores": [], "score_details": []}
        if not query_text.strip():
            return empty_res
        try:
            # Use BM25Retriever's query method
            results = self.bm25.query(query_text, n_results=n_ret)
            doc_ids = results.get("ids", [])
            doc_scores = results.get("scores", [])

            if not doc_ids:
                return empty_res

            details = []
            valid_ids = []
            valid_scores = []
            max_doc_index = len(self.documents) - 1
            for d_id, sc in zip(doc_ids, doc_scores):
                try:
                    idx = int(d_id.split("_")[1])
                    if 0 <= idx <= max_doc_index:
                        details.append({"index": idx, "bm25_score": float(sc)})
                        valid_ids.append(d_id)
                        valid_scores.append(float(sc))
                    else:
                        logger.warning(
                            f"BM25 returned invalid index {idx} for doc_id {d_id}. Max index is {max_doc_index}."
                        )
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Could not parse index from BM25 doc_id '{d_id}': {e}"
                    )

            return {
                "query": query_text,
                "ids": valid_ids,
                "scores": valid_scores,
                "score_details": details,
            }
        except Exception as e:
            logger.error(f"BM25 search error: {e}", exc_info=True)
            return empty_res

    def vectordb_query(self, query_text: str, max_out: int = 3) -> Dict[str, Any]:
        """Combine vector and BM25 search results using reciprocal rank fusion (or weighted sum)."""
        oversample_factor = 3  # Fetch more results for better fusion
        num_to_fetch = max(max_out * oversample_factor, 15)

        vec_res = self._vector_search(query_text, num_to_fetch)
        bm_res = self._bm25_search(query_text, num_to_fetch)

        # Use Reciprocal Rank Fusion (RRF) for combining results
        # k is a constant, often set to 60 (from original paper)
        k_rrf = 60.0
        combined_scores: Dict[
            int, Dict[str, Any]
        ] = {}  # index -> {rrf_score: float, source: str, details: {}}

        def get_idx(doc_id: str) -> int:
            try:
                return int(doc_id.split("_")[1])
            except (ValueError, IndexError):
                return -1

        # Process Vector Results
        max_doc_index = len(self.document_metadata) - 1
        for rank, d_id in enumerate(vec_res["ids"]):
            idx = get_idx(d_id)
            if idx < 0 or idx > max_doc_index:
                continue
            source = self.document_metadata[idx].get("source", "")
            if idx not in combined_scores:
                combined_scores[idx] = {
                    "rrf_score": 0.0,
                    "source": source,
                    "details": {},
                }
            combined_scores[idx]["rrf_score"] += 1.0 / (
                k_rrf + rank + 1
            )  # RRF formula (rank is 0-based)
            combined_scores[idx]["details"]["vector_score"] = vec_res["scores"][rank]
            combined_scores[idx]["details"]["vector_rank"] = rank + 1

        # Process BM25 Results
        for rank, d_id in enumerate(bm_res["ids"]):
            idx = get_idx(d_id)
            if idx < 0 or idx > max_doc_index:
                continue
            source = self.document_metadata[idx].get(
                "source", ""
            )  # Get source again, might be redundant
            if idx not in combined_scores:
                combined_scores[idx] = {
                    "rrf_score": 0.0,
                    "source": source,
                    "details": {},
                }
            combined_scores[idx]["rrf_score"] += 1.0 / (k_rrf + rank + 1)  # RRF formula
            combined_scores[idx]["details"]["bm25_score"] = bm_res["scores"][rank]
            combined_scores[idx]["details"]["bm25_rank"] = rank + 1

        # Add exact match boost *after* RRF calculation? Or factor it in?
        # Let's add it as a simple score boost after RRF.
        try:
            # Case-insensitive whole word match
            exact_pattern = re.compile(r"\b" + re.escape(query_text.lower()) + r"\b")
            for idx, data in combined_scores.items():
                if 0 <= idx < len(self.documents):  # Check index validity
                    if exact_pattern.search(self.documents[idx].lower()):
                        # Add a fixed boost to RRF score
                        boost_amount = 1.0  # Adjust boost amount as needed
                        data["rrf_score"] += boost_amount
                        data["details"]["exact_match_boost"] = boost_amount
                        logger.debug(f"Applying exact match boost to index {idx}")

        except re.error:
            logger.warning(
                f"Could not compile regex for exact match boost: {query_text}"
            )

        # Sort candidates by combined RRF score
        sorted_candidates = sorted(
            combined_scores.items(),  # Sort items (index, data_dict)
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )

        # Select top max_out candidates
        selected = sorted_candidates[:max_out]

        final_ids = []
        final_scores = []  # Store the RRF score
        final_details = []
        final_documents = []
        final_metadatas = []

        for idx, data in selected:
            if 0 <= idx < len(self.documents):  # Final safety check
                final_ids.append(f"doc_{idx}")
                final_scores.append(data["rrf_score"])
                # Add index to details dict
                data["details"]["final_rank"] = len(final_ids)  # 1-based rank
                data["details"]["index"] = idx
                final_details.append(data["details"])
                final_documents.append(self.documents[idx])
                final_metadatas.append(self.document_metadata[idx])
            else:
                logger.warning(
                    f"Skipping invalid index {idx} during final result construction."
                )

        return {
            "query": query_text,
            "ids": final_ids,
            "scores": final_scores,  # RRF scores
            "score_details": final_details,  # Contains original scores, ranks, boost
            "documents": final_documents,
            "metadatas": final_metadatas,
        }

    def close(self) -> None:
        """Close resources. Currently only handles embedder."""
        logger.info("Closing VectorDB resources.")
        if hasattr(self.embedder, "close"):
            try:
                self.embedder.close()
                logger.debug("Called embedder close method.")
            except Exception as e:
                logger.error(f"Error closing embedder: {e}")
        # Add closing logic for BM25 or other components if needed in the future

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

---
### File: src\llamasearch\data_manager.py

```python
#!/usr/bin/env python3
"""
data_manager.py – Dynamic configuration and export utilities for LlamaSearch.

This module stores paths for crawl data, index, models, and logs in a settings
dictionary that is loaded from (and saved to) a JSON file in the base directory.
Users can change these settings at runtime without exiting LlamaSearch.
Additionally, the module provides an export method that packages specified
directories into a tar.gz archive.
"""

import os
import json
import tarfile
import time
from pathlib import Path
from typing import Optional

DEFAULT_SETTINGS = {
    "crawl_data": os.path.join(os.path.expanduser("~"), ".llamasearch", "crawl_data"),
    "index": os.path.join(os.path.expanduser("~"), ".llamasearch", "index"),
    "models": os.path.join(os.path.expanduser("~"), ".llamasearch", "models"),
    "logs": os.path.join(os.path.expanduser("~"), ".llamasearch", "logs")
}

SETTINGS_FILENAME = "settings.json"

class DataManager:
    def __init__(self, base_dir: Optional[Path] = None):
        # Use a base_dir if given, otherwise default to ~/.llamasearch
        self.base_dir = base_dir if base_dir else Path(os.environ.get("LLAMASEARCH_DATA_DIR",
                                                                       os.path.join(os.path.expanduser("~"), ".llamasearch")))
        self.settings_file = self.base_dir / SETTINGS_FILENAME
        self.settings = DEFAULT_SETTINGS.copy()
        self._load_settings()
        self.ensure_directories()

    def _load_settings(self):
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
            except Exception as e:
                print(f"Warning: could not load settings file: {e}")

    def save_settings(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def ensure_directories(self):
        for key in ["crawl_data", "index", "models", "logs"]:
            dir_path = Path(self.settings.get(key, ""))
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

    def get_data_paths(self) -> dict:
        """Return the current data paths for all key directories."""
        return {
            "base": str(self.base_dir),
            "crawl_data": self.settings.get("crawl_data", ""),
            "index": self.settings.get("index", ""),
            "models": self.settings.get("models", ""),
            "logs": self.settings.get("logs", "")
        }

    def set_data_path(self, key: str, path: str):
        """
        Update a given path (e.g., "crawl_data", "index", etc.).
        This change is saved immediately.
        """
        if key in DEFAULT_SETTINGS:
            self.settings[key] = path
            self.ensure_directories()
            self.save_settings()
        else:
            raise ValueError(f"Unknown data path key: {key}")

    def export_data(self, keys: list, output_file: Optional[str] = None) -> str:
        """
        Export the directories specified in keys (e.g., ["crawl_data", "index"]) into a tar.gz archive.
        If output_file is not provided, creates one with a timestamp.
        Returns the path to the archive.
        """
        if not keys:
            raise ValueError("No keys specified for export.")
        export_paths = []
        for key in keys:
            path_str = self.settings.get(key, "")
            if path_str:
                export_paths.append(Path(path_str))
        if not export_paths:
            raise ValueError("No valid paths found for export.")

        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = str(self.base_dir / f"llamasearch_export_{timestamp}.tar.gz")
        else:
            output_file = str(Path(output_file).resolve())

        with tarfile.open(output_file, "w:gz") as tar:
            for exp_path in export_paths:
                tar.add(exp_path, arcname=exp_path.name)
        return output_file

# Singleton instance for convenience
data_manager = DataManager()
```

---
### File: src\llamasearch\exceptions.py

```python
# src/llamasearch/exceptions.py

class ModelNotFoundError(Exception):
    """Custom exception raised when a required model is not found locally."""
    pass

class SetupError(Exception):
    """Custom exception for errors during the setup process."""
    pass
```

---
### File: src\llamasearch\hardware.py

```python
"""
Hardware detection focused solely on CPU and Memory.
Relies only on standard libraries and the 'psutil' package.
"""

import os
import platform
import subprocess
import logging
from typing import Optional

import psutil

# Use pydantic for clear data structures
from pydantic import BaseModel, Field, validator

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Data Models ---


class CPUInfo(BaseModel):
    """Detailed CPU information."""

    logical_cores: int = Field(..., gt=0, description="Number of logical CPU cores.")
    physical_cores: int = Field(..., gt=0, description="Number of physical CPU cores.")
    architecture: str = Field(
        ..., description="CPU architecture (e.g., 'x86_64', 'arm64')."
    )
    model_name: str = Field(..., description="CPU model name.")
    frequency_mhz: Optional[float] = Field(
        None, description="Maximum or current CPU frequency in MHz."
    )
    # Basic instruction set detection (optional, can be expanded)
    supports_avx2: bool = Field(
        False, description="Indicates if AVX2 instructions are supported."
    )

    @validator("model_name", pre=True, always=True)
    def ensure_model_name_string(cls, v):
        # Ensure model name is always a string, even if platform returns None
        return str(v) if v is not None else "Unknown"


class MemoryInfo(BaseModel):
    """System memory (RAM) information."""

    total_gb: float = Field(..., gt=0, description="Total physical RAM in GiB.")
    available_gb: float = Field(
        ..., ge=0, description="Available RAM (usable by new processes) in GiB."
    )
    used_gb: float = Field(..., description="Used RAM in GiB.")
    percent_used: float = Field(
        ..., ge=0, le=100, description="Percentage of RAM currently used."
    )


class HardwareInfo(BaseModel):
    """Container for detected hardware information."""

    cpu: CPUInfo
    memory: MemoryInfo


# --- Detection Functions ---


def _detect_cpu_avx2() -> bool:
    """Attempts to detect AVX2 support."""
    # Prioritize py-cpuinfo if available (more reliable)
    try:
        import py_cpuinfo  # type: ignore

        info = py_cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        if isinstance(flags, list):
            return "avx2" in [flag.lower() for flag in flags]
        logger.debug("py_cpuinfo flags format unexpected, falling back.")
    except ImportError:
        logger.debug("py_cpuinfo not installed, falling back to basic detection.")
    except Exception as e:
        logger.debug(f"Error using py_cpuinfo: {e}, falling back.")

    # Fallback for Linux: Check /proc/cpuinfo
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                content = f.read()
            return " avx2 " in content  # Look for flag with spaces
        except Exception:
            logger.debug("Could not check /proc/cpuinfo for AVX2.")

    # Fallback for macOS: Check sysctl (less common to show specific flags)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "machdep.cpu.features"],  # Check this specific key
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return "AVX2" in result.stdout  # Case-sensitive check? Adjust if needed
        except Exception:
            logger.debug("Could not check sysctl machdep.cpu.features for AVX2.")

    # No reliable method found for Windows without py-cpuinfo or external tools
    logger.warning(
        "Could not reliably determine AVX2 support on this platform without py_cpuinfo."
    )
    return False


def detect_cpu_capabilities() -> CPUInfo:
    """
    Detects CPU capabilities.
    """
    logical_cores = os.cpu_count() or 1
    physical_cores = logical_cores  # Default assumption
    try:
        phys_count = psutil.cpu_count(logical=False)
        if phys_count:
            physical_cores = phys_count
            # Ensure logical isn't less than physical
            logical_cores = max(
                logical_cores, psutil.cpu_count(logical=True) or logical_cores
            )
        else:  # psutil returned None or 0 for physical
            physical_cores = logical_cores // 2 if logical_cores > 1 else 1
            logger.debug("psutil returned no physical core count, estimating.")
    except NotImplementedError:
        logger.warning(
            "psutil could not determine physical core count on this platform."
        )
        physical_cores = logical_cores // 2 if logical_cores > 1 else 1  # Estimate
    except Exception as e:
        logger.warning(f"Error getting physical cores: {e}. Estimating.")
        physical_cores = logical_cores // 2 if logical_cores > 1 else 1  # Estimate

    architecture = platform.machine().lower()
    model_name = "Unknown"
    system = platform.system()

    # Get CPU model name (using previous robust logic)
    try:
        if system == "Windows":
            model_name = platform.processor()
            if not model_name:
                try:
                    # Use shell=True cautiously, ensure command is safe
                    result = subprocess.run(
                        "wmic cpu get name",
                        shell=True,
                        capture_output=True,
                        text=True,
                        check=False,
                        creationflags=subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW,
                    )
                    if (
                        result.returncode == 0
                        and result.stdout
                        and len(result.stdout.splitlines()) > 1
                    ):
                        model_name = result.stdout.splitlines()[1].strip()
                except Exception:
                    pass  # Ignore wmic errors
        elif system == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                model_name = result.stdout.strip()
            except Exception:
                model_name = platform.processor()
        else:  # Linux
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("model name"):
                            model_name = line.split(":", 1)[1].strip()
                            break
                    if model_name == "Unknown":
                        model_name = platform.processor()
            except Exception:
                model_name = platform.processor()
        if not model_name:
            model_name = platform.processor() or "Unknown CPU"
    except Exception as e:
        logger.error(f"Error getting CPU model name: {e}")
        model_name = "Unknown CPU"

    # Get CPU frequency
    frequency_mhz = None
    try:
        freq = psutil.cpu_freq()
        if freq:
            frequency_mhz = (
                freq.max if hasattr(freq, "max") and freq.max else freq.current
            )
    except Exception:
        pass  # Ignore frequency errors

    # Detect AVX2 support
    supports_avx2 = _detect_cpu_avx2()

    return CPUInfo(
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        architecture=architecture,
        model_name=model_name.strip(),
        frequency_mhz=frequency_mhz,
        supports_avx2=supports_avx2,
    )


def detect_memory_info() -> MemoryInfo:
    """
    Detects system memory (RAM) information.
    """
    try:
        mem = psutil.virtual_memory()
        return MemoryInfo(
            total_gb=round(mem.total / (1024**3), 2),
            available_gb=round(mem.available / (1024**3), 2),
            used_gb=round(mem.used / (1024**3), 2),
            percent_used=mem.percent,
        )
    except Exception as e:
        logger.error(f"Failed to get memory info: {e}")
        return MemoryInfo(total_gb=0.0, available_gb=0.0, used_gb=0.0, percent_used=0.0)


# --- Main Public Function ---


def detect_hardware_info() -> HardwareInfo:
    """
    Detects and returns CPU and Memory information.

    Returns:
        HardwareInfo: An object containing detected CPU and Memory details.
    """
    logger.info("Detecting CPU and Memory hardware information...")
    cpu_info = detect_cpu_capabilities()
    memory_info = detect_memory_info()
    logger.info(
        f"CPU: {cpu_info.model_name} ({cpu_info.physical_cores}c/{cpu_info.logical_cores}t), AVX2: {cpu_info.supports_avx2}"
    )
    logger.info(
        f"Memory: {memory_info.total_gb:.1f} GB Total, {memory_info.available_gb:.1f} GB Available"
    )

    return HardwareInfo(cpu=cpu_info, memory=memory_info)


# Example usage (optional)
if __name__ == "__main__":
    hw_info = detect_hardware_info()
    print("\n--- Hardware Information ---")
    print(f"CPU Model:       {hw_info.cpu.model_name}")
    print(f"Architecture:    {hw_info.cpu.architecture}")
    print(f"Physical Cores:  {hw_info.cpu.physical_cores}")
    print(f"Logical Cores:   {hw_info.cpu.logical_cores}")
    print(f"Frequency (MHz): {hw_info.cpu.frequency_mhz or 'N/A'}")
    print(f"AVX2 Support:    {hw_info.cpu.supports_avx2}")
    print("-" * 20)
    print(f"Total RAM:       {hw_info.memory.total_gb:.2f} GB")
    print(f"Available RAM:   {hw_info.memory.available_gb:.2f} GB")
    print(
        f"Used RAM:        {hw_info.memory.used_gb:.2f} GB ({hw_info.memory.percent_used}%)"
    )
    print("-" * 20)
```

---
### File: src\llamasearch\protocols.py

```python
# src/llamasearch/protocols.py
from typing import Protocol, runtime_checkable, Any, Tuple

@runtime_checkable
class ModelInfo(Protocol):
    """Protocol defining the expected attributes for model metadata."""

    @property
    def model_id(self) -> str:
        """Unique identifier for the model."""
        ...

    @property
    def model_engine(self) -> str:
        """Identifier for the backend engine (e.g., 'onnx_teapot', 'transformers')."""
        ...

    @property
    def description(self) -> str:
        """A brief description of the model."""
        ...

    @property
    def context_length(self) -> int:
        """The maximum context length the model supports."""
        ...

@runtime_checkable
class LLM(Protocol):
    """Protocol defining the interface for a Language Model."""

    @property
    def model_info(self) -> ModelInfo:
        """Provides metadata about the model conforming to the ModelInfo protocol."""
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.0,
        **kwargs: Any # Allow flexible keyword arguments
    ) -> Tuple[str, Any]:
        """
        Generates text based on a prompt.

        Args:
            prompt: The input text prompt.
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            repeat_penalty: Penalty for repeating tokens.
            **kwargs: Additional engine-specific generation parameters.

        Returns:
            A tuple containing:
            - The generated text (str).
            - Engine-specific metadata or raw output (Any).
        """
        ...

    def load(self) -> bool:
        """
        Loads the model resources (if not already loaded).
        Returns True if loading was successful or model is already loaded, False otherwise.
        """
        ...

    def unload(self) -> None:
        """
        Unloads the model resources and performs cleanup.
        Should release memory (including GPU memory if applicable).
        """
        ...
```

---
### File: src\llamasearch\setup.py

```python
#!/usr/bin/env python3
"""
setup.py - Command-line utility for downloading and verifying LlamaSearch models.

Downloads only the necessary configuration, tokenizer, and selected ONNX
quantization files using hf_hub_download and assembles them into an
'active_teapot' directory for loading. Ensures the active directory is clean.
"""

import argparse
import sys
import spacy
import subprocess
from pathlib import Path
import time
import shutil
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._hf_folder import HfFolder
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from llamasearch.data_manager import data_manager
from llamasearch.utils import setup_logging
from llamasearch.exceptions import SetupError, ModelNotFoundError
from llamasearch.core.embedder import (
    DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_MODEL,
    EnhancedEmbedder,
)
from llamasearch.core.teapot import (
    TEAPOT_REPO_ID,
    ONNX_SUBFOLDER,
    REQUIRED_ONNX_BASENAMES,
    load_teapot_onnx_llm,
    _select_onnx_quantization,
    _determine_onnx_provider,
    TEAPOT_BASE_FILES,  # Import the constant
)
from llamasearch.core.bm25 import load_nlp_model
from llamasearch.hardware import detect_hardware_info
from typing import Optional

logger = setup_logging("llamasearch.setup")


# --- Helper: Download with Retries ---
def download_file_with_retry(
    repo_id: str,
    filename: str,
    cache_dir: Path,
    force: bool,
    max_retries: int = 2,
    delay: int = 5,
    **kwargs,
):
    """Attempts to download a file with retries on failure."""
    assert isinstance(cache_dir, Path), (
        f"cache_dir must be a Path object, got {type(cache_dir)}"
    )
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt + 1} downloading: {filename}")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force,
                resume_download=True,
                local_files_only=False,
                **kwargs,
            )
            fpath = Path(file_path)
            if not fpath.exists() or fpath.stat().st_size < 10:
                raise FileNotFoundError(
                    f"File {filename} missing/empty DL attempt {attempt + 1}."
                )
            logger.debug(f"Successfully downloaded {filename} to {file_path}")
            return file_path  # Success
        except (ConnectionError, TimeoutError, FileNotFoundError) as e:
            logger.warning(f"Download attempt {attempt + 1} for {filename} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying download of {filename} in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Max retries reached for {filename}. Download failed.")
                raise SetupError(f"Failed download: {filename}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error DL {filename} attempt {attempt + 1}: {e}",
                exc_info=True,
            )
            raise SetupError(f"Failed DL {filename}") from e
    # Added fallback return to satisfy static analysis, though it shouldn't be reached due to exceptions
    raise SetupError(f"Download failed for {filename} after retries.")


# --- Model Check/Download Functions (Embedder and Spacy remain the same) ---
def check_or_download_embedder(models_dir: Path, force: bool = False) -> None:
    model_name = DEFAULT_EMBEDDER_MODEL
    logger.info(f"Checking/Downloading Embedder Model: {model_name}")
    try:
        from huggingface_hub import snapshot_download as embedder_snapshot_download

        embedder_snapshot_download(
            repo_id=model_name,
            cache_dir=models_dir,
            force_download=force,
            resume_download=True,
            local_files_only=not force,
            local_dir_use_symlinks=False,
        )
        logger.info(
            f"Embedder model '{model_name}' cache verified/downloaded in {models_dir}."
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        if not force:
            logger.info(
                f"Embedder model '{model_name}' not found locally. Attempting download..."
            )
            try:
                from huggingface_hub import (
                    snapshot_download as embedder_snapshot_download_retry,
                )

                embedder_snapshot_download_retry(
                    repo_id=model_name,
                    cache_dir=models_dir,
                    force_download=False,
                    resume_download=True,
                    local_files_only=False,
                    local_dir_use_symlinks=False,
                )
                logger.info(f"Embedder model '{model_name}' downloaded successfully.")
            except Exception as download_err:
                raise SetupError(
                    f"Failed to download embedder model {model_name}"
                ) from download_err
        else:
            raise SetupError(f"Failed to get embedder model {model_name} with --force")
    except Exception as e:
        raise SetupError(f"Unexpected error getting embedder model {model_name}") from e


def check_or_download_spacy(force: bool = False) -> None:
    models = ["en_core_web_trf", "en_core_web_sm"]
    all_ok = True
    for model_name in models:
        logger.info(f"Checking/Downloading SpaCy Model: {model_name}")
        try:
            is_installed = spacy.util.is_package(model_name)
            if is_installed and not force:
                logger.info(f"SpaCy model '{model_name}' already installed.")
                continue
            if is_installed and force:
                logger.info(f"Force re-downloading SpaCy model '{model_name}'...")
            py_exec = sys.executable
            cmd = [py_exec, "-m", "spacy", "download", model_name]
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=300
            )
            if result.returncode == 0:
                logger.info(
                    f"Successfully downloaded/verified SpaCy model '{model_name}'."
                )
            else:
                logger.error(
                    f"Failed DL SpaCy model '{model_name}'. RC: {result.returncode}\nStderr:\n{result.stderr}"
                )
                all_ok = False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout DL SpaCy model '{model_name}'.")
            all_ok = False
        except Exception as e:
            logger.error(f"Error DL SpaCy model '{model_name}': {e}", exc_info=True)
            all_ok = False
    if not all_ok:
        raise SetupError("One or more SpaCy models failed to download.")


# --- Modified Teapot ONNX Download & Assembly ---
def check_or_download_teapot_onnx(
    models_dir: Path, quant_pref: str = "auto", force: bool = False
) -> None:
    """Downloads required Teapot files and assembles the 'active_teapot' dir."""
    logger.info(f"Checking/Downloading Teapot ONNX Files (Quantization: {quant_pref})")
    hw_info = detect_hardware_info()
    provider_name, _ = _determine_onnx_provider()
    quant_suffix = _select_onnx_quantization(hw_info, provider_name, None, quant_pref)
    logger.info(f"Targeting ONNX quantization suffix: '{quant_suffix}'")

    active_model_dir = models_dir / "active_teapot"
    active_onnx_dir = active_model_dir / ONNX_SUBFOLDER

    logger.info(f"Ensuring clean active model directory: {active_model_dir}")
    if active_model_dir.exists():
        logger.debug("Removing existing active directory...")
        try:
            shutil.rmtree(active_model_dir)
        except OSError as e:
            logger.error(
                f"Failed to remove existing active directory: {e}", exc_info=True
            )
            raise SetupError(
                f"Failed to clear active directory {active_model_dir}. Please remove it manually and retry."
            ) from e
    active_onnx_dir.mkdir(parents=True, exist_ok=True)

    cache_location = models_dir
    assert isinstance(cache_location, Path), (
        f"cache_location must be a Path object, got {type(cache_location)}"
    )

    files_to_copy_or_link = {}  # Store {target_path: source_path}

    # 1. Download Base Files into cache_location
    logger.info("Downloading/Verifying base Teapot files...")
    for base_file in TEAPOT_BASE_FILES:
        try:
            source_path_str: Optional[str] = None  # Explicitly type hint
            if not force:
                try:
                    source_path_str = hf_hub_download(
                        repo_id=TEAPOT_REPO_ID,
                        filename=base_file,
                        cache_dir=cache_location,
                        local_files_only=True,
                    )
                    logger.debug(f"Base file '{base_file}' found locally.")
                except (LocalEntryNotFoundError, FileNotFoundError):
                    logger.debug(
                        f"Base file '{base_file}' not found locally or symlink broken, proceeding to download."
                    )
                    source_path_str = None

            if source_path_str is None:
                source_path_str = download_file_with_retry(
                    repo_id=TEAPOT_REPO_ID,
                    filename=base_file,
                    cache_dir=cache_location,
                    force=force,
                    repo_type="model",
                )

            # --- Assertion added ---
            assert source_path_str is not None, (
                f"source_path_str should not be None after download for {base_file}"
            )
            target_path = active_model_dir / base_file
            files_to_copy_or_link[target_path] = Path(source_path_str)
        except Exception:
            raise  # Error handled within download_file_with_retry which raises SetupError

    # 2. Download Specific ONNX Files into cache_location
    logger.info(
        f"Downloading/Verifying specific ONNX files for suffix '{quant_suffix}'..."
    )
    onnx_files_to_download = [
        f"{ONNX_SUBFOLDER}/{basename}{quant_suffix}.onnx"
        for basename in REQUIRED_ONNX_BASENAMES
    ]
    for onnx_file_rel_path in onnx_files_to_download:
        try:
            source_path_str: Optional[str] = None  # Explicitly type hint
            if not force:
                try:
                    source_path_str = hf_hub_download(
                        repo_id=TEAPOT_REPO_ID,
                        filename=onnx_file_rel_path,
                        cache_dir=cache_location,
                        local_files_only=True,
                    )
                    logger.debug(f"ONNX file '{onnx_file_rel_path}' found locally.")
                except (LocalEntryNotFoundError, FileNotFoundError):
                    logger.debug(
                        f"ONNX file '{onnx_file_rel_path}' not found locally or symlink broken, proceeding to download."
                    )
                    source_path_str = None

            if source_path_str is None:
                source_path_str = download_file_with_retry(
                    repo_id=TEAPOT_REPO_ID,
                    filename=onnx_file_rel_path,
                    cache_dir=cache_location,
                    force=force,
                    repo_type="model",
                )

            # --- Assertion added ---
            assert source_path_str is not None, (
                f"source_path_str should not be None after download for {onnx_file_rel_path}"
            )
            target_path = active_onnx_dir / Path(onnx_file_rel_path).name
            files_to_copy_or_link[target_path] = Path(source_path_str)
        except EntryNotFoundError:
            logger.error(
                f"Required ONNX file '{onnx_file_rel_path}' not found in repo {TEAPOT_REPO_ID}. Is quant '{quant_suffix}' valid?"
            )
            raise SetupError(
                f"Required ONNX file missing from repository: {onnx_file_rel_path}"
            )
        except Exception:
            raise  # Error handled within download_file_with_retry

    # 3. Assemble the active_model_dir by copying files
    logger.info(f"Assembling active model directory: {active_model_dir}")
    copied_count = 0
    for target, source in files_to_copy_or_link.items():
        if not source.exists():
            logger.error(f"Source file does not exist, cannot copy: {source}")
            raise SetupError(f"Downloaded file missing before copy: {source}")
        try:
            logger.debug(f"Copying {source.name} -> {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_count += 1
        except Exception as e:
            logger.error(f"Failed to copy {source} to {target}: {e}", exc_info=True)
            raise SetupError(f"Failed to assemble active model directory at {target}")

    logger.info(
        f"Successfully downloaded and assembled {copied_count} required files into {active_model_dir}."
    )


# --- Verification Function ---
def verify_setup(onnx_quant_pref: str = "auto"):
    """Attempts to load all required models to verify setup."""
    logger.info("--- Verifying Model Setup ---")
    all_verified = True
    # Verify Embedder
    logger.info("Verifying Embedder model...")
    try:
        embedder = EnhancedEmbedder(auto_optimize=False)
        embedder.close()
        logger.info("Embedder model loaded successfully.")
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: Embedder model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading embedder model: {e}", exc_info=True
        )
        all_verified = False
    # Verify SpaCy
    logger.info("Verifying SpaCy models...")
    try:
        nlp = load_nlp_model()
        del nlp
        logger.info("SpaCy models ('trf' or 'sm') loaded successfully.")
    except ModelNotFoundError as e:
        logger.error(f"Verification Failed: SpaCy model not found. {e}")
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading SpaCy model: {e}", exc_info=True
        )
        all_verified = False
    # Verify Teapot ONNX LLM
    logger.info("Verifying Teapot ONNX LLM...")
    try:
        hw_info = detect_hardware_info()
        provider_name, _ = _determine_onnx_provider()
        expected_quant_suffix = _select_onnx_quantization(
            hw_info, provider_name, None, onnx_quant_pref
        )
        logger.info(
            f"(Verification targets quantization suffix: '{expected_quant_suffix}')"
        )
        llm = load_teapot_onnx_llm(
            onnx_quantization=onnx_quant_pref, preferred_provider="CPUExecutionProvider"
        )
        if llm:
            llm.unload()
            logger.info("Teapot ONNX LLM loaded successfully from active directory.")
        else:
            logger.error("Verification Failed: Teapot ONNX LLM loader returned None.")
            all_verified = False
    except ModelNotFoundError as e:
        logger.error(
            f"Verification Failed: Teapot ONNX model/files not found/invalid in active directory. {e}"
        )
        all_verified = False
    except Exception as e:
        logger.error(
            f"Verification Failed: Error loading Teapot ONNX LLM: {e}", exc_info=True
        )
        all_verified = False
    if not all_verified:
        logger.error("--- Model Verification Failed ---")
        raise SetupError("One or more models failed verification.")
    else:
        logger.info("--- Model Verification Successful ---")


# --- Main Setup Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Download/verify models for LlamaSearch."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force redownload/reassembly"
    )
    parser.add_argument(
        "--onnx-quant",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "int8", "q4", "q4f16", "bnb4", "uint8"],
        help="Specify ONNX quantization (default: auto)",
    )
    args = parser.parse_args()
    logger.info("--- Starting LlamaSearch Model Setup ---")
    if args.force:
        logger.info("Force mode enabled: Active directory will be recreated.")
    try:
        models_dir_str = data_manager.get_data_paths().get("models")
        if not models_dir_str:
            raise SetupError(
                "Models directory path not configured or found in settings."
            )
        models_dir = Path(models_dir_str)
        models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using models directory: {models_dir}")
        try:
            if HfFolder.get_token():
                logger.info("HF token found.")
            else:
                logger.warning("HF token not found.")
        except Exception:
            logger.warning("Could not check HF token.")
        # Download components
        check_or_download_embedder(models_dir, args.force)
        check_or_download_spacy(args.force)
        # Download and assemble Teapot files into 'active_teapot'
        check_or_download_teapot_onnx(models_dir, args.onnx_quant, args.force)
        # Verification Step (will check 'active_teapot')
        verify_setup(args.onnx_quant)
        logger.info("--- LlamaSearch Model Setup Completed Successfully ---")
        sys.exit(0)
    except SetupError as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---
### File: src\llamasearch\ui\__init__.py

```python

```

---
### File: src\llamasearch\ui\app_logic.py

```python
# src/llamasearch/ui/app_logic.py

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- Qt Imports for Signals ---
from PySide6.QtCore import QObject, Signal  # Keep Signal for type hint clarity

from llamasearch.core.llmsearch import LLMSearch
from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.core.teapot import TeapotONNXLLM
from llamasearch.data_manager import data_manager

# --- Corrected: Import utils components ---
from llamasearch.utils import setup_logging, _qt_logging_available, QtLogHandler
from llamasearch.exceptions import ModelNotFoundError

logger = setup_logging(__name__, level=logging.INFO, use_qt_handler=True)


# --- Backend Signal Emitter ---
class AppLogicSignals(QObject):
    """Holds signals emitted by the backend logic."""

    status_updated = Signal(str, str)
    search_completed = Signal(str, bool)
    crawl_index_completed = Signal(str, bool)
    manual_index_completed = Signal(str, bool)
    removal_completed = Signal(str, bool)
    refresh_needed = Signal()
    settings_applied = Signal(str, str)


class LlamaSearchApp:
    """Backend logic handler for LlamaSearch GUI. Runs tasks in threads."""

    def __init__(self, requires_gpu: bool = False, debug: bool = False):
        self.debug = debug
        self.requires_gpu = requires_gpu
        self.data_paths = data_manager.get_data_paths()
        self.llm_search: Optional[LLMSearch] = None
        self.signals = AppLogicSignals()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="LlamaSearchWorker"
        )

        self._log("INFO", f"LlamaSearchApp initializing. Data paths: {self.data_paths}")

        self._current_config = {
            "model_id": "N/A",
            "model_engine": "N/A",
            "context_length": 0,
            "max_results": 3,
            "debug_mode": self.debug,
            "provider": "N/A",
            "quantization": "N/A",
        }
        self._initialize_llm_search()  # Sync init
        if self.llm_search:
            self._log("INFO", "LlamaSearchApp ready.")
        else:
            self._log("ERROR", "LlamaSearchApp init failed. Run 'llamasearch-setup'.")

    def _log(self, level: str, message: str):
        """Wrapper for standard logging."""
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)

    def _initialize_llm_search(self):
        """Initializes LLMSearch synchronously."""
        if self.llm_search:
            self._log("INFO", "Closing existing LLMSearch instance...")
            try:
                self.llm_search.close()
            except Exception as e:
                self._log("ERROR", f"Error closing previous LLMSearch: {e}")
            self.llm_search = None

        index_dir = Path(self.data_paths["index"])
        self._log("INFO", f"Attempting to initialize LLMSearch in: {index_dir}")
        try:
            self.llm_search = LLMSearch(
                storage_dir=index_dir,
                debug=self.debug,
                verbose=self.debug,
                max_results=self._current_config.get("max_results", 3),
            )
            if self.llm_search and self.llm_search.model:
                info = self.llm_search.model.model_info
                self._current_config["model_id"] = info.model_id
                self._current_config["model_engine"] = info.model_engine
                self._current_config["context_length"] = info.context_length
                if isinstance(self.llm_search.model, TeapotONNXLLM):
                    self._current_config["provider"] = getattr(
                        self.llm_search.model, "_provider", "N/A"
                    )
                    model_id_parts = info.model_id.split("-")
                    if len(model_id_parts) >= 3:
                        quant_part = model_id_parts[-1]
                        if quant_part in [
                            "fp32",
                            "fp16",
                            "int8",
                            "q4",
                            "q4f16",
                            "bnb4",
                            "uint8",
                        ]:
                            self._current_config["quantization"] = quant_part
                        else:
                            self._current_config["quantization"] = "unknown"
                    else:
                        self._current_config["quantization"] = "unknown"
                else:
                    self._current_config["provider"] = "N/A"
                    self._current_config["quantization"] = "N/A"
                self._log(
                    "INFO",
                    f"LLMSearch initialized: {info.model_id} (Provider: {self._current_config['provider']}, Quant: {self._current_config['quantization']}, Ctx: {info.context_length})",
                )
            else:
                self._log("ERROR", "LLMSearch initialized, but LLM component failed.")
                self.llm_search = None
                raise ModelNotFoundError("LLM component failed.")
        except ModelNotFoundError as e:
            error_msg = f"Model setup required: {e}. Run 'llamasearch-setup'."
            self._log("ERROR", error_msg)
            logger.error(error_msg)
            self.llm_search = None
        except Exception as e:
            self._log("ERROR", f"Unexpected error initializing LLMSearch: {e}")
            logger.error("LLMSearch unexpected init error", exc_info=True)
            self.llm_search = None

    # --- ASYNC TASK EXECUTION ---
    # --- Corrected: Changed type hint to Any to satisfy pyright with SignalInstance ---
    def _run_in_background(self, task_func, *args, completion_signal: Any):
        """Submits function to thread pool."""
        try:
            future = self.executor.submit(task_func, *args)
            # Pass the *instance* of the signal to the callback
            future.add_done_callback(
                lambda f: self._task_done_callback(f, completion_signal)
            )
        except Exception as e:
            logger.error(f"Failed to submit task: {e}", exc_info=True)
            completion_signal.emit(f"Task Submission Error: {e}", False)

    # --- Corrected: Changed type hint to Any ---
    def _task_done_callback(self, future, completion_signal: Any):
        """Handles task completion."""
        try:
            result_message, success = future.result()
            completion_signal.emit(result_message, success)
        except Exception as e:
            logger.error(f"Exception in background task: {e}", exc_info=True)
            completion_signal.emit(f"Task Error: {e}", False)

    # --- GUI ACTIONS ---
    def submit_search(self, query: str):
        if not self.llm_search:
            self.signals.search_completed.emit(
                "Search failed: LLMSearch not initialized.", False
            )
            return
        if not query:
            self.signals.search_completed.emit("Please enter a query.", False)
            return
        self._log("INFO", f"Submitting search: '{query[:50]}...'")
        self._run_in_background(
            self._execute_search_task,
            query,
            completion_signal=self.signals.search_completed,
        )

    def submit_crawl_and_index(
        self,
        root_urls: List[str],
        target_links: int,
        max_depth: int,
        keywords: Optional[List[str]],
    ):
        self._log("INFO", f"Submitting crawl & index task for {len(root_urls)} URLs...")
        self._run_in_background(
            self._execute_crawl_and_index_task,
            root_urls,
            target_links,
            max_depth,
            keywords,
            completion_signal=self.signals.crawl_index_completed,
        )

    def submit_manual_index(self, path_str: str):
        if not self.llm_search:
            self.signals.manual_index_completed.emit(
                "Indexing failed: LLMSearch not initialized.", False
            )
            return
        source_path = Path(path_str)
        if not source_path.exists():
            self.signals.manual_index_completed.emit(
                f"Error: Path does not exist: {path_str}", False
            )
            return
        self._log("INFO", f"Submitting manual index task for: {source_path}")
        self._run_in_background(
            self._execute_manual_index_task,
            path_str,
            completion_signal=self.signals.manual_index_completed,
        )

    def submit_removal(self, source_id_to_remove: str):
        if not self.llm_search or not self.llm_search.vectordb:
            self.signals.removal_completed.emit(
                "Error: Cannot remove, LLMSearch/VectorDB not ready.", False
            )
            return
        if not isinstance(source_id_to_remove, str) or not source_id_to_remove:
            self.signals.removal_completed.emit(
                "Error: Invalid source ID for removal.", False
            )
            return
        self._log(
            "INFO", f"Submitting removal task for source ID: {source_id_to_remove}"
        )
        self._run_in_background(
            self._execute_removal_task,
            source_id_to_remove,
            completion_signal=self.signals.removal_completed,
        )

    # --- WORKER METHODS ---
    def _execute_search_task(self, query: str) -> Tuple[str, bool]:
        self._log("DEBUG", f"Executing search task for: '{query[:50]}...'")
        try:
            if not self.llm_search:
                return "Search Error: LLMSearch instance lost.", False
            start_time = time.time()
            results = self.llm_search.llm_query(query, debug_mode=self.debug)
            duration = time.time() - start_time
            self._log("INFO", f"Search task completed in {duration:.2f} seconds.")
            response = results.get("formatted_response", "No response generated.")
            return response, True
        except Exception as e:
            self._log("ERROR", f"Search query task failed: {e}")
            return f"Error performing search: {e}", False

    def _execute_crawl_and_index_task(
        self,
        root_urls: List[str],
        target_links: int,
        max_depth: int,
        keywords: Optional[List[str]],
    ) -> Tuple[str, bool]:
        self._log("DEBUG", "Executing crawl & index task...")
        crawl_successful = False
        index_successful = False
        crawl_duration = 0.0
        index_duration = 0.0
        added_chunks = 0
        all_collected_urls = []
        success_count, fail_count = 0, 0
        total_start_time = time.time()
        crawl_dir_base = Path(self.data_paths["crawl_data"])
        raw_output_dir = crawl_dir_base / "raw"
        crawl_dir_base.mkdir(parents=True, exist_ok=True)
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        try:  # Crawl Phase
            for url in root_urls:
                single_crawl_start = time.time()
                self._log(
                    "INFO", f"Crawling URL: {url} (Tgt:{target_links}, D:{max_depth})"
                )
                try:
                    crawler = Crawl4AICrawler(
                        root_urls=[url],
                        base_crawl_dir=crawl_dir_base,
                        target_links=target_links,
                        max_depth=max_depth,
                        relevance_keywords=keywords,
                    )
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    collected_urls = loop.run_until_complete(crawler.run_crawl())
                    loop.close()
                    all_collected_urls.extend(collected_urls)
                    duration = time.time() - single_crawl_start
                    crawl_duration += duration
                    self._log(
                        "INFO",
                        f"Crawl {url} OK ({duration:.2f}s). Got {len(collected_urls)} pages.",
                    )
                    success_count += 1
                    crawl_successful = True
                except Exception as e:
                    duration = time.time() - single_crawl_start
                    crawl_duration += duration
                    self._log("ERROR", f"Crawl FAILED for {url} ({duration:.2f}s): {e}")
                    fail_count += 1
            if not crawl_successful and fail_count > 0:
                raise Exception("All crawl tasks failed.")
        except Exception as crawl_exc:
            total_duration = time.time() - total_start_time
            result_msg = f"Crawl phase FAILED after {total_duration:.2f}s: {crawl_exc}"
            self._log("ERROR", result_msg)
            return result_msg, False
        if crawl_successful:  # Indexing Phase
            try:
                self._log(
                    "INFO",
                    f"Crawl finished ({crawl_duration:.2f}s). Indexing '{raw_output_dir.name}'...",
                )
                if not self.llm_search:
                    raise Exception("LLMSearch not initialized.")
                indexing_start_time = time.time()
                added_chunks = self.llm_search.add_documents_from_directory(
                    raw_output_dir, recursive=True
                )
                index_duration = time.time() - indexing_start_time
                index_successful = True
                self._log(
                    "INFO",
                    f"Auto indexing OK ({index_duration:.2f}s). Added {added_chunks} chunks.",
                )
                self.signals.refresh_needed.emit()
            except Exception as index_exc:
                self._log("ERROR", f"Automatic indexing FAILED: {index_exc}")
                index_successful = False
        # Final Status
        total_duration = time.time() - total_start_time
        unique_collected = len(set(all_collected_urls))
        crawl_status = f"Crawled {success_count}/{len(root_urls)} URLs ({unique_collected} unique pages) in {crawl_duration:.2f}s."
        index_status = ""
        overall_success = crawl_successful
        if crawl_successful:
            if index_successful:
                index_status = (
                    f"Indexed {added_chunks} new chunks in {index_duration:.2f}s."
                )
            else:
                index_status = "Automatic indexing failed."
                overall_success = False
        final_message = (
            f"Finished ({total_duration:.2f}s). {crawl_status} {index_status}".strip()
        )
        self._log("INFO", final_message)
        return final_message, overall_success

    def _execute_manual_index_task(self, path_str: str) -> Tuple[str, bool]:
        self._log("DEBUG", f"Executing manual index task for: {path_str}")
        source_path = Path(path_str)
        try:
            start_time = time.time()
            added_count = 0
            if not self.llm_search:
                return "Indexing Error: LLMSearch instance lost.", False
            if source_path.is_file():
                if hasattr(self.llm_search, "add_document"):
                    added_count = self.llm_search.add_document(source_path)
                else:
                    self._log("ERROR", "add_document method not found.")
                    return "Error: Indexing logic unavailable.", False
            elif source_path.is_dir():
                self._log("INFO", f"Indexing directory recursively: {source_path}")
                added_count = self.llm_search.add_documents_from_directory(
                    source_path, recursive=True
                )
            else:
                return f"Error: Path is not file/dir: {source_path}", False
            duration = time.time() - start_time
            msg = f"Indexing '{source_path.name}' OK ({duration:.2f}s). Added {added_count} new chunks."
            self._log("INFO", msg)
            if added_count > 0:
                self.signals.refresh_needed.emit()
            return msg, True
        except Exception as e:
            msg = f"Error indexing {source_path.name}: {e}"
            self._log("ERROR", msg)
            return msg, False

    def _execute_removal_task(self, source_id_to_remove: str) -> Tuple[str, bool]:
        self._log("DEBUG", f"Executing removal task for: {source_id_to_remove}")
        try:
            if not self.llm_search or not self.llm_search.vectordb:
                return "Removal Error: LLMSearch/VectorDB lost.", False
            display_name = source_id_to_remove
            try:
                path_obj = None
                if source_id_to_remove:
                    path_obj = Path(source_id_to_remove)
                if path_obj and path_obj.exists():
                    display_name = path_obj.name
            except (TypeError, ValueError, OSError):
                pass
            self.llm_search.vectordb._remove_document(source_id_to_remove)
            msg = f"Successfully requested removal for source: {display_name}"
            self._log("INFO", msg)
            self.signals.refresh_needed.emit()
            return msg, True
        except Exception as e:
            display_name = source_id_to_remove
            try:
                path_obj = None
                if source_id_to_remove:
                    path_obj = Path(source_id_to_remove)
                if path_obj and path_obj.exists():
                    display_name = path_obj.name
            except (TypeError, ValueError, OSError):
                pass
            msg = f"Error removing item {display_name}: {e}"
            self._log("ERROR", msg)
            return msg, False

    # --- Other Methods ---
    def get_crawl_data_items(self) -> List[Dict[str, str]]:
        """Retrieves indexed source information (sync)."""
        items = []
        if not self.llm_search or not self.llm_search.vectordb:
            self._log("WARN", "Cannot get index items: Not ready.")
            return items
        vdb = self.llm_search.vectordb
        unique_sources: Dict[str, str] = {}
        try:
            global_lookup_path = (
                Path(self.data_paths["crawl_data"]) / "reverse_lookup.json"
            )
            global_lookup: Dict[str, str] = {}
            if global_lookup_path.exists():
                try:
                    with open(global_lookup_path, "r", encoding="utf-8") as f:
                        global_lookup = json.load(f)
                except Exception as e:
                    self._log("WARN", f"Could not load reverse lookup: {e}")
            if hasattr(vdb, "document_metadata") and isinstance(
                vdb.document_metadata, list
            ):
                all_source_ids = set()
                for meta in vdb.document_metadata:
                    source_id = meta.get("source")  # Can return None
                    # --- Corrected Check ---
                    if isinstance(source_id, str) and source_id:
                        all_source_ids.add(source_id)
                    elif (
                        source_id is not None
                    ):  # Log if not None and not a valid string
                        self._log(
                            "WARN", f"Found non-string/empty source ID: {source_id}"
                        )

                for source_id in all_source_ids:
                    display_name = source_id  # Default display name
                    # --- Add check: source_id must be a string here ---
                    if not isinstance(source_id, str):
                        self._log("WARN", f"Skipping non-string source ID: {source_id}")
                        continue
                    try:
                        # --- Corrected: source_id is now guaranteed to be a str ---
                        path_obj = Path(source_id)
                        potential_hash = path_obj.stem
                        is_local = False
                        try:
                            is_local = path_obj.exists()
                        except OSError:
                            pass  # Handle filesystem errors during check
                        if (
                            not is_local
                            and len(potential_hash) == 16
                            and potential_hash in global_lookup
                        ):
                            display_name = global_lookup[potential_hash]
                        elif is_local:
                            display_name = path_obj.name
                    except (TypeError, ValueError) as path_err:
                        self._log(
                            "WARN",
                            f"Could not interpret source_id '{source_id}': {path_err}",
                        )
                    # --- Corrected: source_id is guaranteed str ---
                    unique_sources[source_id] = display_name
                items = [
                    {"hash": src_id, "url": name}
                    for src_id, name in unique_sources.items()
                ]
                items.sort(key=lambda x: x["url"].lower())
                self._log("DEBUG", f"Found {len(items)} unique sources in index.")
            else:
                self._log("WARN", "VectorDB metadata missing/invalid.")
        except Exception as e:
            self._log("ERROR", f"Failed get indexed items: {e}")
        return items

    def get_current_config(self) -> Dict[str, Any]:
        """Returns current configuration state (sync)."""
        self._current_config["debug_mode"] = self.debug
        if self.llm_search and self.llm_search.model:
            try:
                info = self.llm_search.model.model_info
                self._current_config["model_id"] = info.model_id
                self._current_config["model_engine"] = info.model_engine
                self._current_config["context_length"] = info.context_length
                self._current_config["provider"] = "N/A"
                self._current_config["quantization"] = "N/A"
                if isinstance(self.llm_search.model, TeapotONNXLLM):
                    self._current_config["provider"] = getattr(
                        self.llm_search.model, "_provider", "N/A"
                    )
                    parts = info.model_id.split("-")
                    if len(parts) >= 3:
                        q = parts[-1]
                        self._current_config["quantization"] = (
                            q
                            if q
                            in ["fp32", "fp16", "int8", "q4", "q4f16", "bnb4", "uint8"]
                            else "unknown"
                        )
                    else:
                        self._current_config["quantization"] = "unknown"
            except Exception as e:
                self._log("WARN", f"Could not update config from LLMSearch: {e}")
        elif not self.llm_search:
            self._current_config.update(
                {
                    "model_id": "N/A (Setup Required)",
                    "model_engine": "N/A",
                    "provider": "N/A",
                    "quantization": "N/A",
                    "context_length": 0,
                }
            )
        if self.llm_search:
            self._current_config["max_results"] = getattr(
                self.llm_search, "max_results", self._current_config["max_results"]
            )
        return self._current_config.copy()

    def apply_settings(self, settings: Dict[str, Any]):
        """Applies settings (sync)."""
        restart_needed = False
        config_changed = False
        new_debug = settings.get("debug_mode", self.debug)
        if new_debug != self.debug:
            self.debug = new_debug
            log_level = logging.DEBUG if self.debug else logging.INFO
            base_logger = logging.getLogger("llamasearch")
            base_logger.setLevel(log_level)
            for handler in base_logger.handlers:
                # --- Corrected check using imported variable ---
                if isinstance(
                    handler,
                    (
                        logging.FileHandler,
                        QtLogHandler if _qt_logging_available else type(None),
                    ),
                ):
                    handler.setLevel(log_level)
                elif isinstance(handler, logging.StreamHandler):
                    handler.setLevel(logging.INFO)
            logger.setLevel(log_level)  # Update our specific instance
            self._log("INFO", f"Debug mode set to: {self.debug}")
            if self.llm_search:
                self.llm_search.debug = self.debug
                self.llm_search.verbose = self.debug
            config_changed = True

        new_max_results = settings.get(
            "max_results", self._current_config["max_results"]
        )
        if (
            isinstance(new_max_results, int)
            and new_max_results > 0
            and new_max_results != self._current_config["max_results"]
        ):
            self._current_config["max_results"] = new_max_results
            # --- Corrected: Add nested check for vectordb ---
            if self.llm_search:
                self.llm_search.max_results = new_max_results
                if self.llm_search.vectordb:
                    self.llm_search.vectordb.max_results = new_max_results
            self._log("INFO", f"Max search results set to: {new_max_results}")
            config_changed = True
        elif not isinstance(new_max_results, int) or new_max_results <= 0:
            self._log("WARN", f"Invalid Max Results: {new_max_results}.")

        msg, lvl = "", ""
        if restart_needed:
            msg, lvl = "Settings applied. Restart required for some changes.", "warning"
        elif config_changed:
            msg, lvl = "Settings applied successfully.", "success"
        else:
            msg, lvl = "No setting changes detected.", "info"
        self.signals.settings_applied.emit(msg, lvl)
        return restart_needed

    def close(self):
        """Cleans up resources."""
        self._log("INFO", "LlamaSearchApp closing...")
        self.executor.shutdown(wait=True)
        if self.llm_search:
            try:
                self.llm_search.close()
                self._log("INFO", "LLMSearch closed.")
            except Exception as e:
                self._log("ERROR", f"Error closing LLMSearch: {e}")
            self.llm_search = None
        self._log("INFO", "LlamaSearchApp closed.")
```

---
### File: src\llamasearch\ui\components.py

```python
# src/llamasearch/ui/components.py
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import logging
import importlib.resources  # Use importlib.resources for package data

logger = logging.getLogger(__name__)


def header_component(data_paths: dict):  # data_paths for logging context
    """
    Creates a header widget with a logo and title.
    Uses importlib.resources to find the logo asset bundled with the package.
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)

    # --- Logo Loading using importlib.resources ---
    logo_label = QLabel()
    logo_found = False
    try:
        # Access the 'assets' directory within the 'llamasearch.ui' package
        logo_resource_path = importlib.resources.files("llamasearch.ui").joinpath(
            "assets/llamasearch.png"
        )

        # Use 'as_file' context manager for compatibility
        with importlib.resources.as_file(logo_resource_path) as logo_file_path:
            if logo_file_path.exists():
                pixmap = QPixmap(str(logo_file_path))
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        60,
                        60,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    logo_label.setPixmap(pixmap)
                    logo_label.setFixedSize(60, 60)
                    logger.debug(f"Loaded logo resource: {logo_resource_path}")
                    logo_found = True
                else:
                    logger.warning(f"Logo resource is invalid: {logo_resource_path}")
            else:
                logger.warning(f"Logo resource path DNE: {logo_file_path}")

    except (FileNotFoundError, ModuleNotFoundError, Exception) as e:
        logger.error(f"Error loading logo resource: {e}", exc_info=True)
        base_data_path_str = data_paths.get("base")
        if base_data_path_str:
            logger.error(f"(Data base path: {base_data_path_str})")

    if not logo_found:
        logger.warning("Using placeholder text for logo.")
        logo_label.setText("LS")
        logo_label.setFixedSize(60, 60)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet(
            "QLabel { border: 1px solid #cccccc; background-color: #f0f0f0; font-weight: bold; font-size: 18px; color: #555555; }"
        )

    # --- Title ---
    title_label = QLabel("LlamaSearch")
    title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-left: 10px;")

    layout.addWidget(logo_label)
    layout.addWidget(title_label)
    layout.addStretch()

    widget.setLayout(layout)
    return widget
```

---
### File: src\llamasearch\ui\main.py

```python
#!/usr/bin/env python3
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout
from PySide6.QtCore import Qt
# Ensure the import path is correct relative to the project structure
from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.ui.views.settings_view import SettingsView # Import specific class now
from llamasearch.ui.views.search_view import SearchAndIndexView
from llamasearch.ui.views.terminal_view import TerminalView
from llamasearch.ui.components import header_component
# --- Import setup_logging ---
from llamasearch.utils import setup_logging

# --- Setup logging early, before creating backend ---
# This ensures the QtLogHandler is attached if available
logger = setup_logging("llamasearch.gui_main", use_qt_handler=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LlamaSearch")

        # Initialize backend first
        # Read debug from args/env var if needed here? For now, False.
        # Backend will use the already configured logger
        self.backend = LlamaSearchApp(requires_gpu=False, debug=False)

        central = QWidget()
        self.main_layout = QVBoxLayout(central)

        # --- Add Header ---
        app_header = header_component(self.backend.data_paths)
        self.main_layout.addWidget(app_header)

        # --- Add Tabs ---
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Add search & index tab
        sa_tab = SearchAndIndexView(self.backend)
        self.tabs.addTab(sa_tab, "Search & Index")

        # Add settings tab
        st_tab = SettingsView(self.backend) # Use class directly
        self.tabs.addTab(st_tab, "Settings")

        # --- Add Terminal/Log View Tab ---
        term_tab = TerminalView(self.backend)
        self.tabs.addTab(term_tab, "Logs")

        central.setLayout(self.main_layout)
        self.setCentralWidget(central)
        self.resize(1200, 800)

    def closeEvent(self, event):
        """Ensure backend resources are released on close."""
        logger.info("Close event received, shutting down backend...")
        self.backend.close()
        logger.info("Backend shutdown complete.")
        super().closeEvent(event)

def main():
    # Optional: Set High DPI scaling attributes
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

# Guard for entry point execution
if __name__ == "__main__":
    # Logging is set up at the top level now
    main()
```

---
### File: src\llamasearch\ui\qt_logging.py

```python
# src/llamasearch/ui/qt_logging.py

import logging
from PySide6.QtCore import QObject, Signal

class QtLogSignalEmitter(QObject):
    """Contains a signal to emit log messages."""
    # Signal emits: level_name (str), message (str)
    log_message = Signal(str, str)

class QtLogHandler(logging.Handler):
    """
    A logging handler that emits Qt signals for each log record.
    """
    def __init__(self, signal_emitter: QtLogSignalEmitter, level=logging.NOTSET):
        super().__init__(level=level)
        self.signal_emitter = signal_emitter

    def emit(self, record: logging.LogRecord):
        """
        Formats the log record and emits it via the Qt signal.
        This method can be called from any thread.
        """
        try:
            msg = self.format(record)
            # Emit the signal - Qt handles cross-thread delivery if connected correctly
            self.signal_emitter.log_message.emit(record.levelname, msg)
        except Exception:
            self.handleError(record) # Default error handling

# Global instance of the emitter
# IMPORTANT: This emitter should be created *before* any loggers that use QtLogHandler
# It's often convenient to create it early in the GUI application setup.
qt_log_emitter = QtLogSignalEmitter()
```

---
### File: src\llamasearch\ui\views\search_view.py

```python
# src/llamasearch/ui/views/search_view.py

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QSpinBox,
    QFormLayout,
    QHeaderView,
    QMessageBox,
)
from PySide6.QtCore import (
    Qt,
    QMetaObject,
    Slot,
    QTimer,
    QGenericArgument,
)

from pathlib import Path
import logging
import shlex

logger = logging.getLogger(__name__)


class SearchAndIndexView(QWidget):
    """GUI View for Search, Crawl/Index, Manual Indexing, and Source Management."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()
        # Connect backend signals to UI slots
        self.backend.signals.status_updated.connect(self._set_status)
        self.backend.signals.search_completed.connect(self._on_search_complete)
        self.backend.signals.crawl_index_completed.connect(
            self._on_crawl_index_complete
        )
        self.backend.signals.manual_index_completed.connect(
            self._on_manual_index_complete
        )
        self.backend.signals.removal_completed.connect(self._on_removal_complete)
        self.backend.signals.refresh_needed.connect(self.update_data_display)
        QTimer.singleShot(100, self.update_data_display)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        # Search Section
        search_group_layout = QVBoxLayout()
        query_layout = QHBoxLayout()
        query_layout.addWidget(QLabel("<b>Question:</b>"))
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question...")
        query_layout.addWidget(self.query_input)
        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        query_layout.addWidget(self.search_btn)
        search_group_layout.addLayout(query_layout)
        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        self.search_results.setPlaceholderText("Search results...")
        search_group_layout.addWidget(self.search_results, 1)
        main_layout.addLayout(search_group_layout)
        # Crawling Section
        crawl_group_layout = QVBoxLayout()
        crawl_group_layout.addWidget(
            QLabel("<b>Crawl & Index Websites (Keyword Priority):</b>")
        )
        self.urls_text = QTextEdit()
        self.urls_text.setPlaceholderText("Enter root URLs...")
        self.urls_text.setFixedHeight(60)
        crawl_group_layout.addWidget(self.urls_text)
        crawl_params_layout = QHBoxLayout()
        form = QFormLayout()
        form.setHorizontalSpacing(20)
        self.target_links_spin = QSpinBox()
        self.target_links_spin.setRange(1, 1000)
        self.target_links_spin.setValue(15)
        self.target_links_spin.setToolTip("Max pages per URL.")
        form.addRow("Max Pages:", self.target_links_spin)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(0, 10)
        self.depth_spin.setValue(1)
        self.depth_spin.setToolTip("Max crawl depth")
        form.addRow("Max Depth:", self.depth_spin)
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Optional: guide api ...")
        self.keyword_input.setToolTip("Keywords for priority.")
        form.addRow("Relevance Keywords:", self.keyword_input)
        crawl_params_layout.addLayout(form)
        crawl_params_layout.addStretch()
        self.crawl_and_index_btn = QPushButton("Start Crawl & Index")
        self.crawl_and_index_btn.setStyleSheet("QPushButton { padding: 5px 15px; }")
        crawl_params_layout.addWidget(
            self.crawl_and_index_btn, 0, Qt.AlignmentFlag.AlignBottom
        )
        crawl_group_layout.addLayout(crawl_params_layout)
        main_layout.addLayout(crawl_group_layout)
        # Manual Indexing Section
        local_index_layout = QHBoxLayout()
        local_index_layout.addWidget(QLabel("<b>Manually Index Local Content:</b>"))
        self.index_file_btn = QPushButton("Index File...")
        self.index_dir_btn = QPushButton("Index Directory...")
        self.index_file_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        self.index_dir_btn.setStyleSheet("QPushButton { padding: 5px 10px; }")
        local_index_layout.addWidget(self.index_file_btn)
        local_index_layout.addWidget(self.index_dir_btn)
        local_index_layout.addStretch()
        main_layout.addLayout(local_index_layout)
        # Indexed Sources Table
        main_layout.addWidget(QLabel("<b>Indexed Sources:</b>"))
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(
            ["Source ID / Path", "Display Name / URL", "Actions"]
        )
        self.data_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.data_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.data_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.data_table.setAlternatingRowColors(True)
        main_layout.addWidget(self.data_table, 2)
        # Status Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-style: italic; color: #555;")
        main_layout.addWidget(self.status_label)
        # Connect Signals
        self.search_btn.clicked.connect(self.do_search)
        self.query_input.returnPressed.connect(self.do_search)
        self.crawl_and_index_btn.clicked.connect(self.do_crawl_and_index)
        self.index_file_btn.clicked.connect(self.do_index_file)
        self.index_dir_btn.clicked.connect(self.do_index_dir)

    @Slot(str, str)
    def _set_status(self, message: str, level: str = "info"):
        """Updates the status label via signal."""
        try:
            # Pass method name as bytes, arguments wrapped
            QMetaObject.invokeMethod(
                self.status_label,
                b"setText",
                Qt.ConnectionType.QueuedConnection,
                QGenericArgument(b"QString", f"Status: {message}"),  # type: ignore[reportArgumentType]
            )
            color = {"error": "red", "warning": "orange", "success": "green"}.get(
                level, "#555"
            )
            style = f"font-style: italic; color: {color};"
            QMetaObject.invokeMethod(
                self.status_label,
                b"setStyleSheet",
                Qt.ConnectionType.QueuedConnection,
                QGenericArgument(b"QString", style),  # type: ignore[reportArgumentType]
            )
        except Exception as e:
            logger.error(f"Error invoking method on status_label: {e}")

    @Slot()
    def do_search(self):
        """Initiates a search operation."""
        query = self.query_input.text().strip()
        if not query:
            self._set_status("Enter a query.", level="warning")
            self.search_results.setPlainText("Please enter a query.")
            return
        self._set_status(f"Searching '{query[:30]}...'")
        self._disable_actions(True)
        self.search_results.setPlainText("Submitting search...")
        self.backend.submit_search(query)

    @Slot(str, bool)
    def _on_search_complete(self, result_message: str, success: bool):
        """Handles completion of the search task."""
        self.search_results.setPlainText(result_message)
        self._set_status(
            "Search complete." if success else "Search failed.",
            level="success" if success else "error",
        )
        self._disable_actions(False)

    @Slot()
    def do_crawl_and_index(self):
        """Initiates a crawl and index operation."""
        lines = self.urls_text.toPlainText().splitlines()
        root_urls = [
            ln.strip() for ln in lines if ln.strip().startswith(("http://", "https://"))
        ]
        if not root_urls:
            self._set_status("No valid URLs provided.", level="warning")
            return
        tlinks = self.target_links_spin.value()
        md = self.depth_spin.value()
        keyword_text = self.keyword_input.text().strip()
        keywords_list = None
        if keyword_text:
            try:
                keywords_list = shlex.split(keyword_text)
            except ValueError:
                self._set_status("Error parsing keywords.", level="warning")
                keywords_list = None
        self._set_status(f"Starting crawl & index for {len(root_urls)} URL(s)...")
        self._disable_actions(True)
        self.backend.submit_crawl_and_index(root_urls, tlinks, md, keywords_list)

    @Slot(str, bool)
    def _on_crawl_index_complete(self, result_message: str, success: bool):
        """Handles completion of the crawl/index task."""
        self._set_status(result_message, level="success" if success else "error")
        self._disable_actions(False)

    def _get_start_dir(self) -> str:
        start_dir = str(Path.home())
        try:
            crawl_path = self.backend.data_paths.get("crawl_data")
            index_path = self.backend.data_paths.get("index")
            base_path = self.backend.data_paths.get("base")
            if crawl_path and Path(crawl_path).exists():
                return str(Path(crawl_path))
            if index_path and Path(index_path).exists():
                return str(Path(index_path))
            if base_path and Path(base_path).exists():
                return str(Path(base_path))
        except Exception as e:
            logger.warning(f"Error getting start dir: {e}")
        return start_dir

    @Slot()
    def do_index_file(self):
        start_dir = self._get_start_dir()
        fp, _ = QFileDialog.getOpenFileName(self, "Select File", start_dir)
        if fp:
            self._set_status(f"Indexing file '{Path(fp).name}'...")
            self._disable_actions(True)
            self.backend.submit_manual_index(fp)

    @Slot()
    def do_index_dir(self):
        start_dir = self._get_start_dir()
        dp = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if dp:
            self._set_status(f"Indexing directory '{Path(dp).name}'...")
            self._disable_actions(True)
            self.backend.submit_manual_index(dp)

    @Slot(str, bool)
    def _on_manual_index_complete(self, result_message: str, success: bool):
        self._set_status(result_message, level="success" if success else "error")
        self._disable_actions(False)

    @Slot()
    def remove_item(self, source_id: str):
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Remove source:\n'{source_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._set_status(f"Removing source '{source_id[:30]}...'")
            self._disable_actions(True)
            self.backend.submit_removal(source_id)
        else:
            self._set_status("Removal cancelled.", level="info")

    @Slot(str, bool)
    def _on_removal_complete(self, result_message: str, success: bool):
        self._set_status(result_message, level="success" if success else "error")
        self._disable_actions(False)

    def _disable_actions(self, disable: bool):
        """Enable/disable UI elements during tasks."""
        widgets_to_disable = [
            self.search_btn,
            self.crawl_and_index_btn,
            self.index_file_btn,
            self.index_dir_btn,
            self.data_table,
            self.urls_text,
            self.keyword_input,
            self.target_links_spin,
            self.depth_spin,
            self.query_input,
        ]
        for widget in widgets_to_disable:
            # Pass method name as bytes and argument wrapped
            QMetaObject.invokeMethod(
                widget,
                b"setDisabled",
                Qt.ConnectionType.QueuedConnection,
                QGenericArgument(b"bool", disable), 
            )

    @Slot()
    def update_data_display(self):
        """Updates the indexed sources table via signal."""

        def _update_table_task(self):
            """Task to update table data, run in background."""
            try:
                items = self.backend.get_crawl_data_items()
                self.data_table.setRowCount(0)
                self.data_table.setRowCount(len(items))
                for i, item_data in enumerate(items):
                    source_id = item_data.get("hash", "N/A")
                    display_name = item_data.get("url", "N/A")
                    id_item = QTableWidgetItem(source_id)
                    id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    id_item.setToolTip(source_id)
                    name_item = QTableWidgetItem(display_name)
                    name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    name_item.setToolTip(display_name)
                    self.data_table.setItem(i, 0, id_item)
                    self.data_table.setItem(i, 1, name_item)
                    remove_btn = QPushButton("Remove")
                    remove_btn.setStyleSheet("QPushButton { padding: 2px 5px; }")
                    remove_btn.setToolTip(f"Remove source: {source_id}")
                    # Disconnect previous connections to avoid duplicates if view reloads often (unlikely here)
                    try:
                        remove_btn.clicked.disconnect()
                    except RuntimeError:
                        pass  # Ignore if no connection exists
                    remove_btn.clicked.connect(
                        lambda checked=False, sid=source_id: self.remove_item(sid)
                    )
                    self.data_table.setCellWidget(i, 2, remove_btn)
                logger.debug(f"Updated indexed sources table with {len(items)} items.")
            except Exception as e:
                logger.error(f"Failed to update data display table: {e}", exc_info=True)
                self.backend._log(
                    "ERROR", "Error updating indexed sources list."
                )  # Log via backend

        # Call the nested function directly instead of using invokeMethod
        _update_table_task(self)
```

---
### File: src\llamasearch\ui\views\settings_view.py

```python
# src/llamasearch/ui/views/settings_view.py
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QTabWidget,
    QScrollArea,
    QMessageBox,
)
from PySide6.QtCore import (
    Qt,
    QTimer,
    Slot,
    QMetaObject,
    QUrl,
    QGenericArgument,  # Import QGenericArgument
)
from PySide6.QtGui import QDesktopServices

from typing import Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SettingsView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_label.setVisible(False)
        self.init_ui()
        self.load_settings()
        self.app.signals.settings_applied.connect(self.show_status_message)

    def init_ui(self):
        tabs = QTabWidget()
        general_tab = self._create_general_tab()
        system_tab = self._create_system_tab()
        tabs.addTab(general_tab, "General")
        tabs.addTab(system_tab, "System")
        self.main_layout.addWidget(tabs)
        self.main_layout.addWidget(self.status_label)

    @Slot(str, str)
    def show_status_message(self, message, level="success", duration=3500):
        """Display a status message via signal."""
        # Pass method name as bytes, arguments directly
        QMetaObject.invokeMethod(
            self.status_label,
            b"setText",
            Qt.ConnectionType.QueuedConnection,
            QGenericArgument(b"QString", message),  # type: ignore[reportArgumentType]
        )
        color = {"error": "red", "warning": "orange"}.get(level, "green")
        style = f"color: {color}; font-weight: bold;"
        QMetaObject.invokeMethod(
            self.status_label,
            b"setStyleSheet",
            Qt.ConnectionType.QueuedConnection,
            QGenericArgument(b"QString", style),  # type: ignore[reportArgumentType]
        )
        QMetaObject.invokeMethod(
            self.status_label,
            b"setVisible",
            Qt.ConnectionType.QueuedConnection,
            QGenericArgument(b"bool", True),  # type: ignore[reportArgumentType]
        )
        QTimer.singleShot(
            duration,
            lambda: QMetaObject.invokeMethod(
                self.status_label,
                b"setVisible",
                Qt.ConnectionType.QueuedConnection,
                QGenericArgument(b"bool", False),  # type: ignore[reportArgumentType]
            )
        )

    def _create_general_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group_model = QGroupBox("Active Model Information")
        model_layout = QFormLayout(group_model)
        self.model_id_label = QLabel("N/A")
        self.model_engine_label = QLabel("N/A")
        self.model_provider_label = QLabel("N/A")
        self.model_quant_label = QLabel("N/A")
        self.model_context_label = QLabel("N/A")
        model_layout.addRow("Model ID:", self.model_id_label)
        model_layout.addRow("Engine:", self.model_engine_label)
        model_layout.addRow("ONNX Provider:", self.model_provider_label)
        model_layout.addRow("Quantization:", self.model_quant_label)
        model_layout.addRow("Context Length:", self.model_context_label)
        setup_button = QPushButton("Run Model Setup...")
        setup_button.setToolTip("Show setup instructions")
        setup_button.clicked.connect(self._run_model_setup)
        model_layout.addRow(setup_button)
        group_params = QGroupBox("Search & Generation Parameters")
        params_layout = QFormLayout(group_params)
        self.results_spinner = QSpinBox()
        self.results_spinner.setRange(1, 20)
        self.results_spinner.setToolTip("Max chunks for context.")
        params_layout.addRow("Max Retrieved Chunks:", self.results_spinner)
        self.debug_checkbox = QCheckBox("Enable Debug Logging")
        self.debug_checkbox.setToolTip("Show detailed logs.")
        params_layout.addRow("", self.debug_checkbox)
        self.apply_button = QPushButton("Apply General Settings")
        self.apply_button.clicked.connect(self._apply_general_settings)
        layout.addWidget(group_model)
        layout.addWidget(group_params)
        layout.addStretch()
        layout.addWidget(self.apply_button)
        return tab

    def _create_system_tab(self):
        tab = QScrollArea()
        tab.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        paths_group = QGroupBox("Data Storage Paths")
        paths_layout = QFormLayout(paths_group)
        self.path_labels: Dict[str, QLabel] = {}
        paths = self.app.data_paths
        for key, default_path in paths.items():
            path_label = QLabel(str(default_path))
            path_label.setWordWrap(True)
            path_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            row_layout = QHBoxLayout()
            row_layout.addWidget(path_label)
            button = QPushButton("Open")
            button.clicked.connect(
                lambda checked=False, p=str(default_path): self._open_directory(p)
            )
            button.setToolTip(f"Open {key} directory")
            row_layout.addWidget(button)
            paths_layout.addRow(f"<b>{key.replace('_', ' ').title()}:</b>", row_layout)
            self.path_labels[key] = path_label
        layout.addWidget(paths_group)
        layout.addStretch()
        content.setLayout(layout)
        tab.setWidget(content)
        return tab

    @Slot()
    def _open_directory(self, path_str: str):
        path = Path(path_str)
        if path.exists() and path.is_dir():
            url = QUrl.fromLocalFile(str(path))
            if not QDesktopServices.openUrl(url):
                self.show_status_message(
                    f"Error: Could not open {path_str}", level="error"
                )
                logger.error(f"Failed openUrl: {path_str}")
        else:
            self.show_status_message(f"Error: Not found: {path_str}", level="error")
            logger.warning(f"Dir not found: {path_str}")

    @Slot()
    def _run_model_setup(self):
        QMessageBox.information(
            self,
            "Model Setup",
            "Run:\n\n<code>llamasearch-setup</code>\n\nin your terminal. Restart required.",
            QMessageBox.StandardButton.Ok,
        )

    @Slot()
    def _apply_general_settings(self):
        """Applies settings synchronously and lets backend emit status."""
        settings_to_apply = {
            "max_results": self.results_spinner.value(),
            "debug_mode": self.debug_checkbox.isChecked(),
        }
        try:
            self.app.apply_settings(settings_to_apply)  # Backend emits signal
        except Exception as e:
            self.show_status_message(f"Error: {e}", level="error")
            logger.error(f"Apply settings error: {e}", exc_info=True)

    def load_settings(self):
        """Loads current config into UI elements."""
        config = self.app.get_current_config()
        self.model_id_label.setText(config.get("model_id", "N/A"))
        self.model_engine_label.setText(config.get("model_engine", "N/A"))
        self.model_provider_label.setText(config.get("provider", "N/A"))
        self.model_quant_label.setText(config.get("quantization", "N/A"))
        self.model_context_label.setText(str(config.get("context_length", "N/A")))
        self.results_spinner.setValue(config.get("max_results", 3))
        self.debug_checkbox.setChecked(config.get("debug_mode", False))
        paths = self.app.data_paths
        for key, label in self.path_labels.items():
            label.setText(str(paths.get(key, "N/A")))
```

---
### File: src\llamasearch\ui\views\terminal_view.py

```python
# src/llamasearch/ui/views/terminal_view.py

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel

from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QFont

from ..qt_logging import qt_log_emitter


class TerminalView(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.init_ui()
        # Connect the global log emitter signal to our slot
        # QueuedConnection ensures the slot runs in the GUI thread
        qt_log_emitter.log_message.connect(
            self.append_log_message, Qt.ConnectionType.QueuedConnection
        )

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("<b>Logs</b>")
        layout.addWidget(title)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        font = QFont("Courier New", 10)  # Monospace font
        self.log_view.setFont(font)
        layout.addWidget(self.log_view)

        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.log_view.clear)
        layout.addWidget(clear_btn)

        self.setLayout(layout)

    # Slot receives log messages from the QtLogHandler signal
    @Slot(str, str)
    def append_log_message(self, level_name: str, message: str):
        """Appends a log message to the QTextEdit, potentially with color."""
        # Assured to run in GUI thread by QueuedConnection

        color_map = {
            "DEBUG": QColor("gray"),
            "INFO": QColor("black"),
            "WARNING": QColor("darkOrange"),
            "ERROR": QColor("red"),
            "CRITICAL": QColor("darkRed"),
        }
        color = color_map.get(level_name, QColor("black"))

        text_format = QTextCharFormat()
        text_format.setForeground(color)
        if level_name in ["ERROR", "CRITICAL"]:
            text_format.setFontWeight(QFont.Weight.Bold)

        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        # Insert newline only if needed
        current_text = self.log_view.toPlainText()
        if current_text and not current_text.endswith("\n"):
            cursor.insertText("\n")

        cursor.mergeCharFormat(text_format)
        cursor.insertText(message)
        cursor.setCharFormat(QTextCharFormat())  # Reset format

        self.log_view.ensureCursorVisible()  # Auto-scroll
```

---
### File: src\llamasearch\utils.py

```python
import os
import json
import logging
import numpy as np
from pathlib import Path
import time
from logging.handlers import RotatingFileHandler  # Use rotating file handler

# --- Add import for our Qt logging components ---
# Use a try-except block for environments where UI/Qt might not be installed (e.g., pure CLI usage)
try:
    # --- Corrected import path if qt_logging is inside ui ---
    from .ui.qt_logging import QtLogHandler, qt_log_emitter

    _qt_logging_available = True
except ImportError:
    _qt_logging_available = False
    # Define dummy types if Qt isn't available, so type checking doesn't fail later
    QtLogHandler = type(None)  # Use type(None) as a placeholder type
    qt_log_emitter = None


def is_dev_mode() -> bool:
    """Determine if running in development mode."""
    return os.environ.get("LLAMASEARCH_DEV_MODE", "").lower() in ("1", "true", "yes")


def get_llamasearch_dir() -> Path:
    """Return the base directory for LlamaSearch data."""
    from .data_manager import data_manager  # Local import

    return Path(data_manager.get_data_paths()["base"])


def setup_logging(name, level=logging.INFO, use_qt_handler=True):
    """
    Set up logging to console, file, and optionally Qt signal emitter.
    """
    from .data_manager import data_manager  # Local import

    try:
        log_path_str = data_manager.get_data_paths().get("logs")
        if not log_path_str:
            project_root = Path(data_manager.get_data_paths()["base"])
            logs_dir = project_root / "logs"
        else:
            logs_dir = Path(log_path_str)

        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "llamasearch.log"

        logger = logging.getLogger(name)
        # Prevent adding handlers multiple times if logger already exists
        if logger.hasHandlers():
            logger.setLevel(level)  # Ensure level is updated if changed
            return logger  # Return existing logger

        # Logger doesn't have handlers, configure it
        logger.setLevel(level)

        # --- File Handler (Rotating) ---
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(level)

        # --- Console Handler ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Console fixed at INFO

        # --- Qt Handler (Conditional) ---
        qt_handler = None
        if use_qt_handler and _qt_logging_available and qt_log_emitter is not None:
            qt_handler = QtLogHandler(qt_log_emitter)  # type: ignore[reportCallIssue] # Restored qt_log_emitter argument
            # --- Check qt_handler before calling setLevel ---
            if qt_handler: 
                qt_handler.setLevel(level)
        elif use_qt_handler and not _qt_logging_available:
            logging.warning("Qt logging handler requested but Qt components not found.")

        # --- Formatters ---
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(simple_formatter)
        # --- Corrected: Check if qt_handler was successfully created ---
        if qt_handler:
            qt_handler.setFormatter(simple_formatter)

        # --- Add Handlers ---
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # --- Corrected: Check if qt_handler was successfully created ---
        if qt_handler:
            logger.addHandler(qt_handler)

        logger.propagate = False  # Prevent duplicate logs in root logger

        # Special handling for noisy libraries if needed
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("markdown_it").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.INFO)
        logging.getLogger("onnxruntime").setLevel(logging.WARNING)

        return logger

    except Exception as e:
        # Fallback to basic config if setup fails
        logging.basicConfig(level=logging.INFO)
        logging.error(
            f"Failed to configure custom logging: {e}. Using basic config.",
            exc_info=True,
        )
        return logging.getLogger(name)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and sets."""

    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return list(o)
        return super().default(o)


def log_query(
    query: str,
    chunks: list,
    response: str,
    debug_info: dict,
    full_logging: bool = False,
) -> str:
    """Logs the query, optimized chunks, and response to a JSON Lines file."""
    from .data_manager import data_manager  # Local import

    logger = setup_logging(
        __name__, use_qt_handler=False
    )  # Don't need Qt handler for this specific log
    try:
        logs_dir = Path(data_manager.get_data_paths()["logs"])
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create/access logs directory: {e}. Skipping query log.")
        return ""

    chunks_to_log = chunks
    if not full_logging and chunks:
        simplified_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                sc = {k: chunk.get(k) for k in ["id", "score"] if k in chunk}
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    sc["source"] = chunk["metadata"].get("source", "")
                    if chunk["metadata"].get("entities"):
                        sc["entities"] = chunk["metadata"]["entities"]
                if "text" in chunk and isinstance(chunk["text"], str):
                    sc["text_preview"] = (
                        (chunk["text"][:200] + "...")
                        if len(chunk["text"]) > 200
                        else chunk["text"]
                    )
                simplified_chunks.append(sc)
            else:
                simplified_chunks.append(str(chunk))
        chunks_to_log = simplified_chunks

    optimized_debug_info = {}
    if isinstance(debug_info, dict):
        for key in ["retrieval_time", "generation_time", "total_time", "intent"]:
            if key in debug_info:
                optimized_debug_info[key] = debug_info[key]
        if full_logging:
            for key, value in debug_info.items():
                if key not in optimized_debug_info and key != "chunks":
                    optimized_debug_info[key] = value

    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "chunks": chunks_to_log,
        "response": response,
        "debug_info": optimized_debug_info,
    }

    log_file = logs_dir / "query_log.jsonl"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, cls=NumpyEncoder)
            f.write("\n")
        return str(log_file)
    except Exception as e:
        logger.error(f"Error saving query log to {log_file}: {e}")
        return ""
```

---
### File: tests\__init__.py

```python
# tests/__init__.py
"""Test suite for LlamaSearch."""
```

---

