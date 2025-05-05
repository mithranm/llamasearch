# src/llamasearch/core/llmsearch.py (CPU-Only, mxbai Embedder - CORRECTED SYNTAX)

import json
import threading
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import difflib
import gc  # Import gc for explicit calls

import chromadb
from chromadb.api.types import (
    GetResult,
    QueryResult,
    Metadata,
)
from chromadb.config import Settings as ChromaSettings

from llamasearch.core.bm25 import WhooshBM25Retriever
from llamasearch.core.chunker import chunk_markdown_text, DEFAULT_MIN_CHUNK_LENGTH
from llamasearch.core.embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_NAME
from llamasearch.core.embedder import EnhancedEmbedder  # Uses CPU-Only embedder
from llamasearch.core.teapot import TeapotONNXLLM, load_teapot_onnx_llm

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.protocols import LLM
from llamasearch.utils import setup_logging, log_query

logger = setup_logging(__name__, use_qt_handler=True)

# Constants
CHROMA_COLLECTION_NAME = "llamasearch_docs"
BM25_SUBDIR = "bm25_data"
DEFAULT_MAX_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE_FILTER = DEFAULT_MIN_CHUNK_LENGTH
CHUNK_SIMILARITY_THRESHOLD = 0.85
ChromaMetadataValue = Union[str, int, float, bool]
ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm"}


# Helper function
def _are_chunks_too_similar(text1: str, text2: str, threshold: float) -> bool:
    """Checks if two text chunks are too similar using SequenceMatcher."""
    if not text1 or not text2:
        return False
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity > threshold


class LLMSearch:
    """
    RAG-based search using Teapot ONNX (CPU), ChromaDB, Whoosh BM25,
    and mxbai-embed-large-v1 (CPU-only).
    """

    def __init__(
        self,
        storage_dir: Path,
        shutdown_event: Optional[threading.Event] = None,
        teapot_onnx_quant: str = "auto",
        teapot_provider_opts: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_results: int = 3,
        embedder_model: Optional[str] = None,
        embedder_batch_size: int = 0,
        embedder_truncate_dim: Optional[int] = None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        min_chunk_size_filter: int = DEFAULT_MIN_CHUNK_SIZE_FILTER,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        debug: bool = False,
    ):
        self.verbose = verbose
        self.max_results = max_results
        self.debug = debug
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self._shutdown_event = shutdown_event
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        self.model: Optional[LLM] = None
        self.embedder: Optional[EnhancedEmbedder] = None
        self.chroma_client = None
        self.chroma_collection: Optional[chromadb.Collection] = None
        self.bm25: Optional[WhooshBM25Retriever] = None
        self._reverse_lookup: Dict[str, str] = {}

        self.context_length: int = 0
        self.llm_device_type: str = "cpu"

        # Chunking param validation
        if min_chunk_size_filter < 0:
            logger.warning(
                f"Invalid min_chunk_size_filter ({min_chunk_size_filter}). Setting to 0."
            )
            min_chunk_size_filter = 0
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= max_chunk_size:
            logger.warning(
                f"Invalid chunk_overlap ({chunk_overlap}). Adjusting overlap."
            )
            chunk_overlap = min(max(0, chunk_overlap), max_chunk_size // 3)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size_filter = min_chunk_size_filter
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Chunking Params: MaxSize={self.max_chunk_size}, Overlap={self.chunk_overlap}, MinFilterLen={self.min_chunk_size_filter}"
        )

        try:
            self._load_reverse_lookup()

            # 1. Load LLM (Teapot ONNX - CPU)
            self.logger.info("Initializing Teapot ONNX LLM (CPU-only)...")
            self.model = load_teapot_onnx_llm(
                onnx_quantization=teapot_onnx_quant,
                preferred_provider="CPUExecutionProvider",
                preferred_options=teapot_provider_opts,
            )
            if not self.model:
                raise RuntimeError("load_teapot_onnx_llm returned None")
            model_info = self.model.model_info
            self.context_length = model_info.context_length
            self.llm_device_type = "cpu"
            self.logger.info(
                f"LLM: {model_info.model_id} on CPU. Context: {self.context_length}"
            )

            # 2. Initialize Embedder (CPU-Only, mxbai)
            self.logger.info(
                f"Initializing Embedder '{embedder_model or DEFAULT_EMBEDDER_NAME}' (CPU-only)..."
            )
            self.embedder = EnhancedEmbedder(
                model_name=embedder_model or DEFAULT_EMBEDDER_NAME,
                batch_size=embedder_batch_size,
                truncate_dim=embedder_truncate_dim,
            )
            if self.embedder and self._shutdown_event:
                self.embedder.set_shutdown_event(self._shutdown_event)
            embedding_dim = self.embedder.get_embedding_dimension()
            if not embedding_dim:
                logger.warning("Could not determine embedding dimension.")
            else:
                logger.info(
                    f"Embedder initialized (CPU). Effective Dim: {embedding_dim}"
                )

            # 3. Initialize ChromaDB
            self.logger.info(
                f"Initializing ChromaDB Client (storage: {self.storage_dir})"
            )
            self.chroma_client = chromadb.PersistentClient(
                str(self.storage_dir),
                ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            if self.chroma_client is not None:
                chroma_meta: Metadata = {"hnsw:space": "cosine"}
                if embedding_dim:
                    chroma_meta["embedding_dimension"] = str(int(embedding_dim))
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    CHROMA_COLLECTION_NAME, metadata=chroma_meta
                )
            else:
                raise RuntimeError("Chroma client not initialized!")
            assert self.chroma_collection is not None, (
                "ChromaDB collection creation failed"
            )
            logger.info(
                f"ChromaDB Collection '{CHROMA_COLLECTION_NAME}' ready. Count: {self.chroma_collection.count()}"
            )

            # 4. Initialize WhooshBM25Retriever
            bm25_path = self.storage_dir / BM25_SUBDIR
            self.logger.info(
                f"Initializing Whoosh BM25 Retriever (storage: {bm25_path})"
            )
            self.bm25 = WhooshBM25Retriever(storage_dir=bm25_path)
            logger.info(
                f"Whoosh BM25 Retriever ready. Initial Count: {self.bm25.get_doc_count()}."
            )

            self.logger.info(
                "LLMSearch components initialized successfully (CPU-Only Mode)."
            )

        except (ModelNotFoundError, SetupError) as e:
            self.logger.error(f"LLMSearch init failed: {e}. Run 'llamasearch-setup'.")
            self.close()
            raise
        except ImportError as e:
            if "chromadb" in str(e):
                raise SetupError("ChromaDB not installed.") from e
            elif "whoosh" in str(e):
                raise SetupError("Whoosh not installed.") from e
            else:
                self.logger.error(f"Missing import: {e}", exc_info=True)
                raise
        except Exception as e:
            self.logger.error(f"Unexpected init error: {e}", exc_info=True)
            self.close()
            raise RuntimeError("LLMSearch init failed.") from e

    def _load_reverse_lookup(self):
        """Loads the URL reverse lookup from the crawl data directory."""
        try:
            crawl_data_path = Path(data_manager.get_data_paths()["crawl_data"])
            lookup_file = crawl_data_path / "reverse_lookup.json"
            if lookup_file.exists():
                with open(lookup_file, "r", encoding="utf-8") as f:
                    self._reverse_lookup = json.load(f)
                logger.info(
                    f"Loaded URL reverse lookup ({len(self._reverse_lookup)} entries)."
                )
            else:
                self._reverse_lookup = {}
                logger.info("URL reverse lookup file not found.")
        except Exception as e:
            self._reverse_lookup = {}
            logger.error(f"Error loading reverse lookup: {e}", exc_info=self.debug)

    def add_source(self, source_path_str: str) -> Tuple[int, bool]:
        """Adds source(s). Uses CPU embedder with input_type='document'."""
        assert self.embedder is not None, "Embedder not initialized"
        assert self.chroma_collection is not None, "Chroma collection not initialized"
        assert self.bm25 is not None, "BM25 retriever not initialized"

        if self._shutdown_event and self._shutdown_event.is_set():
            logger.warning("Add source cancelled due to shutdown.")
            return 0, False

        source_path = Path(source_path_str).resolve()
        total_chunks_added = 0

        if not source_path.exists():
            logger.error(f"Source path not found: {source_path}")
            return 0, False

        if source_path.is_file():
            file_ext = source_path.suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                logger.info(
                    f"Skipping file {source_path.name}: Unsupported extension '{file_ext}'."
                )
                return 0, False

            logger.info(f"Processing file: {source_path}")
            if self._is_source_unchanged(source_path_str):
                logger.info(f"File '{source_path.name}' is unchanged. Skipping.")
                return 0, False

            removed_ok, _ = self.remove_source(source_path_str)
            if removed_ok:
                logger.debug(
                    f"Removed existing chunks for changed file: {source_path.name}"
                )

            # Read file content with encoding fallback
            file_content: Optional[str] = None
            try:
                file_content = source_path.read_text(encoding="utf-8", errors="ignore")
                if "�" in file_content[:1000]:
                    content_latin1, content_cp1252 = None, None
                    try:
                        content_latin1 = source_path.read_text(
                            encoding="latin-1", errors="ignore"
                        )
                    except Exception:
                        pass
                    try:
                        content_cp1252 = source_path.read_text(
                            encoding="cp1252", errors="ignore"
                        )
                    except Exception:
                        pass

                    if content_latin1 and "�" not in content_latin1[:1000]:
                        file_content = content_latin1
                        logger.info(f"Using latin-1 encoding for {source_path.name}.")
                    elif content_cp1252 and "�" not in content_cp1252[:1000]:
                        file_content = content_cp1252
                        logger.info(f"Using cp1252 encoding for {source_path.name}.")
                    else:
                        logger.warning(
                            f"Could not resolve encoding issue for {source_path.name}."
                        )
            except Exception as e:
                logger.error(
                    f"Error reading file {source_path}: {e}", exc_info=self.debug
                )
                return 0, False

            if file_content is None or not file_content.strip():
                logger.info(
                    f"Skipping file {source_path.name}: Content empty or read failed."
                )
                return 0, False

            try:
                # Chunking
                logger.debug(f"Chunking {source_path.name}...")
                chunks_with_metadata = chunk_markdown_text(
                    markdown_text=file_content,
                    source=str(source_path),
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    min_chunk_char_length=self.min_chunk_size_filter,
                )
                if not chunks_with_metadata:
                    logger.warning(
                        f"No valid chunks generated from file {source_path.name}. Skipping."
                    )
                    return 0, False

                # Prepare data for embedding and storage
                chunk_texts: List[str] = []
                chunk_metadatas: List[Metadata] = []
                chunk_ids: List[str] = []
                try:
                    mtime_val: Optional[float] = source_path.stat().st_mtime
                except OSError as e:
                    logger.warning(f"Could not stat file {source_path}: {e}")
                    mtime_val = time.time()

                file_hash = hashlib.sha1(source_path_str.encode()).hexdigest()[:8]
                original_url: Optional[str] = self._reverse_lookup.get(source_path.stem)
                if original_url:
                    logger.debug(
                        f"Found original URL for {source_path.name}: {original_url}"
                    )

                base_meta: Metadata = {
                    "source_path": source_path_str,
                    "filename": source_path.name,
                    "mtime": mtime_val,
                }
                if original_url:
                    base_meta["original_url"] = original_url

                valid_chunk_counter = 0
                for c_idx, c in enumerate(chunks_with_metadata):
                    chunk_text = c.get("chunk", "")
                    if not chunk_text:
                        continue

                    chunk_content_hash = hashlib.sha1(
                        chunk_text.encode("utf-8", errors="ignore")
                    ).hexdigest()[:8]
                    chunk_id = f"{file_hash}_{c_idx}_{chunk_content_hash}"
                    chunk_ids.append(chunk_id)
                    valid_chunk_counter += 1

                    meta: Metadata = base_meta.copy()
                    chunker_meta = c.get("metadata", {})
                    if isinstance(chunker_meta, dict):
                        if chunker_meta.get("chunk_index_in_doc") is not None:
                            meta["original_chunk_index"] = chunker_meta[
                                "chunk_index_in_doc"
                            ]
                        if chunker_meta.get("length") is not None:
                            meta["chunk_char_length"] = chunker_meta["length"]
                        if chunker_meta.get("effective_length") is not None:
                            meta["effective_length"] = chunker_meta["effective_length"]
                        if chunker_meta.get("processing_mode") is not None:
                            meta["processing_mode"] = chunker_meta["processing_mode"]

                    meta["filtered_chunk_index"] = valid_chunk_counter - 1
                    meta["chunk_id"] = chunk_id

                    cleaned_meta: Metadata = {}
                    for k, v in meta.items():
                        if isinstance(v, (str, int, float, bool)):
                            cleaned_meta[k] = v
                        elif v is not None:
                            try:
                                cleaned_meta[k] = str(v)
                            except Exception:
                                logger.warning(
                                    f"Skipping non-serializable metadata key '{k}' type {type(v)}"
                                )
                    chunk_metadatas.append(cleaned_meta)
                    chunk_texts.append(chunk_text)

                if not chunk_ids:
                    logger.warning(
                        f"No valid chunks remained after processing for file {source_path.name}."
                    )
                    return 0, False

                # Embedding
                logger.info(
                    f"Embedding {len(chunk_texts)} chunks for {source_path.name} (CPU)..."
                )
                embeddings = self.embedder.embed_strings(
                    chunk_texts, input_type="document", show_progress=self.verbose
                )
                if self._shutdown_event and self._shutdown_event.is_set():
                    return 0, False
                if embeddings is None or embeddings.shape[0] != len(chunk_texts):
                    logger.error(
                        f"Embedding failed or returned incorrect shape for {source_path.name}."
                    )
                    return 0, False

                # Add to ChromaDB
                logger.info(f"Adding {len(chunk_ids)} chunks to ChromaDB...")
                try:
                    self.chroma_collection.upsert(
                        ids=chunk_ids,
                        embeddings=embeddings.tolist(),
                        metadatas=[dict(m) for m in chunk_metadatas],
                        documents=[str(doc) for doc in chunk_texts],
                    )
                except Exception as chroma_e:
                    logger.error(
                        f"ChromaDB upsert failed for {source_path.name}: {chroma_e}",
                        exc_info=self.debug,
                    )

                # Add to Whoosh
                logger.info(f"Adding {len(chunk_texts)} chunks to Whoosh BM25 index...")
                bm25_success_count = 0
                for text, chunk_id in zip(chunk_texts, chunk_ids):
                    if self.bm25.add_document(text, chunk_id):
                        bm25_success_count += 1
                logger.debug(
                    f"Added {bm25_success_count}/{len(chunk_texts)} chunks successfully to Whoosh."
                )
                if bm25_success_count != len(chunk_texts):
                    logger.warning(
                        f"Mismatch in Whoosh add count for {source_path.name}"
                    )

                total_chunks_added = len(chunk_ids)
                logger.info(
                    f"Successfully processed {total_chunks_added} chunks from {source_path.name}."
                )
            except Exception as e:
                logger.error(
                    f"Failed processing file {source_path}: {e}", exc_info=self.debug
                )
                total_chunks_added = 0

        elif source_path.is_dir():
            # Directory processing
            logger.info(f"Processing directory recursively: {source_path}")
            files_processed, files_skipped, files_failed = 0, 0, 0
            try:
                all_items = list(source_path.rglob("*"))
            except Exception as e:
                logger.error(
                    f"Error listing files in directory {source_path}: {e}",
                    exc_info=self.debug,
                )
                return 0, False
            logger.info(f"Found {len(all_items)} items in directory tree. Filtering...")

            for item in all_items:
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.warning("Directory processing cancelled.")
                    break
                if item.is_file():
                    is_hidden = any(
                        part.startswith(".")
                        for part in item.relative_to(source_path).parts
                    ) or item.name.startswith(".")
                    if is_hidden:
                        files_skipped += 1
                        continue

                    if item.suffix.lower() in ALLOWED_EXTENSIONS:
                        try:
                            added, _ = self.add_source(str(item))
                        except Exception as e:
                            logger.error(
                                f"Error processing file {item} during dir scan: {e}",
                                exc_info=self.debug,
                            )
                            files_failed += 1
                            continue

                        if added > 0:
                            total_chunks_added += added
                            files_processed += 1
                        elif self._is_source_unchanged(str(item)):
                            files_skipped += 1
                        else:
                            files_failed += 1
                    else:
                        files_skipped += 1
                elif item.is_dir() and item.name.startswith("."):
                    pass  # Implicitly skip hidden dirs

            logger.info(
                f"Directory scan for '{source_path.name}' complete. "
                f"Added {total_chunks_added} new chunks from {files_processed} processed files. "
                f"{files_skipped} files skipped. {files_failed} files failed."
            )
            if files_failed > 0:
                logger.warning(
                    f"Failed to process/add chunks for {files_failed} files during directory scan."
                )
        else:
            logger.warning(f"Source path is not a file or directory: {source_path}")

        return total_chunks_added, False

    def _is_source_unchanged(self, source_path_str: str) -> bool:
        """Checks if a source file's mtime matches stored mtime in ChromaDB."""
        assert self.chroma_collection is not None
        source_path = Path(source_path_str)
        if not source_path.is_file():
            return False
        try:
            current_mtime = source_path.stat().st_mtime
            results: Optional[GetResult] = self.chroma_collection.get(
                where={"source_path": source_path_str}, limit=1, include=["metadatas"]
            )
            if (
                results
                and results.get("ids")
                and results.get("metadatas")
                and results["metadatas"]
                and isinstance(results["metadatas"][0], dict)
            ):
                stored_meta = results["metadatas"][0]
                if "mtime" in stored_meta:
                    stored_mtime = stored_meta["mtime"]
                    return (
                        isinstance(stored_mtime, (int, float))
                        and abs(float(stored_mtime) - current_mtime) < 0.1
                    )
            return False
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.warning(f"Error checking source status '{source_path_str}': {e}.")
            return False

    def remove_source(self, source_path_str: str) -> Tuple[bool, bool]:
        """Removes all chunks associated with a given source path string."""
        assert self.chroma_collection is not None
        assert self.bm25 is not None
        logger.info(f"Attempting to remove all chunks for source: {source_path_str}")
        removal_occurred = False
        try:
            # Find IDs in Chroma
            ids_to_remove: List[str] = []
            limit = 5000
            offset = 0
            while True:
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.warning("Shutdown during remove (get).")
                    break
                results: Optional[GetResult] = self.chroma_collection.get(
                    where={"source_path": source_path_str},
                    include=["metadatas"],
                    limit=limit,
                    offset=offset,
                )
                if results and results.get("ids"):
                    batch_ids = results["ids"]
                    ids_to_remove.extend(batch_ids)
                    if len(batch_ids) < limit:
                        break
                    offset += limit
                else:
                    break

            if not ids_to_remove:
                logger.info(f"No chunks found for '{source_path_str}'.")
                return False, False

            ids_to_remove = list(set(ids_to_remove))  # Deduplicate
            if self._shutdown_event and self._shutdown_event.is_set():
                logger.warning("Shutdown before remove (delete).")
                return False, False

            # Remove from Chroma
            logger.info(f"Removing {len(ids_to_remove)} chunks from ChromaDB...")
            batch_size = 500
            chroma_delete_errors = 0
            for i in range(0, len(ids_to_remove), batch_size):
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.warning("Shutdown during Chroma delete.")
                    break
                batch_ids = ids_to_remove[i : i + batch_size]
                if batch_ids:
                    try:
                        self.chroma_collection.delete(ids=batch_ids)
                    except Exception as chroma_del_err:
                        logger.error(
                            f"Chroma delete err batch {i}: {chroma_del_err}",
                            exc_info=self.debug,
                        )
                        chroma_delete_errors += 1
            if self._shutdown_event and self._shutdown_event.is_set():
                return False, False
            removal_occurred = True

            # Remove from Whoosh
            logger.info(f"Removing {len(ids_to_remove)} chunks from Whoosh...")
            bm25_removed_count = 0
            bm25_remove_errors = 0
            for chunk_id in ids_to_remove:
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.warning("Shutdown during Whoosh delete.")
                    break
                try:
                    if self.bm25.remove_document(chunk_id):
                        bm25_removed_count += 1
                except Exception as bm25_del_err:
                    logger.error(
                        f"Whoosh remove err {chunk_id}: {bm25_del_err}",
                        exc_info=self.debug,
                    )
                    bm25_remove_errors += 1
            if self._shutdown_event and self._shutdown_event.is_set():
                return False, False

            log_msg = (
                f"Removal '{source_path_str}': Chroma Att={len(ids_to_remove)} Err={chroma_delete_errors}. "
                f"Whoosh Att={len(ids_to_remove)} Rem={bm25_removed_count} Err={bm25_remove_errors}."
            )
            logger.info(log_msg)
            return removal_occurred, False
        except Exception as e:
            logger.error(
                f"Error removing source '{source_path_str}': {e}", exc_info=True
            )
            return False, False

    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """Aggregates chunk metadata to list unique indexed sources."""
        assert self.chroma_collection is not None
        try:
            total_count = self.chroma_collection.count()
            if total_count == 0:
                return []

            logger.debug(f"Fetching {total_count} metadata items...")
            batch_size = 5000
            all_metadata_list: List[Metadata] = []
            fetched_ids = set()
            offset = 0
            while True:  # Fetch metadata
                batch_results: Optional[GetResult] = self.chroma_collection.get(
                    limit=batch_size, offset=offset, include=["metadatas"]
                )
                if not batch_results or not batch_results.get("ids"):
                    break
                ids_in_batch = batch_results["ids"]
                metadatas_in_batch = batch_results.get("metadatas")
                new_items_found = False
                if (
                    ids_in_batch
                    and metadatas_in_batch
                    and len(ids_in_batch) == len(metadatas_in_batch)
                ):
                    for i, doc_id in enumerate(ids_in_batch):
                        if doc_id not in fetched_ids:
                            meta = metadatas_in_batch[i]
                            if isinstance(meta, dict):
                                all_metadata_list.append(meta)
                            fetched_ids.add(doc_id)
                            new_items_found = True
                else:
                    logger.warning(f"Mismatched batch at offset {offset}.")
                    break
                if not new_items_found and len(ids_in_batch) > 0:
                    logger.warning(f"No new IDs at offset {offset}.")
                    break
                offset += len(ids_in_batch)
                if len(ids_in_batch) < batch_size:
                    break

            if not all_metadata_list:
                logger.warning("Chroma get() no metadata.")
                return []

            source_info: Dict[str, Dict[str, Any]] = {}  # Aggregate
            missing_source_path_count = 0
            for meta in all_metadata_list:
                if not isinstance(meta, dict):
                    continue
                src_path_any = meta.get("source_path")
                src_path = str(src_path_any).strip() if src_path_any else ""
                primary_key = str(meta.get("original_url", src_path))
                if not primary_key:
                    missing_source_path_count += 1
                    primary_key = f"unknown_{missing_source_path_count}"
                    src_path = primary_key
                if primary_key not in source_info:
                    filename = "(Unknown)"
                    mtime_val = meta.get("mtime")
                    try:
                        if src_path and not src_path.startswith("unknown_"):
                            filename = Path(src_path).name
                        elif primary_key.startswith("http"):
                            filename = "(Web Content)"
                    except Exception:
                        pass
                    source_info[primary_key] = {
                        "source_path": src_path
                        if not src_path.startswith("unknown_")
                        else "(Missing)",
                        "original_url": meta.get("original_url"),
                        "filename": meta.get("filename", filename),
                        "chunk_count": 0,
                        "mtime": float(mtime_val)
                        if isinstance(mtime_val, (int, float))
                        else None,
                    }
                source_info[primary_key]["chunk_count"] += 1
                current_mtime = source_info[primary_key].get("mtime")
                meta_mtime = meta.get("mtime")
                if (
                    meta_mtime is not None
                    and isinstance(meta_mtime, (int, float))
                    and (current_mtime is None or float(meta_mtime) > current_mtime)
                ):
                    source_info[primary_key]["mtime"] = float(meta_mtime)
            if missing_source_path_count > 0:
                logger.warning(
                    f"{missing_source_path_count} metadata missing source identifier."
                )
            logger.debug(f"Aggregated {len(source_info)} unique sources.")
            source_list = list(source_info.values())
            source_list.sort(
                key=lambda x: str(
                    x.get("original_url")
                    or x.get("filename", "")
                    or x.get("source_path", "")
                ).lower()
            )
            return source_list
        except Exception as e:
            logger.error(f"Error retrieving sources: {e}", exc_info=True)
            return []

    def _get_token_count(self, text: str) -> int:
        """Estimate token count."""
        if not text:
            return 0
        if (
            self.model
            and isinstance(self.model, TeapotONNXLLM)
            and hasattr(self.model, "_tokenizer")
            and self.model._tokenizer is not None
        ):
            try:
                tokenizer = getattr(self.model, "_tokenizer", None)
                if tokenizer and hasattr(tokenizer, "__call__"):
                    ids = tokenizer(text, add_special_tokens=False).get("input_ids")
                    if isinstance(ids, list):
                        return len(ids)
            except Exception:
                pass
        return max(1, len(text) // 4)  # Fallback

    def llm_query(self, query_text: str, debug_mode: bool = False) -> Dict[str, Any]:
        """RAG query using CPU embedder with input_type='query'."""
        start_time_total = time.time()
        assert self.model is not None, "LLM not initialized"
        assert self.embedder is not None, "Embedder not initialized"
        assert self.chroma_collection is not None, "Chroma collection not initialized"
        assert self.bm25 is not None, "BM25 retriever not initialized"

        system_instruction = "Answer the query using *only* the provided Context. If the answer isn't in the Context, state that clearly."
        debug_info: Dict[str, Any] = {}
        retrieval_time, gen_time = -1.0, -1.0
        query_embedding_time = 0.0
        final_context = ""
        retrieved_display = "No relevant chunks retrieved."
        response = "Error: Query processing failed."
        vec_results_count, bm25_results_count = 0, 0

        try:
            # 1. Query Embedding
            query_emb_start = time.time()
            query_embedding = self.embedder.embed_string(query_text, input_type="query")
            query_embedding_time = time.time() - query_emb_start
            debug_info["query_embedding_time"] = f"{query_embedding_time:.3f}s"
            if query_embedding is None:
                raise ValueError("Failed to generate query embedding.")

            # 2. Retrieval Phase
            retrieval_start = time.time()
            num_to_fetch = max(self.max_results * 5, 25)

            # Vector Search (ChromaDB)
            vector_results: Optional[QueryResult] = None
            try:
                logger.debug(f"Querying ChromaDB with {num_to_fetch} results...")
                vector_results = self.chroma_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=num_to_fetch,
                    include=["metadatas", "documents", "distances"],
                )
                if (
                    vector_results
                    and vector_results.get("ids")
                    and vector_results["ids"]
                ):
                    vec_results_count = len(vector_results["ids"][0])
                logger.debug(f"ChromaDB returned {vec_results_count} results.")
            except Exception as e:
                logger.error(f"ChromaDB query failed: {e}", exc_info=debug_mode)

            # Keyword Search (Whoosh BM25)
            bm25_results: Dict[str, Any] = {}
            logger.debug(f"Querying Whoosh BM25 with {num_to_fetch} results...")
            try:
                bm25_results = self.bm25.query(query_text, n_results=num_to_fetch)
                bm25_results_count = len(bm25_results.get("ids", []))
                logger.debug(f"Whoosh BM25 returned {bm25_results_count} results.")
            except Exception as e:
                logger.error(f"BM25 query failed: {e}", exc_info=debug_mode)

            retrieval_time = time.time() - retrieval_start
            debug_info["retrieval_time"] = f"{retrieval_time:.3f}s"
            debug_info["vector_initial_results"] = vec_results_count
            debug_info["bm25_initial_results"] = bm25_results_count

            # 3. RRF Fusion & Ranking
            k_rrf = 60.0
            combined_scores: Dict[str, float] = {}
            doc_lookup: Dict[
                str, Dict[str, Any]
            ] = {}  # Store doc text and metadata by chunk ID

            # Process Vector Results
            if vector_results and vector_results.get("ids") and vector_results["ids"]:
                ids_list = vector_results["ids"][0]
                distances_list = (vector_results.get("distances") or [[]])[0]
                metadatas_list = (vector_results.get("metadatas") or [[]])[0]
                documents_list = (vector_results.get("documents") or [[]])[0]
                # Pad lists if lengths mismatch (should not happen with valid results)
                if len(distances_list) != len(ids_list):
                    distances_list = [2.0] * len(ids_list)
                if len(metadatas_list) != len(ids_list):
                    metadatas_list = [{}] * len(ids_list)
                if len(documents_list) != len(ids_list):
                    documents_list = [""] * len(ids_list)

                for i, chunk_id in enumerate(ids_list):
                    rank = i + 1
                    rrf_score = 1.0 / (k_rrf + rank)
                    combined_scores[chunk_id] = (
                        combined_scores.get(chunk_id, 0.0)
                        + rrf_score * self.vector_weight
                    )
                    if chunk_id not in doc_lookup:
                        doc_lookup[chunk_id] = {
                            "document": documents_list[i] or "",
                            "metadata": metadatas_list[i]
                            if isinstance(metadatas_list[i], dict)
                            else {},
                        }
                    if debug_mode:
                        distance = (
                            distances_list[i]
                            if isinstance(distances_list[i], (float, int))
                            else 2.0
                        )
                        vector_score = max(
                            0.0, 1.0 - (distance / 2.0)
                        )  # Cosine similarity score heuristic
                        if chunk_id not in debug_info.setdefault("chunk_details", {}):
                            debug_info["chunk_details"][chunk_id] = {}
                        debug_info["chunk_details"][chunk_id].update(
                            {
                                "vector_rank": rank,
                                "vector_dist": f"{distance:.4f}",
                                "vector_score": f"{vector_score:.4f}",
                                "vector_rrf": f"{rrf_score * self.vector_weight:.4f}",
                            }
                        )

            # Process Whoosh BM25 Results
            bm25_ids = bm25_results.get("ids", [])
            bm25_scores = bm25_results.get("scores", [])
            if len(bm25_scores) != len(bm25_ids):
                bm25_scores = [0.0] * len(bm25_ids)
            for i, chunk_id in enumerate(bm25_ids):
                rank = i + 1
                score_val = (
                    bm25_scores[i] if isinstance(bm25_scores[i], (float, int)) else 0.0
                )
                rrf_score = 1.0 / (k_rrf + rank)
                combined_scores[chunk_id] = (
                    combined_scores.get(chunk_id, 0.0) + rrf_score * self.bm25_weight
                )
                doc_lookup.setdefault(
                    chunk_id, {"document": "", "metadata": {}}
                )  # Ensure entry exists
                if debug_mode:
                    if chunk_id not in debug_info.setdefault("chunk_details", {}):
                        debug_info["chunk_details"][chunk_id] = {}
                    debug_info["chunk_details"][chunk_id].update(
                        {
                            "bm25_rank": rank,
                            "bm25_raw_score": f"{score_val:.4f}",
                            "bm25_rrf": f"{rrf_score * self.bm25_weight:.4f}",
                        }
                    )

            debug_info["combined_unique_chunks"] = len(combined_scores)
            sorted_chunks = sorted(
                combined_scores.items(), key=lambda item: item[1], reverse=True
            )

            # 4. Duplicate Filtering
            num_candidates = min(len(sorted_chunks), self.max_results * 3)
            top_candidates_with_scores = sorted_chunks[:num_candidates]
            filtered_top_chunks_with_scores = []
            added_chunk_texts = []
            skipped_due_similarity = 0
            for chunk_id, score in top_candidates_with_scores:
                if len(filtered_top_chunks_with_scores) >= self.max_results:
                    break

                if chunk_id not in doc_lookup or not doc_lookup[chunk_id].get(
                    "document"
                ):
                    try:
                        chroma_get = self.chroma_collection.get(
                            ids=[chunk_id], include=["documents", "metadatas"]
                        )
                        if (
                            chroma_get
                            and chroma_get.get("ids") 
                            and chroma_get.get("documents") is not None
                            and chroma_get.get("metadatas") is not None
                        ):
                            docs = chroma_get["documents"]
                            metas = chroma_get["metadatas"]
                            doc_lookup[chunk_id] = {
                                "document": docs[0] if docs else "",
                                "metadata": metas[0] if metas else {},
                            }
                        else:
                            continue  # Skip if fetch fails
                    except Exception:
                        continue

                current_text = doc_lookup[chunk_id].get("document", "")
                if not current_text.strip():
                    continue

                is_duplicate = any(
                    _are_chunks_too_similar(
                        current_text, et, CHUNK_SIMILARITY_THRESHOLD
                    )
                    for et in added_chunk_texts
                )
                if is_duplicate:
                    skipped_due_similarity += 1
                else:
                    filtered_top_chunks_with_scores.append((chunk_id, score))
                    added_chunk_texts.append(current_text)

            top_chunk_ids_with_scores = filtered_top_chunks_with_scores
            top_chunk_ids = [cid for cid, score in top_chunk_ids_with_scores]
            debug_info["skipped_similar_chunk_count"] = skipped_due_similarity
            debug_info["final_selected_chunk_count"] = len(top_chunk_ids)

            if top_chunk_ids:
                self.logger.info(f"Top {len(top_chunk_ids)} UNIQUE chunks selected.")
            else:
                self.logger.warning("No unique chunks selected after filtering.")

            # 5. Build Context String
            if not top_chunk_ids:
                response = "Could not find relevant information."
                query_time = (
                    retrieval_time if retrieval_time > 0 else query_embedding_time
                )
                return {
                    "response": response,
                    "debug_info": debug_info if debug_mode else {},
                    "retrieved_context": retrieved_display,
                    "formatted_response": f"<h2>AI Answer</h2><p>{response}</p><h2>Retrieved Context</h2>{retrieved_display}",
                    "query_time_seconds": query_time,
                    "generation_time_seconds": 0,
                }

            temp_context_parts = []
            temp_display_list = []
            prompt_base = (
                f"{system_instruction}\n\nContext:\n\n\nQuery: {query_text}\n\nAnswer:"
            )
            prompt_base_len = self._get_token_count(prompt_base)
            generation_reservation = max(250, self.context_length // 4)
            available_context_tokens = max(
                0, self.context_length - prompt_base_len - generation_reservation
            )
            current_context_len = 0
            context_chunks_details = []

            for i, (chunk_id, score) in enumerate(top_chunk_ids_with_scores):
                if chunk_id not in doc_lookup:
                    continue
                doc_data = doc_lookup[chunk_id]
                doc_text = doc_data.get("document", "")
                metadata = doc_data.get("metadata", {})
                if not doc_text.strip():
                    continue

                (
                    source_path,
                    filename,
                    original_url,
                    display_source,
                ) = "N/A", "N/A", None, "N/A"
                if isinstance(metadata, dict):
                    source_path, filename, original_url = (
                        metadata.get("source_path", "N/A"),
                        metadata.get("filename", "N/A"),
                        metadata.get("original_url"),
                    )
                    if original_url:
                        display_source = original_url
                    elif filename != "N/A":
                        display_source = filename
                    elif source_path != "N/A":
                        try:
                            display_source = Path(source_path).name
                        except Exception:
                            pass

                header_for_llm = (
                    f"[Source: {display_source} | Rank: {i + 1}]\n"  # Simplified header
                )
                source_display_html = (
                    f'<a href="{original_url}">{original_url}</a>'
                    if original_url
                    else source_path
                )
                header_for_display = f"--- Rank {i + 1} (Score: {score:.4f}) ---\nSource: {source_display_html}\n"
                doc_chunk_full_text = f"{header_for_llm}{doc_text}\n\n"
                doc_chunk_len = self._get_token_count(doc_chunk_full_text)

                if current_context_len + doc_chunk_len <= available_context_tokens:
                    temp_context_parts.append(doc_chunk_full_text)
                    temp_display_list.append(f"{header_for_display}{doc_text}\n\n")
                    current_context_len += doc_chunk_len
                    context_chunks_details.append(
                        {
                            "id": chunk_id,
                            "rank": i + 1,
                            "score": score,
                            "token_count": doc_chunk_len,
                            "source": display_source,
                        }
                    )
                else:
                    logger.warning(
                        f"Stopping context build rank {i + 1}. Limit reached."
                    )
                    debug_info["context_truncated_at_chunk_rank"] = i + 1
                    break

            final_context = "".join(temp_context_parts).strip()
            retrieved_display = (
                "".join(temp_display_list).strip()
                if temp_display_list
                else "No relevant chunks in context."
            )
            debug_info["final_context_token_count"] = current_context_len
            debug_info["final_context_chars"] = len(final_context)
            debug_info["final_context_chunks_details"] = context_chunks_details

            if not final_context:
                response = (
                    "Error: Could not build context."
                    if top_chunk_ids
                    else "Could not find info."
                )
                query_time = (
                    retrieval_time if retrieval_time > 0 else query_embedding_time
                )
                return {
                    "response": response,
                    "debug_info": debug_info if debug_mode else {},
                    "retrieved_context": retrieved_display,
                    "formatted_response": f"<h2>AI Answer</h2><p>{response}</p><h2>Retrieved Context</h2>{retrieved_display}",
                    "query_time_seconds": query_time,
                    "generation_time_seconds": 0,
                }

        except Exception as e:  # Catch retrieval/context errors
            logger.error(
                f"Error during retrieval/context build: {e}", exc_info=debug_mode
            )
            response = f"Error retrieving context: {e}"
            query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
            return {
                "response": response,
                "debug_info": debug_info if debug_mode else {},
                "retrieved_context": retrieved_display,
                "formatted_response": f"<h2>AI Answer</h2><p>{response}</p><h2>Retrieved Context</h2>{retrieved_display}",
                "query_time_seconds": query_time,
                "generation_time_seconds": 0,
            }

        # 6. LLM Generation
        prompt = f"{system_instruction}\n\nContext:\n{final_context}\n\nQuery: {query_text}\n\nAnswer:"
        prompt_token_count = self._get_token_count(prompt)
        debug_info["final_prompt_chars"] = len(prompt)
        debug_info["final_prompt_tokens_estimated"] = prompt_token_count
        if debug_mode:
            logger.debug(
                f"--- LLM Prompt ({prompt_token_count} tokens) ---\n{prompt}\n--- LLM Prompt End ---"
            )

        if prompt_token_count >= self.context_length:
            response = "Error: Prompt too long."
            query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
            return {
                "response": response,
                "debug_info": debug_info if debug_mode else {},
                "retrieved_context": retrieved_display,
                "formatted_response": f"<h2>AI Answer</h2><p>{response}</p><h2>Retrieved Context</h2>{retrieved_display}",
                "query_time_seconds": query_time,
                "generation_time_seconds": 0,
            }

        self.logger.info("Generating response with LLM (CPU)...")
        gen_start = time.time()
        text_response = "LLM Error."
        raw_llm_output = None
        assert self.model is not None

        try:
            if self._shutdown_event and self._shutdown_event.is_set():
                raise InterruptedError("Shutdown before LLM.")
            max_gen_tokens = max(50, self.context_length - prompt_token_count - 15)
            debug_info["llm_max_gen_tokens"] = max_gen_tokens
            text_response, raw_llm_output = self.model.generate(
                prompt=prompt,
                max_tokens=max_gen_tokens,
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.15,
            )
            response = text_response.strip() if text_response else ""
        except InterruptedError:
            response = "LLM cancelled."
        except Exception as e:
            logger.error(f"LLM gen error: {e}", exc_info=debug_mode)
            raw_llm_output = {"error": str(e)}
            response = f"LLM Error: {e}"

        gen_time = time.time() - gen_start
        self.logger.info(f"LLM gen took {gen_time:.2f}s.")
        debug_info["llm_generation_time"] = f"{gen_time:.3f}s"
        if (
            raw_llm_output
            and isinstance(raw_llm_output, dict)
            and raw_llm_output.get("error")
        ):
            response = f"LLM Error: {raw_llm_output['error']}"
        elif not response or response.isspace():
            response = "(LLM empty response)"

        # 7. Final Result
        total_time = time.time() - start_time_total
        debug_info["total_query_processing_time"] = f"{total_time:.3f}s"
        formatted_response = f"<h2>AI Answer</h2><p>{response}</p><h2>Retrieved Context</h2>{retrieved_display}"
        try:
            log_query(
                query_text,
                debug_info.get("final_context_chunks_details", []),
                response,
                debug_info,
                full_logging=debug_mode,
            )
        except Exception as log_e:
            logger.warning(f"Failed log query: {log_e}")
        query_time_final = (
            retrieval_time if retrieval_time > 0 else query_embedding_time
        )
        self.logger.info(f"Search Result:\n{response}")
        return {
            "response": response,
            "debug_info": debug_info if debug_mode else {},
            "retrieved_context": retrieved_display,
            "formatted_response": formatted_response,
            "query_time_seconds": query_time_final,
            "generation_time_seconds": gen_time if gen_time > 0 else 0,
        }

    def _safe_unload_llm(self, timeout: float = 3.0) -> None:
        """Safely unload LLM (CPU-only version)."""
        if self.model is None or not hasattr(self.model, "unload"):
            logger.debug("No LLM model or unload method.")
            return
        logger.info("Skip explicit Teapot unload on CPU.")
        self.model = None
        gc.collect()
        return

    def close(self) -> None:
        """Unload models and release resources (CPU-Only)."""
        logger.info("Closing LLMSearch components (CPU-Only)...")
        if self._shutdown_event and not self._shutdown_event.is_set():
            self._shutdown_event.set()
        if self.model:
            self._safe_unload_llm()
        if self.embedder:
            self.embedder.close()
            self.embedder = None
        if self.bm25:
            try:
                self.bm25.close()
            except Exception:
                pass
            finally:
                self.bm25 = None
        if self.chroma_client:
            # No explicit close for PersistentClient, rely on GC
            self.chroma_client = None
        logger.info("LLMSearch closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
