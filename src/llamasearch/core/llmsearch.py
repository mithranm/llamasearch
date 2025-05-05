# src/llamasearch/core/llmsearch.py

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import chromadb
from chromadb.api.types import GetResult, QueryResult
from chromadb.config import Settings as ChromaSettings
import torch

from llamasearch.core.bm25 import BM25Retriever
from llamasearch.core.chunker import MarkdownChunker
from llamasearch.core.embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_NAME
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.teapot import TeapotONNXLLM, load_teapot_onnx_llm
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.protocols import LLM
from llamasearch.utils import NumpyEncoder, setup_logging, log_query

logger = setup_logging(__name__, use_qt_handler=True)

# Constants
CHROMA_COLLECTION_NAME = "llamasearch_docs"
BM25_SUBDIR = "bm25_data"


class LLMSearch:
    """
    RAG-based search using Teapot ONNX, ChromaDB, and BM25.
    Manages LLM, Embedder, ChromaDB Client/Collection, and BM25Retriever instances.
    """

    def __init__(
        self,
        storage_dir: Path,
        shutdown_event: Optional[threading.Event] = None,
        teapot_onnx_quant: str = "auto",
        teapot_provider: Optional[str] = None,
        teapot_provider_opts: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        max_results: int = 3,
        embedder_model: Optional[str] = None,
        embedder_batch_size: int = 32,
        max_chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 128,
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
        self.bm25: Optional[BM25Retriever] = None
        self.chunker: Optional[MarkdownChunker] = None

        self.context_length: int = 0
        self.llm_device_type: str = "cpu"

        try:
            # 1. Load LLM
            self.logger.info("Initializing Teapot ONNX LLM...")
            self.model = load_teapot_onnx_llm(
                onnx_quantization=teapot_onnx_quant,
                preferred_provider=teapot_provider,
                preferred_options=teapot_provider_opts,
            )
            if not self.model:
                raise RuntimeError("load_teapot_onnx_llm returned None")
            model_info = self.model.model_info
            self.context_length = model_info.context_length
            # Determine device type safely
            if hasattr(self.model, 'device') and isinstance(getattr(self.model, 'device', None), torch.device):
                self.llm_device_type = self.model.device.type
            elif isinstance(self.model, TeapotONNXLLM) and hasattr(self.model, '_provider'):
                provider = getattr(self.model, '_provider')
                if "CUDA" in provider:
                    self.llm_device_type = "cuda"
                elif "ROCM" in provider:
                    self.llm_device_type = "rocm"
                elif "CoreML" in provider:
                    self.llm_device_type = "mps"
                else:
                    self.llm_device_type = "cpu"
            else:
                self.llm_device_type = "cpu"
            self.logger.info(f"LLM: {model_info.model_id} on {self.llm_device_type}. Context: {self.context_length}")

            # 2. Initialize Embedder
            self.logger.info("Initializing Embedder...")
            self.embedder = EnhancedEmbedder(
                model_name=embedder_model or DEFAULT_EMBEDDER_NAME,
                batch_size=embedder_batch_size,
            )
            if self.embedder and self._shutdown_event:
                self.embedder.set_shutdown_event(self._shutdown_event)

            # 3. Initialize Chunker
            self.logger.info("Initializing Markdown Chunker...")
            self.chunker = MarkdownChunker(
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
                overlap_percent=(chunk_overlap / max_chunk_size) if max_chunk_size > 0 else 0.1
            )

            # 4. Initialize ChromaDB
            self.logger.info(f"Initializing ChromaDB Client (storage: {self.storage_dir})")
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.storage_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True)
            )
            if self.chroma_client is not None:
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=CHROMA_COLLECTION_NAME,
                )
            else:
                raise RuntimeError("Chroma client not initialized!")
            assert self.chroma_collection is not None, "ChromaDB collection creation failed"
            logger.info(f"ChromaDB Collection '{CHROMA_COLLECTION_NAME}' ready. Count: {self.chroma_collection.count()}")

            # 5. Initialize BM25Retriever
            bm25_path = self.storage_dir / BM25_SUBDIR
            self.logger.info(f"Initializing BM25 Retriever (storage: {bm25_path})")
            self.bm25 = BM25Retriever(storage_dir=bm25_path)
            logger.info(f"BM25 Retriever ready. Initial Count: {self.bm25.get_doc_count()}. Index rebuild required: {self.bm25._index_needs_rebuild}")

            self.logger.info("LLMSearch components initialized successfully.")

        except ModelNotFoundError as e:
            self.logger.error(f"LLMSearch init failed: {e}. Run 'llamasearch-setup'.")
            self.close()
            raise
        except ImportError as e:
            if 'chromadb' in str(e):
                self.logger.error("ChromaDB not installed. Install: pip install chromadb")
                raise SetupError("ChromaDB not installed.") from e
            else:
                self.logger.error(f"Missing import during init: {e}", exc_info=True)
                raise
        except Exception as e:
            self.logger.error(f"Unexpected error during LLMSearch init: {e}", exc_info=True)
            self.close()
            raise RuntimeError("LLMSearch failed to initialize.") from e

    def _process_file_to_markdown(self, file_path: Path) -> Tuple[Optional[str], Optional[Path]]:
        """ Converts DOCX, PDF etc. to Markdown using pandoc. Returns content and temp path if created."""
        ext = file_path.suffix.lower()
        supported_conversion_ext = [".docx", ".doc", ".pdf", ".odt", ".rtf", ".epub"]
        passthrough_ext = [".md", ".markdown", ".txt", ".html", ".htm"]

        if ext in passthrough_ext:
            try:
                return file_path.read_text(encoding='utf-8', errors='ignore'), None
            except Exception as e:
                logger.error(f"Error reading passthrough file {file_path}: {e}")
                return None, None
        elif ext in supported_conversion_ext:
            pandoc_path = shutil.which("pandoc")
            if not pandoc_path:
                logger.warning(f"pandoc not found, cannot convert {file_path.name}. Skipping.")
                return None, None
            temp_dir = self.storage_dir / "temp_conversions"
            temp_dir.mkdir(exist_ok=True)
            md_temp_file = temp_dir / f"{file_path.stem}_{uuid4().hex[:8]}.md.temp"
            # Use markdown_strict+pipe_tables for better table handling, disable native spans
            cmd = [pandoc_path, str(file_path), "-t", "markdown_strict+pipe_tables-native_spans", "--wrap=none", "-o", str(md_temp_file)]
            logger.info(f"Converting '{file_path.name}' using pandoc...")
            try:
                creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if sys.platform == 'win32' else 0
                res = subprocess.run(cmd, capture_output=True, check=False, timeout=300, creationflags=creationflags)
                if res.returncode != 0:
                    stderr = res.stderr.decode("utf-8", "ignore").strip() if res.stderr else "No stderr."
                    logger.error(f"Pandoc error (Code {res.returncode}) for {file_path.name}. Command: {' '.join(cmd)}\nStderr: {stderr}")
                    md_temp_file.unlink(missing_ok=True)
                    return None, None
                if not md_temp_file.exists() or md_temp_file.stat().st_size == 0:
                    logger.warning(f"Pandoc ran but produced empty/no output for {file_path.name}. Treating as empty content.")
                    md_temp_file.unlink(missing_ok=True)
                    return "", None # Return empty string for empty content
                logger.debug(f"Successfully converted '{file_path.name}' to Markdown.")
                content = md_temp_file.read_text(encoding='utf-8', errors='ignore')
                return content, md_temp_file
            except subprocess.TimeoutExpired:
                logger.error(f"Pandoc conversion timed out for {file_path.name}.")
                md_temp_file.unlink(missing_ok=True)
                return None, None
            except Exception as e:
                logger.error(f"Pandoc conversion exception for {file_path.name}: {e}", exc_info=True)
                md_temp_file.unlink(missing_ok=True)
                return None, None
        else:
            logger.debug(f"File type '{ext}' not configured for processing, skipping: {file_path.name}")
            return None, None

    def add_source(self, source_path_str: str) -> Tuple[int, bool]:
        """
        Adds a single source file or directory recursively.
        Returns:
            Tuple[int, bool]: (Number of chunks added, BM25 index needs rebuild)
        """
        assert self.embedder is not None, "Embedder not initialized"
        assert self.chunker is not None, "Chunker not initialized"
        assert self.chroma_collection is not None, "Chroma collection not initialized"
        assert self.bm25 is not None, "BM25 retriever not initialized"
        if self._shutdown_event and self._shutdown_event.is_set():
            logger.warning("Add source cancelled due to shutdown.")
            return 0, False

        source_path = Path(source_path_str).resolve()
        total_chunks_added = 0
        bm25_needs_rebuild = False # Track if BM25 was modified in this call

        if not source_path.exists():
            logger.error(f"Source path not found: {source_path}")
            self.remove_source(source_path_str)
            return 0, False

        if source_path.is_file():
            logger.info(f"Processing file: {source_path}")
            if self._is_source_unchanged(source_path_str):
                logger.info(f"File '{source_path.name}' is unchanged based on mtime. Skipping.")
                return 0, False

            # Removal might trigger BM25 rebuild flag if successful
            removed_ok, removal_needs_rebuild = self.remove_source(source_path_str)
            if removal_needs_rebuild:
                 bm25_needs_rebuild = True

            markdown_content, temp_file = self._process_file_to_markdown(source_path)

            if markdown_content is None:
                logger.warning(f"Skipping file {source_path.name}: Content processing failed or format unsupported.")
                if temp_file:
                    temp_file.unlink(missing_ok=True)
                return 0, bm25_needs_rebuild
            if not markdown_content.strip():
                 logger.info(f"Skipping file {source_path.name}: Content is empty.")
                 if temp_file:
                     temp_file.unlink(missing_ok=True)
                 if not removed_ok: # Ensure old data is removed if removal failed before
                     self.remove_source(source_path_str)
                 return 0, bm25_needs_rebuild

            try:
                chunks = list(self.chunker.chunk_document(markdown_content, source=source_path_str))
                if not chunks:
                    logger.warning(f"No chunks generated from file {source_path.name}. Skipping add.")
                    if temp_file:
                        temp_file.unlink(missing_ok=True)
                    if not removed_ok:
                        self.remove_source(source_path_str)
                    return 0, bm25_needs_rebuild

                chunk_texts, chunk_metadatas, chunk_ids = [], [], []
                base_meta: Dict[str, Any] = {
                    "source_path": source_path_str,
                    "filename": source_path.name
                }
                try:
                    base_meta["mtime"] = source_path.stat().st_mtime
                except OSError as e:
                    logger.warning(f"Could not stat file {source_path}: {e}")
                    base_meta["mtime"] = time.time()

                for i, c in enumerate(chunks):
                    chunk_id = f"{source_path.stem.replace('.', '_')}_{uuid4().hex[:12]}"
                    chunk_ids.append(chunk_id)
                    meta = base_meta.copy()
                    chunker_meta = c.get("metadata", {})
                    if isinstance(chunker_meta, dict):
                        meta.update(chunker_meta)
                    meta['chunk_id'] = chunk_id
                    chunk_metadatas.append(meta)
                    chunk_text = c.get("chunk", "")
                    if chunk_text is None:
                        chunk_text = ""
                    chunk_texts.append(chunk_text)

                logger.info(f"Embedding {len(chunk_texts)} chunks for {source_path.name}...")
                embeddings = self.embedder.embed_strings(chunk_texts, show_progress=True)

                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.warning("Add source embedding cancelled due to shutdown.")
                    if temp_file:
                        temp_file.unlink(missing_ok=True)
                    return 0, bm25_needs_rebuild

                if embeddings is None or embeddings.shape[0] != len(chunk_texts):
                    logger.error(f"Embedding failed or returned incorrect shape for {source_path.name}.")
                    if temp_file:
                        temp_file.unlink(missing_ok=True)
                    return 0, bm25_needs_rebuild

                logger.info(f"Adding {len(chunk_ids)} chunks to ChromaDB...")
                try:
                    embeddings_list = embeddings.tolist()
                    self.chroma_collection.upsert(
                        ids=chunk_ids,
                        embeddings=embeddings_list,
                        metadatas=chunk_metadatas,
                        documents=chunk_texts
                    )
                except Exception as chroma_e:
                    logger.error(f"ChromaDB upsert failed for {source_path.name}: {chroma_e}", exc_info=True)
                    raise

                logger.info(f"Adding {len(chunk_ids)} chunks to BM25 tracker...")
                bm25_added_count = 0
                for text, chunk_id in zip(chunk_texts, chunk_ids):
                    if self.bm25.add_document(text, chunk_id):
                        bm25_added_count += 1

                if bm25_added_count > 0:
                    bm25_needs_rebuild = True

                total_chunks_added = len(chunk_ids)
                logger.info(f"Successfully processed {total_chunks_added} chunks from {source_path.name}.")

            except Exception as e:
                logger.error(f"Failed processing file {source_path}: {e}", exc_info=True)
                self.remove_source(source_path_str)
                total_chunks_added = 0
            finally:
                if temp_file:
                    temp_file.unlink(missing_ok=True)

        elif source_path.is_dir():
            logger.info(f"Processing directory recursively: {source_path}")
            files_processed, files_failed = 0, 0
            supported_suffixes = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx", ".doc", ".odt", ".rtf", ".epub"}
            any_bm25_modified = False

            all_files = list(source_path.rglob("*"))
            logger.info(f"Found {len(all_files)} items in directory tree. Filtering...")

            for item in all_files:
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.warning("Directory processing cancelled during iteration.")
                    break
                if item.is_file() and item.suffix.lower() in supported_suffixes:
                    try:
                        added, file_needs_rebuild = self.add_source(str(item))
                        if added > 0:
                            total_chunks_added += added
                            files_processed += 1
                        elif not self._is_source_unchanged(str(item)):
                            files_failed += 1
                        if file_needs_rebuild:
                            any_bm25_modified = True
                    except Exception as e:
                        logger.error(f"Error processing file {item} within directory scan: {e}", exc_info=self.debug)
                        files_failed += 1
                elif item.is_file():
                    logger.debug(f"Skipping unsupported file type: {item.name}")

            logger.info(f"Directory scan for {source_path.name} complete. Added {total_chunks_added} new chunks from {files_processed} files.")
            if files_failed > 0:
                logger.warning(f"Failed to process/add chunks for {files_failed} files within directory scan.")
            if any_bm25_modified:
                bm25_needs_rebuild = True

        else:
            logger.warning(f"Source path is not a file or directory: {source_path}")

        return total_chunks_added, bm25_needs_rebuild

    def build_bm25_index_if_needed(self) -> bool:
        """ Builds the BM25 index if it's marked as needing rebuild. Returns True on success/no-op, False on failure. """
        assert self.bm25 is not None, "BM25 retriever not initialized"
        success = True
        if self.bm25._index_needs_rebuild:
            logger.info("BM25 index needs rebuild. Starting build...")
            success = self.bm25.build_index()
            if success:
                logger.info("BM25 index rebuilt successfully.")
            else:
                logger.error("BM25 index build failed. BM25 search might not work correctly.")
        else:
            logger.debug("BM25 index is up-to-date. No rebuild needed.")
        return success

    def _is_source_unchanged(self, source_path_str: str) -> bool:
        assert self.chroma_collection is not None
        source_path = Path(source_path_str)
        if not source_path.is_file():
            return False
        try:
            current_mtime = source_path.stat().st_mtime
            results: Optional[GetResult] = self.chroma_collection.get(
                where={"source_path": source_path_str},
                limit=1,
                include=["metadatas"]
            )
            if results and results['ids'] and results['metadatas'] and results['metadatas'][0]:
                stored_meta = results['metadatas'][0]
                if isinstance(stored_meta, dict) and 'mtime' in stored_meta:
                    stored_mtime = stored_meta['mtime']
                    if isinstance(stored_mtime, (int, float)) and abs(float(stored_mtime) - current_mtime) < 1e-4:
                         return True
                    else:
                         return False
                else:
                    return False
            else:
                return False
        except FileNotFoundError:
             logger.warning(f"File not found during mtime check: {source_path_str}. Assuming changed.")
             return False
        except Exception as e:
            logger.warning(f"Error checking source status for '{source_path_str}': {e}. Assuming changed.", exc_info=self.debug)
            return False

    def remove_source(self, source_path_str: str) -> Tuple[bool, bool]:
        """
        Removes all chunks associated with a given source_path.
        Returns:
            Tuple[bool, bool]: (If any removal occurred, If BM25 index needs rebuild)
        """
        assert self.chroma_collection is not None
        assert self.bm25 is not None
        logger.info(f"Attempting to remove all chunks for source: {source_path_str}")
        chroma_removed_count = 0
        bm25_removed_count = 0
        initial_bm25_rebuild_needed = self.bm25._index_needs_rebuild # Check state before removal
        try:
            ids_to_remove: List[str] = []
            results: Optional[GetResult] = self.chroma_collection.get(
                where={"source_path": source_path_str},
                include=[]
            )
            if results and results['ids']:
                ids_to_remove = results['ids']

            if not ids_to_remove:
                logger.info(f"No indexed chunks found for source path '{source_path_str}'. Nothing to remove.")
                return False, initial_bm25_rebuild_needed # No removal, return initial rebuild state

            logger.info(f"Removing {len(ids_to_remove)} chunks from ChromaDB for '{source_path_str}'...")
            if ids_to_remove:
                self.chroma_collection.delete(ids=ids_to_remove)
                chroma_removed_count = len(ids_to_remove)

            logger.info(f"Removing {len(ids_to_remove)} corresponding chunks from BM25 tracker...")
            for chunk_id in ids_to_remove:
                if self.bm25.remove_document(chunk_id):
                    bm25_removed_count += 1

            # BM25 needs rebuild if documents were actually removed from it
            final_bm25_rebuild_needed = self.bm25._index_needs_rebuild

            logger.info(f"Successfully removed {chroma_removed_count} chunks (Chroma) and {bm25_removed_count} chunks (BM25) for source: {source_path_str}")
            # Indicate removal occurred if either Chroma or BM25 removed something
            removal_occurred = chroma_removed_count > 0 or bm25_removed_count > 0
            return removal_occurred, final_bm25_rebuild_needed

        except Exception as e:
            logger.error(f"Error removing source '{source_path_str}': {e}", exc_info=True)
            # Return False for removal, but keep original rebuild state as error might be intermediate
            return False, initial_bm25_rebuild_needed

    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """ Aggregates chunk metadata to list unique indexed sources. """
        assert self.chroma_collection is not None
        try:
            all_results: Optional[GetResult] = self.chroma_collection.get(include=["metadatas"])

            if not all_results or not all_results.get('ids') or not all_results.get('metadatas'):
                return []

            all_metadata = all_results['metadatas']
            if not all_metadata:
                return []

            source_info: Dict[str, Dict[str, Any]] = {}
            valid_meta_count = 0
            for meta in all_metadata:
                if not isinstance(meta, dict):
                    continue
                valid_meta_count += 1
                src_path = meta.get('source_path')

                if not isinstance(src_path, str) or not src_path:
                    logger.warning(f"Found chunk metadata with invalid/missing source_path: {meta}")
                    continue

                if src_path not in source_info:
                    filename = "N/A"
                    try:
                        p = Path(src_path)
                        filename = p.name
                    except Exception:
                        pass

                    source_info[src_path] = {
                        'source_path': src_path,
                        'filename': meta.get('filename', filename),
                        'chunk_count': 0,
                        'mtime': meta.get('mtime')
                    }
                source_info[src_path]['chunk_count'] += 1
                current_mtime = source_info[src_path].get('mtime')
                meta_mtime = meta.get('mtime')
                if meta_mtime is not None and (current_mtime is None or meta_mtime > current_mtime):
                     source_info[src_path]['mtime'] = meta_mtime


            logger.debug(f"Processed {valid_meta_count} metadata entries, aggregated into {len(source_info)} unique sources.")

            source_list = list(source_info.values())
            source_list.sort(key=lambda x: str(x.get('filename', '')).lower())
            return source_list
        except Exception as e:
            logger.error(f"Error retrieving indexed sources: {e}", exc_info=True)
            return []

    def _get_token_count(self, text: str) -> int:
        """Estimate token count. Uses tokenizer if available, otherwise heuristic."""
        if (self.model and isinstance(self.model, TeapotONNXLLM) and
            hasattr(self.model, "_tokenizer") and self.model._tokenizer is not None):
            try:
                encode_method = getattr(self.model._tokenizer, "encode", None)
                if callable(encode_method):
                    encoded_ids = encode_method(text, add_special_tokens=False)
                    # Check if the result is a list (most common)
                    if isinstance(encoded_ids, list):
                        return len(encoded_ids)
                    # Check if the result has a 'shape' attribute (like a tensor)
                    elif hasattr(encoded_ids, 'shape'):
                        shape_attr = getattr(encoded_ids, 'shape', None) # Safely get shape
                        # Check if shape_attr is not None, sequence-like, and non-empty
                        if shape_attr is not None and hasattr(shape_attr, '__len__') and hasattr(shape_attr, '__getitem__') and len(shape_attr) > 0:
                            try:
                                # Assume the last dimension is the token count
                                return int(shape_attr[-1]) # Ensure it's an int
                            except (TypeError, IndexError) as shape_err:
                                logger.warning(f"Could not get token count from shape attribute {shape_attr}: {shape_err}")
                        else:
                            logger.warning(f"Tokenizer encode returned object with shape attribute of unexpected type or structure: {type(shape_attr)}")
                    else:
                        logger.warning(f"Tokenizer encode returned unexpected type: {type(encoded_ids)}")
            except Exception as e:
                logger.warning(f"Tokenizer encode failed: {e}. Falling back to heuristic.")
        # Fallback heuristic
        return max(1, len(text) // 4)


    def llm_query(self, query_text: str, debug_mode: bool = False) -> Dict[str, Any]:
        """ RAG-based retrieval (Chroma + BM25) + LLM generation. """
        start_time_total = time.time()
        assert self.model is not None, "LLM not initialized"
        assert self.embedder is not None, "Embedder not initialized"
        assert self.chroma_collection is not None, "Chroma collection not initialized"
        assert self.bm25 is not None, "BM25 retriever not initialized"

        system_instruction = "Answer the query using *only* the provided Context. If the answer isn't in the Context, state that clearly."
        debug_info: Dict[str, Any] = {}
        retrieval_time, gen_time = -1.0, -1.0
        query_embedding_time = 0.0 # Initialize here
        final_context, retrieved_display = "", "No relevant chunks retrieved."
        response = "Error: Query processing failed."
        vec_ids: List[str] = [] # Initialize here

        try:
            query_emb_start = time.time()
            query_embedding = self.embedder.embed_string(query_text)
            query_embedding_time = time.time() - query_emb_start
            debug_info["query_embedding_time"] = f"{query_embedding_time:.3f}s"
            if query_embedding is None:
                raise ValueError("Failed to generate query embedding.")

            retrieval_start = time.time()
            num_to_fetch = max(self.max_results * 4, 20)

            vector_results: Optional[QueryResult] = None
            try:
                 vector_results = self.chroma_collection.query(
                     query_embeddings=[query_embedding.tolist()],
                     n_results=num_to_fetch,
                     include=["metadatas", "documents", "distances"]
                 )
            except Exception as chroma_err:
                 logger.error(f"ChromaDB query failed: {chroma_err}", exc_info=self.debug)

            bm25_results = self.bm25.query(query_text, n_results=num_to_fetch)
            retrieval_time = time.time() - retrieval_start
            debug_info["retrieval_time"] = f"{retrieval_time:.3f}s"

            k_rrf = 60.0
            combined_scores: Dict[str, float] = {}
            doc_lookup: Dict[str, Dict[str, Any]] = {}

            # Process Vector Results safely
            if vector_results:
                ids_list = vector_results.get("ids")
                distances_list = vector_results.get("distances")
                metadatas_list = vector_results.get("metadatas")
                documents_list = vector_results.get("documents")

                # Check if the lists exist and are not empty before accessing [0]
                if ids_list and len(ids_list) > 0:
                    vec_ids = ids_list[0]
                    vec_distances = distances_list[0] if distances_list and len(distances_list) > 0 else []
                    vec_metadatas = metadatas_list[0] if metadatas_list and len(metadatas_list) > 0 else []
                    vec_documents = documents_list[0] if documents_list and len(documents_list) > 0 else []

                    for i, chunk_id in enumerate(vec_ids):
                        rrf_score = 1.0 / (k_rrf + i + 1)
                        combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + rrf_score * self.vector_weight

                        if chunk_id not in doc_lookup:
                             doc_lookup[chunk_id] = {
                                 "document": vec_documents[i] if i < len(vec_documents) else "",
                                 "metadata": vec_metadatas[i] if i < len(vec_metadatas) else {}
                             }
                        if debug_mode:
                            distance = vec_distances[i] if i < len(vec_distances) else 2.0
                            vector_score = max(0.0, 1.0 - (distance / 2.0))
                            if chunk_id not in debug_info:
                                debug_info[chunk_id] = {}
                            debug_info[chunk_id]['vector_rank'] = i + 1
                            debug_info[chunk_id]['vector_score'] = f"{vector_score:.4f}"
                            debug_info[chunk_id]['vector_rrf'] = f"{rrf_score * self.vector_weight:.4f}"

            # Process BM25 Results
            bm25_ids = bm25_results.get("ids", [])
            bm25_scores = bm25_results.get("scores", [])
            bm25_documents = bm25_results.get("documents", [])
            for rank, chunk_id in enumerate(bm25_ids):
                 rrf_score = 1.0 / (k_rrf + rank + 1)
                 combined_scores[chunk_id] = combined_scores.get(chunk_id, 0.0) + rrf_score * self.bm25_weight

                 if chunk_id not in doc_lookup:
                      doc_lookup[chunk_id] = {
                          "document": bm25_documents[rank] if rank < len(bm25_documents) else "",
                          "metadata": {"source_path": "N/A (BM25 only)", "filename": "N/A"}
                      }
                 if debug_mode:
                      if chunk_id not in debug_info:
                          debug_info[chunk_id] = {}
                      debug_info[chunk_id]['bm25_rank'] = rank + 1
                      debug_info[chunk_id]['bm25_score'] = f"{bm25_scores[rank]:.4f}" if rank < len(bm25_scores) else 'N/A'
                      debug_info[chunk_id]['bm25_rrf'] = f"{rrf_score * self.bm25_weight:.4f}"

            debug_info["vector_initial_results"] = len(vec_ids)
            debug_info["bm25_initial_results"] = len(bm25_ids)
            debug_info["combined_unique_chunks"] = len(combined_scores)

            sorted_chunks = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
            top_chunk_ids = [chunk_id for chunk_id, score in sorted_chunks[:self.max_results]]
            debug_info["final_selected_chunk_ids"] = top_chunk_ids
            debug_info["final_selected_chunk_count"] = len(top_chunk_ids)

            if not top_chunk_ids:
                logger.warning(f"No relevant chunks found for query after fusion: '{query_text[:100]}...'")
                response = "Could not find relevant information in the indexed documents to answer the query."
                query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
                return {
                    "response": response,
                    "debug_info": debug_info if debug_mode else {},
                    "retrieved_context": retrieved_display,
                    "formatted_response": f"## AI Answer\n{response}\n\n## Retrieved Context\n{retrieved_display}",
                    "query_time_seconds": query_time,
                    "generation_time_seconds": 0,
                }
            else:
                logger.info(f"Selected {len(top_chunk_ids)} chunks after fusion for context.")
                temp_context, temp_display_list = "", []
                prompt_base = f"{system_instruction}\n\nContext:\n\n\nQuery: {query_text}\n\nAnswer:"
                prompt_base_len = self._get_token_count(prompt_base)
                generation_reservation = max(150, self.context_length // 4)
                available_context_tokens = self.context_length - prompt_base_len - generation_reservation

                if available_context_tokens <= 0:
                     logger.error(f"Insufficient context space after accounting for prompt base and generation reservation. Available: {available_context_tokens}")
                     raise ValueError("Not enough context length available for retrieved documents.")

                debug_info["available_context_tokens"] = available_context_tokens
                current_context_len = 0

                for i, chunk_id in enumerate(top_chunk_ids):
                    if chunk_id not in doc_lookup:
                        logger.warning(f"Data for selected chunk_id {chunk_id} not found in lookup. Skipping.")
                        continue

                    doc_data = doc_lookup[chunk_id]
                    doc_text = doc_data.get("document", "")
                    metadata = doc_data.get("metadata", {})
                    final_score = combined_scores.get(chunk_id, 0.0)

                    if not doc_text:
                         logger.debug(f"Skipping empty document text for chunk {chunk_id}")
                         continue

                    source_path, filename = "N/A", "N/A"
                    if isinstance(metadata, dict):
                        source_path = metadata.get('source_path', 'N/A')
                        filename = metadata.get('filename', 'N/A')
                        if filename == "N/A" and source_path != "N/A":
                            try:
                                filename = Path(source_path).name
                            except Exception:
                                pass

                    header = f"[Doc {i + 1} | Source: {filename} | Score: {final_score:.3f}]\n"
                    doc_chunk_full_text = f"{header}{doc_text}\n\n"
                    display_chunk_text = f"--- Chunk {i + 1} (Score: {final_score:.3f}) ---\nSource: {source_path}\n{doc_text}\n\n"
                    doc_chunk_len = self._get_token_count(doc_chunk_full_text)

                    if current_context_len + doc_chunk_len <= available_context_tokens:
                        temp_context += doc_chunk_full_text
                        temp_display_list.append(display_chunk_text)
                        current_context_len += doc_chunk_len
                    else:
                        logger.warning(f"Stopping context build at chunk {i + 1}/{len(top_chunk_ids)} due to token limit ({current_context_len}+{doc_chunk_len} > {available_context_tokens}).")
                        debug_info["context_truncated_at_chunk"] = i + 1
                        break

                final_context = temp_context.strip()
                retrieved_display = "".join(temp_display_list).strip()
                debug_info["final_context_token_count"] = current_context_len

        except Exception as e:
            logger.error(f"Error during retrieval/fusion: {e}", exc_info=self.debug)
            response = f"Error during context retrieval: {e}"
            query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
            return {
                "response": response,
                "debug_info": debug_info if debug_mode else {},
                "retrieved_context": retrieved_display,
                "formatted_response": f"## AI Answer\n{response}\n\n## Retrieved Context\n{retrieved_display}",
                "query_time_seconds": query_time,
                "generation_time_seconds": 0,
            }

        prompt = f"{system_instruction}\n\nContext:\n{final_context}\n\nQuery: {query_text}\n\nAnswer:"
        prompt_token_count = self._get_token_count(prompt)
        debug_info["final_prompt_chars"] = len(prompt)
        debug_info["final_prompt_tokens_estimated"] = prompt_token_count

        if self.debug:
            log_prompt = prompt[:2000] + "..." if len(prompt) > 2000 else prompt
            logger.debug(f"--- LLM Prompt Start ({prompt_token_count} tokens est.) ---\n{log_prompt}\n--- LLM Prompt End ---")

        if prompt_token_count >= self.context_length:
            logger.error(f"Final prompt token count ({prompt_token_count}) exceeds model context limit ({self.context_length}). Cannot generate response.")
            response = "Error: The retrieved context and query are too long for the language model's context limit."
            query_time = retrieval_time if retrieval_time > 0 else query_embedding_time
            return {
                "response": response,
                "debug_info": debug_info if debug_mode else {},
                "retrieved_context": retrieved_display,
                "formatted_response": f"## AI Answer\n{response}\n\n## Retrieved Context\n{retrieved_display}",
                "query_time_seconds": query_time,
                "generation_time_seconds": 0,
            }

        logger.info("Generating response with LLM...")
        gen_start = time.time()
        text_response = "Error: LLM generation failed."
        raw_llm_output = None
        assert self.model is not None, "LLM became None before generation"

        try:
            if self._shutdown_event and self._shutdown_event.is_set():
                raise InterruptedError("Shutdown requested before LLM generation.")

            max_gen_tokens = max(50, self.context_length - prompt_token_count - 10)
            debug_info["llm_max_gen_tokens"] = max_gen_tokens

            text_response, raw_llm_output = self.model.generate(
                prompt=prompt,
                max_tokens=max_gen_tokens,
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.15,
            )
            response = text_response.strip()
        except InterruptedError as e:
            logger.warning(f"LLM generation cancelled: {e}")
            response = "LLM response generation was cancelled."
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=self.debug)
            raw_llm_output = {"error": str(e)}
            response = f"Error during LLM generation: {e}"

        gen_time = time.time() - gen_start
        logger.info(f"LLM generation took {gen_time:.2f}s. Response length: {len(response)} chars.")
        debug_info["llm_generation_time"] = f"{gen_time:.3f}s"

        if self.debug and raw_llm_output is not None:
            try:
                debug_info["raw_llm_output"] = json.dumps(raw_llm_output, cls=NumpyEncoder, indent=2)
            except TypeError:
                 debug_info["raw_llm_output"] = f"Raw LLM output contains non-serializable data: {type(raw_llm_output)}"
            except Exception as json_err:
                 debug_info["raw_llm_output"] = f"Error serializing raw LLM output: {json_err}"

        if raw_llm_output and raw_llm_output.get("error"):
            logger.error(f"LLM generation failed: {raw_llm_output['error']}")
            response = f"Error during LLM generation: {raw_llm_output['error']}"
        elif not response or len(response.strip()) < 5:
            logger.warning(f"LLM response is empty or very short ({len(response.strip())} chars). May indicate an issue.")
            response = "(LLM returned empty response)" # Provide clearer feedback

        total_time = time.time() - start_time_total
        debug_info["total_query_processing_time"] = f"{total_time:.3f}s"
        formatted_response = f"## AI Answer\n{response}\n\n## Retrieved Context\n{retrieved_display}"

        try:
            log_query(query_text, [], response, debug_info, full_logging=debug_mode)
        except Exception as log_e:
            logger.warning(f"Failed to log query details: {log_e}")

        query_time_final = retrieval_time if retrieval_time > 0 else query_embedding_time
        return {
            "response": response,
            "debug_info": debug_info if debug_mode else {},
            "retrieved_context": retrieved_display,
            "formatted_response": formatted_response,
            "query_time_seconds": query_time_final,
            "generation_time_seconds": gen_time if gen_time > 0 else 0,
        }

    def _safe_unload_llm(self, timeout: float = 3.0) -> None:
        """
        Try to unload the LLM in a background daemon thread so that GUI/CLI
        shutdown is never blocked. The call returns as soon as    (1) unload
        finished, (2) the timeout elapsed, or (3) no unload method exists.
        """
        # --- Fast‑path: CPU provider hangs when calling unload.
        if self.model is None or not hasattr(self.model, "unload"):
            return
        try:
            provider = getattr(self.model, "_provider", "CPUExecutionProvider")
            if provider == "CPUExecutionProvider":
                logger.info("Skip explicit Teapot unload on CPU provider – "
                            "let GC reclaim objects during interpreter shutdown.")
                self.model = None
                import gc # Redundant import for linters
                gc.collect()
                return
        except Exception:
            # Best‑effort – continue with normal path if provider detection failed
            pass
        # --- End Fast-path ---

        finished = threading.Event()

        def _worker():
            try:
                assert self.model is not None # Explicit assertion for Pyright
                self.model.unload()
            except Exception as e:         # noqa: BLE001
                logger.error(f"Safe‑unload error: {e}", exc_info=self.debug)
            finally:
                finished.set()

        t = threading.Thread(target=_worker, name="TeapotUnload", daemon=True)
        t.start()
        if not finished.wait(timeout):
            logger.warning(
                f"LLM unload exceeded {timeout}s – continuing app shutdown."
            )
        # regardless of outcome, drop the reference so GC can run
        self.model = None
        import gc
        gc.collect()

    def close(self) -> None:
        """ Unload models and release resources. """
        logger.info("Closing LLMSearch and its components...")
        if self._shutdown_event and not self._shutdown_event.is_set():
            logger.debug("Signalling shutdown during LLMSearch close.")
            self._shutdown_event.set()

        if self.model:
            logger.debug("Attempting to unload LLM asynchronously…")
            self._safe_unload_llm()           # <- new non‑blocking call

        if self.embedder:
            logger.debug("Attempting to unload Embedder...")
            try:
                # No explicit unload needed for SentenceTransformer typically
                # Releasing reference is sufficient for GC
                # if hasattr(self.embedder, 'unload') and callable(self.embedder.unload):
                #     self.embedder.unload()
                #     logger.info("Embedder unloaded successfully.")
                # else:
                logger.debug("Embedder reference release is sufficient; no unload method called.")
            except Exception as e:
                logger.error(f"Error during Embedder cleanup (unexpected): {e}", exc_info=self.debug)
            finally:
                self.embedder = None
                logger.debug("Embedder reference cleared.")

        if self.bm25:
            logger.debug("Attempting to clear BM25 reference...")
            try:
                # Saving is handled by add/remove methods, no explicit save needed on close.
                # self.bm25.save_index() # Ensure index is saved on close
                # logger.info("BM25 state saved successfully.")
                logger.debug("BM25 reference cleared. State assumed saved by prior operations.")
            except Exception as e:
                logger.error(f"Error during BM25 cleanup (unexpected): {e}", exc_info=self.debug)
            finally:
                self.bm25 = None
                logger.debug("BM25 reference cleared.")

        # Chroma client doesn't typically have an explicit close/unload unless using server
        # Just releasing the reference is usually sufficient for the embedded version.
        if self.chroma_collection:
            logger.debug("Clearing ChromaDB collection reference.")
            self.chroma_collection = None
        if self.chroma_client:
            logger.debug("Attempting to reset ChromaDB client...")
            try:
                # client.reset() might be too aggressive, causing data loss if not intended.
                # Simple garbage collection should handle embedded client resources.
                logger.debug("ChromaDB client reference released for GC.")
            except Exception as e:
                logger.error(f"Error during ChromaDB client handling on close: {e}", exc_info=self.debug)
            finally:
                self.chroma_client = None
                logger.debug("Chroma client reference cleared.")

        logger.debug("Clearing Chunker reference.")
        self.chunker = None
        logger.info("LLMSearch closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()