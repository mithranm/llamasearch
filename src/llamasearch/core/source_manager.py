# src/llamasearch/core/source_manager.py
import hashlib
import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from chromadb.api.types import GetResult, Metadata, Where
from whoosh import index as whoosh_index

from llamasearch.data_manager import data_manager
from llamasearch.utils import NumpyEncoder
from .bm25 import WhooshBM25Retriever
from .chunker import chunk_markdown_text, DEFAULT_MIN_CHUNK_LENGTH
from .embedder import EnhancedEmbedder

# Type Aliases consistent with search_engine.py
ChromaWhereClause = Where
ChromaMetadataValue = Union[str, int, float, bool]
ChromaMetadataDict = Dict[str, ChromaMetadataValue]

ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm"}
DEFAULT_MAX_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE_FILTER = DEFAULT_MIN_CHUNK_LENGTH

logger = logging.getLogger(__name__)  # Use module-level logger


# Mixin Class for Source Management Logic
class _SourceManagementMixin:
    # These attributes are expected to be set by the main LLMSearch class
    embedder: Optional[EnhancedEmbedder]
    chroma_collection: Optional[chromadb.Collection]
    bm25: Optional[WhooshBM25Retriever]
    _shutdown_event: Optional[threading.Event]
    _reverse_lookup: Dict[str, str]
    max_chunk_size: int
    chunk_overlap: int
    min_chunk_size_filter: int
    verbose: bool
    debug: bool
    storage_dir: Path  # Needed for BM25 path within LLMSearch

    def _load_reverse_lookup(self):
        """Loads the hash -> URL mapping from the crawl data directory."""
        try:
            crawl_data_path_str = data_manager.get_data_paths().get("crawl_data")
            if not crawl_data_path_str:
                logger.warning(
                    "Crawl data path not configured, cannot load reverse lookup."
                )
                self._reverse_lookup = {}
                return
            crawl_data_path = Path(crawl_data_path_str)
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

    def _save_reverse_lookup(self):
        """Saves the current reverse lookup dictionary to the file."""
        try:
            crawl_data_path_str = data_manager.get_data_paths().get("crawl_data")
            if not crawl_data_path_str:
                logger.warning(
                    "Crawl data path not configured, cannot save reverse lookup."
                )
                return
            crawl_data_path = Path(crawl_data_path_str)
            lookup_file = crawl_data_path / "reverse_lookup.json"
            # Ensure parent directory exists
            lookup_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lookup_file, "w", encoding="utf-8") as f:
                json.dump(
                    self._reverse_lookup,
                    f,
                    cls=NumpyEncoder,
                    indent=4,
                )
            logger.info(
                f"Saved URL reverse lookup ({len(self._reverse_lookup)} entries)."
            )
        except Exception as e:
            logger.error(f"Error saving reverse lookup: {e}", exc_info=self.debug)

    def add_source(
        self, source_path_str: str, internal_call: bool = False
    ) -> Tuple[int, bool]:
        """
        Adds source(s) and handles metadata including original URL for crawled files.
        Disallows adding files from within crawl_data/raw unless internal_call is True.
        Returns (chunks_added, was_blocked).
        """
        assert self.embedder is not None, "Embedder not initialized"
        assert self.chroma_collection is not None, "Chroma collection not initialized"
        assert self.bm25 is not None, "BM25 retriever not initialized"

        if self._shutdown_event and self._shutdown_event.is_set():
            logger.warning("Add source cancelled due to shutdown.")
            return 0, False

        source_path = Path(source_path_str).resolve()
        total_chunks_added = 0
        crawl_raw_dir = None

        # Check if source is inside managed crawl directory
        is_in_crawl_dir = False
        try:
            crawl_data_path_str = data_manager.get_data_paths().get("crawl_data")
            if crawl_data_path_str:
                crawl_raw_dir = Path(crawl_data_path_str).resolve() / "raw"
                if (
                    source_path.exists()
                    and crawl_raw_dir.exists()
                    and crawl_raw_dir in source_path.parents
                ):
                    is_in_crawl_dir = True
        except Exception as path_check_err:
            logger.error(
                f"Error checking source path location: {path_check_err}",
                exc_info=self.debug,
            )

        # --- Block adding from crawl dir if not internal call ---
        if is_in_crawl_dir and not internal_call:
            logger.warning(
                f"Cannot manually add source from managed crawl directory: {source_path}. Use crawl feature or move the file."
            )
            return 0, True  # Indicate blocked

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

            logger.info(
                f"Processing file: {source_path}{' (internal call)' if internal_call else ''}"
            )

            # Determine identifier and potential URL
            identifier_for_removal = source_path_str
            url_from_lookup = None
            file_hash_stem = None
            # Check if this looks like a crawled file (MD + 16 hex chars stem)
            if is_in_crawl_dir and file_ext == ".md" and len(source_path.stem) == 16:
                try:
                    int(source_path.stem, 16)  # Check if stem is hex
                    file_hash_stem = source_path.stem
                    url_from_lookup = self._reverse_lookup.get(file_hash_stem)
                    if url_from_lookup:
                        identifier_for_removal = url_from_lookup
                        logger.debug(
                            f"Identified as crawled file. Using URL '{url_from_lookup}' as identifier."
                        )
                    else:
                        logger.warning(
                            f"Crawled file format but no URL found in lookup for hash: {file_hash_stem}"
                        )
                except ValueError:
                    pass  # Stem is not hex, treat as normal file

            if self._is_source_unchanged(source_path_str, url_from_lookup):
                logger.info(f"File '{source_path.name}' is unchanged. Skipping.")
                return 0, False

            removed_ok, _ = self.remove_source(identifier_for_removal)
            if removed_ok:
                logger.debug(
                    f"Removed existing chunks for changed/re-added source: {identifier_for_removal}"
                )

            # Read file content
            file_content: Optional[str] = None
            try:
                file_content = source_path.read_text(encoding="utf-8", errors="ignore")
                if (
                    file_content and "�" in file_content[:1000]
                ):  # Check for replacement char
                    logger.debug(
                        f"Potential encoding issue detected in {source_path.name}. Trying alternatives..."
                    )
                    # (Encoding fallback logic remains the same)
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
                            f"Could not resolve encoding issue for {source_path.name}. Proceeding with UTF-8 (ignore)."
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

                # Prepare data
                chunk_texts: List[str] = []
                chunk_metadatas: List[ChromaMetadataDict] = []
                chunk_ids: List[str] = []
                try:
                    mtime_val: Optional[float] = source_path.stat().st_mtime
                except OSError as e:
                    logger.warning(f"Could not stat file {source_path}: {e}")
                    mtime_val = time.time()  # Use current time as fallback

                id_prefix_source = (
                    url_from_lookup if url_from_lookup else source_path_str
                )
                id_prefix = hashlib.sha1(
                    id_prefix_source.encode("utf-8", errors="ignore")
                ).hexdigest()[:8]

                base_meta: ChromaMetadataDict = {
                    "source_path": source_path_str,
                    "filename": source_path.name,
                    "mtime": float(mtime_val) if mtime_val is not None else 0.0,
                }
                # --- Add original_url if available ---
                if url_from_lookup:
                    base_meta["original_url"] = url_from_lookup
                    logger.debug(
                        f"Adding 'original_url': {url_from_lookup} to metadata."
                    )
                # ------------------------------------

                valid_chunk_counter = 0
                for c_idx, c in enumerate(chunks_with_metadata):
                    chunk_text = c.get("chunk", "")
                    if not chunk_text:
                        continue

                    chunk_content_hash = hashlib.sha1(
                        chunk_text.encode("utf-8", errors="ignore")
                    ).hexdigest()[:8]
                    chunk_id = f"{id_prefix}_{c_idx}_{chunk_content_hash}"
                    chunk_ids.append(chunk_id)
                    valid_chunk_counter += 1

                    meta: ChromaMetadataDict = base_meta.copy()
                    chunker_meta = c.get("metadata", {})
                    if isinstance(chunker_meta, dict):
                        original_chunk_index = chunker_meta.get("chunk_index_in_doc")
                        length = chunker_meta.get("length")
                        eff_length = chunker_meta.get("effective_length")
                        proc_mode = chunker_meta.get("processing_mode")
                        if original_chunk_index is not None:
                            meta["original_chunk_index"] = int(original_chunk_index)
                        if length is not None:
                            meta["chunk_char_length"] = int(length)
                        if eff_length is not None:
                            meta["effective_length"] = int(eff_length)
                        if proc_mode is not None:
                            meta["processing_mode"] = str(proc_mode)

                    meta["filtered_chunk_index"] = valid_chunk_counter - 1
                    meta["chunk_id"] = chunk_id

                    cleaned_meta: ChromaMetadataDict = {
                        k: v
                        for k, v in meta.items()
                        if isinstance(v, (str, int, float, bool))
                    }
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
                    valid_metadatas = [
                        m for m in chunk_metadatas if isinstance(m, dict)
                    ]
                    if len(valid_metadatas) != len(chunk_ids):
                        logger.error(f"Metadata count mismatch for {source_path.name}.")
                        return 0, False
                    self.chroma_collection.upsert(
                        ids=chunk_ids,
                        embeddings=embeddings.tolist(),
                        metadatas=valid_metadatas,  # type: ignore
                        documents=[str(doc) for doc in chunk_texts],
                    )
                except Exception as chroma_e:
                    logger.error(
                        f"ChromaDB upsert failed: {chroma_e}", exc_info=self.debug
                    )
                    return 0, False

                # Add to Whoosh
                logger.info(f"Adding {len(chunk_texts)} chunks to Whoosh BM25 index...")
                bm25_success_count = 0
                bm25_failed_ids = []
                for text, chunk_id in zip(chunk_texts, chunk_ids):
                    if self.bm25.add_document(text, chunk_id):
                        bm25_success_count += 1
                    else:
                        bm25_failed_ids.append(chunk_id)
                logger.debug(
                    f"Added {bm25_success_count}/{len(chunk_texts)} chunks to Whoosh."
                )
                if bm25_success_count != len(chunk_texts):
                    logger.warning(
                        f"Mismatch in Whoosh add count for {source_path.name}. Failed IDs: {bm25_failed_ids}"
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
            logger.info(f"Processing directory recursively: {source_path}")
            files_processed, files_skipped, files_failed = 0, 0, 0
            try:
                all_items = list(source_path.rglob("*"))
            except Exception as e:
                logger.error(
                    f"Error listing files in {source_path}: {e}", exc_info=self.debug
                )
                return 0, False
            logger.info(f"Found {len(all_items)} items. Filtering...")
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
                            added, blocked = self.add_source(
                                str(item), internal_call=False
                            )
                        except Exception as e:
                            logger.error(
                                f"Error processing file {item}: {e}",
                                exc_info=self.debug,
                            )
                            files_failed += 1
                            continue
                        if blocked:
                            files_skipped += 1
                        elif added > 0:
                            total_chunks_added += added
                            files_processed += 1
                        elif self._is_source_unchanged(str(item)):
                            files_skipped += 1
                        else:
                            files_processed += 1
                    else:
                        files_skipped += 1
                elif item.is_dir():
                    if any(
                        part.startswith(".")
                        for part in item.relative_to(source_path).parts
                    ) or item.name.startswith("."):
                        logger.debug(f"Skipping hidden directory: {item}")

            logger.info(
                f"Dir scan '{source_path.name}': Added {total_chunks_added} chunks from {files_processed} files. Skipped {files_skipped}. Failed {files_failed}."
            )
            if files_failed > 0:
                logger.warning(
                    f"Failed to process/add chunks for {files_failed} files during dir scan."
                )

        else:
            logger.warning(f"Source path is not a file or directory: {source_path}")

        return total_chunks_added, False

    def _is_source_unchanged(
        self, source_path_str: str, known_url: Optional[str] = None
    ) -> bool:
        """Checks if a source file is already indexed and unchanged based on mtime."""
        assert self.chroma_collection is not None
        source_path = Path(source_path_str)
        if not source_path.is_file():
            return False

        identifier_key: str = "source_path"
        identifier_value: ChromaMetadataValue = source_path_str
        if known_url:
            identifier_key = "original_url"
            identifier_value = known_url
        else:
            url_from_lookup = None
            if source_path.suffix.lower() == ".md" and len(source_path.stem) == 16:
                try:
                    int(source_path.stem, 16)
                    file_hash_stem = source_path.stem
                    url_from_lookup = self._reverse_lookup.get(file_hash_stem)
                    if url_from_lookup:
                        identifier_key = "original_url"
                        identifier_value = url_from_lookup
                except ValueError:
                    pass

        where_filter: ChromaWhereClause = {identifier_key: identifier_value}

        try:
            current_mtime = source_path.stat().st_mtime
            results: Optional[GetResult] = self.chroma_collection.get(
                where=where_filter,  # type: ignore
                limit=1,
                include=["metadatas"],
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
                    is_unchanged = (
                        isinstance(stored_mtime, (int, float))
                        and abs(float(stored_mtime) - current_mtime) < 0.1
                    )
                    if is_unchanged:
                        logger.debug(
                            f"Source '{identifier_value}' is unchanged (mtime)."
                        )
                        return True
                    else:
                        logger.debug(f"Source '{identifier_value}' mtime differs.")
            logger.debug(f"Source '{identifier_value}' changed or not found.")
            return False
        except FileNotFoundError:
            logger.debug(f"Source file not found during unchanged check: {source_path}")
            return False
        except Exception as e:
            logger.warning(f"Error checking source status '{source_path_str}': {e}.")
            return False

    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """
        Retrieves aggregated information about indexed sources, prioritizing URL identification.
        Uses _reverse_lookup based on source_path to reliably identify crawled sources.
        """
        assert self.chroma_collection is not None, "Chroma collection not initialized"
        logger.debug("Fetching metadata for aggregation...")

        source_info: Dict[str, Dict[str, Any]] = {}
        missing_identifier_count = 0

        # Need crawl_raw_dir path for checking file locations
        crawl_raw_dir_str = data_manager.get_data_paths().get("crawl_data")
        crawl_raw_dir = None
        if crawl_raw_dir_str:
            crawl_raw_dir = Path(crawl_raw_dir_str).resolve() / "raw"

        try:
            # Fetch all metadata in batches to avoid memory issues with huge indexes
            total_docs = self.chroma_collection.count()
            logger.debug(f"Fetching metadata for {total_docs} total chunks...")
            batch_size = 5000 # Adjust batch size as needed
            all_metadatas: List[Metadata] = []

            for offset in range(0, total_docs, batch_size):
                if self._shutdown_event and self._shutdown_event.is_set():
                    logger.warning("Metadata fetch cancelled due to shutdown.")
                    return []

                logger.debug(f"Fetching metadata batch: offset={offset}, limit={batch_size}")
                results: GetResult = self.chroma_collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"]
                )
                if results and results["metadatas"]:
                    all_metadatas.extend(results["metadatas"])
                else:
                    # Should not happen if offset < total_docs, but handle defensively
                    logger.warning(f"No metadata returned for offset {offset}.")
                    break # Stop if no more data

            logger.debug(f"Total unique metadata items fetched: {len(all_metadatas)}")

            # --- Aggregate Source Info --- Now iterate through the fetched metadata
            for metadata_item in all_metadatas:
                if not metadata_item: # Skip if metadata is None or empty
                    continue

                current_source_path = metadata_item.get("source_path")
                current_filename = metadata_item.get("filename", "N/A")
                current_mtime = metadata_item.get("mtime")

                primary_key = None
                is_url_source = False
                looked_up_url = None
                source_path_obj = None

                # --- Determine Primary Identifier using Reverse Lookup --- START
                if isinstance(current_source_path, str) and current_source_path != "N/A":
                    source_path_obj = Path(current_source_path)
                    # Check if it looks like a file within our managed crawl directory
                    if (
                        crawl_raw_dir
                        and crawl_raw_dir.exists()
                        and crawl_raw_dir in source_path_obj.parents
                        and source_path_obj.suffix.lower() == ".md"
                        and len(source_path_obj.stem) == 16
                    ):
                        try:
                            # Verify stem is hex and look up
                            int(source_path_obj.stem, 16)
                            file_hash_stem = source_path_obj.stem
                            looked_up_url = self._reverse_lookup.get(file_hash_stem)
                            if looked_up_url:
                                primary_key = looked_up_url.strip()
                                is_url_source = True
                                logger.log(logging.NOTSET, # Use lower level to avoid spam
                                        f"Identified URL source via lookup: {primary_key} from {current_source_path}")
                            else:
                                logger.warning(
                                     f"Crawled file format detected ({current_source_path}) but no URL in lookup for hash {file_hash_stem}. Falling back to path."
                                )
                        except ValueError:
                            # Stem wasn't hex, treat as normal file path below
                            pass

                    # If not identified as URL source via lookup, use the path
                    if not primary_key:
                        primary_key = current_source_path.strip()
                        is_url_source = False
                # --- Determine Primary Identifier using Reverse Lookup --- END

                if not primary_key:
                    # Fallback: Use original_url from metadata if present and path wasn't usable
                    current_original_url = metadata_item.get("original_url")
                    if isinstance(current_original_url, str) and current_original_url:
                         primary_key = current_original_url.strip()
                         is_url_source = True
                         logger.warning(f"Used original_url metadata fallback for {primary_key} as path was missing/invalid.")
                    else:
                        logger.warning(f"Missing usable identifier (path or URL) in metadata: {metadata_item}")
                        missing_identifier_count += 1
                        continue # Skip this chunk

                # --- Aggregate Info --- (Simplified)
                if primary_key not in source_info:
                    # Initialize entry using the determined identifier and type
                    source_info[primary_key] = {
                        "identifier": primary_key,
                        "original_url": primary_key if is_url_source else None,
                        "source_path": current_source_path if not is_url_source else "N/A", # Store path only if it's the primary ID
                        "filename": current_filename if isinstance(current_filename, str) else "N/A",
                        "chunk_count": 0,
                        "mtime": None,
                        "is_url_source": is_url_source,
                    }
                    # Set initial mtime
                    if isinstance(current_mtime, (int, float)):
                        source_info[primary_key]["mtime"] = float(current_mtime)
                    # If it's a URL source, store the corresponding path if we have it
                    if is_url_source and isinstance(current_source_path, str):
                         source_info[primary_key]["source_path"] = current_source_path

                # --- Update Existing Entry --- START
                # Always increment chunk count
                source_info[primary_key]["chunk_count"] += 1

                # Ensure most recent mtime is stored
                if isinstance(current_mtime, (int, float)):
                    existing_mtime = source_info[primary_key].get("mtime")
                    if existing_mtime is None or current_mtime > existing_mtime:
                        source_info[primary_key]["mtime"] = float(current_mtime)

                # If this chunk provides a missing piece of info (URL or path)
                if is_url_source:
                    # If we have a path for this chunk but it wasn't stored yet
                    if (isinstance(current_source_path, str)
                        and source_info[primary_key].get("source_path") == "N/A"): # If URL source, path might be missing
                         source_info[primary_key]["source_path"] = current_source_path
                         # Update filename if it was N/A and path gives us one
                         if source_info[primary_key].get("filename") == "N/A" and source_path_obj:
                             source_info[primary_key]["filename"] = source_path_obj.name
                else: # If it's a path source
                    # Check if this chunk *happened* to have original_url metadata (e.g., from old index)
                    # We prioritize the path lookup, but store the URL if found for completeness
                    current_original_url = metadata_item.get("original_url")
                    if (isinstance(current_original_url, str)
                        and source_info[primary_key].get("original_url") is None):
                        source_info[primary_key]["original_url"] = current_original_url

                # Ensure filename is updated if initially N/A
                if source_info[primary_key].get("filename") == "N/A":
                     path_val = source_info[primary_key].get("source_path")
                     if path_val and path_val != "N/A":
                          try:
                              source_info[primary_key]["filename"] = Path(path_val).name
                          except Exception:
                              pass
                # --- Update Existing Entry --- END

            logger.debug(
                f"Aggregation complete. Found {len(source_info)} unique sources."
            )
            if missing_identifier_count > 0:
                logger.warning(
                    f"Could not determine a primary identifier for {missing_identifier_count} metadata entries."
                )

            return list(source_info.values())

        except Exception as e:
            logger.error(f"Error retrieving indexed sources: {e}", exc_info=True)
            return []

    def remove_source(self, source_identifier: str) -> Tuple[bool, bool]:
        """
        Removes all chunks for a source, identified by URL or path.
        Also removes the associated crawled file if applicable and removes URL from lookup cache.
        Returns (removal_occurred, was_blocked).
        """
        assert self.chroma_collection is not None
        assert self.bm25 is not None
        logger.info(f"Attempting to remove source identifier: '{source_identifier}'")
        removal_occurred = False
        is_url_identifier = source_identifier.startswith(
            "http://"
        ) or source_identifier.startswith("https://")
        where_filter: ChromaWhereClause
        if is_url_identifier:
            where_filter = {"original_url": source_identifier}
            logger.debug("Removing based on 'original_url'.")
        else:
            where_filter = {"source_path": source_identifier}
            logger.debug("Removing based on 'source_path'.")

        target_file_path_from_meta: Optional[Path] = None
        original_url_from_meta: Optional[str] = None
        file_hash_key_from_meta: Optional[str] = None

        try:
            # 1. Find all chunk IDs and potentially the file path/URL from the first chunk
            ids_to_remove: List[str] = []
            limit = 5000
            offset = 0
            first_meta_found = False
            while True:
                if self._shutdown_event and self._shutdown_event.is_set():
                    raise InterruptedError("Shutdown finding IDs.")
                try:
                    results: Optional[GetResult] = self.chroma_collection.get(
                        where=where_filter,  # type: ignore
                        include=["metadatas"] if not first_meta_found else [],
                        limit=limit,
                        offset=offset,
                    )
                except Exception as get_err:
                    logger.error(
                        f"Error querying ChromaDB for removal: {get_err}",
                        exc_info=self.debug,
                    )
                    break
                if results and results.get("ids"):
                    batch_ids = results["ids"]
                    ids_to_remove.extend(batch_ids)
                    if not first_meta_found:
                        batch_metas = results.get("metadatas")
                        if batch_metas and isinstance(batch_metas[0], dict):
                            meta = batch_metas[0]
                            path_str = meta.get("source_path")
                            url_str = meta.get("original_url")
                            if isinstance(path_str, str):
                                target_file_path_from_meta = Path(path_str)
                                if (
                                    url_str
                                    and path_str.endswith(".md")
                                    and len(Path(path_str).stem) == 16
                                ):
                                    try:
                                        int(Path(path_str).stem, 16)
                                        file_hash_key_from_meta = Path(path_str).stem
                                    except ValueError:
                                        pass
                            if isinstance(url_str, str):
                                original_url_from_meta = url_str
                            first_meta_found = True
                    if len(batch_ids) < limit:
                        break
                    offset += limit
                else:
                    break

            if not ids_to_remove:
                logger.info(f"No chunks found for identifier '{source_identifier}'.")
                return False, False
            ids_to_remove = list(set(ids_to_remove))
            logger.info(
                f"Found {len(ids_to_remove)} chunk IDs to remove for '{source_identifier}'."
            )

            # 2. Identify managed crawl file
            file_to_delete: Optional[Path] = None
            crawl_raw_dir = None
            try:
                crawl_data_path_str = data_manager.get_data_paths().get("crawl_data")
                if crawl_data_path_str and target_file_path_from_meta:
                    crawl_raw_dir = Path(crawl_data_path_str).resolve() / "raw"
                    resolved_target_file = target_file_path_from_meta.resolve()
                    if (
                        crawl_raw_dir.exists()
                        and crawl_raw_dir in resolved_target_file.parents
                    ):
                        file_to_delete = resolved_target_file
                        logger.debug(
                            f"Identified managed crawl file for deletion: {file_to_delete}"
                        )
            except Exception as path_err:
                logger.error(
                    f"Error resolving file path for deletion: {path_err}",
                    exc_info=self.debug,
                )

            if self._shutdown_event and self._shutdown_event.is_set():
                raise InterruptedError("Shutdown before deletes.")

            # 3. Remove from ChromaDB
            logger.info(f"Removing {len(ids_to_remove)} chunks from ChromaDB...")
            chroma_delete_errors = 0
            try:
                if ids_to_remove:
                    self.chroma_collection.delete(ids=ids_to_remove)
                    removal_occurred = True
            except Exception as chroma_del_err:
                logger.error(
                    f"ChromaDB delete failed: {chroma_del_err}", exc_info=self.debug
                )
                chroma_delete_errors += 1
                return False, False

            # 4. Remove from Whoosh BM25
            logger.info(f"Removing {len(ids_to_remove)} chunks from Whoosh...")
            bm25_removed_count, bm25_remove_errors = 0, 0
            writer = None
            try:
                if self.bm25.ix is None:
                    raise RuntimeError("Whoosh index is None.")
                writer = self.bm25.ix.writer(timeout=60.0)
                with writer:
                    for chunk_id in ids_to_remove:
                        if self._shutdown_event and self._shutdown_event.is_set():
                            raise InterruptedError("Shutdown during Whoosh delete.")
                        try:
                            num_deleted = writer.delete_by_term("chunk_id", chunk_id)
                            if num_deleted > 0:
                                bm25_removed_count += 1
                        except Exception as bm25_del_err:
                            logger.warning(
                                f"Whoosh remove error for chunk_id '{chunk_id}': {bm25_del_err}"
                            )
                            bm25_remove_errors += 1
                if bm25_remove_errors > 0:
                    logger.warning(
                        f"{bm25_remove_errors} Whoosh remove errors occurred."
                    )
            except whoosh_index.LockError as lock_err:
                logger.error(f"Failed Whoosh lock for removal: {lock_err}")
                logger.critical(
                    f"INCONSISTENCY: Chroma OK but Whoosh FAILED for '{source_identifier}'."
                )
            except Exception as writer_err:
                logger.error(
                    f"Error during Whoosh removal: {writer_err}", exc_info=self.debug
                )
                logger.critical(
                    f"INCONSISTENCY: Chroma OK but Whoosh FAILED for '{source_identifier}'."
                )

            # 5. Delete the managed crawl file
            file_deleted_successfully = False
            if file_to_delete:
                if file_to_delete.exists():
                    if not (self._shutdown_event and self._shutdown_event.is_set()):
                        logger.info(f"Deleting managed crawl file: {file_to_delete}")
                        try:
                            os.remove(file_to_delete)
                            file_deleted_successfully = True
                        except FileNotFoundError:
                            file_deleted_successfully = True
                        except Exception as del_err:
                            logger.error(
                                f"Error deleting {file_to_delete}: {del_err}",
                                exc_info=self.debug,
                            )
                    else:
                        logger.warning(
                            f"Skipping file deletion due to shutdown: {file_to_delete}"
                        )
                else:
                    file_deleted_successfully = True

            # 6. Remove from reverse lookup cache and save (if applicable)
            if file_hash_key_from_meta and file_hash_key_from_meta in self._reverse_lookup:
                try:
                    del self._reverse_lookup[file_hash_key_from_meta]
                    logger.info(
                        f"Removed '{file_hash_key_from_meta}' (URL: {original_url_from_meta or 'unknown'}) from reverse lookup cache."
                    )
                    self._save_reverse_lookup() # Persist the change
                    removal_occurred = True # Mark removal occurred if lookup entry was removed
                except KeyError:
                    logger.warning(
                        f"Key '{file_hash_key_from_meta}' already missing from reverse lookup."
                    )
                except Exception as lookup_err:
                     logger.error(
                        f"Error removing key '{file_hash_key_from_meta}' from reverse lookup: {lookup_err}",
                        exc_info=self.debug,
                    )

            if self._shutdown_event and self._shutdown_event.is_set():
                return False, False

            log_msg = f"Removal '{source_identifier}': Chunks Found={len(ids_to_remove)}. Chroma OK={len(ids_to_remove) - chroma_delete_errors}. Whoosh OK={bm25_removed_count}. File Deleted: {file_deleted_successfully}."
            logger.info(log_msg)
            return removal_occurred, False
        except InterruptedError:
            logger.warning(f"Removal interrupted for {source_identifier}")
            return False, False
        except Exception as e:
            logger.error(
                f"Unexpected error removing '{source_identifier}': {e}", exc_info=True
            )
            return False, False
