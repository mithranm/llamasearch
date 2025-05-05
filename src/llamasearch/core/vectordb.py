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
