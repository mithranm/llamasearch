import os
import numpy as np
import logging
import gc
import tempfile
import shutil
from typing import Dict, Any, Optional, Union, Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from .embedding import Embedder
from .embedding_pca import PCAReducer
from ..utils import find_project_root

# Import our new chunker
from .chunker import MarkdownChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorDB:
    """
    Enhanced vector database designed for working with code-text pairs.
    Embeds only the text portion but stores and retrieves both code and text.
    Optionally uses PCA for dimensionality reduction.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedder: Optional[Embedder] = None,
        chunk_size: int = 500,
        text_embedding_size: int = 512,  # Token limit for text to be embedded
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,  # Minimum size for chunks
        batch_size: int = 8,
        similarity_threshold: float = 0.25,  # Lowered for better recall
        max_chunks: int = 5000,
        persist: bool = True,
        storage_dir: Optional[str] = None,
        use_pca: bool = True,
        pca_components: int = 128,
        pca_training_size: int = 1000,  # Number of embeddings to use for PCA training
    ):
        """Initialize enhanced vector database with optional PCA support."""
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.text_embedding_size = text_embedding_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.batch_size = batch_size
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
        self.persist = persist
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca_training_size = pca_training_size

        # Set up storage paths
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            project_root = find_project_root()
            self.storage_dir = os.path.join(
                project_root, f"vector_db_{'pca' if use_pca else 'no_pca'}"
            )

        # Clean up storage directory if not persisting
        if not self.persist and os.path.exists(self.storage_dir):
            logger.info(f"Clearing vector database directory at {self.storage_dir}")
            try:
                # Clear the directory contents
                for filename in os.listdir(self.storage_dir):
                    filepath = os.path.join(self.storage_dir, filename)
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
                logger.info("Vector database directory cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing vector database directory: {str(e)}")

        # Ensure directory exists
        os.makedirs(self.storage_dir, exist_ok=True)

        # Key file paths
        self.metadata_path = os.path.join(
            self.storage_dir, f"{collection_name}_meta.json"
        )
        self.embeddings_path = os.path.join(
            self.storage_dir, f"{collection_name}_embeddings.npy"
        )

        # Initialize embedder with appropriate batch size for memory efficiency
        self.embedder = embedder or Embedder(
            batch_size=4, max_length=text_embedding_size
        )

        # Initialize PCA reducer if enabled
        self.pca_reducer = None
        if self.use_pca:
            self.pca_reducer = PCAReducer(
                n_components=self.pca_components,
                storage_dir=self.storage_dir,
                model_name=f"{collection_name}_pca",
            )
            # Try to load existing PCA model
            if self.pca_reducer.load_model():
                logger.info(
                    f"Loaded existing PCA model with {self.pca_components} components"
                )

        # Initialize chunker with our token limit for text embedding
        self.chunker = MarkdownChunker(
            chunk_size=chunk_size,
            text_embedding_size=text_embedding_size,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks,
            batch_size=batch_size,
            ignore_link_urls=True,
            code_context_window=3,  # Look 3 paragraphs before/after code
            include_section_headers=True,
            always_create_chunks=True,  # Ensure chunks are always created
        )

        # Load metadata with memory care
        self.documents = []  # Full content to return (text + code)
        self.document_metadata = []
        self._load_metadata()

        # Track whether we have enough embeddings for PCA
        self.has_pca_training_data = (
            self._get_embedding_count() >= self.pca_training_size
        )

        logger.info(f"Initialized VectorDB with {len(self.documents)} documents")
        logger.info(f"Storage directory: {self.storage_dir}")
        logger.info(f"Embeddings stored at: {self.embeddings_path}")
        logger.info(f"Metadata stored at: {self.metadata_path}")
        logger.info(f"Persistence mode: {'ON' if self.persist else 'OFF'}")
        logger.info(f"PCA mode: {'ON' if self.use_pca else 'OFF'}")
        if self.use_pca:
            logger.info(f"PCA components: {self.pca_components}")
            logger.info(
                f"PCA training data: {'READY' if self.has_pca_training_data else 'NOT READY'}"
            )
        logger.info(f"Text embedding size: {self.text_embedding_size} tokens")

    def _load_metadata(self):
        """Load document metadata without loading all embeddings."""
        try:
            if os.path.exists(self.metadata_path):
                logger.info(f"Loading metadata from {self.metadata_path}")
                with open(self.metadata_path, "r") as f:
                    metadata = json.load(f)

                self.documents = metadata.get("documents", [])
                self.document_metadata = metadata.get("metadata", [])

                logger.info(f"Loaded {len(self.documents)} document entries")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self.documents = []
            self.document_metadata = []

    def _save_metadata(self):
        """Save document metadata in a memory-efficient way."""
        try:
            logger.info(f"Saving metadata ({len(self.documents)} documents)")

            # Write to a temporary file first
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as tmp_file:
                temp_path = tmp_file.name
                # Create metadata structure
                metadata = {
                    "documents": self.documents,
                    "metadata": self.document_metadata,
                }

                # Write metadata
                json.dump(metadata, tmp_file)

            # Replace the original file
            os.replace(temp_path, self.metadata_path)

            logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            # Clean up temp file if it exists
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _is_document_processed(self, file_path: str) -> bool:
        """Check if a document has been processed before and hasn't been modified since then."""
        if not self.document_metadata:
            return False

        file_mtime = os.path.getmtime(file_path)
        for meta in self.document_metadata:
            if meta.get("source") == file_path:
                # Check if the file has been modified since last processing
                if (
                    abs(meta.get("mtime", 0) - file_mtime) < 0.001
                ):  # Small epsilon for float comparison
                    return True
                else:
                    # File has been modified, remove old entries
                    self.documents = [
                        doc
                        for i, doc in enumerate(self.documents)
                        if self.document_metadata[i].get("source") != file_path
                    ]
                    self.document_metadata = [
                        meta
                        for meta in self.document_metadata
                        if meta.get("source") != file_path
                    ]
                    return False
        return False

    def _train_or_update_pca(self, temp_embeddings_path=None):
        """
        Train or update PCA model with new embeddings.

        Args:
            temp_embeddings_path: Optional path to temporary new embeddings
        """
        if not self.use_pca or not self.pca_reducer:
            return

        # Check if we have existing embeddings
        existing_embeddings_exist = (
            os.path.exists(self.embeddings_path)
            and os.path.getsize(self.embeddings_path) > 0
        )

        # If we have new embeddings but not enough training data yet
        if (
            not self.has_pca_training_data
            and temp_embeddings_path
            and os.path.exists(temp_embeddings_path)
        ):
            # If we have existing embeddings, we need to combine them with the new ones
            if existing_embeddings_exist:
                # Load existing embeddings
                existing = np.load(self.embeddings_path)
                # Load new embeddings
                new = np.load(temp_embeddings_path)
                # Combine both
                combined = np.vstack([existing, new])

                # If we now have enough data for PCA training
                if combined.shape[0] >= self.pca_training_size:
                    logger.info(
                        f"Training PCA model with {combined.shape[0]} embeddings"
                    )
                    self.pca_reducer.fit(combined)
                    self.has_pca_training_data = True

                    # Transform all embeddings
                    reduced = self.pca_reducer.transform(combined)

                    # Save reduced embeddings
                    with open(self.embeddings_path, "wb") as f:
                        np.save(f, reduced, allow_pickle=False)

                    # Cleanup
                    del existing, new, combined, reduced
                    gc.collect()

                    logger.info("PCA model trained and all embeddings reduced")
                    return True

                # Not enough data yet, just save the combined embeddings
                with open(self.embeddings_path, "wb") as f:
                    np.save(f, combined, allow_pickle=False)

                # Cleanup
                del existing, new, combined
                gc.collect()

                logger.info(
                    f"Combined embeddings saved, but not enough for PCA training yet ({combined.shape[0]}/{self.pca_training_size})"
                )
                return False

            # No existing embeddings, just use the new ones
            elif temp_embeddings_path and os.path.exists(temp_embeddings_path):
                new = np.load(temp_embeddings_path)

                # Check if we have enough data now
                if new.shape[0] >= self.pca_training_size:
                    logger.info(f"Training PCA model with {new.shape[0]} embeddings")
                    self.pca_reducer.fit(new)
                    self.has_pca_training_data = True

                    # Transform the embeddings
                    reduced = self.pca_reducer.transform(new)

                    # Save reduced embeddings
                    with open(self.embeddings_path, "wb") as f:
                        np.save(f, reduced, allow_pickle=False)

                    # Cleanup
                    del new, reduced
                    gc.collect()

                    logger.info("PCA model trained and embeddings reduced")
                    return True

                # Not enough data yet, just save the embeddings
                shutil.copy(temp_embeddings_path, self.embeddings_path)

                # Cleanup
                del new
                gc.collect()

                logger.info(
                    f"New embeddings saved, but not enough for PCA training yet ({new.shape[0]}/{self.pca_training_size})"
                )
                return False

        # If we already have a trained PCA model and are adding new embeddings
        elif (
            self.has_pca_training_data
            and temp_embeddings_path
            and os.path.exists(temp_embeddings_path)
        ):
            # Load new embeddings
            new = np.load(temp_embeddings_path)

            # Transform the new embeddings using existing PCA model
            reduced_new = self.pca_reducer.transform(new)

            # If we have existing embeddings, combine them
            if existing_embeddings_exist:
                # Load existing reduced embeddings
                existing = np.load(self.embeddings_path)

                # Combine with newly reduced embeddings
                combined = np.vstack([existing, reduced_new])

                # Save combined reduced embeddings
                with open(self.embeddings_path, "wb") as f:
                    np.save(f, combined, allow_pickle=False)

                # Cleanup
                del existing, new, reduced_new, combined
                gc.collect()
            else:
                # No existing embeddings, just save the reduced ones
                with open(self.embeddings_path, "wb") as f:
                    np.save(f, reduced_new, allow_pickle=False)

                # Cleanup
                del new, reduced_new
                gc.collect()

            logger.info("New embeddings transformed with existing PCA model and saved")
            return True

        return False

    def _merge_embeddings_without_pca(self, temp_path: str):
        """
        Merge new embeddings with existing ones without applying PCA.
        """
        logger.info("Merging new embeddings with existing database")

        if (
            os.path.exists(self.embeddings_path)
            and os.path.getsize(self.embeddings_path) > 0
        ):
            try:
                # Load new embeddings
                new_embeddings = np.load(temp_path)
                new_embeddings = np.ascontiguousarray(new_embeddings, dtype=np.float32)

                # Load existing embeddings
                existing_embeddings = np.load(self.embeddings_path)
                existing_embeddings = np.ascontiguousarray(
                    existing_embeddings, dtype=np.float32
                )

                # Combine the embeddings
                merged = np.vstack([existing_embeddings, new_embeddings])
                merged = np.ascontiguousarray(merged, dtype=np.float32)

                # Save merged embeddings - use direct numpy save to avoid pickle
                with open(self.embeddings_path, "wb") as f:
                    np.save(f, merged, allow_pickle=False)

                # Clean up
                del existing_embeddings, new_embeddings, merged

            except Exception as merge_e:
                logger.error(f"Error merging embeddings: {merge_e}")
                raise
        else:
            # No existing embeddings file, just use the temp file
            logger.info(
                "No existing embeddings found. Using new embeddings as initial set."
            )
            shutil.copy(temp_path, self.embeddings_path)

        # Save metadata
        self._save_metadata()

        logger.info("Successfully merged embeddings into database")

    def add_document(
        self, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Process document with separate text embeddings and content storage."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Skip processing if the document has been processed before and persist is True
        if self.persist and self._is_document_processed(file_path):
            logger.info(f"Document already processed: {file_path}")
            # Count chunks for this document
            chunk_count = sum(
                1 for meta in self.document_metadata if meta.get("source") == file_path
            )
            return chunk_count

        # Get document's last modified time
        doc_mtime = os.path.getmtime(file_path)

        base_metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "mtime": doc_mtime,
            **(metadata or {}),
        }

        added_chunks = 0
        temp_path = None

        try:
            # Create temporary file for new embeddings
            with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_file:
                temp_path = tmp_file.name
                logger.info(f"Created temporary embeddings file: {temp_path}")

            # Process document in batches using our enhanced chunker
            for batch_idx, batch in enumerate(
                self.chunker.process_file_in_batches(file_path, self.batch_size)
            ):
                if not batch:
                    continue

                # Extract texts for embedding and full content
                embedding_texts = [item["embedding_text"] for item in batch]
                full_chunks = [item["chunk"] for item in batch]

                # Extract and enhance metadata
                batch_metadata = []
                for i, item in enumerate(batch):
                    chunk_metadata = {
                        **base_metadata,
                        **(item.get("metadata", {})),
                        "batch_idx": batch_idx,
                        "chunk_idx": i,
                    }
                    batch_metadata.append(chunk_metadata)

                # Generate embeddings for the embedding_text only
                logger.info(
                    f"Generating embeddings for batch {batch_idx+1} ({len(embedding_texts)} chunks)"
                )
                embeddings = self.embedder.embed_batch(embedding_texts)

                # Ensure we have contiguous memory layout
                embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

                # Save to temp file in an incremental way
                try:
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        # Load existing embeddings
                        existing = np.load(temp_path)
                        existing = np.ascontiguousarray(existing, dtype=np.float32)

                        # Combine with new embeddings
                        combined = np.vstack([existing, embeddings])
                        combined = np.ascontiguousarray(combined, dtype=np.float32)

                        # Save combined embeddings - direct numpy save without pickle
                        with open(temp_path, "wb") as f:
                            np.save(f, combined, allow_pickle=False)

                        # Clean up
                        del existing, combined
                    else:
                        # First batch - save directly without pickle
                        with open(temp_path, "wb") as f:
                            np.save(f, embeddings, allow_pickle=False)
                except Exception as inner_e:
                    logger.error(f"Error saving embeddings to temp file: {inner_e}")
                    raise

                # Update document lists - store full chunks (text + code)
                self.documents.extend(full_chunks)
                self.document_metadata.extend(batch_metadata)

                # Track added chunks
                added_chunks += len(batch)

                # Force cleanup
                del embedding_texts, full_chunks, batch_metadata, embeddings
                gc.collect()

                logger.info(
                    f"Processed batch {batch_idx+1}: {len(batch)} chunks (total: {added_chunks})"
                )

            # Now apply PCA and merge with existing embeddings
            if added_chunks > 0:
                if self.use_pca:
                    # Apply PCA to the new embeddings if needed
                    self._train_or_update_pca(temp_path)
                else:
                    # Just merge without PCA
                    self._merge_embeddings_without_pca(temp_path)

            # Clean up temp file if it exists
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info("Cleaned up temporary file")

            return added_chunks

        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            # Clean up temp file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            raise

    def _get_dynamic_threshold(self, query_text: str) -> float:
        """
        Dynamically adjust the similarity threshold based on query characteristics.

        Args:
            query_text: The query text

        Returns:
            Adjusted similarity threshold
        """
        # Default threshold
        threshold = self.similarity_threshold

        # Lower threshold for very short queries (like "hello")
        if len(query_text.split()) <= 2 and len(query_text) < 10:
            threshold = max(0.1, threshold - 0.2)  # Much lower threshold for greetings

        # Lower threshold for longer, more specific queries
        elif len(query_text.split()) >= 6:
            threshold = max(
                0.15, threshold - 0.05
            )  # Slightly lower for specific questions

        # Handle questions specifically
        if query_text.strip().endswith("?"):
            threshold = max(0.15, threshold - 0.05)  # Questions need more context

        return threshold

    def _text_overlap_score(self, text1: str, text2: str) -> float:
        """
        Calculate text overlap score between two strings.
        Higher score means more content overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Overlap score between 0 and 1
        """
        # Normalize and tokenize texts
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _format_chunk_for_context(self, doc: str, meta: Dict, similarity: float) -> str:
        """
        Format a chunk in a clean, readable way for the context.

        Args:
            doc: Document text
            meta: Metadata for the document
            similarity: Similarity score

        Returns:
            Formatted chunk text
        """
        chunk_type = meta.get("type", "unknown")
        header = meta.get("header", "")
        source = meta.get("source", "")
        filename = (
            os.path.basename(source) if source else meta.get("filename", "unknown")
        )

        # Start with source information
        formatted = f"Source: {filename}\n"

        # Add header if available
        if header:
            formatted += f"Section: {header}\n"

        # Format based on chunk type
        if chunk_type == "code_text_pair":
            # Try to separate code and text if possible
            if "```" in doc:
                # Already has markdown code blocks
                formatted += f"\n{doc}\n"
            else:
                # Try to identify code portion
                lines = doc.split("\n")
                code_lines = []
                text_lines = []

                in_code_block = False
                for line in lines:
                    # Heuristic: lines with special characters are likely code
                    if any(c in line for c in "{}[]()=><;:") and not in_code_block:
                        in_code_block = True
                        if text_lines:
                            # Add an empty line before code block
                            text_lines.append("")
                        code_lines.append(line)
                    elif in_code_block:
                        if line.strip() == "" and len(code_lines) > 0:
                            # Empty line might end a code block
                            in_code_block = False
                            text_lines.append(line)
                        else:
                            code_lines.append(line)
                    else:
                        text_lines.append(line)

                # If we identified code and text portions
                if code_lines and text_lines:
                    formatted += "\nText:\n" + "\n".join(text_lines)
                    formatted += "\n\nCode:\n```\n" + "\n".join(code_lines) + "\n```\n"
                else:
                    # Just use the original content
                    formatted += f"\n{doc}\n"
        else:
            # Regular text chunk
            formatted += f"\n{doc}\n"

        return formatted

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents using the code-text pair approach with dynamic thresholds."""
        if not self.documents:
            logger.warning("No documents to search")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

        try:
            # Handle empty or very short queries specially
            if not query_text or len(query_text.strip()) < 2:
                logger.info("Query too short, returning empty results")
                return {
                    "query": query_text,
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                }

            # Ensure query isn't too long to avoid memory issues
            if len(query_text) > 2000:
                query_text = query_text[:2000]
                logger.info("Truncated long query to 2000 chars")

            # Get dynamic threshold based on query characteristics
            effective_threshold = self._get_dynamic_threshold(query_text)
            logger.info(f"Using dynamic threshold: {effective_threshold}")

            # Generate query embedding
            logger.info("Generating embedding for query")
            query_embedding = self.embedder.embed_string(query_text)
            query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

            # Apply PCA reduction to the query if using PCA
            if self.use_pca and self.pca_reducer and self.pca_reducer.is_fitted:
                logger.info("Applying PCA reduction to query embedding")
                query_embedding = self.pca_reducer.transform(
                    query_embedding.reshape(1, -1)
                )[0]

            # Get embedding count
            embedding_count = self._get_embedding_count()
            logger.info(f"Searching against {embedding_count} embeddings")

            # Process in small chunks to avoid loading everything at once
            similarities = np.zeros(embedding_count)

            # Use a batch size for processing
            batch_size = min(100, embedding_count)
            logger.info(f"Processing in batches of {batch_size}")

            # Load document embeddings with memory mapping
            document_embeddings = np.load(self.embeddings_path, mmap_mode="r")

            # Calculate similarities in small batches
            for i in range(0, embedding_count, batch_size):
                end = min(i + batch_size, embedding_count)

                # Load this batch into memory
                batch = np.ascontiguousarray(
                    document_embeddings[i:end], dtype=np.float32
                )

                # Calculate similarity
                batch_similarities = cosine_similarity([query_embedding], batch)[0]

                # Store results
                similarities[i:end] = batch_similarities

                # Clean up
                del batch, batch_similarities
                gc.collect()

                # Log progress for large collections
                if embedding_count > 1000 and (i + batch_size) % 1000 == 0:
                    logger.info(
                        f"Processed {i + batch_size}/{embedding_count} embeddings"
                    )

            # Find results above threshold
            filtered_indices = np.where(similarities >= effective_threshold)[0]

            # If we don't have any results, try an even lower threshold for simple queries
            if len(filtered_indices) == 0 and len(query_text.split()) <= 3:
                logger.info(
                    f"No results above threshold {effective_threshold}, trying lower threshold"
                )
                filtered_indices = np.where(similarities >= 0.1)[
                    0
                ]  # Very low threshold for simple queries

            if len(filtered_indices) == 0:
                logger.info(f"No results above threshold {effective_threshold}")
                return {
                    "query": query_text,
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                }

            # Sort indices by similarity (descending)
            sorted_indices = filtered_indices[
                np.argsort(-similarities[filtered_indices])
            ]

            # Calculate text overlap for reranking
            if len(sorted_indices) > 1:
                # Get top results
                top_n = min(
                    len(sorted_indices), n_results * 2
                )  # Get twice as many for reranking
                top_indices = sorted_indices[:top_n]

                # Calculate text overlap scores
                overlap_scores = []
                for idx in top_indices:
                    doc_text = self.documents[idx]
                    overlap = self._text_overlap_score(query_text, doc_text)

                    # Create a combined score: 0.7 * similarity + 0.3 * overlap
                    combined_score = (0.7 * similarities[idx]) + (0.3 * overlap)
                    overlap_scores.append(combined_score)

                # Rerank based on combined scores
                reranked_indices = [
                    top_indices[i] for i in np.argsort(-np.array(overlap_scores))
                ]
                sorted_indices = np.array(
                    reranked_indices
                    + [idx for idx in sorted_indices if idx not in top_indices]
                )

            # Add deduplication
            unique_doc_contents = set()
            unique_doc_sources = (
                {}
            )  # THIS IS THE FIX: Initialize as a dictionary, not a set
            unique_indices = []

            for idx in sorted_indices:
                doc_content = self.documents[idx]
                doc_source = self.document_metadata[idx].get("source", "")

                # Create a content hash (first 100 chars to avoid memory issues)
                content_hash = hash(doc_content[:100])

                # Only add if:
                # 1. We don't have too many chunks from the same source already, and
                # 2. This exact content isn't already included
                if (
                    doc_source not in unique_doc_sources
                    or unique_doc_sources[doc_source] < 2
                ):
                    if content_hash not in unique_doc_contents:
                        # Update source count
                        unique_doc_sources[doc_source] = (
                            unique_doc_sources.get(doc_source, 0) + 1
                        )
                        unique_doc_contents.add(content_hash)
                        unique_indices.append(idx)

                        # Break if we have enough results
                        if len(unique_indices) >= n_results:
                            break

            # Use unique indices only
            sorted_indices = np.array(unique_indices)

            # Gather results
            result = {
                "query": query_text,
                "ids": [f"doc_{i}" for i in sorted_indices],
                "documents": [self.documents[i] for i in sorted_indices],
                "metadatas": [self.document_metadata[i] for i in sorted_indices],
                "distances": [1.0 - similarities[i] for i in sorted_indices],
                "similarities": [similarities[i] for i in sorted_indices],
            }

            # Log search summary
            logger.info(
                f"Found {len(sorted_indices)} unique results above threshold {effective_threshold}"
            )

            # Clean up
            del document_embeddings, similarities, filtered_indices, sorted_indices
            gc.collect()

            return result

        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

    def _get_embedding_count(self) -> int:
        """Safely get the number of embeddings stored."""
        if os.path.exists(self.embeddings_path):
            try:
                embeddings = np.load(self.embeddings_path, mmap_mode="r")
                count = embeddings.shape[0]
                del embeddings
                return count
            except Exception as e:
                logger.error(f"Error checking embedding count: {str(e)}")
        return 0

    def get_context_for_query(
        self,
        query_text: str,
        n_results: int = 5,
        debug_mode: bool = False,
        return_chunks: bool = False,
    ) -> Union[str, Tuple[str, List], Tuple[str, Dict, List]]:
        """Format search results as context for LLM, with optional debug info and chunks."""
        results = self.query(query_text, n_results=n_results)

        if not results["documents"]:
            if debug_mode:
                return "", {"chunks": []}, []
            elif return_chunks:
                return "", []
            else:
                return ""

        # Assemble context carefully to avoid memory issues
        context_parts = []
        debug_info = {"query": query_text, "chunks": [], "similarity_scores": []}

        # Store chunks for return_chunks parameter
        returned_chunks = []

        # Process all chunks and format them properly
        for i, (doc, meta, distance, similarity) in enumerate(
            zip(
                results["documents"],
                results["metadatas"],
                results["distances"],
                results.get("similarities", [0] * len(results["documents"])),
            )
        ):
            chunk_type = meta.get("type", "unknown")

            # Format the chunk using the specialized formatter
            formatted_chunk = self._format_chunk_for_context(doc, meta, similarity)
            context_parts.append(formatted_chunk)

            # Always collect chunks for both debug mode and return_chunks mode
            chunk_info = {
                "id": f"Chunk {i+1}",
                "text": doc,
                "metadata": meta,
                "similarity": similarity,
                "type": chunk_type,
            }

            if debug_mode:
                debug_info["chunks"].append(chunk_info)
                debug_info["similarity_scores"].append(similarity)

            # Always collect returned chunks, regardless of mode
            returned_chunks.append(chunk_info)

        # Join the context parts with clear separators
        context = "\n\n" + "\n\n---\n\n".join(context_parts)

        # Return different results based on mode, but always include chunks in debug mode
        if debug_mode:
            return (context, debug_info, returned_chunks)
        elif return_chunks:
            return (context, returned_chunks)
        else:
            return context

    def close(self):
        """Clean up resources and save metadata."""
        try:
            self._save_metadata()

            # Clear memory
            self.documents = []
            self.document_metadata = []
            gc.collect()

            logger.info("VectorDB resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
