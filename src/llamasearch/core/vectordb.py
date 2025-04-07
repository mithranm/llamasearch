import os
import re
import numpy as np
import json
import shutil
from typing import Dict, Any, Optional, List, Set, cast, Tuple
from pathlib import Path
from llamasearch.utils import setup_logging
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.chunker import MarkdownChunker, HtmlChunker
from llamasearch.core.bm25 import BM25Retriever
from llamasearch.core.grapher import KnowledgeGraph, get_phonetic_representation

from transformers.pipelines import pipeline
from sklearn.metrics.pairwise import cosine_similarity

logger = setup_logging(__name__)

class VectorDB:
    """
    Enhanced VectorDB integrates semantic (vector) retrieval with BM25 and knowledge-graph
    for improved retrieval performance.
    
    Key features:
    - Unified NER approach across components
    - Hybrid search combining vector and keyword-based retrieval
    - Entity-aware chunk deduplication
    - Multilingual entity recognition support
    - Enhanced name query handling with improved ranking
    """

    def __init__(
        self,
        storage_dir: Path,
        collection_name: str,
        embedder: Optional[EnhancedEmbedder] = None,
        max_chunk_size: int = 512,
        text_embedding_size: int = 768,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        chunker_batch_size: int = 32,
        embedder_batch_size: int = 32,
        similarity_threshold: float = 0.2,
        max_results: int = 3,
        graph_weight: float = 0.3,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.4,
        ner_model: str = "Babelscape/wikineural-multilingual-ner",
        device: str = "cpu",
        enable_deduplication: bool = True,
        dedup_similarity_threshold: float = 0.8,  # Threshold for Jaccard similarity deduplication
    ):
        self.collection_name = collection_name
        self.max_chunk_size = max_chunk_size
        self.text_embedding_size = text_embedding_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.embedder_batch_size = embedder_batch_size
        self.chunker_batch_size = chunker_batch_size
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self.graph_weight = graph_weight
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.storage_dir = storage_dir
        self.ner_model = ner_model
        self.device = device
        self.enable_deduplication = enable_deduplication
        self.dedup_similarity_threshold = dedup_similarity_threshold

        # Create the main collection directory and subdirectories
        collection_dir = os.path.join(self.storage_dir, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        self.vector_dir = os.path.join(collection_dir, "vector")
        self.bm25_dir = os.path.join(collection_dir, "bm25")
        self.kg_dir = os.path.join(collection_dir, "kg")
        os.makedirs(self.vector_dir, exist_ok=True)
        os.makedirs(self.bm25_dir, exist_ok=True)
        os.makedirs(self.kg_dir, exist_ok=True)

        # Define file paths
        self.metadata_path = os.path.join(self.vector_dir, "meta.json")
        self.embeddings_path = os.path.join(self.vector_dir, "embeddings.npy")
        
        # Initialize embedder
        self.embedder = embedder or EnhancedEmbedder(batch_size=self.embedder_batch_size)
        
        # Initialize NER pipeline with common multilingual model to be shared
        # across all components for consistency
        logger.info(f"Loading NER model: {self.ner_model}")
        self.ner_pipeline = pipeline(
            "ner", 
            model=self.ner_model,
            device=self.device,
            aggregation_strategy="simple"
        )
        logger.info("NER pipeline initialized successfully")
        
        # Initialize chunkers
        chunker_params = {
            'max_chunk_size': self.max_chunk_size,
            'min_chunk_size': self.min_chunk_size,
            'batch_size': self.chunker_batch_size,
            'overlap_size': self.chunk_overlap,
            'debug_output': False,
        }
        self.markdown_chunker = MarkdownChunker(**chunker_params)
        self.html_chunker = HtmlChunker(**chunker_params)
        self.chunker = self.markdown_chunker  # default

        # Initialize BM25 retriever
        self.bm25 = BM25Retriever(storage_dir=self.bm25_dir)

        # Initialize the knowledge graph
        self.kg = KnowledgeGraph(storage_dir=self.kg_dir)
        
        # Initialize document storage
        self.documents: List[str] = []
        self.document_metadata: List[Dict[str, Any]] = []
        self._data_modified = False
        
        # Track processed chunks for deduplication
        self.processed_chunks: List[Set[str]] = []
        
        # Load existing data if available
        self._load_metadata()

    def _load_metadata(self) -> bool:
        """Load document metadata from disk if available"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.document_metadata = data.get("metadata", [])
                
                # Rebuild processed chunks tokens for deduplication
                self.processed_chunks = []
                for doc in self.documents:
                    self.processed_chunks.append(self._get_tokens(doc))
                
                if len(self.documents) != len(self.document_metadata):
                    logger.error(f"Mismatch in loaded data: {len(self.documents)} docs vs {len(self.document_metadata)} metadata entries")
                    self.documents = []
                    self.document_metadata = []
                    self.processed_chunks = []
                    return False
                logger.info(f"Loaded {len(self.documents)} document entries from metadata")
                return True
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.documents = []
                self.document_metadata = []
                self.processed_chunks = []
                return False
        else:
            logger.info(f"No metadata file found at {self.metadata_path}")
            return False

    def _save_metadata(self) -> None:
        """Save document metadata to disk"""
        logger.info(f"Saving metadata for {len(self.documents)} documents")
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        temp_fn = f"temp_{os.path.basename(self.metadata_path)}"
        temp_path = os.path.join(os.path.dirname(self.metadata_path), temp_fn)
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                meta = {
                    "documents": self.documents, 
                    "metadata": self.document_metadata
                }
                json.dump(meta, f)
            os.replace(temp_path, self.metadata_path)
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def is_document_processed(self, file_path: str) -> bool:
        """Check if a document has already been processed"""
        if not self.document_metadata:
            return False
        if not os.path.exists(file_path):
            return False
        disk_mtime = os.path.getmtime(file_path)
        for md in self.document_metadata:
            if md.get("source") == file_path:
                if abs(md.get("mtime", 0) - disk_mtime) < 1e-4:
                    return True
                else:
                    # File has changed, remove all chunks from this source
                    self._remove_document(file_path)
                    return False
        return False
    
    def _remove_document(self, file_path: str) -> None:
        """Remove all chunks from a specific document source"""
        old_len = len(self.documents)
        
        # Get indices to remove
        indices_to_remove = [i for i, meta in enumerate(self.document_metadata) 
                           if meta.get("source") == file_path]
        
        # Remove from BM25 index
        for idx in indices_to_remove:
            if idx < len(self.documents):
                doc_id = f"doc_{idx}"
                try:
                    self.bm25.remove_document(doc_id)
                except AttributeError:
                    # If BM25 doesn't support direct removal, we'll rebuild it later
                    pass
        
        # Remove items while preserving indices
        new_documents = []
        new_metadata = []
        new_processed_chunks = []
        
        for i in range(len(self.documents)):
            if i not in indices_to_remove:
                new_documents.append(self.documents[i])
                new_metadata.append(self.document_metadata[i])
                if i < len(self.processed_chunks):
                    new_processed_chunks.append(self.processed_chunks[i])
        
        self.documents = new_documents
        self.document_metadata = new_metadata
        self.processed_chunks = new_processed_chunks
        
        # Flag that data has been modified
        self._data_modified = True
        logger.info(f"Removed {old_len - len(self.documents)} chunks from {file_path}")

    def _merge_embeddings(self, temp_path: str) -> None:
        """Merge newly created embeddings with existing database"""
        logger.info("Merging newly created embeddings with existing DB")
        if os.path.exists(self.embeddings_path) and os.path.getsize(self.embeddings_path) > 0:
            try:
                arr_old = np.load(self.embeddings_path, allow_pickle=False)
                arr_new = np.load(temp_path, allow_pickle=False)
                merged = np.vstack([arr_old, arr_new])
                np.save(self.embeddings_path, merged, allow_pickle=False)
                logger.info("Embeddings merged successfully")
            except Exception as e:
                logger.error(f"Error merging embeddings: {e}")
                raise
        else:
            shutil.copy2(temp_path, self.embeddings_path)
            logger.info("New embeddings file created")

    def _get_tokens(self, text: str) -> Set[str]:
        """
        Get tokens from text for Jaccard similarity comparison.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Set of tokens
        """
        # Normalize and tokenize the text
        return set(re.findall(r'\b\w+\b', text.lower()))

    def _is_duplicate_chunk(self, text: str) -> bool:
        """
        Check if a chunk is a duplicate using Jaccard similarity.
        
        Args:
            text: Chunk text to check for duplication
            
        Returns:
            True if the chunk is similar enough to an existing chunk
        """
        text_tokens = self._get_tokens(text)
        if not text_tokens:
            return False  # Empty chunks are not considered duplicates
        
        # Only compare with recent chunks to avoid O(n²) comparisons
        recent_limit = 100
        recent_chunks = self.processed_chunks[-recent_limit:] if len(self.processed_chunks) > 0 else []
        
        for existing_tokens in recent_chunks:
            # Skip empty token sets
            if not existing_tokens:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(text_tokens.intersection(existing_tokens))
            union = len(text_tokens.union(existing_tokens))
            
            if union > 0:  # Avoid division by zero
                similarity = intersection / union
                
                if similarity > self.dedup_similarity_threshold:
                    return True
                    
        return False

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """
        Extract named entities from text using transformer-based NER pipeline.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entity strings
        """
        entities = []
            
        try:
            # Limit text length to avoid issues with long inputs
            if len(text) > 1000:
                text = text[:1000]
                
            # Extract entities using NER pipeline
            results = self.ner_pipeline(text)
            
            # Convert results to a list of entity strings
            if results is not None:
                for item in results:
                    if isinstance(item, dict):
                        entity_text = item.get("word")
                        if entity_text and len(entity_text) > 1 and len(entity_text) < 50:
                            entities.append(entity_text)

            return entities
        except Exception as e:
            logger.warning(f"Error extracting entities with NER pipeline: {e}")
            return entities

    def is_named_entity_query(self, query_text: str) -> Tuple[bool, Optional[str]]:
        """
        Use the NER pipeline to determine if this is a named entity query.
        Leverages existing entity extraction without duplicating logic.
        
        Args:
            query_text: The query text to analyze
            
        Returns:
            Tuple of (is_entity_query, extracted_entity)
        """
        # First try to extract entities using our existing method
        entities = self._extract_potential_entities(query_text)
        
        # Check for common query patterns that indicate entity queries
        extracted_entity = None
        patterns = [
            (r"(?:tell|talk|explain)\s+(?:me|us)?\s*(?:about|on)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", 1),
            (r"who\s+(?:is|was)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", 1)
        ]
        
        # Try to extract entity from patterns
        for pattern, group in patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                extracted_entity = match.group(group)
                logger.info(f"Extracted named entity '{extracted_entity}' from query pattern")
                break
        
        # If we found entities with NER or extracted from pattern
        if entities or extracted_entity:
            # If we have extracted entity from pattern and it's not in NER results, add it
            if extracted_entity and extracted_entity not in entities:
                entities.append(extracted_entity)
            
            # If we haven't extracted from pattern but have NER entities, use the first one
            if not extracted_entity and entities:
                extracted_entity = entities[0]
                
            logger.info(f"Query identified as named entity query with target: {extracted_entity}")
            return True, extracted_entity
        
        # Special case for single-word queries - might be names not recognized by NER
        if len(query_text.split()) == 1:
            word = query_text.split()[0]
            
            # Check in knowledge graph name components if available
            if hasattr(self, 'kg') and hasattr(self.kg, 'name_components'):
                if word.lower() in self.kg.name_components:
                    logger.info(f"Single word '{word}' found in name components index")
                    return True, word
                    
        # Not a named entity query
        return False, None

    def _vector_search(self, text: str, n_ret: int = 10) -> Dict[str, Any]:
        """
        Perform vector search based on cosine similarity.
        
        Args:
            text: Query text
            n_ret: Number of results to return
            
        Returns:
            Search results dictionary
        """
        empty_result = {
            "query": text,
            "ids": [],
            "scores": [],
            "distances": [],
            "score_details": []
        }
        if not text.strip():
            return empty_result
        if len(text) > 2000:
            text = text[:2000]
        try:
            if not os.path.exists(self.embeddings_path):
                logger.warning(f"No embeddings file found at {self.embeddings_path}")
                return empty_result
            embeddings = np.load(self.embeddings_path, allow_pickle=False)
            embeddings = np.array(embeddings, dtype=np.float32)
            query_embedding = self.embedder.embed_string(text)
            scores = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
            
            # Log raw scores for debugging
            logger.info(f"Raw cosine similarity scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
            top_indices = np.argsort(scores)[::-1][:n_ret]
            top_scores = scores[top_indices]
            
            score_details = []
            for idx, score in zip(top_indices, top_scores):
                detail = {
                    "index": int(idx),
                    "cosine_similarity": float(score),
                    "formula": f"cos_sim(doc_{idx}) = {score:.4f}",
                    "source": self.document_metadata[idx].get("source", "") if idx < len(self.document_metadata) else ""
                }
                score_details.append(detail)
            
            mask = top_scores >= self.similarity_threshold
            filtered_indices = top_indices[mask]
            filtered_scores = top_scores[mask]
            
            # Filter score_details based on the threshold
            score_details = [d for d, s in zip(score_details, top_scores) if s >= self.similarity_threshold]
            
            if len(filtered_indices) == 0:
                logger.info(f"No results above threshold {self.similarity_threshold}")
                return empty_result
                
            result = {
                "query": text,
                "ids": [f"doc_{i}" for i in filtered_indices],
                "scores": filtered_scores.tolist(),
                "distances": (1.0 - filtered_scores).tolist(),
                "score_details": score_details
            }
            logger.info(f"Vector search returned {len(result['ids'])} results")
            return result
        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            return empty_result

    def _bm25_search(self, text: str, n_ret: int = 10) -> Dict[str, Any]:
        """
        Perform BM25 keyword search.
        
        Args:
            text: Query text
            n_ret: Number of results to return
            
        Returns:
            Search results dictionary
        """
        empty_result = {
            "query": text,
            "ids": [],
            "scores": [],
            "documents": [],
            "score_details": []
        }
        
        if not text.strip():
            return empty_result
            
        try:
            bm25_results = self.bm25.query(query_text=text, n_results=n_ret)
            
            if not bm25_results.get("ids"):
                logger.info("BM25 search returned no results")
                return empty_result
                
            # Convert BM25 results to match vector search format
            score_details = []
            for i, (doc_id, score) in enumerate(zip(bm25_results.get("ids", []), 
                                                  bm25_results.get("scores", []))):
                # Extract numeric ID from doc_id string
                try:
                    idx = int(doc_id.split("_")[1])
                    detail = {
                        "index": idx,
                        "bm25_score": float(score),
                        "formula": f"bm25(doc_{idx}) = {score:.4f}",
                        "source": self.document_metadata[idx].get("source", "") if idx < len(self.document_metadata) else ""
                    }
                    score_details.append(detail)
                except (ValueError, IndexError):
                    continue
                    
            return {
                "query": text,
                "ids": bm25_results.get("ids", []),
                "scores": bm25_results.get("scores", []),
                "documents": bm25_results.get("documents", []),
                "score_details": score_details
            }
        except Exception as e:
            logger.error(f"BM25 search error: {e}", exc_info=True)
            return empty_result

    def _kg_search(self, text: str, n_ret: int = 10) -> Dict[str, Any]:
        """
        Perform knowledge graph search based on entity relationships.
        
        Args:
            text: Query text
            n_ret: Number of results to return
            
        Returns:
            Search results dictionary
        """
        empty_result = {
            "query": text,
            "ids": [],
            "scores": [],
            "sources": [],
            "score_details": []
        }
        
        if not text.strip():
            return empty_result
            
        try:
            # Extract entities from query
            query_entities = self._extract_entities_from_text(text)
            
            if not query_entities:
                logger.info("No entities found in query for KG search")
                return empty_result
                
            # Use knowledge graph to find relevant documents
            kg_results = self.kg.search(query_entities, limit=n_ret)
            
            if not kg_results:
                logger.info("KG search returned no results")
                return empty_result
                
            # Convert KG results to same format as vector search
            document_scores = {}
            score_details = []
            
            # KG search will return document sources, need to map to doc IDs
            for entity, related_docs in kg_results.items():
                for doc_source, score in related_docs.items():
                    if doc_source not in document_scores:
                        document_scores[doc_source] = score
                    else:
                        document_scores[doc_source] = max(document_scores[doc_source], score)
            
            # Create a mapping from source to doc_ids
            source_to_ids = {}
            for i, meta in enumerate(self.document_metadata):
                source = meta.get("source", "")
                if source:
                    if source not in source_to_ids:
                        source_to_ids[source] = []
                    source_to_ids[source].append(i)
            
            # Build results
            doc_ids = []
            scores = []
            sources = []
            
            for source, score in sorted(document_scores.items(), 
                                      key=lambda x: x[1], reverse=True):
                if source in source_to_ids:
                    for idx in source_to_ids[source]:
                        doc_id = f"doc_{idx}"
                        doc_ids.append(doc_id)
                        scores.append(score)
                        sources.append(source)
                        
                        detail = {
                            "index": idx,
                            "kg_score": float(score),
                            "formula": f"kg_relevance(doc_{idx}) = {score:.4f}",
                            "source": source
                        }
                        score_details.append(detail)
                        
            return {
                "query": text,
                "ids": doc_ids[:n_ret],
                "scores": scores[:n_ret],
                "sources": sources[:n_ret],
                "score_details": score_details[:n_ret]
            }
        except Exception as e:
            logger.error(f"KG search error: {e}", exc_info=True)
            return empty_result

    def _hybrid_search(self, 
                      query_text: str, 
                      max_results: int = 10) -> Dict[str, Any]:
        """
        Perform a hybrid search that combines vector, BM25, and knowledge graph results.
        Enhanced with named entity detection and boosting.
        
        Args:
            query_text: The query text
            max_results: Maximum number of results to return
            
        Returns:
            Combined search results
        """
        # Check if this is a named entity query
        is_entity_query, target_entity = self.is_named_entity_query(query_text)
        
        # Track document scores and already seen documents
        doc_scores = {}
        doc_sources = set()
        exact_match_docs = set()  # Track documents with exact match to query
        
        # Helper to extract doc index from doc_id
        def get_doc_idx(doc_id):
            try:
                return int(doc_id.split("_")[1])
            except (ValueError, IndexError):
                return -1
        
        # For entity queries, identify documents with exact entity matches
        if is_entity_query and target_entity:
            logger.info(f"Entity query detected: '{target_entity}'")
            target_lower = target_entity.lower()
            
            for i, doc in enumerate(self.documents):
                doc_lower = doc.lower()
                if target_lower in doc_lower:
                    exact_match_docs.add(f"doc_{i}")
                    
                    # Higher boost if target appears in document title or beginning
                    if target_lower in doc_lower[:150]:
                        logger.info(f"Target entity '{target_entity}' found in prominent position in doc_{i}")
                    
                # Check entity lists too
                entities = self.document_metadata[i].get("entities", [])
                for entity in entities:
                    if target_lower in entity.lower():
                        exact_match_docs.add(f"doc_{i}")
                        logger.info(f"Target entity '{target_entity}' found in entity list for doc_{i}")
                        
            logger.info(f"Found {len(exact_match_docs)} documents with exact matches for '{target_entity}'")
        
        # Run the three search methods in parallel
        vector_results = self._vector_search(query_text, max_results * 2)
        bm25_results = self._bm25_search(query_text, max_results * 2)
        kg_results = self._kg_search(query_text, max_results * 2)
        
        # Process vector results
        for doc_id, score in zip(vector_results.get("ids", []), 
                               vector_results.get("scores", [])):
            doc_idx = get_doc_idx(doc_id)
            if doc_idx >= 0:
                source = self.document_metadata[doc_idx].get("source", "") if doc_idx < len(self.document_metadata) else ""
                if source and source in doc_sources:
                    # Skip duplicate sources
                    continue
                    
                # Apply score with vector weight
                base_score = score * self.vector_weight
                
                # Boost for entity queries with exact matches
                if is_entity_query and doc_id in exact_match_docs:
                    base_score += 0.5  # Significant boost for exact entity matches
                
                doc_scores[doc_id] = base_score
                if source:
                    doc_sources.add(source)
        
        # Process BM25 results
        for doc_id, score in zip(bm25_results.get("ids", []), 
                               bm25_results.get("scores", [])):
            doc_idx = get_doc_idx(doc_id)
            if doc_idx >= 0:
                source = self.document_metadata[doc_idx].get("source", "") if doc_idx < len(self.document_metadata) else ""
                if source and source in doc_sources:
                    # Skip duplicate sources unless it's an exact match for name query
                    if not (is_entity_query and doc_id in exact_match_docs):
                        continue
                    
                # Normalize BM25 scores - use a better normalization that preserves relative ranking
                normalized_score = min(score / 5.0, 1.0)  # Less aggressive normalization
                
                # Special handling for entity queries
                if is_entity_query:
                    # Additional boost for exact entity matches
                    if doc_id in exact_match_docs:
                        normalized_score += 0.3
                
                # Apply BM25 weight and add to existing score or set new score
                if doc_id in doc_scores:
                    doc_scores[doc_id] += normalized_score * self.bm25_weight
                else:
                    doc_scores[doc_id] = normalized_score * self.bm25_weight
                    
                if source:
                    doc_sources.add(source)
        
        # Process KG results
        for doc_id, score in zip(kg_results.get("ids", []), 
                               kg_results.get("scores", [])):
            doc_idx = get_doc_idx(doc_id)
            if doc_idx >= 0:
                source = self.document_metadata[doc_idx].get("source", "") if doc_idx < len(self.document_metadata) else ""
                if source and source in doc_sources:
                    # Skip duplicate sources
                    if not (is_entity_query and doc_id in exact_match_docs):
                        continue
                
                # Apply knowledge graph weight with potential entity query boost
                kg_score = score * self.graph_weight
                
                # Apply entity boost
                if is_entity_query and doc_id in exact_match_docs:
                    kg_score += 0.2  # Additional boost for exact entity matches
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += kg_score
                else:
                    doc_scores[doc_id] = kg_score
                    
                if source:
                    doc_sources.add(source)
        
        # For entity queries, reorder results to prioritize exact matches
        if is_entity_query and exact_match_docs:
            exact_matches = []
            other_matches = []
            
            # Separate exact matches from other matches
            for doc_id, score in doc_scores.items():
                if doc_id in exact_match_docs:
                    exact_matches.append((doc_id, score))
                else:
                    other_matches.append((doc_id, score))
            
            # Sort both groups by score and combine with exact matches first
            sorted_results = (sorted(exact_matches, key=lambda x: x[1], reverse=True) + 
                            sorted(other_matches, key=lambda x: x[1], reverse=True))
            
            logger.info(f"Reordered results for entity query: {len(exact_matches)} exact matches prioritized")
        else:
            # Standard sort by combined score
            sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results
        ids = []
        scores = []
        combined_score_details = []
        
        for doc_id, score in sorted_results[:max_results]:
            ids.append(doc_id)
            scores.append(score)
            
            doc_idx = get_doc_idx(doc_id)
            if doc_idx >= 0:
                source = self.document_metadata[doc_idx].get("source", "") if doc_idx < len(self.document_metadata) else ""
                
                # Find individual scores from each method
                vector_score = 0.0
                for detail in vector_results.get("score_details", []):
                    if detail.get("index") == doc_idx:
                        vector_score = detail.get("cosine_similarity", 0.0)
                        break
                        
                bm25_score = 0.0
                for detail in bm25_results.get("score_details", []):
                    if detail.get("index") == doc_idx:
                        bm25_score = detail.get("bm25_score", 0.0)
                        break
                        
                kg_score = 0.0
                for detail in kg_results.get("score_details", []):
                    if detail.get("index") == doc_idx:
                        kg_score = detail.get("kg_score", 0.0)
                        break
                
                # Check if this had an exact match boost
                exact_match_boost = 0.0
                if is_entity_query and doc_id in exact_match_docs:
                    exact_match_boost = 0.5  # Same value used above for entity boost
                
                # Create detailed score breakdown
                formula = f"combined(doc_{doc_idx}) = {vector_score:.2f}*{self.vector_weight} + {bm25_score:.2f}*{self.bm25_weight} + {kg_score:.2f}*{self.graph_weight}"
                if exact_match_boost > 0:
                    formula += f" + {exact_match_boost} (exact entity match boost)"
                
                combined_score_details.append({
                    "index": doc_idx,
                    "combined_score": score,
                    "vector_score": vector_score,
                    "bm25_score": bm25_score,
                    "kg_score": kg_score,
                    "exact_match_boost": exact_match_boost if exact_match_boost > 0 else None,
                    "formula": formula,
                    "source": source
                })
        
        return {
            "query": query_text,
            "ids": ids,
            "scores": scores,
            "score_details": combined_score_details
        }

    def _extract_potential_entities(self, query_text: str) -> List[str]:
        """
        Extract potential named entities from the query text using the NER pipeline.
        Leverages existing NER model to extract entities from queries.
        
        Args:
            query_text: The query text
        
        Returns:
            List of potential entity strings
        """
        # Initialize with empty list
        entities = []
        
        # Clean and normalize query text
        cleaned_text = query_text.strip()
        
        # Try using the transformer-based NER pipeline
        try:
            results = self.ner_pipeline(cleaned_text)
            
            # Also try with capitalized version for potential better detection of names
            if cleaned_text.islower():
                cap_text = ' '.join(word.capitalize() for word in cleaned_text.split())
                cap_results = cast(list, self.ner_pipeline(cap_text))
                
                # Merge results from both runs, preserving original form from query
                for cap_result in cap_results:
                    if isinstance(cap_result, dict):
                        word = cap_result.get("word")
                        entity_type = cap_result.get("entity_group", "")
                        
                        # Find corresponding text in original query
                        if word and word.lower() in cleaned_text.lower():
                            # Get the exact form from the original query
                            start_pos = cleaned_text.lower().find(word.lower())
                            if start_pos >= 0:
                                end_pos = start_pos + len(word)
                                original_form = cleaned_text[start_pos:end_pos]
                                entities.append(original_form)
            
            # Extract entity texts from NER results
            if results:
                for result in results:
                    if isinstance(result, dict):
                        word = result.get("word")
                        if word:
                            entities.append(word)
            
            # If we got entities from the model, return them with duplicates removed
            if entities:
                return list(set(entities))
        except Exception as e:
            logger.warning(f"NER pipeline failed for query: {e}")
            # Fall back to regex patterns
        
        # Extract potential name from query patterns like "tell me about X" or "who is X"
        name_patterns = [
            r"(?:tell|talk|explain)\s+(?:me|us)?\s*(?:about|on)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"who\s+(?:is|was)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                name = match.group(1)
                if name and name not in entities:
                    entities.append(name)
                    logger.info(f"Extracted name from query pattern: '{name}'")
        
        # If no entities found, try special handling for potential name queries
        words = cleaned_text.split()
        
        # Check if query might be a simple name query
        if len(words) <= 3:
            # If 1-3 words, this could be a name
            # For exact single-word queries (like "Georgi"), add it directly as an entity
            if len(words) == 1 and len(words[0]) > 1:
                entities.append(words[0])  # Add single word as potential entity
                
            # Check against name component index if available
            query_is_name = False
            
            # If this is a single word query, check if it's in our name component index
            if len(words) == 1 and hasattr(self, 'kg') and hasattr(self.kg, 'name_components'):
                if words[0].lower() in self.kg.name_components:
                    query_is_name = True
                    # Add the query as a potential entity
                    entities.append(words[0])
                    
                    # Also add any known full names containing this component
                    for full_name in self.kg.name_components.get(words[0].lower(), []):
                        if full_name not in entities:
                            entities.append(full_name)
            
            # If whole query looks like a name, add it
            if not query_is_name:
                # Check for typical name patterns
                name_pattern = re.compile(r'^[A-Za-z]+(?:\s+[A-Za-z]+){0,2}$')
                if name_pattern.match(cleaned_text):
                    # Add the whole query as a potential name entity
                    entities.append(cleaned_text)
        
        # Fallback to regex patterns for non-name entities
        if not entities:
            # Pattern for detecting capitalized words that might be names
            entity_pattern = re.compile(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b')
            
            # Extract entities using regex
            for match in entity_pattern.finditer(cleaned_text):
                entity = match.group(0).strip()
                # Skip common words that start sentences
                if entity.lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'there'}:
                    continue
                entities.append(entity)
            
            # Look for lowercase words that might be names in question patterns
            # This helps with queries like "who is georgi"
            word_pattern = re.compile(r'\b(?:who|what|where|when|how|why)\s+(?:is|are|was|were)\s+([a-z][a-z]+)\b', re.IGNORECASE)
            for match in word_pattern.finditer(cleaned_text):
                entities.append(match.group(1))
        
        return entities

    def expand_context(self, initial_results: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        """
        Expands retrieved context by exploring adjacent chunks and chunks with shared entities.
        Enhanced for better handling of name queries.
        
        Args:
            initial_results: The initial retrieval results
            query_text: The original query text
            
        Returns:
            Enhanced retrieval results with expanded context
        """
        if not initial_results.get("ids"):
            return initial_results
            
        # Track already seen sources to avoid duplicates
        retrieved_sources = set()
        
        # Extract document sources and chunk indices from initial results
        retrieved_meta = initial_results.get("metadatas", [])
        
        # Track which documents we've already included
        included_doc_ids = set(initial_results.get("ids", []))
        
        # Extract sources from initial results
        for meta in retrieved_meta:
            if meta and "source" in meta:
                source = meta.get("source")
                if source:
                    retrieved_sources.add(source)
        
        # Extract entities mentioned in retrieved chunks
        retrieved_entities = set()
        for meta in retrieved_meta:
            if meta and "entities" in meta:
                retrieved_entities.update(meta.get("entities", []))
        
        # Extract potential entities from the query
        query_entities = self._extract_potential_entities(query_text)
        
        # Enhanced name query detection
        query_might_be_name = len(query_text.split()) <= 3
        if query_might_be_name:
            # Check for name components
            if hasattr(self, 'kg') and hasattr(self.kg, 'name_components'):
                for word in query_text.lower().split():
                    if len(word) > 2 and word in self.kg.name_components:
                        # This is likely a name query - add all known variations
                        for full_name in self.kg.name_components.get(word, []):
                            query_entities.append(full_name)
                            name_parts = full_name.split()
                            for part in name_parts:
                                if part.lower() != word:
                                    query_entities.append(part)
            
            # Also check phonetic variations
            if hasattr(self, 'kg') and hasattr(self.kg, 'phonetic_map'):
                phonetic = get_phonetic_representation(query_text)
                    
                if phonetic and phonetic in self.kg.phonetic_map:
                    for variant in self.kg.phonetic_map.get(phonetic, []):
                        query_entities.append(variant)
        
        # Add all query entities to retrieved entities
        for entity in query_entities:
            retrieved_entities.add(entity)
        
        # Keep track of related chunks and their scores
        related_chunks = []
        related_scores = []
        
        # Enhanced entity matching with better support for name variations
        for i, meta in enumerate(self.document_metadata):
            doc_id = f"doc_{i}"
            if doc_id in included_doc_ids:
                continue
                
            source = meta.get("source")
            if source in retrieved_sources:
                # Skip documents from sources we've already included
                # to prevent duplicate sources in results
                continue
                
            # Check if this is from the same source as any retrieved chunk
            same_source = any(rmeta.get("source") == source for rmeta in retrieved_meta)
            
            # Check if chunk contains entities from query or retrieved documents
            chunk_entities = meta.get("entities", [])
            
            # Check for exact entity matches
            shared_entities = bool(set(chunk_entities) & retrieved_entities)
            
            # Enhanced name matching logic for better handling of names
            name_matches = False
            if not shared_entities and query_might_be_name:
                # For each query entity that might be a name
                for q_entity in query_entities:
                    q_lower = q_entity.lower()
                    
                    # For each entity in the chunk
                    for c_entity in chunk_entities:
                        c_lower = c_entity.lower()
                        
                        # Check name variations:
                        
                        # 1. First name / last name matches
                        q_parts = q_lower.split()
                        c_parts = c_lower.split()
                        
                        # Check for shared name components
                        common_parts = set(q_parts) & set(c_parts)
                        if common_parts:
                            name_matches = True
                            break
                        
                        # 2. Check if query is substring of entity or vice versa
                        if q_lower in c_lower or c_lower in q_lower:
                            name_matches = True
                            break
                        
                        # 3. Check for phonetic similarity
                        if hasattr(self.kg, 'get_phonetic_representation'):
                            q_phonetic = get_phonetic_representation(q_entity)
                            c_phonetic = get_phonetic_representation(c_entity)
                            if q_phonetic and c_phonetic and (q_phonetic == c_phonetic or 
                                                            q_phonetic in c_phonetic or
                                                            c_phonetic in q_phonetic):
                                name_matches = True
                                break
                    
                    if name_matches:
                        break
            
            # Check if this chunk is adjacent to any retrieved chunk
            is_adjacent = False
            for rmeta in retrieved_meta:
                if rmeta.get("source") == source:
                    # Check if this chunk is adjacent (based on batch_idx and chunk_idx)
                    if (abs(meta.get("batch_idx", -1) - rmeta.get("batch_idx", -2)) <= 1 and
                        abs(meta.get("chunk_idx", -1) - rmeta.get("chunk_idx", -2)) <= 1):
                        is_adjacent = True
                        break
            
            # Add related chunks with a relevance score
            direct_query_match = bool(set(chunk_entities) & set(query_entities))
            
            if same_source and (is_adjacent or shared_entities or direct_query_match or 
                            name_matches):
                related_chunks.append(i)
                
                # Calculate relevance score 
                score = 0.0
                if direct_query_match:
                    score += 0.9  # Direct match with query entity gets highest score
                if shared_entities:
                    score += 0.7  # Chunks with exact entity matches
                if name_matches:
                    score += 0.8  # Name variations/components match
                if is_adjacent:
                    score += 0.4  # Adjacent chunks get boost
                if same_source:
                    score += 0.2  # Same source gives small boost
                    
                related_scores.append(score)
        
        # Sort related chunks by score and take top ones up to a limit
        expansion_limit = 3  # Limit to top 3 related chunks
        if related_chunks:
            sorted_pairs = sorted(zip(related_chunks, related_scores), key=lambda x: x[1], reverse=True)
            top_related = sorted_pairs[:expansion_limit]
            
            # Add top related chunks to the results
            for chunk_idx, score in top_related:
                doc_id = f"doc_{chunk_idx}"
                source = self.document_metadata[chunk_idx].get("source", "")
                
                # Skip if source already in retrieved_sources
                if source in retrieved_sources:
                    continue
                    
                initial_results["ids"].append(doc_id)
                initial_results["documents"].append(self.documents[chunk_idx])
                initial_results["metadatas"].append(self.document_metadata[chunk_idx])
                initial_results["scores"].append(score)
                
                # Add source to tracking set
                if source:
                    retrieved_sources.add(source)
                
                # Add scoring details for debugging
                if "score_details" in initial_results:
                    detail = {
                        "index": chunk_idx,
                        "score": score,
                        "reason": "Context expansion",
                        "entities": self.document_metadata[chunk_idx].get("entities", [])
                    }
                    initial_results["score_details"].append(detail)
        
        return initial_results

    def vectordb_query(
        self,
        query_text: str,
        show_retrieved_chunks: bool = True,
        query_entities: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        expand_context: bool = True,
        use_hybrid_search: bool = True
    ) -> Dict[str, Any]:
        """
        Query the vector database using hybrid search and context expansion.
        Enhanced with improved named entity detection.
        
        Args:
            query_text: The query text
            show_retrieved_chunks: Whether to include retrieved chunks in the response
            query_entities: Optional list of entities to boost in the search
            max_results: Maximum number of results to return
            expand_context: Whether to perform context expansion
            use_hybrid_search: Whether to use hybrid search (vector+BM25+KG)
            
        Returns:
            Dictionary with query results
        """
        empty_response = {
            "query": query_text,
            "ids": [],
            "documents": [],
            "metadatas": [],
            "scores": [],
            "distances": [],
            "score_details": [],
            "debug_chunks": [],
            "error": None
        }
        if not self.documents:
            logger.warning("No documents in database")
            empty_response["error"] = "No documents in database"
            return empty_response

        # Check if this is a named entity query (using existing NER)
        is_entity_query, target_entity = self.is_named_entity_query(query_text)

        # For entity queries, adjust max_results to improve recall
        effective_max_results = max_results or self.max_results
        if is_entity_query:
            effective_max_results = min(effective_max_results + 3, 6)  # Increase for entity queries
            logger.info(f"Entity query detected, using {effective_max_results} max results")

        # Determine search method
        if use_hybrid_search:
            search_results = self._hybrid_search(query_text, effective_max_results)
        else:
            # Fallback to vector search only
            search_results = self._vector_search(query_text, effective_max_results)
        
        vector_ids = search_results.get("ids", [])
        indices = []
        seen_sources = set()  # Track unique sources to avoid duplicates
        
        for doc_id in vector_ids:
            try:
                i = int(doc_id.split("_")[1])
                
                # Check if this document's source is already included
                source = self.document_metadata[i].get("source", "") if i < len(self.document_metadata) else ""
                if source in seen_sources:
                    continue
                    
                indices.append(i)
                if source:
                    seen_sources.add(source)
            except Exception:
                continue
                
        retrieved_docs = [self.documents[i] for i in indices if i < len(self.documents)]
        retrieved_meta = [self.document_metadata[i] for i in indices if i < len(self.document_metadata)]
        
        # Ensure we don't have duplicate entries
        unique_indices = []
        unique_docs = []
        unique_meta = []
        
        for i, (idx, doc, meta) in enumerate(zip(indices, retrieved_docs, retrieved_meta)):
            if i == 0 or doc not in unique_docs:
                unique_indices.append(idx)
                unique_docs.append(doc)
                unique_meta.append(meta)
        
        response = {
            "query": query_text,
            "ids": [f"doc_{i}" for i in unique_indices],
            "documents": unique_docs,
            "metadatas": unique_meta,
            "scores": search_results.get("scores", [])[:len(unique_indices)],
            "distances": [1.0 - s for s in search_results.get("scores", [])[:len(unique_indices)]],
            "score_details": search_results.get("score_details", [])[:len(unique_indices)],
            "debug_chunks": [],
            "error": None
        }
        
        # Apply context expansion if enabled
        if expand_context:
            response = self.expand_context(response, query_text)
        
        # Fallback for entity queries - if target entity is not found in retrieved documents
        if is_entity_query and target_entity and response["documents"]:
            # Check if the entity appears in any of the retrieved documents
            entity_found = any(target_entity.lower() in doc.lower() for doc in response["documents"])
            
            # If target entity not found in retrieved docs, search directly
            if not entity_found:
                logger.info(f"Target entity '{target_entity}' not found in retrieved docs, doing direct search")
                direct_matches = []
                included_doc_ids = set(response["ids"])
                
                # Search for direct mentions of the entity
                for i, doc in enumerate(self.documents):
                    doc_id = f"doc_{i}"
                    if (doc_id not in included_doc_ids and 
                        target_entity.lower() in doc.lower()):
                        
                        # Calculate a score for direct entity matches - higher for specific queries
                        direct_score = 0.95  # High fixed score for direct entity matches
                        direct_matches.append((i, direct_score))
                
                # Add top direct matches to results if found (max 2)
                if direct_matches:
                    for idx, score in sorted(direct_matches, key=lambda x: x[1], reverse=True)[:2]:
                        doc_id = f"doc_{idx}"
                        logger.info(f"Adding direct entity match: {doc_id}")
                        
                        # Insert at the beginning to prioritize direct entity matches
                        response["ids"].insert(0, doc_id)
                        response["documents"].insert(0, self.documents[idx])
                        response["metadatas"].insert(0, self.document_metadata[idx])
                        response["scores"].insert(0, score)
                        
                        if "score_details" in response:
                            detail = {
                                "index": idx,
                                "score": score,
                                "reason": f"Direct entity match for '{target_entity}'",
                                "source": self.document_metadata[idx].get("source", "")
                            }
                            response["score_details"].insert(0, detail)
        
        return response

    def add_document_chunks(self, file_path: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Adds document chunks from a file to the vector database.
        Updates BM25 index, metadata, embeddings, and the knowledge graph.
        Applies deduplication to prevent duplicate chunks.
        
        Args:
            file_path: Path to the source document
            chunks: List of chunk dictionaries
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning(f"No chunks provided for {file_path}")
            return 0
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if self.is_document_processed(file_path):
            logger.info(f"Document already processed: {file_path}")
            return sum(1 for m in self.document_metadata if m.get("source") == file_path)
        
        doc_mtime = os.path.getmtime(file_path)
        base_meta: Dict[str, Any] = {
            "source": file_path, 
            "filename": os.path.basename(file_path), 
            "mtime": doc_mtime
        }
        
        # Extract full text for knowledge graph
        full_text = "\n\n".join(c["chunk"] for c in chunks if "chunk" in c)
        
        # Update the knowledge graph
        self.kg.add_document(file_path, full_text)
        
        # Prepare for embeddings
        added_chunks = 0
        dest_dir = os.path.dirname(self.embeddings_path)
        os.makedirs(dest_dir, exist_ok=True)
        temp_fn = f"temp_embeddings_{os.path.basename(self.embeddings_path)}"
        temp_path = os.path.join(dest_dir, temp_fn)
        
        try:
            batch_size = self.chunker_batch_size or 100
            
            # Apply deduplication if enabled, but always keep at least one chunk per document
            if self.enable_deduplication:
                filtered_chunks = []
                # Always keep at least the first chunk to ensure document is represented
                have_kept_one = False
                
                for chunk in chunks:
                    if "chunk" in chunk:
                        chunk_text = chunk["chunk"]
                        
                        # Always keep at least one chunk
                        if not have_kept_one:
                            filtered_chunks.append(chunk)
                            have_kept_one = True
                            self.processed_chunks.append(self._get_tokens(chunk_text))
                            continue
                            
                        # Skip if too similar to existing chunks
                        if self._is_duplicate_chunk(chunk_text):
                            logger.debug(f"Skipping similar chunk: {chunk_text[:50]}...")
                            continue
                            
                        # Add to filtered chunks
                        filtered_chunks.append(chunk)
                        self.processed_chunks.append(self._get_tokens(chunk_text))
                
                chunks = filtered_chunks
                logger.info(f"After deduplication: {len(chunks)} unique chunks")
            
            # Organize chunks into batches for processing
            all_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            total_chunks = len(chunks)
            logger.info(f"{os.path.basename(file_path)} => {total_chunks} chunks in {len(all_batches)} batches")
            doc_start_idx = len(self.documents)
            
            for batch_idx, batch in enumerate(all_batches):
                if not batch:
                    continue
                    
                texts = [x["chunk"] for x in batch if "chunk" in x]
                emb_texts = [x.get("embedding_text") or x["chunk"] for x in batch if "chunk" in x]
                
                # Extract entities from each chunk using the NER pipeline
                meta_list: List[Dict[str, Any]] = []
                for i, bdat in enumerate(batch):
                    met = dict(base_meta)
                    met["batch_idx"] = batch_idx
                    met["chunk_idx"] = i
                    
                    # Extract entities from this chunk
                    chunk_text = bdat["chunk"]
                    chunk_entities = self._extract_entities_from_text(chunk_text)
                    
                    # Add entities to metadata
                    met["entities"] = chunk_entities
                    
                    # Add any other metadata from the chunk
                    if "metadata" in bdat:
                        for k, v in bdat["metadata"].items():
                            if k == "source" and v:
                                met["source"] = v
                            else:
                                met[k] = v
                                
                    meta_list.append(met)
                
                # Generate embeddings for this batch
                embeddings = self.embedder.embed_batch(emb_texts)
                embeddings = np.array(embeddings, dtype=np.float32)
                
                # Save or merge embeddings
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    old = np.load(temp_path, allow_pickle=False)
                    merged = np.vstack([old, embeddings])
                    np.save(temp_path, merged, allow_pickle=False)
                else:
                    np.save(temp_path, embeddings, allow_pickle=False)
                
                # Add chunks to document list
                self.documents.extend(texts)
                self.document_metadata.extend(meta_list)
                
                # Update BM25 index
                for i, (t, m) in enumerate(zip(texts, meta_list)):
                    doc_idx = doc_start_idx + i
                    doc_id = f"doc_{doc_idx}"
                    m['id'] = doc_idx
                    self.bm25.add_document(t, doc_id)
                    
                added_chunks += len(batch)
                
            # Finalize processing
            if added_chunks > 0:
                self._merge_embeddings(temp_path)
                self._data_modified = True
                if hasattr(self.bm25, "_build_entity_indices"):
                    self.bm25._build_entity_indices()
                self._save_metadata()
                
            # Clean up temp files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
            logger.info(f"Done indexing {file_path}: {added_chunks} chunks")
            return added_chunks
            
        except Exception as e:
            logger.error(f"Error adding chunks from {file_path}: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise