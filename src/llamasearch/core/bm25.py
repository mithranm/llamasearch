import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import spacy
import spacy.util
from rank_bm25 import BM25Okapi

from llamasearch.exceptions import ModelNotFoundError
from llamasearch.utils import setup_logging

# Use the global logger setup
logger = setup_logging(__name__, use_qt_handler=True)


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
    Filters out stopwords, punctuation, and spaces.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        # Keep proper nouns and entities as they are
        if token.pos_ == "PROPN" or token.ent_type_:
            if not token.is_stop and not token.is_punct and not token.is_space:
                tokens.append(token.text)
        # Lowercase others, ignore stopwords, punctuation and space tokens
        elif not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.text.lower())
    if not tokens and text.strip(): # Log if non-empty text yields no tokens
        logger.debug(f"Tokenizer yielded empty list for non-empty input: '{text[:100]}...'")
    return tokens


class BM25Retriever:
    """
    BM25Retriever builds an index over documents and uses BM25Okapi for keyword retrieval.
    Uses spaCy for tokenization, checking for model availability first.
    Supports explicit index building and removal. Data stored in specified directory.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir # Store as Path object
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[str] = []
        self.doc_ids: List[str] = [] # Stores chunk IDs (e.g., 'uuid_...')
        self.tokenized_corpus: List[List[str]] = []
        self.metadata_path = self.storage_dir / "bm25_meta.json"

        try:
            self.nlp = load_nlp_model()
        except ModelNotFoundError as e:
            logger.error(f"Failed to initialize BM25Retriever: {e}")
            raise

        self.bm25: Optional[BM25Okapi] = None
        self.term_indices: Dict[str, Set[int]] = {}
        self._index_needs_rebuild = True
        self._load_or_init_index()
        if self._index_needs_rebuild and self.tokenized_corpus:
            self.build_index() # Initial build if loaded data requires it

    def _load_or_init_index(self) -> None:
        """Loads existing metadata or initializes empty lists."""
        if self.metadata_path.exists():
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
                        f"Mismatch in loaded BM25 data lengths ({self.metadata_path}). Resetting index."
                    )
                    self.documents, self.doc_ids, self.tokenized_corpus = [], [], []
                    self._index_needs_rebuild = True # Needs rebuild after reset
                elif self.tokenized_corpus:
                    # Mark for rebuild, but don't build immediately here
                    self._index_needs_rebuild = True
                    logger.info(
                        f"Loaded {len(self.documents)} BM25 documents from {self.metadata_path.name}. Index requires rebuild on next access or explicit call."
                    )
                else:
                    self._index_needs_rebuild = False # No data, index is trivially 'built' (empty)
                    logger.info(f"Loaded empty BM25 metadata from {self.metadata_path.name}.")
            except Exception as e:
                logger.error(f"Error loading BM25 metadata from {self.metadata_path}: {e}. Resetting index.")
                self.documents, self.doc_ids, self.tokenized_corpus = [], [], []
                self._index_needs_rebuild = True # Needs rebuild after reset
        else:
            logger.info(
                f"No BM25 metadata found at {self.metadata_path}. Starting new index."
            )
            self.documents, self.doc_ids, self.tokenized_corpus = [], [], []
            self._index_needs_rebuild = False # Empty index is considered 'built'

    def build_index(self) -> bool:
        """
        Explicitly builds/rebuilds the BM25Okapi index and term indices.
        Returns True if successful or if no data to index, False on error.
        """
        if not self.tokenized_corpus:
            logger.info("BM25: No documents to index. Clearing existing index.")
            self.bm25 = None
            self.term_indices = {}
            self._index_needs_rebuild = False
            return True # Considered success as there's nothing to index

        # Filter out completely empty token lists before building
        valid_corpus = [tokens for tokens in self.tokenized_corpus if tokens]
        if not valid_corpus:
             logger.warning("BM25: Corpus contains only empty token lists after filtering. Cannot build index.")
             self.bm25 = None
             self.term_indices = {}
             self._index_needs_rebuild = False # Cannot build, but state is consistent (empty index)
             return True # Considered success in the sense that state is valid, though index is unusable

        logger.info(
            f"BM25: Building index for {len(valid_corpus)} non-empty documents..."
        )
        try:
            # Pass the filtered corpus to BM25Okapi
            self.bm25 = BM25Okapi(valid_corpus)
            # Rebuild term indices based on the original corpus indices
            self._build_term_indices() # Uses self.tokenized_corpus
            self._index_needs_rebuild = False
            logger.info("BM25: Index built successfully.")
            return True
        except ZeroDivisionError:
             # This specific error might indicate issues with the BM25 library or very unusual corpus
             logger.error("BM25: Failed to build index due to ZeroDivisionError in rank_bm25 library. "
                          "This might happen with very sparse or unusual token distributions.", exc_info=True)
             self.bm25 = None # Ensure index is None on failure
             self._index_needs_rebuild = True
             return False
        except Exception as e:
            logger.error(f"BM25: Failed to build index: {e}", exc_info=True)
            self.bm25 = None # Ensure index is None on failure
            self._index_needs_rebuild = True
            return False

    def is_index_built(self) -> bool:
        """Checks if the BM25 index is considered built and ready."""
        # Index is ready if _index_needs_rebuild is False.
        # self.bm25 might be None if the corpus was empty, which is still considered 'ready'.
        return not self._index_needs_rebuild

    def _build_term_indices(self) -> None:
        """Builds a term-to-document index (mapping term to set of doc indices)."""
        self.term_indices = {}
        # Iterate over the original tokenized_corpus to maintain correct indices
        for idx, tokens in enumerate(self.tokenized_corpus):
            for token in tokens:
                if token not in self.term_indices:
                    self.term_indices[token] = set()
                self.term_indices[token].add(idx)

    def save(self) -> None:
        """Saves the index metadata (documents, doc_ids, tokenized_corpus)."""
        # --- Build index before saving if needed ---
        if self._index_needs_rebuild:
            logger.info("BM25 index requires rebuild before saving. Building now...")
            if not self.build_index():
                logger.error("BM25 index build failed. Cannot save potentially invalid state.")
                return # Do not save if build failed

        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "tokenized_corpus": self.tokenized_corpus,
        }
        # Use Path object methods
        temp_path = self.metadata_path.with_suffix(".json.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            # os.replace is generally more atomic than rename on some OSes
            os.replace(temp_path, self.metadata_path)
            logger.debug(f"BM25 metadata saved ({len(self.doc_ids)} docs).")
        except Exception as e:
            logger.error(f"Error saving BM25 metadata to {self.metadata_path}: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def add_document(self, text: str, doc_id: str) -> bool:
        """
        Adds a document's text and unique ID (e.g., chunk UUID).
        Tokenizes the text and marks the index for rebuild.
        Returns True if added, False if ID exists.
        """
        if doc_id in self.doc_ids:
            logger.debug(f"BM25: Document ID {doc_id} already exists. Skipping add.")
            return False

        tokens = improved_tokenizer(text, self.nlp)
        if not tokens:
             logger.warning(f"BM25: Document ID {doc_id} resulted in zero tokens after processing. Adding empty token list.")
             # Still add the document to keep lists consistent, but log warning.
             # The build_index step will filter these out later.

        self.documents.append(text)
        self.doc_ids.append(doc_id)
        self.tokenized_corpus.append(tokens)
        self._index_needs_rebuild = True # Mark that index needs update
        return True

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document by its unique ID.
        Marks index for rebuild. Returns True if removed, False if not found.
        """
        try:
            idx = self.doc_ids.index(doc_id)
            self.documents.pop(idx)
            self.doc_ids.pop(idx)
            self.tokenized_corpus.pop(idx)
            self._index_needs_rebuild = True # Mark that index needs update
            return True
        except ValueError:
            logger.debug(f"BM25: Document ID '{doc_id}' not found for removal.")
            return False
        except Exception as e:
            logger.error(f"BM25: Error removing document '{doc_id}': {e}")
            return False

    # Removed remove_documents_by_source as it's handled by LLMSearch

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Performs a BM25 query. Builds index if stale."""
        empty_res = {"query": query_text, "ids": [], "scores": [], "documents": []}

        if not self.is_index_built():
            logger.warning("BM25 index not built/stale. Building now before query...")
            if not self.build_index():
                logger.error("BM25 index build failed. Returning empty query results.")
                return empty_res
            # Check again after build attempt
            if not self.is_index_built():
                 logger.error("BM25 index still not ready after build attempt. Returning empty.")
                 return empty_res
            logger.info("BM25 index was rebuilt before querying.")

        # Index is now considered built (even if empty)
        bm25_instance = self.bm25
        if bm25_instance is None or not self.tokenized_corpus or not self.doc_ids:
             logger.warning("BM25 query: Index is None or corpus/doc_ids are empty. Returning empty results.")
             return empty_res

        query_tokens = improved_tokenizer(query_text, self.nlp)
        if not query_tokens:
             logger.debug("BM25 query tokenization resulted in empty list.")
             return empty_res

        try:
            # Note: get_scores expects the original (unfiltered) tokenized corpus length
            scores = bm25_instance.get_scores(query_tokens)
        except ValueError: # Can happen if no query terms are in the corpus vocabulary
            logger.debug(f"BM25 query '{query_text[:50]}...' contained no indexed terms.")
            return empty_res
        except IndexError as e: # Can happen if corpus/index structure is bad or mismatched
             logger.error(f"BM25 get_scores IndexError (corrupted index or mismatch?): {e}")
             return empty_res
        except Exception as e:
            logger.error(f"BM25 get_scores error: {e}", exc_info=True)
            return empty_res

        # --- Ensure scores length matches document count ---
        if len(scores) != len(self.doc_ids):
             logger.error(f"BM25 score length ({len(scores)}) mismatch with doc_ids length ({len(self.doc_ids)}). Index likely corrupt. Returning empty.")
             return empty_res

        actual_n = min(n_results, len(self.documents))
        if actual_n == 0:
            return empty_res

        # Efficiently get top N indices from scores
        if len(scores) <= actual_n:
            # If asking for more than available, sort all
             top_indices = np.argsort(scores)[::-1]
        else:
             # Use argpartition for efficiency if fetching a small subset
             top_indices_unsorted = np.argpartition(scores, -actual_n)[-actual_n:]
             top_indices = top_indices_unsorted[np.argsort(scores[top_indices_unsorted])][::-1]

        # Filter by score > 0.0 and limit to n_results
        final_indices = [idx for idx in top_indices if scores[idx] > 1e-6][:n_results] # Use small epsilon

        # Check bounds before accessing lists
        max_valid_index = len(self.doc_ids) - 1
        valid_final_indices = [idx for idx in final_indices if 0 <= idx <= max_valid_index]
        if len(valid_final_indices) != len(final_indices):
            logger.warning(f"BM25 query returned indices out of bounds. Max index: {max_valid_index}, Got: {final_indices}")

        top_ids = [self.doc_ids[i] for i in valid_final_indices]
        top_docs = [self.documents[i] for i in valid_final_indices]
        top_scores = [float(scores[i]) for i in valid_final_indices]

        return {
            "query": query_text,
            "ids": top_ids, # Chunk IDs
            "scores": top_scores,
            "documents": top_docs, # Chunk text
        }

    def get_doc_count(self) -> int:
        """Returns the number of documents currently tracked (before potential filtering)."""
        return len(self.doc_ids)