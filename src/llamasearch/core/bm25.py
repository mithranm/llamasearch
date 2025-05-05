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
