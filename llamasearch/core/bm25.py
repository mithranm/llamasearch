import logging
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Dict, Any, Optional, List

# Import SpaCy for improved text processing
try:
    import spacy
except ImportError:
    pass

logger = logging.getLogger(__name__)


def load_nlp_model(model_name: str = "en_core_web_sm"):
    """
    Load SpaCy NLP model for text processing.

    Args:
        model_name: Name of the SpaCy model to load

    Returns:
        Loaded SpaCy model
    """
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning(
            f"SpaCy model '{model_name}' not found. Will use simple tokenization."
        )
        return None


def extract_proper_nouns(text: str, nlp) -> List[str]:
    """
    Extract proper nouns from text using SpaCy.

    Args:
        text: Input text to extract proper nouns from
        nlp: SpaCy model

    Returns:
        List of proper nouns found in the text
    """
    if nlp is None:
        return []  # Return empty list if SpaCy is not available

    doc = nlp(text)
    proper_nouns = []

    # Extract proper nouns (PROPN) and named entities
    for token in doc:
        if (
            token.pos_ == "PROPN"
            and len(token.text) > 1
            and token.text.lower() not in proper_nouns
        ):
            proper_nouns.append(token.text.lower())

    # Add named entities if not already included
    for ent in doc.ents:
        if len(ent.text) > 1 and ent.text.lower() not in proper_nouns:
            proper_nouns.append(ent.text.lower())

    return proper_nouns


class BM25Retriever:
    """
    Enhanced BM25 retriever for keyword-based search with SpaCy integration.
    Complements vector-based search with exact keyword matching and proper noun detection.
    """

    def __init__(self, use_spacy: bool = True):
        self.documents = []  # List of document texts
        self.document_metadata = []  # List of document metadata
        self.tokenized_corpus = []  # Tokenized corpus for BM25
        self.bm25 = None  # BM25 model
        self.use_spacy = use_spacy
        self.nlp = load_nlp_model() if use_spacy else None

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using SpaCy if available, otherwise fallback to simple tokenization."""
        if self.nlp is not None:
            doc = self.nlp(text.lower())
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 1
            ]
            return tokens
        else:
            return text.lower().split()

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a single document to the BM25 index."""
        tokens = self.tokenize_text(text)

        self.documents.append(text)
        self.document_metadata.append(metadata or {})
        self.tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        self._build_index()

    def add_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add multiple documents at once to the BM25 index."""
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for text, metadata in zip(texts, metadatas):
            tokens = self.tokenize_text(text)
            self.documents.append(text)
            self.document_metadata.append(metadata)
            self.tokenized_corpus.append(tokens)

        self._build_index()

    def _build_index(self):
        """Build or rebuild the BM25 index."""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Built BM25 index with {len(self.tokenized_corpus)} documents")

    def query(
        self, query_text: str, n_results: int = 5, boost_proper_nouns: bool = True
    ) -> Dict[str, Any]:
        """Search for documents using BM25 with optional proper noun boosting."""
        if not self.bm25:
            logger.warning("No documents indexed for BM25 search")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
            }

        tokenized_query = self.tokenize_text(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Optionally boost docs containing proper nouns
        if boost_proper_nouns and self.nlp is not None:
            proper_nouns = extract_proper_nouns(query_text, self.nlp)
            if proper_nouns:
                logger.debug(f"Found proper nouns to boost: {proper_nouns}")
                for i, doc_tokens in enumerate(self.tokenized_corpus):
                    for noun in proper_nouns:
                        noun_tokens = self.tokenize_text(noun)
                        if any(token in doc_tokens for token in noun_tokens):
                            # Boost the score for documents containing those nouns
                            scores[i] *= 1.5

        top_indices = np.argsort(scores)[::-1][:n_results]
        top_indices = [idx for idx in top_indices if scores[idx] > 0]

        result = {
            "query": query_text,
            "ids": [f"doc_{i}" for i in top_indices],
            "documents": [self.documents[i] for i in top_indices],
            "metadatas": [self.document_metadata[i] for i in top_indices],
            "scores": [scores[i] for i in top_indices],
        }

        return result
