import logging
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Dict, Any, Optional, List
import spacy

logger = logging.getLogger(__name__)


def load_nlp_model(model_name: str = "en_core_web_sm") -> Optional[spacy.language.Language]:
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
        logger.warning(f"SpaCy model '{model_name}' not found. Will use simple tokenization.")
        return None


def extract_proper_nouns(text: str, nlp: Optional[spacy.language.Language]) -> List[str]:
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
    proper_nouns: List[str] = []

    # Extract proper nouns (PROPN) and named entities
    for token in doc:
        if token.pos_ == "PROPN" and len(token.text) > 1 and token.text.lower() not in proper_nouns:
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

    def __init__(self, use_spacy: bool = True) -> None:
        self.documents: List[str] = []  # List of document texts
        self.document_metadata: List[Dict[str, Any]] = []  # List of document metadata
        self.tokenized_corpus: List[List[str]] = []  # Tokenized corpus for BM25
        self.bm25: Optional[BM25Okapi] = None  # BM25 model
        self.use_spacy = use_spacy
        self.nlp: Optional[spacy.language.Language] = load_nlp_model() if use_spacy else None

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using SpaCy if available, otherwise fallback to simple tokenization."""
        if self.nlp is not None:
            doc = self.nlp(text.lower())
            tokens: List[str] = [
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 1
            ]
            return tokens
        else:
            return text.lower().split()

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a single document to the BM25 index."""
        tokens = self.tokenize_text(text)

        self.documents.append(text)
        self.document_metadata.append(metadata or {})
        self.tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        self._build_index()

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add multiple documents at once to the BM25 index."""
        if metadatas is None:
            metadatas = [{} for _ in texts]

        for text, metadata in zip(texts, metadatas):
            tokens = self.tokenize_text(text)
            self.documents.append(text)
            self.document_metadata.append(metadata)
            self.tokenized_corpus.append(tokens)

        self._build_index()

    def _build_index(self) -> None:
        """Build or rebuild the BM25 index."""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Built BM25 index with {len(self.tokenized_corpus)} documents")

    def query(self, query_text: str, n_results: int = 5, boost_proper_nouns: bool = True) -> Dict[str, Any]:
        """Search for documents using BM25 with optional proper noun boosting and filtering."""
        if not self.bm25:
            logger.warning("No documents indexed for BM25 search")
            return {
                "query": query_text,
                "ids": [],
                "documents": [],
                "metadatas": [],
                "scores": [],
                "score_details": [],
            }

        # Tokenize the query normally
        tokenized_query = self.tokenize_text(query_text)
        base_scores = self.bm25.get_scores(tokenized_query)
        scores = base_scores.copy()
        
        score_details: List[Dict[str, Any]] = []
        
        # Extract proper nouns using a title-cased query to help spaCy recognize them
        proper_nouns: List[str] = []
        if boost_proper_nouns and self.nlp is not None:
            proper_nouns = extract_proper_nouns(query_text.title(), self.nlp)
            logger.debug(f"Extracted proper nouns for query: {proper_nouns}")
        
        # Iterate over each document score
        for i, score in enumerate(base_scores):
            detail: Dict[str, Any] = {
                "index": i,
                "base_bm25": score,
                "proper_noun_boost": 1.0,
                "formula": f"BM25(doc_{i}) = {score:.4f}"
            }
            doc_tokens = self.tokenized_corpus[i]
            if proper_nouns:
                # Check if any proper noun from the query is present in the document tokens
                if any(noun in doc_tokens for noun in [noun.lower() for noun in proper_nouns]):
                    # Apply boost factor
                    detail["proper_noun_boost"] = 1.5
                    scores[i] *= 1.5
                    detail["formula"] += f" × 1.5 (proper noun boost) = {scores[i]:.4f}"
                    detail["latex"] = f"\\text{{BM25}}_{{{i}}} = {score:.4f} \\times 1.5 = {scores[i]:.4f}"
                else:
                    # If no proper noun match is found, filter out this document
                    scores[i] = 0
                    detail["filtered"] = True
                    detail["reason"] = "No proper noun match from query"
            else:
                detail["latex"] = f"\\text{{BM25}}_{{{i}}} = {score:.4f}"
            score_details.append(detail)

        # Determine top indices based on the (possibly boosted/filtered) scores
        top_indices = np.argsort(scores)[::-1][:n_results]
        top_indices = [idx for idx in top_indices if scores[idx] > 0]

        result: Dict[str, Any] = {
            "query": query_text,
            "ids": [f"doc_{i}" for i in top_indices],
            "documents": [self.documents[i] for i in top_indices],
            "metadatas": [self.document_metadata[i] for i in top_indices],
            "scores": [scores[i] for i in top_indices],
            "score_details": [score_details[i] for i in top_indices]
        }

        return result