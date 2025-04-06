import logging
from rank_bm25 import BM25Okapi
from typing import Dict, Any, Optional, List
import spacy
import os
import json

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


def find_project_root():
    """Finds the root of the project by looking for pyproject.toml."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, "pyproject.toml")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise RuntimeError("Could not find project root. Please check your project structure.")


class BM25Retriever:
    """Enhanced BM25 retriever for keyword-based search with SpaCy integration."""

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize BM25 retriever.

        Args:
            storage_dir: Directory to store BM25 index files. If None, uses default index/bm25 directory.
        """
        if storage_dir is None:
            project_root = find_project_root()
            storage_dir = os.path.join(project_root, "index", "bm25")

        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.index_path = os.path.join(self.storage_dir, "bm25_index.pkl")
        self.docs_path = os.path.join(self.storage_dir, "documents.json")
        self.vocab_path = os.path.join(self.storage_dir, "vocabulary.json")

        # Initialize components
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenized_corpus = []
        self.documents = []
        self.bm25 = None

        # Load existing index if available
        self._load_or_init_index()

    def _load_or_init_index(self) -> None:
        """Load existing index data or initialize new index."""
        try:
            with open(self.docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                self.tokenized_corpus = vocab_data.get("tokenized_corpus", [])
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Loaded BM25 index from {self.storage_dir}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.documents = []
            self.tokenized_corpus = []
            self.bm25 = None
            logger.info("Initialized new BM25 index")

    def save(self) -> None:
        """Save index data to storage directory."""
        os.makedirs(self.storage_dir, exist_ok=True)

        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        vocab_data = {
            "tokenized_corpus": self.tokenized_corpus
        }
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

        logger.info(f"Saved BM25 index to {self.storage_dir}")

    def add_document(self, text: str, doc_id: str) -> None:
        """Add a document to the index."""
        tokens = [token.text.lower() for token in self.nlp(text) if not token.is_stop and not token.is_punct]
        self.documents.append({"id": doc_id, "text": text})
        self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.save()

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Search the index for relevant documents."""
        if not self.bm25:
            return {"ids": [], "scores": [], "documents": [], "score_details": []}

        # Tokenize query
        query_tokens = [token.text.lower() for token in self.nlp(query_text) if not token.is_stop and not token.is_punct]

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort and get top results - convert numpy float64 to native float
        top_n = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)[:n_results]

        results = {
            "ids": [self.documents[idx]["id"] for idx, _ in top_n],
            "scores": [score for _, score in top_n],
            "documents": [self.documents[idx]["text"] for idx, _ in top_n],
            "score_details": [
                {
                    "index": idx,
                    "score": score,
                    "formula": f"BM25({query_text}, doc_{idx}) = {score:.4f}"
                }
                for idx, score in top_n
            ]
        }

        return results