import os
import json
import logging
from typing import Dict, Any, Optional, List, Set

import spacy
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

def load_nlp_model_for_lang(lang: str = "en") -> spacy.language.Language:
    """
    Loads a spaCy model appropriate for the given language.
    For our purposes (tokenization for BM25), we use a transformer-based model if available.
    Supported languages: "en" for English, "zh" for Chinese, "ru" for Russian.
    Falls back to a blank model if necessary.
    """
    lang = lang.lower()
    model_name = ""
    if lang == "en":
        model_name = "en_core_web_trf"
        fallback_model = "en_core_web_sm"
    elif lang == "zh":
        model_name = "zh_core_web_trf"
        fallback_model = "zh_core_web_sm"
    elif lang == "ru":
        model_name = "ru_core_news_trf"
        fallback_model = "ru_core_web_sm"
    else:
        logger.warning(f"Language '{lang}' not explicitly supported; defaulting to English model.")
        model_name = "en_core_web_trf"
        fallback_model = "en_core_web_sm"
    
    try:
        nlp = spacy.load(model_name)
        logger.info(f"Loaded spaCy model '{model_name}' for language '{lang}'.")
        return nlp
    except Exception as e:
        logger.warning(f"Failed to load model '{model_name}' for language '{lang}': {e}. Trying fallback model.")
        try:
            nlp = spacy.load(fallback_model)
            logger.info(f"Loaded fallback model '{fallback_model}' for language '{lang}'.")
            return nlp
        except Exception as e2:
            logger.warning(f"Failed to load fallback model '{fallback_model}': {e2}. Using blank model.")
            return spacy.blank(lang)

def improved_tokenizer(text: str, nlp: spacy.language.Language) -> List[str]:
    """
    Tokenizes text using spaCy.
    Proper nouns and recognized entities are preserved in their original case;
    all other tokens are lowercased.
    This tokenizer is shared with the knowledge graph so that BM25 and KG use a consistent tokenization.
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
    BM25Retriever builds an index over a collection of documents and uses BM25Okapi for keyword-based retrieval.
    
    This version is multilingual: it loads a spaCy model for tokenization based on a provided language code.
    It also provides a placeholder for building term-to-document indices (_build_entity_indices).
    """
    def __init__(self, storage_dir: str, lang: str = "en", boost_proper_nouns: bool = True, max_results: int = 5) -> None:
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.boost_proper_nouns = boost_proper_nouns
        self.similarity_threshold = 0.2
        self.max_results = max_results
        self.metadata_path = os.path.join(self.storage_dir, "meta.json")
        self.nlp = load_nlp_model_for_lang(lang)
        self.bm25: Optional[BM25Okapi] = None
        self.term_indices: Dict[str, Set[int]] = {}
        self._load_or_init_index()

    def _load_or_init_index(self) -> None:
        """
        Loads existing index metadata (documents, doc_ids, tokenized_corpus) from disk.
        If not present, initializes empty lists.
        """
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                self.doc_ids = data.get("doc_ids", [])
                self.tokenized_corpus = data.get("tokenized_corpus", [])
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                    self._build_entity_indices()
                logger.info(f"Loaded {len(self.documents)} documents from metadata.")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.documents = []
                self.doc_ids = []
                self.tokenized_corpus = []
                self.bm25 = None
                self.term_indices = {}
        else:
            logger.info(f"No metadata found at {self.metadata_path}. Starting a new index.")
            self.documents = []
            self.doc_ids = []
            self.tokenized_corpus = []
            self.bm25 = None
            self.term_indices = {}

    def _build_entity_indices(self) -> None:
        """
        Build a simple term-to-document index.
        For each token in the tokenized corpus, map it to the set of document indices in which it appears.
        This can later be used to boost BM25 scores based on entity presence.
        """
        self.term_indices = {}
        for idx, tokens in enumerate(self.tokenized_corpus):
            for token in tokens:
                if token not in self.term_indices:
                    self.term_indices[token] = set()
                self.term_indices[token].add(idx)
        logger.info(f"Built entity indices for {len(self.term_indices)} unique terms.")

    def save(self) -> None:
        """
        Saves the current index metadata (documents, doc_ids, tokenized_corpus) to disk.
        """
        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "tokenized_corpus": self.tokenized_corpus
        }
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        temp_path = os.path.join(self.storage_dir, "temp_meta.json")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(temp_path, self.metadata_path)
            logger.info("Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def add_document(self, text: str, doc_id: str) -> None:
        """
        Adds a document to the BM25 index.
        Tokenizes the text and updates the BM25 index.
        """
        tokens = improved_tokenizer(text, self.nlp)
        self.documents.append(text)
        self.doc_ids.append(doc_id)
        self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.save()
        # Rebuild the term-to-document indices
        self._build_entity_indices()

    def remove_document(self, doc_id: str) -> bool:
        """
        Removes a document from the BM25 index.
        
        Args:
            doc_id: ID of the document to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if doc_id not in self.doc_ids:
                logger.warning(f"Document ID '{doc_id}' not found in index.")
                return False
                
            # Find index of document
            idx = self.doc_ids.index(doc_id)
            
            # Remove document
            self.documents.pop(idx)
            self.doc_ids.pop(idx)
            self.tokenized_corpus.pop(idx)
            
            # Rebuild BM25 index
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                self._build_entity_indices()
            else:
                self.bm25 = None
                self.term_indices = {}
                
            self.save()
            logger.info(f"Successfully removed document '{doc_id}' from index.")
            return True
        except Exception as e:
            logger.error(f"Error removing document '{doc_id}': {e}")
            return False

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Performs a BM25 query on the index.
        Tokenizes the query, computes BM25 scores, sorts documents by score,
        and returns the top n_results.
        
        Returns a dictionary with:
          - query: the query text
          - ids: list of document IDs for the top results
          - scores: list of BM25 scores corresponding to these documents
          - documents: list of document texts for the top results
        """
        if self.bm25 is None:
            logger.warning("BM25 index not initialized.")
            return {"query": query_text, "ids": [], "scores": [], "documents": []}
            
        query_tokens = improved_tokenizer(query_text, self.nlp)
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply boost for documents containing proper nouns in query
        if self.boost_proper_nouns:
            doc = self.nlp(query_text)
            proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
            
            for token in proper_nouns:
                if token in self.term_indices:
                    # Get documents containing this proper noun
                    doc_indices = self.term_indices.get(token, set())
                    for idx in doc_indices:
                        if idx < len(scores):
                            # Apply boost to score
                            scores[idx] *= 1.5
        
        sorted_indices = np.argsort(scores)[::-1]
        
        # Filter out low-scoring documents
        filtered_indices = []
        filtered_scores = []
        
        for idx in sorted_indices:
            score = scores[idx]
            if score > 0.01:  # Minimum threshold for relevance
                filtered_indices.append(idx)
                filtered_scores.append(score)
                
                if len(filtered_indices) >= n_results:
                    break
        
        top_ids = [self.doc_ids[i] for i in filtered_indices]
        top_docs = [self.documents[i] for i in filtered_indices]
        
        return {
            "query": query_text,
            "ids": top_ids,
            "scores": [float(scores[i]) for i in filtered_indices],
            "documents": top_docs
        }