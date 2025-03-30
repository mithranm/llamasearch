"""
Exposes an API for computing embeddings of a natural language string.
"""

import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        try:
            self.model = SentenceTransformer(model_name)
        except (OSError, ValueError) as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            raise

    def embed_string(self, string: str) -> np.ndarray:
        if not isinstance(string, str):
            raise TypeError("Input must be a string.")
        embeddings = self.model.encode(string)
        return embeddings

if __name__ == "__main__":
    try:
        embedder = Embedder()
        text = "This is an example sentence."
        embedding = embedder.embed_string(text)
        print(f"Embedding shape: {embedding.shape}")
    except Exception as e:
        print(f"Error: {e}")