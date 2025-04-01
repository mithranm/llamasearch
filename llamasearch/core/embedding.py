"""
Ultra memory-efficient embeddings with batch-by-batch processing and
aggressive memory management for M2 MacBooks using all-MiniLM-L6-v2.
"""

import numpy as np
import logging
import torch
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 2,
    ):  # Increased max_length to 512
        """
        Initialize the embedder with the sentence-transformers model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('cpu' or 'cuda')
            max_length: Maximum token length for input text
            batch_size: Batch size for processing multiple texts
        """
        try:
            self.device = device
            self.max_length = max_length
            self.batch_size = batch_size

            # Load the model
            logger.info(f"Loading sentence-transformer model: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)

            logger.info(f"Loaded sentence-transformer model {model_name} on {device}")
        except (OSError, ValueError) as e:
            logger.error(f"Error loading model '{model_name}': {e}")
            raise

    def embed_string(self, string: str) -> np.ndarray:
        """
        Get embedding for a single string.

        Args:
            string: Input text string

        Returns:
            Numpy array containing the embedding
        """
        if not isinstance(string, str):
            raise TypeError("Input must be a string.")

        logger.info(f"Embedding single string ({len(string)} chars)")

        # Process the string character by character to enforce a max length
        if len(string) > 2000:  # Increased from 1000 to 2000 for longer sequence length
            string = string[:2000]
            logger.info("Truncated string to 2000 chars to reduce memory usage")

        # Get embedding with explicit memory management
        with torch.no_grad():
            embedding = self.model.encode(string, convert_to_numpy=True)

        # Clean up to ensure memory is released
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return embedding

    def embed_batch(self, strings: list) -> np.ndarray:
        """
        Get embeddings for a batch of strings, processing in small batches
        for memory efficiency on limited systems like M2 MacBooks.

        Args:
            strings: List of input text strings

        Returns:
            Numpy array containing embeddings
        """
        if not strings:
            return np.array([])

        logger.info(f"Embedding batch of {len(strings)} strings")

        # For memory efficiency, process in small batches
        all_embeddings = []

        with tqdm(
            total=len(strings), desc="Generating embeddings", unit="text"
        ) as pbar:
            for i in range(0, len(strings), self.batch_size):
                batch = strings[i : i + self.batch_size]
                logger.info(
                    f"Processing batch {i//self.batch_size + 1}/{(len(strings) + self.batch_size - 1)//self.batch_size} ({len(batch)} strings)"
                )

                # Truncate very long texts to reduce memory pressure
                truncated_batch = []
                for text in batch:
                    if len(text) > 2000:  # Increased from 1000 to 2000
                        truncated_batch.append(text[:2000])
                        logger.info("Truncated string to 2000 chars")
                    else:
                        truncated_batch.append(text)

                # Get embeddings for this batch
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        truncated_batch, convert_to_numpy=True
                    )

                all_embeddings.append(batch_embeddings)

                # Force garbage collection after each batch
                gc.collect()

                # Update progress bar
                pbar.update(len(batch))

                # Log progress for every few batches
                if (i + self.batch_size) % (self.batch_size * 5) == 0 or (
                    i + self.batch_size >= len(strings)
                ):
                    logger.info(
                        f"Processed {min(i + self.batch_size, len(strings))}/{len(strings)} strings"
                    )

        # Combine all into a numpy array
        combined = np.vstack(all_embeddings)

        # Force garbage collection again
        all_embeddings = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return combined


if __name__ == "__main__":
    try:
        # Configure logging for standalone testing
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize with smallest possible batch size
        embedder = Embedder(batch_size=1)

        # Test embedding
        text = "This is an example sentence."
        embedding = embedder.embed_string(text)
        logger.info(f"Embedding shape: {embedding.shape}")

        # Test batch embedding with minimal memory usage
        batch = ["First test sentence.", "Second test sentence."]
        batch_embeddings = embedder.embed_batch(batch)
        logger.info(f"Batch embedding shape: {batch_embeddings.shape}")
    except Exception as e:
        logger.error(f"Error: {e}")
