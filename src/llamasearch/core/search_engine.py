# src/llamasearch/core/search_engine.py (CPU-Only, Syntax Fixes)

import gc
import threading
from pathlib import Path
from typing import Dict, Optional, Union, List, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction, Embeddings, Embeddable
from chromadb.config import Settings as ChromaSettings

from .bm25 import WhooshBM25Retriever
from .embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_NAME
from .embedder import EnhancedEmbedder
from .onnx_model import load_onnx_llm
from .source_manager import (
    _SourceManagementMixin, DEFAULT_MAX_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MIN_CHUNK_SIZE_FILTER,
)
from .query_processor import _QueryProcessingMixin
from ..exceptions import ModelNotFoundError, SetupError
from ..protocols import LLM
from ..utils import setup_logging
from sentence_transformers import SentenceTransformer # Import needed for isinstance check

logger = setup_logging(__name__, use_qt_handler=True)

CHROMA_COLLECTION_NAME = "llamasearch_docs"
BM25_SUBDIR = "bm25_data"
ChromaMetadataValue = Union[str, int, float, bool]
ChromaMetadataDict = Dict[str, ChromaMetadataValue]

# --- ChromaDB Embedding Function Wrapper ---
class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Embeddable]):
    """Wrapper to adapt SentenceTransformer for ChromaDB's EmbeddingFunction interface."""
    def __init__(self, model: SentenceTransformer):
        self._model = model

    def __call__(self, input: Embeddable) -> Embeddings:
        # Runtime check: SentenceTransformer only handles text documents (List[str])
        if not isinstance(input, list) or not all(isinstance(doc, str) for doc in input):
            raise TypeError(
                "Input must be a list of strings (Documents) for SentenceTransformerEmbeddingFunction."
            )
        # Use cast to inform the type checker after the runtime check
        embeddings_np = self._model.encode(cast(List[str], input), convert_to_numpy=True)
        return embeddings_np.tolist()

class LLMSearch(_SourceManagementMixin, _QueryProcessingMixin):
    """Orchestrates CPU-only embedding, indexing, and querying."""
    def __init__(
        self,
        storage_dir: Path,
        shutdown_event: Optional[threading.Event] = None,
        llm_onnx_quant: str = "auto",
        # llm_provider_opts removed - CPU only
        verbose: bool = True,
        max_results: int = 3,
        embedder_model: Optional[str] = None,
        embedder_batch_size: int = 0,
        embedder_truncate_dim: Optional[int] = None,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        min_chunk_size_filter: int = DEFAULT_MIN_CHUNK_SIZE_FILTER,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        debug: bool = False,
    ):
        self.verbose = verbose
        self.max_results = max_results
        self.debug = debug
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True) # Fixed E701
        self._shutdown_event = shutdown_event
        self.bm25_weight, self.vector_weight = bm25_weight, vector_weight
        self.model: Optional[LLM] = None
        self.embedder: Optional[EnhancedEmbedder] = None
        self.chroma_client: Optional[ClientAPI] = None
        self.chroma_collection: Optional[chromadb.Collection] = None
        self.bm25: Optional[WhooshBM25Retriever] = None
        self._reverse_lookup: Dict[str, str] = {}
        self.context_length: int = 0
        self.llm_device_type: str = "cpu" # Hardcoded CPU

        self.min_chunk_size_filter = max(0, min_chunk_size_filter)
        self.max_chunk_size = max(1, max_chunk_size)
        self.chunk_overlap = min(max(0, chunk_overlap), self.max_chunk_size // 2)
        if self.min_chunk_size_filter > self.max_chunk_size:
            logger.warning("min_chunk_filter > max_chunk_size.")
        logger.info(f"Chunk Params: Max={self.max_chunk_size}, Overlap={self.chunk_overlap}, MinFilter={self.min_chunk_size_filter}")

        try:
            self._load_reverse_lookup()

            # 2. Initialize LLM (CPU-Only ONNX)
            logger.info("Initializing Generic ONNX LLM (CPU-Only)...")
            self.model = load_onnx_llm(
                onnx_quantization=llm_onnx_quant,
                preferred_provider="CPUExecutionProvider", # Explicitly force CPU
                preferred_options=None, # No options needed/used for CPU
            )
            if not self.model:
                raise RuntimeError("load_onnx_llm returned None.")
            model_info = self.model.model_info
            self.context_length = model_info.context_length
            logger.info(f"LLM OK: {model_info.model_id} on CPU. Context: {self.context_length}")

            # 3. Initialize Embedder (CPU-only)
            embed_model_name = embedder_model or DEFAULT_EMBEDDER_NAME
            logger.info(f"Initializing Embedder '{embed_model_name}' (CPU-only)...")
            self.embedder = EnhancedEmbedder(model_name=embed_model_name, batch_size=embedder_batch_size, truncate_dim=embedder_truncate_dim)
            if self.embedder and self._shutdown_event:
                self.embedder.set_shutdown_event(self._shutdown_event)
            embedding_dim = self.embedder.get_embedding_dimension()
            if embedding_dim:
                logger.info(f"Embedder OK (CPU). Dim: {embedding_dim}")
            else:
                logger.warning("Could not determine embedding dimension.")

            # 4. Initialize ChromaDB Client and Collection (CPU Embedder)
            logger.info(f"Initializing ChromaDB Client (storage: {self.storage_dir})")
            chroma_settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=str(self.storage_dir))
            self.chroma_client = chromadb.Client(chroma_settings)
            if self.chroma_client is None:
                raise RuntimeError("Failed init Chroma Client.")

            chroma_meta = {"hnsw:space": "cosine"} if embedding_dim else None
            if self.embedder and self.embedder.model and isinstance(self.embedder.model, SentenceTransformer):
                # Create the wrapper instance using the loaded model
                embedding_function_wrapper = SentenceTransformerEmbeddingFunction(self.embedder.model)
                logger.debug("Created SentenceTransformerEmbeddingFunction wrapper for ChromaDB.")
            else:
                logger.error("Embedder instance not valid for Chroma.")
                raise SetupError("Failed to get valid embedder instance for Chroma.")

            self.chroma_collection = self.chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_function_wrapper, metadata=chroma_meta) # Use the wrapper
            if self.chroma_collection:
                logger.info(f"ChromaDB OK: '{CHROMA_COLLECTION_NAME}'. Count: {self.chroma_collection.count()}")
            else:
                raise RuntimeError("Failed get/create Chroma collection.")

            # 5. Initialize BM25 Retriever
            bm25_path = self.storage_dir / BM25_SUBDIR
            logger.info(f"Initializing Whoosh BM25 Retriever (storage: {bm25_path})")
            self.bm25 = WhooshBM25Retriever(storage_dir=bm25_path)
            logger.info(f"Whoosh BM25 OK. Doc Count: {self.bm25.get_doc_count()}.")

            logger.info("LLMSearch engine components initialized successfully (CPU-Only).")

        except (ModelNotFoundError, SetupError) as e:
            logger.error(f"LLMSearch init failed (setup): {e}. Run setup.")
            self.close() # Fixed E702
            raise # Fixed E702
        except ImportError as e:
            raise SetupError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected LLMSearch init error: {e}", exc_info=True) # Fixed E702
            self.close() # Fixed E702
            raise RuntimeError("LLMSearch init failed.") from e

    # --- Lifecycle Methods (CPU-Only) ---
    def _safe_unload_llm(self) -> None:
        if self.model is None or not hasattr(self.model, "unload"):
            return
        logger.info("Unloading LLM model...")
        try:
            unload = getattr(self.model, "unload", None)
            if callable(unload): # Fixed E701
                unload()
            # Fixed E701 (implicit else)
            del self.model
        except Exception as e:
            logger.error(f"Error during LLM unload: {e}", exc_info=True)
        finally:
            self.model = None # Fixed E701
            gc.collect() # Fixed E701

    def close(self) -> None:
        logger.info("Closing LLMSearch engine components (CPU-Only)...")
        if self._shutdown_event and not self._shutdown_event.is_set(): # Fixed E701
            self._shutdown_event.set()
        if self.model: # Fixed E701
            self._safe_unload_llm()
        if self.embedder:
            try: # Fixed E701
                self.embedder.close()
            except Exception as e: # Fixed E701
                logger.warning(f"Err closing embedder: {e}")
            finally: # Fixed E701
                self.embedder = None
        if self.bm25:
            try: # Fixed E701
                self.bm25.close()
            except Exception as e: # Fixed E701
                logger.warning(f"Err closing BM25: {e}")
            finally: # Fixed E701, moved statements to new lines
                self.bm25 = None
        if self.chroma_client:
            try: # Fixed E701
                pass # No explicit close needed, release refs
            finally: # Fixed E701, moved statements to new lines
                self.chroma_client = None
                self.chroma_collection = None
                logger.debug("Chroma refs released.")
        logger.info("LLMSearch engine closed (CPU-Only).")
        gc.collect() # Fixed E701

    def __enter__(self):
        return self # Fixed E701
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() # Fixed E701