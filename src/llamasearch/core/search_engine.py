# src/llamasearch/core/search_engine.py
import gc
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Embeddable, EmbeddingFunction, Embeddings
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from ..exceptions import ModelNotFoundError, SetupError
from ..protocols import LLM
from ..utils import setup_logging
from .bm25 import WhooshBM25Retriever
from .embedder import DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_NAME
from .embedder import EnhancedEmbedder
from .onnx_model import load_onnx_llm
from .query_processor import \
    _QueryProcessingMixin  # Changed to _QueryProcessingMixin
from .source_manager import (DEFAULT_CHUNK_OVERLAP, DEFAULT_MAX_CHUNK_SIZE,
                             DEFAULT_MIN_CHUNK_SIZE_FILTER,
                             _SourceManagementMixin)

logger = setup_logging(__name__, use_qt_handler=True)


CHROMA_COLLECTION_NAME = "llamasearch_docs"
BM25_SUBDIR = "bm25_data"
ChromaMetadataValue = Union[str, int, float, bool]
ChromaMetadataDict = Dict[str, ChromaMetadataValue]


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Embeddable]):
    def __init__(self, model: SentenceTransformer):
        self._model = model

    def __call__(self, input_data: Embeddable) -> Embeddings:
        if not isinstance(input_data, list) or not all(
            isinstance(doc, str) for doc in input_data
        ):
            raise TypeError(
                "Input must be a list of strings (Documents) for SentenceTransformerEmbeddingFunction."
            )
        if not callable(getattr(self._model, "encode", None)):
            raise TypeError(
                "Provided model for SentenceTransformerEmbeddingFunction does not have a callable 'encode' method."
            )

        embeddings_np = self._model.encode(
            cast(List[str], input_data), convert_to_numpy=True
        )
        return embeddings_np.tolist()


class LLMSearch(
    _SourceManagementMixin, _QueryProcessingMixin
):  # Inherits from _QueryProcessingMixin
    def __init__(
        self,
        storage_dir: Path,
        shutdown_event: Optional[threading.Event] = None,
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
        # LLM Generation Parameters (can be stored on self and passed to process_llm_query)
        temperature: float = 0.1,
        top_p: float = 1.0,
        top_k: int = 50,  # Although GenericONNXLLM might not use all of these directly
        repetition_penalty: float = 1.0,
    ):
        self.verbose = verbose
        self.max_results = max_results
        self.debug = debug
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown_event = shutdown_event
        self.bm25_weight, self.vector_weight = bm25_weight, vector_weight
        self.model: Optional[LLM] = None
        self.embedder: Optional[EnhancedEmbedder] = None
        self.chroma_client: Optional[ClientAPI] = None
        self.chroma_collection: Optional[chromadb.Collection] = None
        self.bm25: Optional[WhooshBM25Retriever] = None
        self._reverse_lookup: Dict[str, str] = {}
        self.context_length: int = 0
        self.llm_device_type: str = "cpu"

        # Store LLM generation parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        self.min_chunk_size_filter = max(0, min_chunk_size_filter)
        self.max_chunk_size = max(1, max_chunk_size)
        self.chunk_overlap = min(max(0, chunk_overlap), self.max_chunk_size // 2)
        if self.min_chunk_size_filter > self.max_chunk_size:
            logger.warning("min_chunk_filter > max_chunk_size.")
        logger.info(
            f"Chunk Params: Max={self.max_chunk_size}, Overlap={self.chunk_overlap}, MinFilter={self.min_chunk_size_filter}"
        )

        embedding_function_wrapper: Optional[SentenceTransformerEmbeddingFunction] = (
            None
        )

        try:
            self._load_reverse_lookup()  # From _SourceManagementMixin
            logger.info("Initializing Generic ONNX LLM (CPU-Only)...")
            self.model = load_onnx_llm()
            if not self.model:
                raise RuntimeError("load_onnx_llm returned None.")
            model_info = self.model.model_info
            self.context_length = model_info.context_length
            logger.info(
                f"LLM OK: {model_info.model_id} on CPU. Context: {self.context_length}"
            )

            embed_model_name = embedder_model or DEFAULT_EMBEDDER_NAME
            logger.info(f"Initializing Embedder '{embed_model_name}' (CPU-only)...")
            self.embedder = EnhancedEmbedder(
                model_name=embed_model_name,
                batch_size=embedder_batch_size,
                truncate_dim=embedder_truncate_dim,
            )

            if self.embedder and self._shutdown_event:
                self.embedder.set_shutdown_event(self._shutdown_event)

            embedding_dim = (
                self.embedder.get_embedding_dimension() if self.embedder else None
            )

            if embedding_dim:
                logger.info(f"Embedder OK (CPU). Dim: {embedding_dim}")
            else:
                logger.warning("Could not determine embedding dimension.")

            logger.info(f"Initializing ChromaDB Client (storage: {self.storage_dir})")
            chroma_settings = ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
                is_persistent=True,
                persist_directory=str(self.storage_dir),
            )
            self.chroma_client = chromadb.Client(chroma_settings)
            if self.chroma_client is None:
                raise RuntimeError("Failed init Chroma Client.")

            chroma_meta = {"hnsw:space": "cosine"} if embedding_dim else None

            if self.embedder and self.embedder.model:
                actual_model_object = self.embedder.model
                if isinstance(actual_model_object, SentenceTransformer):
                    embedding_function_wrapper = SentenceTransformerEmbeddingFunction(
                        actual_model_object
                    )
                    logger.debug(
                        "Created SentenceTransformerEmbeddingFunction wrapper for ChromaDB."
                    )
                else:
                    logger.error(
                        f"Embedder's internal model is not a SentenceTransformer instance. Type: {type(actual_model_object)}"
                    )
                    raise SetupError(
                        "Embedder's internal model is not a valid SentenceTransformer for ChromaDB."
                    )
            else:
                embedder_state = f"self.embedder is {self.embedder}"
                model_state = "N/A (no embedder)"
                if self.embedder:
                    model_attr = getattr(self.embedder, "model", "MISSING .model ATTR")
                    model_state = f"self.embedder.model is {model_attr}"
                logger.error(
                    f"Embedder or its internal SentenceTransformer model not available. {embedder_state}, {model_state}"
                )
                raise SetupError(
                    "Failed to get valid SentenceTransformer model from embedder for ChromaDB."
                )

            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_function_wrapper,
                metadata=chroma_meta,
            )
            if self.chroma_collection:
                logger.info(
                    f"ChromaDB OK: '{CHROMA_COLLECTION_NAME}'. Count: {self.chroma_collection.count()}"
                )
            else:
                raise RuntimeError("Failed get/create Chroma collection.")

            bm25_path = self.storage_dir / BM25_SUBDIR
            logger.info(f"Initializing Whoosh BM25 Retriever (storage: {bm25_path})")
            self.bm25 = WhooshBM25Retriever(storage_dir=bm25_path)
            logger.info(f"Whoosh BM25 OK. Doc Count: {self.bm25.get_doc_count()}.")

            logger.info(
                "LLMSearch engine components initialized successfully (CPU-Only)."
            )

        except (ModelNotFoundError, SetupError) as e:
            logger.error(f"LLMSearch init failed (setup): {e}. Run setup.")
            self.close()
            raise
        except ImportError as e:
            logger.error(
                f"LLMSearch init failed due to missing dependency: {e}", exc_info=True
            )
            self.close()
            raise SetupError(f"Missing dependency: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected LLMSearch init error: {e}", exc_info=True)
            self.close()
            raise RuntimeError("LLMSearch init failed.") from e

    # llm_query method is inherited from _QueryProcessingMixin

    def _safe_unload_llm(self) -> None:
        if self.model is None or not hasattr(self.model, "unload"):
            return
        logger.info("Unloading LLM model...")
        try:
            unload = getattr(self.model, "unload", None)
            if callable(unload):
                unload()
        except Exception as e:
            logger.error(f"Error during LLM unload: {e}", exc_info=True)
        finally:
            self.model = None
            gc.collect()

    def close(self) -> None:
        logger.info("Closing LLMSearch engine components (CPU-Only)...")
        if self._shutdown_event and not self._shutdown_event.is_set():
            self._shutdown_event.set()
        if self.model:
            self._safe_unload_llm()
        if self.embedder:
            try:
                self.embedder.close()
            except Exception as e:
                logger.warning(f"Err closing embedder: {e}")
            finally:
                self.embedder = None
        if self.bm25:
            try:
                self.bm25.close()
            except Exception as e:
                logger.warning(f"Err closing BM25: {e}")
            finally:
                self.bm25 = None
        if self.chroma_client:
            try:
                # ChromaDB client doesn't have an explicit close, rely on GC for persistent client
                # For in-memory, reset might be needed if we want to clear it.
                # For persistent, just releasing reference should be fine.
                pass
            except Exception as e:
                logger.warning(f"Err during Chroma client cleanup: {e}")
            finally:
                self.chroma_client = None
                self.chroma_collection = None
                logger.debug("Chroma refs released.")
        logger.info("LLMSearch engine closed (CPU-Only).")
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
