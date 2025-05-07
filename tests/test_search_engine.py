# tests/test_search_engine.py
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
from pathlib import Path
import threading
import logging
import json
import numpy as np

from sentence_transformers import SentenceTransformer

from llamasearch.core.search_engine import (
    LLMSearch,
    SentenceTransformerEmbeddingFunction,
)
from llamasearch.core.onnx_model import GenericONNXLLM, GenericONNXModelInfo
from llamasearch.core.bm25 import WhooshBM25Retriever
from llamasearch.exceptions import ModelNotFoundError, SetupError
from chromadb.api import ClientAPI as ChromaClientAPI  # For spec
from chromadb import Collection as ChromaCollectionType  # For spec
from chromadb.api.types import Embeddable, Embeddings


MOCK_DATAMANAGER_PATH_FOR_SOURCE_MIXIN = "llamasearch.core.source_manager.data_manager"
MOCK_SEARCH_ENGINE_SETUP_LOGGING_TARGET = "llamasearch.core.search_engine.setup_logging"
mock_search_engine_logger_global_instance = MagicMock(spec=logging.Logger)


class TestSearchEngineModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger_patcher = patch(
            MOCK_SEARCH_ENGINE_SETUP_LOGGING_TARGET,
            return_value=mock_search_engine_logger_global_instance,
        )
        cls.logger_patcher.start()

        cls.data_manager_source_mixin_patcher = patch(
            MOCK_DATAMANAGER_PATH_FOR_SOURCE_MIXIN
        )
        cls.mock_data_manager_source_mixin = (
            cls.data_manager_source_mixin_patcher.start()
        )

    @classmethod
    def tearDownClass(cls):
        cls.logger_patcher.stop()
        cls.data_manager_source_mixin_patcher.stop()

    def setUp(self):
        mock_search_engine_logger_global_instance.reset_mock()
        self.mock_data_manager_source_mixin.reset_mock()

        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_search_engine_")
        self.storage_dir = Path(self.temp_dir_obj.name)
        self.shutdown_event = threading.Event()

        self.mock_load_onnx_llm = patch(
            "llamasearch.core.search_engine.load_onnx_llm"
        ).start()

        # Patch EnhancedEmbedder's _load_model to prevent actual model loading attempts
        self.patch_ee_load_model = patch(
            "llamasearch.core.embedder.EnhancedEmbedder._load_model"
        ).start()

        self.mock_chromadb_client_cls = patch(
            "llamasearch.core.search_engine.chromadb.Client"
        ).start()
        self.mock_whoosh_bm25_cls = patch(
            "llamasearch.core.search_engine.WhooshBM25Retriever"
        ).start()

        self.mock_llm_instance = MagicMock(spec=GenericONNXLLM)
        self.mock_llm_model_info = MagicMock(spec=GenericONNXModelInfo)
        type(self.mock_llm_model_info).context_length = PropertyMock(return_value=4096)
        type(self.mock_llm_model_info).model_id = PropertyMock(
            return_value="mock-llm-fp32"
        )
        type(self.mock_llm_instance).model_info = PropertyMock(
            return_value=self.mock_llm_model_info
        )
        self.mock_load_onnx_llm.return_value = self.mock_llm_instance

        self.mock_st_model_internal = MagicMock(spec=SentenceTransformer)
        self.mock_st_model_internal.encode.return_value = np.array([[0.1]])
        self.mock_st_model_internal.get_sentence_embedding_dimension.return_value = 384

        # When EnhancedEmbedder is initialized by LLMSearch, its _load_model is mocked.
        # We then need to ensure its 'model' attribute is set correctly.
        # The easiest way is to patch the EnhancedEmbedder class used by search_engine
        # and make its return_value's .model attribute point to our mock_st_model_internal.
        self.patch_enhanced_embedder_cls_in_se = patch(
            "llamasearch.core.search_engine.EnhancedEmbedder"
        ).start()
        self.mock_ee_instance_in_se = (
            self.patch_enhanced_embedder_cls_in_se.return_value
        )
        self.mock_ee_instance_in_se.model = (
            self.mock_st_model_internal
        )  # Key assignment
        self.mock_ee_instance_in_se.get_embedding_dimension.return_value = 384

        self.mock_chroma_client_instance = MagicMock(spec=ChromaClientAPI)
        self.mock_chroma_collection_instance = MagicMock(spec=ChromaCollectionType)
        self.mock_chroma_collection_instance.count.return_value = 0
        self.mock_chroma_client_instance.get_or_create_collection.return_value = (
            self.mock_chroma_collection_instance
        )
        self.mock_chromadb_client_cls.return_value = self.mock_chroma_client_instance

        self.mock_bm25_instance = MagicMock(spec=WhooshBM25Retriever)
        self.mock_bm25_instance.get_doc_count.return_value = 0
        self.mock_whoosh_bm25_cls.return_value = self.mock_bm25_instance

        self.mock_crawl_data_path_for_mixin = self.storage_dir / "crawl_data_for_mixin"
        self.mock_crawl_data_path_for_mixin.mkdir(parents=True, exist_ok=True)
        (self.mock_crawl_data_path_for_mixin / "reverse_lookup.json").write_text(
            json.dumps({"key1": "url1"})
        )
        self.mock_data_manager_source_mixin.get_data_paths.return_value = {
            "crawl_data": str(self.mock_crawl_data_path_for_mixin)
        }

    def tearDown(self):
        patch.stopall()
        self.temp_dir_obj.cleanup()
        self.shutdown_event.clear()

    def test_sentence_transformer_embedding_function_call_valid(self):
        mock_st_model = MagicMock(spec=SentenceTransformer)
        mock_embeddings_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_st_model.encode.return_value = mock_embeddings_np

        embed_func = SentenceTransformerEmbeddingFunction(mock_st_model)
        input_docs: Embeddable = ["doc1", "doc2"]

        result_embeddings: Embeddings = embed_func(input_docs)

        mock_st_model.encode.assert_called_once_with(
            ["doc1", "doc2"], convert_to_numpy=True
        )
        self.assertTrue(np.array_equal(np.array(result_embeddings), mock_embeddings_np))

    def test_sentence_transformer_embedding_function_call_invalid_type(self):
        mock_st_model = MagicMock(spec=SentenceTransformer)
        embed_func = SentenceTransformerEmbeddingFunction(mock_st_model)

        with self.assertRaisesRegex(TypeError, "Input must be a list of strings"):
            embed_func(123)  # type: ignore
        with self.assertRaisesRegex(TypeError, "Input must be a list of strings"):
            embed_func(["doc1", 123])  # type: ignore

    def test_llmsearch_init_model_not_found_error(self):
        self.mock_load_onnx_llm.side_effect = ModelNotFoundError("LLM setup failed")
        with self.assertRaises(ModelNotFoundError):
            LLMSearch(storage_dir=self.storage_dir)

    def test_llmsearch_init_setup_error_embedder(self):
        # Simulate EnhancedEmbedder's constructor (or its _load_model) raising SetupError
        self.patch_enhanced_embedder_cls_in_se.side_effect = SetupError(
            "Embedder setup failed"
        )
        with self.assertRaises(SetupError):
            LLMSearch(storage_dir=self.storage_dir)
        if self.mock_load_onnx_llm.call_count > 0:
            self.mock_llm_instance.unload.assert_called_once()
        # self.mock_ee_instance_in_se.close.assert_not_called() # Instance not created
        self.mock_bm25_instance.close.assert_not_called()

if __name__ == "__main__":
    unittest.main()
