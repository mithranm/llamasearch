# tests/test_search_engine.py
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
from pathlib import Path
import threading
import logging

import numpy as np
from sentence_transformers import SentenceTransformer 

from llamasearch.core.search_engine import LLMSearch, SentenceTransformerEmbeddingFunction
from llamasearch.core.embedder import EnhancedEmbedder, DEFAULT_MODEL_NAME as DEFAULT_EMBEDDER_NAME
from llamasearch.core.onnx_model import GenericONNXLLM, GenericONNXModelInfo
from llamasearch.core.bm25 import WhooshBM25Retriever
from llamasearch.exceptions import ModelNotFoundError, SetupError
from chromadb.api.types import Embeddable, Embeddings 


MOCK_DATAMANAGER_PATH_FOR_SOURCE_MIXIN = "llamasearch.core.source_manager.data_manager"
MOCK_SEARCH_ENGINE_SETUP_LOGGING_TARGET = "llamasearch.core.search_engine.setup_logging"
mock_search_engine_logger_global_instance = MagicMock(spec=logging.Logger)


class TestSearchEngineModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logger_patcher = patch(MOCK_SEARCH_ENGINE_SETUP_LOGGING_TARGET, return_value=mock_search_engine_logger_global_instance)
        cls.logger_patcher.start()
        
        cls.data_manager_source_mixin_patcher = patch(MOCK_DATAMANAGER_PATH_FOR_SOURCE_MIXIN)
        cls.mock_data_manager_source_mixin = cls.data_manager_source_mixin_patcher.start()


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

        self.mock_load_onnx_llm = patch("llamasearch.core.search_engine.load_onnx_llm").start()
        self.mock_enhanced_embedder_cls = patch("llamasearch.core.search_engine.EnhancedEmbedder").start()
        self.mock_chromadb_client_cls = patch("llamasearch.core.search_engine.chromadb.Client").start()
        self.mock_whoosh_bm25_cls = patch("llamasearch.core.search_engine.WhooshBM25Retriever").start()

        self.mock_llm_instance = MagicMock(spec=GenericONNXLLM)
        self.mock_llm_model_info = MagicMock(spec=GenericONNXModelInfo)
        type(self.mock_llm_model_info).context_length = PropertyMock(return_value=4096)
        type(self.mock_llm_model_info).model_id = PropertyMock(return_value="mock-llm-fp32")
        type(self.mock_llm_instance).model_info = PropertyMock(return_value=self.mock_llm_model_info)
        self.mock_load_onnx_llm.return_value = self.mock_llm_instance

        self.mock_embedder_instance = MagicMock(spec=EnhancedEmbedder)
        self.mock_embedder_instance.get_embedding_dimension.return_value = 384
        
        # Make self.mock_embedder_instance.model a mock that "is" a SentenceTransformer
        self.mock_embedder_instance.model = MagicMock(spec=SentenceTransformer)

        self.mock_enhanced_embedder_cls.return_value = self.mock_embedder_instance

        self.mock_chroma_client_instance = MagicMock()
        self.mock_chroma_collection_instance = MagicMock()
        self.mock_chroma_client_instance.get_or_create_collection.return_value = self.mock_chroma_collection_instance
        self.mock_chromadb_client_cls.return_value = self.mock_chroma_client_instance

        self.mock_bm25_instance = MagicMock(spec=WhooshBM25Retriever)
        self.mock_whoosh_bm25_cls.return_value = self.mock_bm25_instance
        
        self.mock_data_manager_source_mixin.get_data_paths.return_value = {"crawl_data": str(self.storage_dir / "crawl_data_for_mixin")}


    def tearDown(self):
        patch.stopall()
        self.temp_dir_obj.cleanup()
        self.shutdown_event.clear()

    def test_sentence_transformer_embedding_function_call_valid(self):
        mock_st_model_actual_instance = MagicMock(spec=SentenceTransformer)
        mock_embeddings_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_st_model_actual_instance.encode.return_value = mock_embeddings_np

        embed_func = SentenceTransformerEmbeddingFunction(mock_st_model_actual_instance)
        input_docs: Embeddable = ["doc1", "doc2"]
        
        result_embeddings: Embeddings = embed_func(input_docs)

        mock_st_model_actual_instance.encode.assert_called_once_with(["doc1", "doc2"], convert_to_numpy=True)
        self.assertListEqual(result_embeddings, mock_embeddings_np.tolist())


    def test_sentence_transformer_embedding_function_call_invalid_type(self):
        mock_st_model = MagicMock(spec=SentenceTransformer)
        embed_func = SentenceTransformerEmbeddingFunction(mock_st_model)
        
        with self.assertRaisesRegex(TypeError, "Input must be a list of strings"):
            embed_func(123) # type: ignore
        with self.assertRaisesRegex(TypeError, "Input must be a list of strings"):
            embed_func(["doc1", 123]) # type: ignore


    def test_llmsearch_init_success(self):
        with patch('builtins.open', MagicMock(side_effect=FileNotFoundError)): 
            llms = LLMSearch(storage_dir=self.storage_dir, shutdown_event=self.shutdown_event)

        self.mock_load_onnx_llm.assert_called_once()
        self.mock_enhanced_embedder_cls.assert_called_once_with(
            model_name=DEFAULT_EMBEDDER_NAME, batch_size=0, truncate_dim=None
        )
        self.mock_embedder_instance.set_shutdown_event.assert_called_once_with(self.shutdown_event)
        self.mock_chromadb_client_cls.assert_called_once() 
        
        self.mock_chroma_client_instance.get_or_create_collection.assert_called_once()
        args, kwargs = self.mock_chroma_client_instance.get_or_create_collection.call_args
        self.assertEqual(kwargs['name'], "llamasearch_docs")
        self.assertIsInstance(kwargs['embedding_function'], SentenceTransformerEmbeddingFunction)
        # Check that the model passed to SentenceTransformerEmbeddingFunction is the one from our mock_embedder_instance
        self.assertIs(kwargs['embedding_function']._model, self.mock_embedder_instance.model)
        self.assertEqual(kwargs['metadata'], {"hnsw:space": "cosine"})
        
        self.mock_whoosh_bm25_cls.assert_called_once_with(storage_dir=self.storage_dir / "bm25_data")
        self.assertIs(llms.model, self.mock_llm_instance)
        self.assertIs(llms.embedder, self.mock_embedder_instance)
        self.assertIs(llms.chroma_collection, self.mock_chroma_collection_instance)
        self.assertIs(llms.bm25, self.mock_bm25_instance)
        self.assertEqual(llms.context_length, 4096)


    def test_llmsearch_init_model_not_found_error(self):
        self.mock_load_onnx_llm.side_effect = ModelNotFoundError("LLM setup failed")
        with self.assertRaises(ModelNotFoundError):
            LLMSearch(storage_dir=self.storage_dir)
        self.mock_llm_instance.unload.assert_not_called() 
        self.mock_embedder_instance.close.assert_not_called() 
        self.mock_bm25_instance.close.assert_not_called()


    def test_llmsearch_init_setup_error_embedder(self):
        self.mock_enhanced_embedder_cls.side_effect = SetupError("Embedder setup failed")
        with self.assertRaises(SetupError):
            LLMSearch(storage_dir=self.storage_dir)
        self.mock_llm_instance.unload.assert_called_once() 
        self.mock_embedder_instance.close.assert_not_called() 
        self.mock_bm25_instance.close.assert_not_called()


    @patch('llamasearch.core.search_engine.gc.collect')
    def test_llmsearch_close(self, mock_gc_collect):
        with patch('builtins.open', MagicMock(side_effect=FileNotFoundError)):
             llms = LLMSearch(storage_dir=self.storage_dir, shutdown_event=self.shutdown_event)
        
        self.mock_llm_instance.reset_mock()
        self.mock_embedder_instance.reset_mock()
        self.mock_bm25_instance.reset_mock()
        self.shutdown_event.clear() 

        llms.close()

        self.assertTrue(self.shutdown_event.is_set())
        self.mock_llm_instance.unload.assert_called_once()
        self.mock_embedder_instance.close.assert_called_once()
        self.mock_bm25_instance.close.assert_called_once()
        self.assertIsNone(llms.model)
        self.assertIsNone(llms.embedder)
        self.assertIsNone(llms.bm25)
        self.assertIsNone(llms.chroma_client)
        self.assertIsNone(llms.chroma_collection)
        mock_gc_collect.assert_called()


    def test_llmsearch_context_manager(self):
        with patch.object(LLMSearch, 'close') as mock_close_method:
            with patch('builtins.open', MagicMock(side_effect=FileNotFoundError)): 
                with LLMSearch(storage_dir=self.storage_dir):
                    pass 
            mock_close_method.assert_called_once()


if __name__ == '__main__':
    unittest.main()