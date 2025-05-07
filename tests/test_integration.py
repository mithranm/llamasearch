# tests/test_integration.py
import unittest
import tempfile
import shutil
import os
import json
from pathlib import Path
from typing import Optional, List, Any # Removed cast
import threading

os.environ["LLAMASEARCH_TESTING"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from llamasearch.core.search_engine import LLMSearch
import llamasearch.data_manager as dm_module
from llamasearch.data_manager import DataManager
from llamasearch.utils import setup_logging
from llamasearch.core.onnx_model import (
    ONNX_SUBFOLDER,
    MODEL_ONNX_BASENAME,
    ONNX_REQUIRED_LOAD_FILES,
    GenericONNXLLM,
)
from sentence_transformers import SentenceTransformer 

from unittest.mock import MagicMock, patch, PropertyMock 
from chromadb.api.types import ( 
    GetResult as ChromaGetResultType, Metadata, QueryResult as ChromaQueryResultType
)
import numpy as np
from optimum.onnxruntime import ORTModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from chromadb.api import ClientAPI as ChromaClientAPI # Unused based on last Ruff output
from chromadb import Collection as ChromaCollectionType


logger = setup_logging("llamasearch.test_integration")


class TestBasicIntegration(unittest.TestCase):
    llm_search: Optional[LLMSearch]
    original_data_manager_instance: Optional[DataManager] = None

    # Attributes for mocks configured in setUp
    mock_st_model: MagicMock 
    mock_embedder_instance: MagicMock


    @classmethod
    def setUpClass(cls):
        cls.temp_dir_main_obj = tempfile.TemporaryDirectory(prefix="ls_int_main_")
        cls.base_test_dir = Path(cls.temp_dir_main_obj.name)
        cls.original_data_manager_instance = dm_module.data_manager
        test_data_manager = DataManager(base_dir=cls.base_test_dir)
        dm_module.data_manager = test_data_manager
        cls.test_index_dir = Path(dm_module.data_manager.settings["index"])
        cls.test_models_dir = Path(dm_module.data_manager.settings["models"])
        cls.test_crawl_data_dir = Path(dm_module.data_manager.settings["crawl_data"])
        cls.test_logs_dir = Path(dm_module.data_manager.settings["logs"])
        for p in [cls.test_index_dir, cls.test_models_dir, cls.test_crawl_data_dir, cls.test_logs_dir]:
            p.mkdir(parents=True, exist_ok=True)
        active_model_dir = cls.test_models_dir / "active_model"
        active_onnx_dir = active_model_dir / ONNX_SUBFOLDER
        active_onnx_dir.mkdir(parents=True, exist_ok=True)
        for fname in ONNX_REQUIRED_LOAD_FILES:
            (active_model_dir / fname).touch()
        (active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx").touch()
        (active_onnx_dir / f"{MODEL_ONNX_BASENAME}.onnx_data").touch()
        os.makedirs(active_model_dir, exist_ok=True)
        with open(active_model_dir / "config.json", "w") as f:
            json.dump({"model_type": "onnx", "_comment": "Dummy", "hf_model_id": "d"}, f)

    @classmethod
    def tearDownClass(cls):
        if cls.original_data_manager_instance is not None:
            dm_module.data_manager = cls.original_data_manager_instance
        cls.temp_dir_main_obj.cleanup()

    def setUp(self):
        self.llm_search = None
        patcher_load_onnx_llm = patch('llamasearch.core.search_engine.load_onnx_llm')
        self.mock_load_onnx_llm = patcher_load_onnx_llm.start()
        self.addCleanup(patcher_load_onnx_llm.stop)

        patcher_auto_tokenizer = patch('llamasearch.core.onnx_model.AutoTokenizer.from_pretrained')
        self.mock_auto_tokenizer_loader = patcher_auto_tokenizer.start()
        self.addCleanup(patcher_auto_tokenizer.stop)

        patcher_ort_model = patch('llamasearch.core.onnx_model.ORTModelForCausalLM.from_pretrained')
        self.mock_ort_model_loader = patcher_ort_model.start()
        self.addCleanup(patcher_ort_model.stop)

        self.patch_ee_internal_load_model = patch('llamasearch.core.embedder.EnhancedEmbedder._load_model', MagicMock())
        self.patch_ee_internal_load_model.start()
        self.addCleanup(self.patch_ee_internal_load_model.stop)

        # This is the mock for the SentenceTransformer object that .model property should return
        self.mock_st_model = MagicMock(spec=SentenceTransformer, name="MockSTModel")
        
        # This is the mock instance of EnhancedEmbedder we want to be used.
        self.mock_embedder_instance = MagicMock(name="MockEnhancedEmbedderInstance")
        
        # Configure 'model' as a PropertyMock on the *type* of this specific instance.
        # This intercepts access to 'instance.model'.
        type(self.mock_embedder_instance).model = PropertyMock(return_value=self.mock_st_model)
        
        # Configure other necessary methods directly on the instance
        self.mock_embedder_instance.get_embedding_dimension.return_value = 384
        self.mock_embedder_instance.embed_strings = MagicMock() 
        self.mock_embedder_instance.set_shutdown_event = MagicMock()
        self.mock_embedder_instance.close = MagicMock()

        # Patch the EnhancedEmbedder CLASS constructor.
        # When EnhancedEmbedder(...) is called, it will return our pre-configured instance.
        patcher_ee_constructor = patch('llamasearch.core.search_engine.EnhancedEmbedder', 
                                       return_value=self.mock_embedder_instance)
        self.MockEnhancedEmbedderConstructor_class_patch = patcher_ee_constructor.start()
        self.addCleanup(patcher_ee_constructor.stop)
        
        patcher_chromadb_client = patch('llamasearch.core.search_engine.chromadb.Client')
        self.mock_chromadb_client_constructor = patcher_chromadb_client.start()
        self.addCleanup(patcher_chromadb_client.stop)
        
        dummy_llm_model = MagicMock(spec=GenericONNXLLM)
        dummy_llm_model.model_info = MagicMock(context_length=4096, model_id="dummy-onnx")
        dummy_llm_model.generate = MagicMock(return_value={"response": "Llamas.", "retrieved_context": ["ctx"]})
        self.mock_load_onnx_llm.return_value = dummy_llm_model

        current_test_data_manager = dm_module.data_manager
        test_index_dir_for_setup = Path(current_test_data_manager.settings["index"])
        test_crawl_data_dir_for_setup = Path(current_test_data_manager.settings["crawl_data"])

        if test_index_dir_for_setup.exists():
            shutil.rmtree(test_index_dir_for_setup)
        test_index_dir_for_setup.mkdir(parents=True, exist_ok=True)

        raw_crawl_dir = test_crawl_data_dir_for_setup / "raw"
        if raw_crawl_dir.exists():
            shutil.rmtree(raw_crawl_dir)
        raw_crawl_dir.mkdir(parents=True, exist_ok=True)

        lookup_file = test_crawl_data_dir_for_setup / "reverse_lookup.json"
        if lookup_file.exists():
            lookup_file.unlink()
        with open(lookup_file, 'w') as f:
            json.dump({}, f)

    def tearDown(self):
        if self.llm_search:
            self.llm_search.close()

    def test_add_file_and_search_integration(self):
        self.mock_ort_model_loader.return_value = MagicMock(spec=ORTModelForCausalLM, config=MagicMock(eos_token_id=2))
        self.mock_auto_tokenizer_loader.return_value = MagicMock(spec=PreTrainedTokenizerBase, eos_token_id=2)
        
        mock_chroma_collection_instance = MagicMock(spec=ChromaCollectionType)
        mock_chroma_collection_instance.count.return_value = 0
        mock_chroma_collection_instance.upsert = MagicMock()
        def mock_chroma_get_general(**kwargs: Any) -> ChromaGetResultType:
            ids = kwargs.get('ids', [])
            return ChromaGetResultType(ids=ids, documents=[None]*len(ids), metadatas=[None]*len(ids)) # type: ignore
        mock_chroma_collection_instance.get.side_effect = mock_chroma_get_general
        self.mock_chromadb_client_constructor.return_value.get_or_create_collection.return_value = mock_chroma_collection_instance

        current_test_data_manager = dm_module.data_manager
        test_crawl_data_dir_for_test = Path(current_test_data_manager.settings["crawl_data"])
        test_index_dir_for_test = Path(current_test_data_manager.settings["index"])

        dummy_query_ids = [['id1', 'id2']]
        dummy_metadatas: List[List[Metadata]] = [[{'source_path': 's1'}], [{'source_path': 's2'}]]
        dummy_docs: List[List[Optional[str]]] = [['doc1'], ['doc2']]
        mock_chroma_collection_instance.query.return_value = ChromaQueryResultType(ids=dummy_query_ids, metadatas=dummy_metadatas, documents=dummy_docs, distances=[[0.1,0.2]], embeddings=None, uris=None, data=None, included=['metadatas', 'documents', 'distances']) # type: ignore

        # Sanity check the PropertyMock setup *before* LLMSearch is initialized
        self.assertIsNotNone(self.mock_embedder_instance.model, "Pre-check: .model (via PropertyMock) is None")
        self.assertIs(self.mock_embedder_instance.model, self.mock_st_model, 
                      "Pre-check: PropertyMock for .model not returning the ST mock")
        self.assertEqual(self.mock_embedder_instance.get_embedding_dimension(), 384)


        logger.info(f"Integration test using index: {test_index_dir_for_test}")
        self.llm_search = LLMSearch(storage_dir=test_index_dir_for_test, shutdown_event=threading.Event())
        
        self.MockEnhancedEmbedderConstructor_class_patch.assert_called_once()
        self.assertIsNotNone(self.llm_search)
        self.assertIs(self.llm_search.embedder, self.mock_embedder_instance)
        
        # Add detailed logging for the embedder and its model as LLMSearch sees it
        logger.info(f"LLMSearch.embedder object ID: {id(self.llm_search.embedder)}")
        logger.info(f"Expected mock_embedder_instance ID: {id(self.mock_embedder_instance)}")
        if hasattr(self.llm_search.embedder, 'model'):
            logger.info(f"LLMSearch.embedder.model object ID: {id(self.llm_search.embedder.model)}") # type: ignore
            logger.info(f"LLMSearch.embedder.model type: {type(self.llm_search.embedder.model)}") # type: ignore
            logger.info(f"LLMSearch.embedder.model itself: {self.llm_search.embedder.model}") # type: ignore
            logger.info(f"Expected mock_st_model ID: {id(self.mock_st_model)}")
        else:
            logger.info("LLMSearch.embedder has NO 'model' attribute just before the assertion.")


        self.assertIsNotNone(self.llm_search.embedder.model, "LLMSearch.embedder.model is None or Falsy after init.")
        self.assertIs(self.llm_search.embedder.model, self.mock_st_model)
        self.assertIsInstance(self.llm_search.embedder.model, SentenceTransformer)
        self.assertEqual(self.llm_search.embedder.get_embedding_dimension(), 384)

        dummy_md_content = "# Title\n" + "Llama text. " * 20
        md_file_path = test_crawl_data_dir_for_test / "llamas_to_index.md"
        md_file_path.write_text(dummy_md_content)
        self.llm_search.max_chunk_size = 60
        self.llm_search.min_chunk_size_filter = 10
        num_expected_chunks = 6
        
        self.mock_embedder_instance.embed_strings.return_value = np.random.rand(num_expected_chunks, 384).astype(np.float32)

        added_chunks, was_blocked = self.llm_search.add_source(str(md_file_path), internal_call=True)
        self.assertFalse(was_blocked)
        self.assertEqual(added_chunks, num_expected_chunks, f"Chunks: got {added_chunks}, expected {num_expected_chunks}")
        self.mock_embedder_instance.embed_strings.assert_called_once()
        mock_chroma_collection_instance.upsert.assert_called_once()

        query = "What is llama?"
        results = self.llm_search.llm_query(query)
        self.mock_load_onnx_llm.return_value.generate.assert_called_once()
        self.assertIn("response", results)
        self.assertIn("Llamas.", results["response"])
        mock_chroma_collection_instance.query.assert_called_once()
        md_file_path.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()