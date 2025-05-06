# tests/test_query_processor.py
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import threading
import numpy as np
from typing import Optional, Dict, List, Any 

from llamasearch.core.query_processor import _QueryProcessingMixin, _are_chunks_too_similar
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.bm25 import WhooshBM25Retriever
from llamasearch.core.onnx_model import GenericONNXLLM, GenericONNXModelInfo
from chromadb import Collection as ChromaCollection
from chromadb.api.types import QueryResult as ChromaQueryResultType, GetResult as ChromaGetResultType


# Dummy class that uses the mixin
class DummySearcher(_QueryProcessingMixin):
    doc_lookup_default: Dict[str, Dict[str, Any]] 

    def __init__(self):
        self.model: Optional[GenericONNXLLM] = MagicMock(spec=GenericONNXLLM)
        if self.model: 
            self.model._tokenizer = MagicMock() 
            if self.model._tokenizer is not None:
                 self.model._tokenizer.return_value = {"input_ids": []} 

            self.model.generate = MagicMock(return_value=("LLM Response", {}))
            
            mock_model_info_instance = MagicMock(spec=GenericONNXModelInfo)
            type(mock_model_info_instance).context_length = PropertyMock(return_value=4096)
            self.model.model_info = mock_model_info_instance


        self.embedder: Optional[EnhancedEmbedder] = MagicMock(spec=EnhancedEmbedder)
        self.chroma_collection: Optional[ChromaCollection] = MagicMock(spec=ChromaCollection)
        self.bm25: Optional[WhooshBM25Retriever] = MagicMock(spec=WhooshBM25Retriever)
        
        self.max_results: int = 3
        self.bm25_weight: float = 0.4
        self.vector_weight: float = 0.6
        self.context_length: int = 4096 
        if self.model and self.model.model_info: 
             self.context_length = self.model.model_info.context_length 

        self.debug: bool = False
        self._shutdown_event: Optional[threading.Event] = MagicMock(spec=threading.Event)
        if self._shutdown_event: 
            self._shutdown_event.is_set.return_value = False
        
        self.doc_lookup_default = {}


class TestQueryProcessingMixin(unittest.TestCase):

    def setUp(self):
        self.searcher = DummySearcher()
        if self.searcher.model: 
            self.searcher.model.generate.reset_mock(return_value=True, side_effect=True)
            if self.searcher.model._tokenizer is not None: 
                self.searcher.model._tokenizer.reset_mock(return_value=True, side_effect=True) #type: ignore
                self.searcher.model._tokenizer.side_effect = lambda text, **kwargs: {"input_ids": list(range(max(1, len(text) // 4)))} 
            self.searcher.model.generate.return_value = ("LLM Response", {"output_token_count": 5})


        if self.searcher.embedder: 
            self.searcher.embedder.embed_strings.reset_mock(return_value=True, side_effect=True)
            self.searcher.embedder.embed_strings.return_value = np.array([[0.1, 0.2, 0.3]])
        if self.searcher.chroma_collection: 
            self.searcher.chroma_collection.query.reset_mock(return_value=True, side_effect=True)
            self.searcher.chroma_collection.get.reset_mock(return_value=True, side_effect=True)
            
            query_result: ChromaQueryResultType = {
                'ids': [['id1', 'id2']],
                'distances': [[0.1, 0.2]],
                'metadatas': [[{'source_path': 'path1', 'filename': 'f1.md'}, {'source_path': 'path2', 'filename':'f2.md'}]],
                'documents': [['doc1 text', 'doc2 text']],
                'uris': None, 'data': None, 'embeddings': None, 'included': ['metadatas', 'documents', 'distances'] 
            }
            self.searcher.chroma_collection.query.return_value = query_result

        if self.searcher.bm25: 
            self.searcher.bm25.query.reset_mock(return_value=True, side_effect=True)
            self.searcher.bm25.query.return_value = {
                "ids": ["id_bm25_1", "id_bm25_2"],
                "scores": [10.0, 9.0],
                "documents": [None, None] 
            }
        if self.searcher._shutdown_event: 
            self.searcher._shutdown_event.is_set.return_value = False

        self.searcher.doc_lookup_default = {
            'id1': {'document': 'doc1 text', 'metadata': {'source_path': 'path1', 'filename': 'file1.md', 'original_chunk_index': 0}},
            'id2': {'document': 'doc2 text', 'metadata': {'source_path': 'path2', 'filename': 'file2.md', 'original_chunk_index': 1}},
            'id_bm25_1': {'document': 'bm25 doc1 text', 'metadata': {'source_path': 'path_bm25_1', 'filename': 'bm25_file1.md', 'original_chunk_index': 0}},
            'id_bm25_2': {'document': 'bm25 doc2 text', 'metadata': {'source_path': 'path_bm25_2', 'filename': 'bm25_file2.md', 'original_chunk_index': 3}},
        }


    @patch('llamasearch.core.query_processor.log_query')
    @patch('llamasearch.core.query_processor.time.time') 
    def test_llm_query_success(self, mock_time_time, mock_log_query): 
        # Use a function that returns increasing values instead of a fixed list
        current_time = 100.0
        def time_side_effect():
            nonlocal current_time
            current_time += 0.05
            return current_time
        mock_time_time.side_effect = time_side_effect
        
        self.searcher.debug = True
        if self.searcher.chroma_collection:
            query_return_val: ChromaQueryResultType = {
                'ids': [['id1', 'id2']], 'distances': [[0.1,0.2]], 
                'metadatas': [[
                    self.searcher.doc_lookup_default['id1']['metadata'], 
                    self.searcher.doc_lookup_default['id2']['metadata']
                ]], 
                'documents': [['doc1 text', 'doc2 text']],
                'uris':None, 'data':None, 'embeddings':None, 'included':['metadatas','documents','distances']
            }
            self.searcher.chroma_collection.query.return_value = query_return_val
        if self.searcher.bm25:
            self.searcher.bm25.query.return_value['ids'] = ['id_bm25_1'] # type: ignore

        if self.searcher.chroma_collection:
            def get_side_effect(ids, **kwargs):
                results_docs: List[Optional[str]] = []
                results_metas: List[Optional[Dict[str,Any]]] = []
                for _id in ids:
                    if _id in self.searcher.doc_lookup_default:
                        results_docs.append(self.searcher.doc_lookup_default[_id]['document'])
                        results_metas.append(self.searcher.doc_lookup_default[_id]['metadata'])
                    else: 
                        results_docs.append(None)
                        results_metas.append(None)
                
                get_result: ChromaGetResultType = {'ids':ids, 'documents':results_docs, 'metadatas':results_metas, 'embeddings':None, 'uris':None, 'data':None, 'included':['documents', 'metadatas']} # type: ignore
                return get_result
            self.searcher.chroma_collection.get.side_effect = get_side_effect


        with patch('llamasearch.core.query_processor._are_chunks_too_similar', return_value=False):
            results = self.searcher.llm_query("test query", debug_mode=True)

        self.assertIn("response", results)
        self.assertEqual(results["response"], "LLM Response")
        
        # Test for approximate time since we're using an incrementing timer
        self.assertIsInstance(results["query_time_seconds"], float)
        self.assertIsInstance(results["generation_time_seconds"], float)
        self.assertGreater(results["query_time_seconds"], 0)
        self.assertGreater(results["generation_time_seconds"], 0)
        
        debug_info = results["debug_info"]
        self.assertTrue("query_embedding_time" in debug_info)
        self.assertTrue("retrieval_time" in debug_info)
        self.assertTrue("llm_generation_time" in debug_info)
        
        if self.searcher.model: 
            self.searcher.model.generate.assert_called_once()
            # Verify LLM prompt contents
            llm_prompt_arg = self.searcher.model.generate.call_args[1]['prompt']
            self.assertIn("User Question", llm_prompt_arg)
            self.assertIn("test query", llm_prompt_arg)
            self.assertIn("Context", llm_prompt_arg)


    @patch('llamasearch.core.query_processor.log_query')
    @patch('llamasearch.core.query_processor.time.time')
    def test_llm_query_no_context_found(self, mock_time_time, mock_log_query):
        # Use a function that returns increasing values instead of a fixed list
        current_time = 200.0
        def time_side_effect():
            nonlocal current_time
            current_time += 0.05
            return current_time
        mock_time_time.side_effect = time_side_effect

        if self.searcher.chroma_collection: 
            no_results_query: ChromaQueryResultType = {
                'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]], 
                'uris': None, 'data': None, 'embeddings': None, 'included': []
            }
            self.searcher.chroma_collection.query.return_value = no_results_query
        if self.searcher.bm25: 
            self.searcher.bm25.query.return_value = {"ids": [], "scores": [], "documents": []}

        results = self.searcher.llm_query("query with no context", debug_mode=False)

        self.assertEqual(results["response"], "Could not find relevant information.")
        self.assertEqual(results["retrieved_context"], "")
        self.assertEqual(results["formatted_response"], "...")
        self.assertGreater(results["query_time_seconds"], 0)
        self.assertEqual(results["generation_time_seconds"], 0)
        
        if self.searcher.model: 
            self.searcher.model.generate.assert_not_called()
        mock_log_query.assert_not_called() 

    @patch('llamasearch.core.query_processor.log_query')
    def test_llm_query_llm_error(self, mock_log_query):
        if self.searcher.model: 
            self.searcher.model.generate.return_value = ("LLM Error: Something went wrong", {"error": "Something went wrong"})

        results = self.searcher.llm_query("test query causing LLM error")
        
        self.assertEqual(results["response"], "LLM Error: Something went wrong")
        mock_log_query.assert_called_once()

    @patch('llamasearch.core.query_processor._are_chunks_too_similar', return_value=True) 
    @patch('llamasearch.core.query_processor.log_query')
    def test_llm_query_all_chunks_too_similar(self, mock_log_query, mock_similar_check):
        num_initial_candidates = self.searcher.max_results * 3 
        if self.searcher.chroma_collection: 
             similar_query_results: ChromaQueryResultType = {
                'ids': [[f'id{i}' for i in range(num_initial_candidates)]],
                'distances': [[0.1 + i*0.01 for i in range(num_initial_candidates)]],
                'metadatas': [[{'source_path': f'p{i}', 'filename':f'f{i}.md'}] for i in range(num_initial_candidates)], 
                'documents': [['doc text similar a'] * num_initial_candidates], 
                'uris': None, 'data': None, 'embeddings': None, 'included': ['metadatas', 'documents', 'distances']
            }
             self.searcher.chroma_collection.query.return_value = similar_query_results
        if self.searcher.bm25: 
            self.searcher.bm25.query.return_value = {"ids": [], "scores": [], "documents": []} 

        if self.searcher.chroma_collection: 
            def mock_chroma_get(ids: List[str], **kwargs) -> ChromaGetResultType:
                if ids and ids[0].startswith('id'):
                    get_res: ChromaGetResultType = {'ids':[ids[0]], 'documents':['doc text similar a'], 'metadatas':[{'source_path': 'p_get', 'filename':'f_get.md'}], 'embeddings':None, 'uris':None, 'data':None, 'included': ['metadatas', 'documents']}
                    return get_res
                empty_res: ChromaGetResultType = {'ids':[], 'documents':[], 'metadatas':[], 'embeddings':None, 'uris':None, 'data':None, 'included': []}
                return empty_res
            self.searcher.chroma_collection.get.side_effect = mock_chroma_get
        
        results = self.searcher.llm_query("test similarity filter", debug_mode=True)

        self.assertIn("LLM Response", results["response"]) 
        self.assertEqual(results["debug_info"]["skipped_similar_chunk_count"], num_initial_candidates - 1)
        self.assertEqual(results["debug_info"]["final_selected_chunk_count"], 1) 
        self.assertGreater(mock_similar_check.call_count, 0)

    @patch('llamasearch.core.query_processor.log_query')
    def test_llm_query_shutdown_event(self, mock_log_query):
        if self.searcher._shutdown_event: 
            self.searcher._shutdown_event.is_set.return_value = True 

        if self.searcher.model: 
            self.searcher.model.generate.side_effect = InterruptedError("Shutdown during LLM")
        
        results = self.searcher.llm_query("test query during shutdown")

        self.assertEqual(results["response"], "LLM generation cancelled during shutdown.")
        self.assertGreaterEqual(results["query_time_seconds"], 0)
        self.assertGreaterEqual(results["generation_time_seconds"], 0)


    def test_get_token_count_with_tokenizer(self):
        if self.searcher.model and self.searcher.model._tokenizer is not None: 
            self.searcher.model._tokenizer.side_effect = None 
            self.searcher.model._tokenizer.return_value = {"input_ids": [1,2,3,4,5]}
            count = self.searcher._get_token_count("some text")
            self.assertEqual(count, 5)
            self.searcher.model._tokenizer.assert_called_once_with("some text", add_special_tokens=False, truncation=False) #type: ignore

    def test_get_token_count_tokenizer_error(self):
        if self.searcher.model and self.searcher.model._tokenizer is not None: 
            self.searcher.model._tokenizer.side_effect = Exception("Tokenizer broke") 
            count = self.searcher._get_token_count("some text with error")
            self.assertEqual(count, len("some text with error") // 4) 

    def test_get_token_count_no_model_or_tokenizer(self):
        self.searcher.model = None 
        count = self.searcher._get_token_count("text no model")
        self.assertEqual(count, len("text no model") // 4)

        self.setUp() 
        if self.searcher.model: 
            self.searcher.model._tokenizer = None 
            count = self.searcher._get_token_count("text no tokenizer")
            self.assertEqual(count, len("text no tokenizer") // 4)


    def test_are_chunks_too_similar(self):
        self.assertTrue(_are_chunks_too_similar("apple banana cherry grape", "apple banana cherry date", 0.85)) 
        self.assertFalse(_are_chunks_too_similar("apple banana cherry", "apple banana kiwi", 0.90)) 
        self.assertFalse(_are_chunks_too_similar("apple banana cherry", "dog cat bird", 0.80))
        self.assertFalse(_are_chunks_too_similar("", "text", 0.80))
        self.assertFalse(_are_chunks_too_similar("text", "", 0.80))


if __name__ == '__main__':
    unittest.main()