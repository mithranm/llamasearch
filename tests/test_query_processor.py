import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import numpy as np
import logging
import threading
import time
import itertools
import html

# Module to test
from src.llamasearch.core.query_processor import (
    _QueryProcessingMixin,
    _are_chunks_too_similar,
    GenericONNXLLM,
    CHUNK_SIMILARITY_THRESHOLD
)
# Dependent types
from src.llamasearch.protocols import LLM
from src.llamasearch.core.embedder import EnhancedEmbedder
from src.llamasearch.core.bm25 import WhooshBM25Retriever
import chromadb # For chromadb.Collection type hint

# Disable logging during tests unless explicitly testing logging
logging.disable(logging.CRITICAL)

class TestQueryProcessingMixin(unittest.TestCase):

    def setUp(self):
        self.mixin = _QueryProcessingMixin()
        self.mixin.model = MagicMock(spec=GenericONNXLLM)
        self.mixin.embedder = MagicMock(spec=EnhancedEmbedder)
        self.mixin.chroma_collection = MagicMock(spec=chromadb.Collection)
        self.mixin.bm25 = MagicMock(spec=WhooshBM25Retriever)
        self.mixin.max_results = 5
        self.mixin.bm25_weight = 0.5
        self.mixin.vector_weight = 0.5
        self.mixin.context_length = 2048
        self.mixin.debug = False
        self.mixin._shutdown_event = MagicMock(spec=threading.Event)
        self.mixin._shutdown_event.is_set.return_value = False

        # Mock tokenizer for GenericONNXLLM if it's an instance of GenericONNXLLM
        # self.mixin.model is already a mock spec'd as GenericONNXLLM
        # Its _tokenizer attribute will be a new MagicMock by default if accessed.
        # We will set it explicitly in tests if needed or let it be a MagicMock.

    def test_are_chunks_too_similar(self):
        self.assertFalse(_are_chunks_too_similar("", "text2", 0.8))
        self.assertFalse(_are_chunks_too_similar("text1", "", 0.8))
        self.assertTrue(_are_chunks_too_similar("hello world", "hello world", 0.8))
        self.assertTrue(_are_chunks_too_similar("this is a test sentence", "this is test sentence", 0.8))
        self.assertFalse(_are_chunks_too_similar("completely different", "another string entirely", 0.8))
        self.assertTrue(_are_chunks_too_similar("abc", "abd", 0.6))
        self.assertFalse(_are_chunks_too_similar("abc", "abd", 0.9))

    def test_get_token_count_empty_text(self):
        self.assertEqual(self.mixin._get_token_count(""), 0)

    def test_get_token_count_no_model(self):
        self.mixin.model = None
        self.assertEqual(self.mixin._get_token_count("test"), 1) # Fallback

    def test_get_token_count_model_not_generic_onnx_llm(self):
        self.mixin.model = MagicMock(spec=LLM) # Not GenericONNXLLM
        self.assertEqual(self.mixin._get_token_count("test text"), 2) # Fallback

    def test_get_token_count_generic_onnx_llm_no_tokenizer_attr(self):
        # Test case where GenericONNXLLM instance does not have _tokenizer attribute
        mock_model_instance = MagicMock(spec=GenericONNXLLM)
        # Make hasattr(mock_model_instance, "_tokenizer") return False
        # One way is to ensure _tokenizer is not in its __dict__ and not in its spec's attributes that MagicMock would auto-create.
        # If GenericONNXLLM spec implies _tokenizer exists, delattr is needed.
        # If _tokenizer is not part of spec, MagicMock won't create it unless accessed.
        # The simplest is to rely on the `and self.model._tokenizer is not None` part.
        # For this test, let's ensure `_tokenizer` is truly absent by deleting if mock auto-creates.
        try:
            delattr(mock_model_instance, "_tokenizer")
        except AttributeError:
            pass # It was not auto-created, good for this test.
        self.mixin.model = mock_model_instance
        self.assertEqual(self.mixin._get_token_count("test text"), 2) # Fallback

    def test_get_token_count_generic_onnx_llm_tokenizer_is_none(self):
        self.mixin.model = MagicMock(spec=GenericONNXLLM)
        self.mixin.model._tokenizer = None # Explicitly set to None
        self.assertEqual(self.mixin._get_token_count("test text"), 2) # Fallback

    def test_get_token_count_generic_onnx_llm_tokenizer_not_callable(self):
        self.mixin.model = MagicMock(spec=GenericONNXLLM)
        self.mixin.model._tokenizer = "not_callable_string"
        self.assertEqual(self.mixin._get_token_count("test text"), 2) # Fallback

    def test_get_token_count_generic_onnx_llm_tokenizer_call_fails(self):
        self.mixin.model = MagicMock(spec=GenericONNXLLM)
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = Exception("Tokenizer boom!")
        self.mixin.model._tokenizer = mock_tokenizer
        self.mixin.debug = True
        # Suppress logging to console for this specific test if it's noisy
        # logging.disable(logging.WARNING)
        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='WARNING') as log:
            count = self.mixin._get_token_count("test text")
            self.assertEqual(count, 2) # Fallback
            self.assertTrue(any("Tokenizer error: Tokenizer boom!" in msg for msg in log.output))
        # logging.disable(logging.CRITICAL) # Re-disable general logging
        self.mixin.debug = False


    def test_get_token_count_generic_onnx_llm_tokenizer_success(self):
        self.mixin.model = MagicMock(spec=GenericONNXLLM)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        self.mixin.model._tokenizer = mock_tokenizer
        self.assertEqual(self.mixin._get_token_count("some text here"), 5)
        mock_tokenizer.assert_called_once_with("some text here", add_special_tokens=False, truncation=False)

    def test_get_token_count_generic_onnx_llm_tokenizer_success_no_input_ids(self):
        self.mixin.model = MagicMock(spec=GenericONNXLLM)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"attention_mask": [1, 1, 1]} # Missing input_ids
        self.mixin.model._tokenizer = mock_tokenizer
        self.assertEqual(self.mixin._get_token_count("test text"), 2) # Fallback

    def _setup_default_mocks_for_llm_query(self):
        self.mixin.embedder.embed_strings.return_value = np.array([[0.1, 0.2, 0.3]])

        self.mock_chroma_results_data = {
            "ids": [["vec_id1", "vec_id2", "shared_id1"]],
            "documents": [["doc_vec1", "doc_vec2", "doc_shared1"]],
            "metadatas": [[
                {"source_path": "vec_source1", "original_chunk_index": 0},
                {"source_path": "vec_source2", "original_chunk_index": 1},
                {"source_path": "shared_source1", "original_chunk_index": 2}
            ]],
            "distances": [[0.1, 0.2, 0.3]]
        }
        self.mixin.chroma_collection.query.return_value = MagicMock(
            **{f"get.side_effect": lambda k, d=None: self.mock_chroma_results_data.get(k, d)}
        )
        self.mixin.chroma_collection.get.side_effect = self._mock_chroma_get

        self.mixin.bm25.query.return_value = {
            "ids": ["bm25_id1", "shared_id1", "bm25_id2"],
            "scores": [0.9, 0.8, 0.7],
        }
        self.mixin.model.generate.return_value = ("Generated response", {"tokens_generated": 10})

        self.mock_get_token_count_patch = patch.object(self.mixin, '_get_token_count', autospec=True)
        self.started_mock_get_token_count = self.mock_get_token_count_patch.start()
        self.addCleanup(self.mock_get_token_count_patch.stop)
        self.started_mock_get_token_count.return_value = 10 # Default simple token count

        self.mock_log_query_patch = patch('src.llamasearch.core.query_processor.log_query', autospec=True)
        self.started_mock_log_query = self.mock_log_query_patch.start()
        self.addCleanup(self.mock_log_query_patch.stop)

    def _mock_chroma_get(self, ids, include):
        results_docs, results_metas = [], []
        doc_map = {
            "vec_id1": ("doc_vec1_fetched", {"source_path": "vec_source1_fetched", "original_chunk_index": 0}),
            "vec_id2": ("doc_vec2_fetched", {"source_path": "vec_source2_fetched", "original_chunk_index": 1}),
            "shared_id1": ("doc_shared1_fetched", {"source_path": "shared_source1_fetched", "original_chunk_index": 2}),
            "bm25_id1": ("doc_bm25_1_fetched", {"source_path": "bm25_source1_fetched", "original_chunk_index": 3}),
            "bm25_id2": ("doc_bm25_2_fetched", {"source_path": "bm25_source2_fetched", "original_chunk_index": 4}),
        }
        for an_id in ids:
            doc, meta = doc_map.get(an_id, (None, None))
            results_docs.append(doc)
            results_metas.append(meta)
        return {"documents": results_docs, "metadatas": results_metas, "ids": ids}

    @patch('time.time', side_effect=itertools.count(1000, 0.1)) # Auto-incrementing time
    def test_llm_query_happy_path(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.debug = False
        self.mixin.max_results = 2
        self.started_mock_get_token_count.side_effect = lambda x: max(1, len(x.split())) # Realistic token count

        query_text = "test query"
        result = self.mixin.llm_query(query_text)

        self.assertEqual(result["response"], "Generated response")
        self.assertIn("Retrieved Context", result["formatted_response"])
        self.assertNotIn("debug_info", result["debug_info"]) # debug_info is {} if debug=False
        self.assertGreater(result["query_time_seconds"], 0)
        self.assertGreater(result["generation_time_seconds"], 0)

        self.mixin.embedder.embed_strings.assert_called_once_with([query_text], input_type="query")

        num_candidates = max(self.mixin.max_results * 5, 25)
        vector_weight_ratio = self.mixin.vector_weight / (self.mixin.vector_weight + self.mixin.bm25_weight + 1e-9)
        expected_n_vec = max(1, int(num_candidates * vector_weight_ratio))
        expected_n_bm25 = max(1, int(num_candidates * (1.0 - vector_weight_ratio)))

        self.mixin.chroma_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=expected_n_vec,
            include=["metadatas", "documents", "distances"]
        )
        self.mixin.bm25.query.assert_called_once_with(query_text, n_results=expected_n_bm25)

        self.mixin.model.generate.assert_called_once()
        llm_call_args_kwargs = self.mixin.model.generate.call_args[1]
        self.assertIn(query_text, llm_call_args_kwargs['prompt'])
        # Check that some context from mocked docs is present
        self.assertTrue("doc_shared1" in llm_call_args_kwargs['prompt'] or "doc_bm25_1" in llm_call_args_kwargs['prompt'])

        self.started_mock_log_query.assert_called_once()
        log_call_args = self.started_mock_log_query.call_args[0]
        logged_chunks = log_call_args[1] # chunks
        self.assertEqual(len(logged_chunks), 2) # max_results


    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_happy_path_debug_mode(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.debug = True # Instance debug
        self.mixin.max_results = 1
        self.started_mock_get_token_count.return_value = 5

        result = self.mixin.llm_query("debug query", debug_mode=True) # Explicitly pass debug_mode

        self.assertEqual(result["response"], "Generated response")
        self.assertIsInstance(result["debug_info"], dict)
        self.assertGreater(len(result["debug_info"]), 0)
        expected_keys = [
            "query_embedding_time", "retrieval_time", "vector_initial_results",
            "bm25_initial_results", "combined_unique_chunks", "final_selected_chunk_count",
            "final_context_content_token_count", "llm_generation_time", "chunk_details",
            "total_query_processing_time"
        ]
        for key in expected_keys:
            self.assertIn(key, result["debug_info"])
        self.started_mock_log_query.assert_called_once()
        self.assertTrue(self.started_mock_log_query.call_args[1]['full_logging'])


    def test_llm_query_assertions_not_initialized(self):
        query_text = "test"
        for attr_name, error_msg_part in [
            ("model", "LLM not initialized"),
            ("embedder", "Embedder not initialized"),
            ("chroma_collection", "Chroma collection not initialized"),
            ("bm25", "BM25 retriever not initialized"),
        ]:
            original_attr = getattr(self.mixin, attr_name)
            setattr(self.mixin, attr_name, None)
            with self.assertRaisesRegex(AssertionError, error_msg_part):
                self.mixin.llm_query(query_text)
            setattr(self.mixin, attr_name, original_attr) # Restore


    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_embedder_fails_value_error(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.embedder.embed_strings.side_effect = ValueError("Embedding failed")
        self.mixin.debug = True
        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='ERROR') as log:
            result = self.mixin.llm_query("query")
            self.assertTrue(any("Retrieval phase failed: Embedding failed" in msg for msg in log.output))
        # Should proceed with BM25 if possible
        self.assertEqual(result["response"], "Generated response") # From BM25
        self.mixin.chroma_collection.query.assert_not_called()
        self.assertEqual(result["debug_info"]["vector_initial_results"], 0)


    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_no_results_from_retrieval(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.chroma_collection.query.return_value = MagicMock(
            **{f"get.side_effect": lambda k, d=None: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}.get(k,d)}
        )
        self.mixin.bm25.query.return_value = {"ids": [], "scores": []}

        result = self.mixin.llm_query("query")
        self.assertEqual(result["response"], "Could not find relevant information.")
        self.mixin.model.generate.assert_not_called()

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_context_truncation(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.max_results = 3
        self.mixin.context_length = 100 # Small to force truncation
        self.mixin.debug = True

        # instruction_len ~10 (mocked), template_overhead_estimate=50, generation_reservation=max(300, 100//4)=300
        # available_context_tokens = 100 - 10 - 50 - 300 -> 0
        self.started_mock_get_token_count.side_effect = lambda text: 30 if "Context" not in text and "Query" not in text else (10 if "Query" in text else 500)


        result = self.mixin.llm_query("query")
        self.assertEqual(result["response"], "Could not find relevant information.")
        self.assertIn("context_truncated_at_chunk_rank", result["debug_info"])
        self.assertEqual(result["debug_info"]["context_truncated_at_chunk_rank"], 1)
        self.mixin.model.generate.assert_not_called()

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_estimated_prompt_too_long(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.debug = True
        # Force prompt_for_llm to be very long via _get_token_count
        def mock_token_count_for_prompt(text):
            if "Context:" in text: # This is prompt_for_llm
                return self.mixin.context_length + 100
            return 10 # For instruction and individual chunks
        self.started_mock_get_token_count.side_effect = mock_token_count_for_prompt

        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='ERROR') as log:
            result = self.mixin.llm_query("query")
            self.assertTrue(any("Estimated prompt too long" in msg for msg in log.output))

        self.assertIn("Error: Estimated prompt too long", result["response"])
        self.mixin.model.generate.assert_not_called()

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_llm_generate_exception(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.model.generate.side_effect = Exception("LLM boom!")
        self.mixin.debug = True
        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='ERROR') as log:
            result = self.mixin.llm_query("query")
            self.assertTrue(any("LLM generation error: LLM boom!" in msg for msg in log.output))
        self.assertEqual(result["response"], "LLM Error: LLM boom!")

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_llm_returns_empty_response(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.model.generate.return_value = ("  ", {"meta": "data"}) # Whitespace
        self.mixin.debug = True
        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='WARNING') as log:
            result = self.mixin.llm_query("query")
            self.assertTrue(any("(LLM returned empty response after splitting)" in msg for msg in log.output))
        self.assertEqual(result["response"], "(LLM returned empty response after splitting)")

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_shutdown_event_set_before_llm(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin._shutdown_event.is_set.return_value = True # Shutdown signaled
        self.mixin.debug = True
        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='WARNING') as log:
            result = self.mixin.llm_query("query")
            # The InterruptedError will be caught and logged inside self.model.generate if it's called.
            # However, the check `if self._shutdown_event and self._shutdown_event.is_set():` is before the call.
            self.assertTrue(any("LLM generation cancelled during shutdown." in msg for msg in log.output))

        self.assertEqual(result["response"], "LLM generation cancelled during shutdown.")
        self.mixin.model.generate.assert_not_called() # Crucial: LLM not called

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_chroma_get_incomplete_data_for_bm25_chunk(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        self.mixin.chroma_collection.query.return_value = MagicMock(
             **{f"get.side_effect": lambda k, d=None: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}.get(k,d)}
        )
        self.mixin.bm25.query.return_value = {"ids": ["bm25_only_id"], "scores": [0.9]}
        self.mixin.chroma_collection.get.return_value = {
            "documents": [None], "metadatas": [[{"s": "m"}]], "ids": ["bm25_only_id"]
        }
        self.mixin.debug = True
        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='WARNING') as log:
            result = self.mixin.llm_query("query")
            self.assertTrue(any("Chroma fetch for bm25_only_id failed/incomplete." in msg for msg in log.output))
        self.assertEqual(result["response"], "Could not find relevant information.")

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_skip_empty_chunk_text(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        # Provide one chunk that has empty text
        self.mock_chroma_results_data = {
            "ids": [["empty_id", "valid_id"]],
            "documents": [["   ", "valid_doc"]], # First doc is whitespace only
            "metadatas": [[{"source_path": "s_empty"}, {"source_path": "s_valid"}]],
            "distances": [[0.1, 0.2]]
        }
        self.mixin.chroma_collection.query.return_value = MagicMock(
            **{f"get.side_effect": lambda k, d=None: self.mock_chroma_results_data.get(k, d)}
        )
        self.mixin.bm25.query.return_value = {"ids": [], "scores": []} # No BM25 to simplify
        self.mixin.debug = True
        self.mixin.max_results = 1

        # To capture DEBUG logs, need to set level for that logger
        logger_qp = logging.getLogger('src.llamasearch.core.query_processor')
        original_level = logger_qp.level
        logger_qp.setLevel(logging.DEBUG)

        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='DEBUG') as log:
            result = self.mixin.llm_query("query")
            self.assertTrue(any("Skipping chunk empty_id: Empty text." in msg for msg in log.output))

        logger_qp.setLevel(original_level) # Restore level

        # "valid_id" should still be processed
        self.assertEqual(result["response"], "Generated response")
        self.started_mock_log_query.assert_called_once()
        logged_chunks = self.started_mock_log_query.call_args[0][1]
        self.assertEqual(len(logged_chunks), 1)
        self.assertEqual(logged_chunks[0]['id'], 'valid_id')

    @patch('time.time', side_effect=itertools.count(1000, 0.1))
    def test_llm_query_final_context_empty_error(self, mock_time):
        self._setup_default_mocks_for_llm_query()
        # All chunks are valid, but token counts make them too large for context
        self.started_mock_get_token_count.return_value = self.mixin.context_length + 1000
        self.mixin.debug = True

        with self.assertLogs(logger='src.llamasearch.core.query_processor', level='ERROR') as log:
             result = self.mixin.llm_query("query")
             self.assertTrue(any("Error: Could not build context from selected chunks." in msg for msg in log.output))

        self.assertEqual(result["response"], "Error: Could not build context from selected chunks.")
        self.assertIn("final_context_content_token_count", result["debug_info"])
        self.assertEqual(result["debug_info"]["final_context_content_token_count"], 0)
        # Chunks were selected but couldn't be added to context
        self.assertGreater(result["debug_info"]["final_selected_chunk_count"], 0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)