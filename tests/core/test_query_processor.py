# llamasearch/tests/test_query_processor.py
import threading
import unittest
from typing import Any, Dict, List, Optional, cast
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
from chromadb import Collection as ChromaCollection
from chromadb.api.types import Document
from chromadb.api.types import GetResult as ChromaGetResultType
from chromadb.api.types import Metadata
from chromadb.api.types import QueryResult as ChromaQueryResultType

from llamasearch.core.bm25 import WhooshBM25Retriever
from llamasearch.core.embedder import EnhancedEmbedder
from llamasearch.core.onnx_model import GenericONNXLLM, GenericONNXModelInfo
from llamasearch.core.query_processor import (_are_chunks_too_similar,
                                              _QueryProcessingMixin)


# Dummy class that uses the mixin
class DummySearcher(_QueryProcessingMixin):
    doc_lookup_default: Dict[str, Dict[str, Any]]

    def __init__(self):
        self.model: MagicMock = MagicMock(
            spec=GenericONNXLLM
        )  # Changed to MagicMock for easier attribute setting

        # Setup _tokenizer as a MagicMock attribute of self.model
        self.model._tokenizer = MagicMock()
        self.model._tokenizer.return_value = {"input_ids": []}

        self.model.generate = MagicMock(return_value=("LLM Response", {}))

        mock_model_info_instance = MagicMock(spec=GenericONNXModelInfo)
        type(mock_model_info_instance).context_length = PropertyMock(return_value=4096)  # type: ignore
        type(mock_model_info_instance).model_id = PropertyMock(return_value="dummy-onnx")  # type: ignore
        self.model.model_info = mock_model_info_instance

        self.embedder: Optional[EnhancedEmbedder] = MagicMock(spec=EnhancedEmbedder)
        self.chroma_collection: Optional[ChromaCollection] = MagicMock(
            spec=ChromaCollection
        )
        self.bm25: Optional[WhooshBM25Retriever] = MagicMock(spec=WhooshBM25Retriever)

        self.max_results: int = 3
        self.bm25_weight: float = 0.4
        self.vector_weight: float = 0.6

        # Ensure context_length is set based on the mocked model_info
        self.context_length: int = (
            self.model.model_info.context_length
            if self.model and self.model.model_info
            else 4096
        )

        self.debug: bool = False
        self._shutdown_event: Optional[threading.Event] = MagicMock(
            spec=threading.Event
        )
        if self._shutdown_event:
            cast(MagicMock, self._shutdown_event).is_set.return_value = False  # type: ignore

        self.doc_lookup_default = {}

        # Attributes for _QueryProcessingMixin generation parameters
        self.temperature: float = 0.1
        self.top_p: float = 1.0
        self.top_k: int = 50
        self.repetition_penalty: float = 1.0


class TestQueryProcessingMixin(unittest.TestCase):

    def setUp(self):
        self.searcher = DummySearcher()

        self.searcher.model.generate.reset_mock(return_value=True, side_effect=True)  # type: ignore

        cast(MagicMock, self.searcher.model._tokenizer).reset_mock(
            return_value=True, side_effect=True
        )
        cast(MagicMock, self.searcher.model._tokenizer).side_effect = (
            lambda text, **kwargs: {"input_ids": list(range(max(1, len(text) // 4)))}
        )
        self.searcher.model.generate.return_value = (
            "LLM Response",
            {"output_token_count": 5},
        )

        if self.searcher.embedder:
            self.searcher.embedder.embed_strings.reset_mock(return_value=True, side_effect=True)  # type: ignore
            self.searcher.embedder.embed_strings.return_value = np.array([[0.1, 0.2, 0.3]])  # type: ignore
        if self.searcher.chroma_collection:
            self.searcher.chroma_collection.query.reset_mock(return_value=True, side_effect=True)  # type: ignore
            self.searcher.chroma_collection.get.reset_mock(return_value=True, side_effect=True)  # type: ignore

            # Ensure query_result keys match QueryResult TypedDict
            query_result: ChromaQueryResultType = {
                "ids": [["id1", "id2"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [
                    [
                        {"source_path": "path1", "filename": "f1.md"},
                        {"source_path": "path2", "filename": "f2.md"},
                    ]
                ],
                "documents": [["doc1 text", "doc2 text"]],
                "uris": None,
                "data": None,
                "embeddings": None,
                "included": ["metadatas", "documents", "distances"],
            }
            self.searcher.chroma_collection.query.return_value = query_result  # type: ignore

        if self.searcher.bm25:
            self.searcher.bm25.query.reset_mock(return_value=True, side_effect=True)  # type: ignore
            self.searcher.bm25.query.return_value = {  # type: ignore
                "ids": ["id_bm25_1", "id_bm25_2"],
                "scores": [10.0, 9.0],
                "documents": [None, None],
            }
        if self.searcher._shutdown_event:
            cast(MagicMock, self.searcher._shutdown_event).is_set.return_value = False  # type: ignore

        self.searcher.doc_lookup_default = {
            "id1": {
                "document": "doc1 text",
                "metadata": {
                    "source_path": "path1",
                    "filename": "file1.md",
                    "original_chunk_index": 0,
                },
            },
            "id2": {
                "document": "doc2 text",
                "metadata": {
                    "source_path": "path2",
                    "filename": "file2.md",
                    "original_chunk_index": 1,
                },
            },
            "id_bm25_1": {
                "document": "bm25 doc1 text",
                "metadata": {
                    "source_path": "path_bm25_1",
                    "filename": "bm25_file1.md",
                    "original_chunk_index": 0,
                },
            },
            "id_bm25_2": {
                "document": "bm25 doc2 text",
                "metadata": {
                    "source_path": "path_bm25_2",
                    "filename": "bm25_file2.md",
                    "original_chunk_index": 3,
                },
            },
        }

    @patch("llamasearch.core.query_processor.log_query")
    @patch("llamasearch.core.query_processor.time.time")
    def test_llm_query_success(self, mock_time_time, mock_log_query):
        current_time = 100.0

        def time_side_effect():
            nonlocal current_time
            current_time += 0.05
            return current_time

        mock_time_time.side_effect = time_side_effect

        self.searcher.debug = True
        if self.searcher.chroma_collection:
            doc1_meta = self.searcher.doc_lookup_default["id1"]["metadata"]
            doc2_meta = self.searcher.doc_lookup_default["id2"]["metadata"]

            query_return_val: ChromaQueryResultType = {
                "ids": [["id1", "id2"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [[cast(Metadata, doc1_meta), cast(Metadata, doc2_meta)]],
                "documents": [["doc1 text", "doc2 text"]],
                "uris": None,
                "data": None,
                "embeddings": None,
                "included": ["metadatas", "documents", "distances"],
            }
            self.searcher.chroma_collection.query.return_value = query_return_val  # type: ignore
        if self.searcher.bm25:
            self.searcher.bm25.query.return_value["ids"] = ["id_bm25_1"]  # type: ignore

        if self.searcher.chroma_collection:

            def get_side_effect(ids, **kwargs) -> ChromaGetResultType:
                results_docs: List[Optional[Document]] = []
                results_metas: List[Optional[Metadata]] = []
                final_ids_get: List[str] = []

                for _id_val in ids:
                    final_ids_get.append(_id_val)
                    if _id_val in self.searcher.doc_lookup_default:
                        results_docs.append(
                            cast(
                                Document,
                                self.searcher.doc_lookup_default[_id_val]["document"],
                            )
                        )
                        results_metas.append(
                            cast(
                                Metadata,
                                self.searcher.doc_lookup_default[_id_val]["metadata"],
                            )
                        )
                    else:
                        results_docs.append(None)
                        results_metas.append(None)

                get_result: ChromaGetResultType = {"ids": final_ids_get, "documents": results_docs, "metadatas": results_metas, "embeddings": None, "uris": None, "data": None, "included": ["documents", "metadatas"]}  # type: ignore
                return get_result

            self.searcher.chroma_collection.get.side_effect = get_side_effect  # type: ignore

        with patch(
            "llamasearch.core.query_processor._are_chunks_too_similar",
            return_value=False,
        ):
            results = self.searcher.llm_query("test query", debug_mode=True)

        self.assertIn("response", results)
        self.assertEqual(results["response"], "LLM Response")

        self.assertIsInstance(results["query_time_seconds"], float)
        # FIX: Changed key name
        self.assertIsInstance(results["generation_time_sec"], float)
        self.assertGreater(results["query_time_seconds"], 0)
        # FIX: Changed key name
        self.assertGreater(results["generation_time_sec"], 0)

        debug_info = results["debug_info"]
        self.assertTrue("query_embedding_time" in debug_info)
        self.assertTrue("retrieval_time" in debug_info)
        self.assertTrue("llm_generation_time" in debug_info)

        self.searcher.model.generate.assert_called_once()  # type: ignore
        llm_prompt_arg = self.searcher.model.generate.call_args[1]["prompt"]  # type: ignore
        self.assertIn("User Question", llm_prompt_arg)
        self.assertIn("test query", llm_prompt_arg)
        self.assertIn("Context", llm_prompt_arg)
        self.assertIn("formatted_response", results)
        self.assertIn("LLM Response", results["formatted_response"])
        self.assertIn("Retrieved Sources", results["formatted_response"])
        mock_log_query.assert_called_once()  # Verify logging happens on success

    @patch("llamasearch.core.query_processor.log_query")
    @patch("llamasearch.core.query_processor.time.time")
    def test_llm_query_no_context_found(self, mock_time_time, mock_log_query):
        current_time = 200.0

        def time_side_effect():
            nonlocal current_time
            current_time += 0.05
            return current_time

        mock_time_time.side_effect = time_side_effect

        if self.searcher.chroma_collection:
            no_results_query: ChromaQueryResultType = {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "uris": None,
                "data": None,
                "embeddings": None,
                "included": [],
            }
            self.searcher.chroma_collection.query.return_value = no_results_query  # type: ignore
        if self.searcher.bm25:
            self.searcher.bm25.query.return_value = {"ids": [], "scores": [], "documents": []}  # type: ignore

        results = self.searcher.llm_query("query with no context", debug_mode=False)

        self.assertEqual(results["response"], "Could not find relevant information.")
        self.assertEqual(results["retrieved_context"], [])  # Changed from ""
        self.assertIn(
            "Could not find relevant information.", results["formatted_response"]
        )
        self.assertGreater(results["query_time_seconds"], 0)
        # FIX: Check the key is present and has the default value 0.0
        self.assertIn("generation_time_sec", results)
        self.assertEqual(results["generation_time_sec"], 0.0)

        self.searcher.model.generate.assert_not_called()  # type: ignore
        mock_log_query.assert_not_called()

    @patch("llamasearch.core.query_processor.log_query")
    def test_llm_query_llm_error(self, mock_log_query):
        # Simulate the model's generate method raising an exception instead of returning error text
        error_msg = "Something went wrong in LLM"
        self.searcher.model.generate.side_effect = RuntimeError(error_msg)  # type: ignore

        results = self.searcher.llm_query("test query causing LLM error")

        # The response text now includes the exception message
        self.assertEqual(results["response"], f"LLM Error: {error_msg}")
        self.assertIn("generation_time_sec", results)
        self.assertGreaterEqual(
            results["generation_time_sec"], 0.0
        )  # Time might be >0 if error is late
        mock_log_query.assert_called_once()  # Log should still happen

    @patch(
        "llamasearch.core.query_processor._are_chunks_too_similar", return_value=True
    )
    @patch("llamasearch.core.query_processor.log_query")
    def test_llm_query_all_chunks_too_similar(self, mock_log_query, mock_similar_check):
        num_initial_candidates = self.searcher.max_results * 3
        if self.searcher.chroma_collection:
            # Ensure metadatas is List[List[Optional[Metadata]]]
            metadatas_similar: List[List[Optional[Metadata]]] = [
                [
                    cast(
                        Optional[Metadata],
                        {"source_path": f"p{i}", "filename": f"f{i}.md"},
                    )
                    for i in range(num_initial_candidates)
                ]
            ]
            # Ensure documents is List[List[Optional[Document]]]
            documents_similar: List[List[Optional[Document]]] = [
                [cast(Optional[Document], "doc text similar a")]
                * num_initial_candidates
            ]

            similar_query_results: ChromaQueryResultType = {
                "ids": [[f"id{i}" for i in range(num_initial_candidates)]],
                "distances": [[0.1 + i * 0.01 for i in range(num_initial_candidates)]],
                "metadatas": metadatas_similar,
                "documents": documents_similar,
                "uris": None,
                "data": None,
                "embeddings": None,
                "included": ["metadatas", "documents", "distances"],
            }  # type: ignore
            self.searcher.chroma_collection.query.return_value = similar_query_results  # type: ignore
        if self.searcher.bm25:
            self.searcher.bm25.query.return_value = {"ids": [], "scores": [], "documents": []}  # type: ignore

        if self.searcher.chroma_collection:

            def mock_chroma_get(ids: List[str], **kwargs) -> ChromaGetResultType:  # type: ignore
                if ids and ids[0].startswith("id"):
                    get_res: ChromaGetResultType = {
                        "ids": [ids[0]],
                        "documents": [cast(Document, "doc text similar a")],
                        "metadatas": [
                            cast(
                                Metadata,
                                {"source_path": "p_get", "filename": "f_get.md"},
                            )
                        ],
                        "embeddings": None,
                        "uris": None,
                        "data": None,
                        "included": ["metadatas", "documents"],
                    }
                    return get_res
                empty_res: ChromaGetResultType = {
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "embeddings": None,
                    "uris": None,
                    "data": None,
                    "included": [],
                }
                return empty_res

            self.searcher.chroma_collection.get.side_effect = mock_chroma_get  # type: ignore

        results = self.searcher.llm_query("test similarity filter", debug_mode=True)

        self.assertIn("LLM Response", results["response"])
        self.assertIn("generation_time_sec", results)  # Check key presence
        self.assertEqual(
            results["debug_info"]["skipped_similar_chunk_count"],
            num_initial_candidates - 1,
        )
        self.assertEqual(results["debug_info"]["final_selected_chunk_count"], 1)
        self.assertGreater(mock_similar_check.call_count, 0)
        mock_log_query.assert_called_once()  # Log should happen

    @patch("llamasearch.core.query_processor.log_query")
    def test_llm_query_shutdown_event(self, mock_log_query):
        if self.searcher._shutdown_event:
            cast(MagicMock, self.searcher._shutdown_event).is_set.return_value = True  # type: ignore

        # Simulate shutdown *during* LLM call
        self.searcher.model.generate.side_effect = InterruptedError("Shutdown during LLM")  # type: ignore

        results = self.searcher.llm_query("test query during shutdown")

        self.assertEqual(
            results["response"], "LLM generation cancelled during shutdown."
        )
        self.assertGreaterEqual(results["query_time_seconds"], 0)
        # FIX: Check the key is present and >= 0.0 (might be small if interrupted early)
        self.assertIn("generation_time_sec", results)
        self.assertGreaterEqual(results["generation_time_sec"], 0.0)
        # Log should still happen even if interrupted
        mock_log_query.assert_called_once()

    def test_get_token_count_with_tokenizer(self):
        cast(MagicMock, self.searcher.model._tokenizer).side_effect = None  # type: ignore
        cast(MagicMock, self.searcher.model._tokenizer).return_value = {"input_ids": [1, 2, 3, 4, 5]}  # type: ignore
        count = self.searcher._get_token_count("some text")
        self.assertEqual(count, 5)
        cast(MagicMock, self.searcher.model._tokenizer).assert_called_once_with("some text", add_special_tokens=False, truncation=False)  # type: ignore

    def test_get_token_count_tokenizer_error(self):
        cast(MagicMock, self.searcher.model._tokenizer).side_effect = Exception("Tokenizer broke")  # type: ignore
        count = self.searcher._get_token_count("some text with error")
        self.assertEqual(count, len("some text with error") // 4)

    def test_get_token_count_no_model_or_tokenizer(self):
        self.searcher.model = None  # type: ignore
        count = self.searcher._get_token_count("text no model")
        self.assertEqual(count, len("text no model") // 4)

        self.setUp()  # Re-init searcher to get model back
        self.searcher.model._tokenizer = None  # type: ignore
        count = self.searcher._get_token_count("text no tokenizer")
        self.assertEqual(count, len("text no tokenizer") // 4)

    def test_are_chunks_too_similar(self):
        self.assertTrue(
            _are_chunks_too_similar(
                "apple banana cherry grape", "apple banana cherry date", 0.85
            )
        )
        self.assertFalse(
            _are_chunks_too_similar("apple banana cherry", "apple banana kiwi", 0.90)
        )
        self.assertFalse(
            _are_chunks_too_similar("apple banana cherry", "dog cat bird", 0.80)
        )
        self.assertFalse(_are_chunks_too_similar("", "text", 0.80))
        self.assertFalse(_are_chunks_too_similar("text", "", 0.80))


if __name__ == "__main__":
    unittest.main()
