import unittest
import shutil
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock, call, ANY

# IMPORTANT: Patch setup_logging at the very beginning,
# before importing the module under test.
# This ensures that when bm25.py is loaded, it uses our mock setup_logging.
MOCK_SETUP_LOGGING_TARGET = 'llamasearch.utils.setup_logging'
mock_logger_instance = MagicMock()
global_logger_patcher = patch(MOCK_SETUP_LOGGING_TARGET, return_value=mock_logger_instance)
global_logger_patcher.start()

# Now import the module under test and other necessary components
from llamasearch.core.bm25 import WhooshBM25Retriever, BM25Schema, DEFAULT_WRITER_TIMEOUT
from whoosh import index as whoosh_real_index # For integration tests
from whoosh.fields import Schema # For integration tests, actual schema type
from whoosh.qparser import QueryParser as WhooshQueryParser
from whoosh.scoring import BM25F
# Whoosh exceptions are part of whoosh.index
from whoosh.index import EmptyIndexError, LockError


# To stop the global patcher if other test files exist (optional)
# def tearDownModule():
#     global_logger_patcher.stop()

class TestWhooshBM25Retriever(unittest.TestCase):

    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir_obj.name)
        
        # Each test method can re-patch llamasearch.core.bm25.logger if it needs a fresh mock
        # For general logging, mock_logger_instance (global) will capture them.
        # We can reset it if needed:
        mock_logger_instance.reset_mock()


    def tearDown(self):
        self.temp_dir_obj.cleanup()

    @patch('llamasearch.core.bm25.whoosh_index.create_in')
    @patch('llamasearch.core.bm25.whoosh_index.open_dir')
    @patch('llamasearch.core.bm25.whoosh_index.exists_in')
    @patch('llamasearch.core.bm25.QueryParser', spec=WhooshQueryParser) # Mock QueryParser class
    @patch('llamasearch.core.bm25.logger') # Specific logger for this module
    def test_init_creates_new_index(self, mock_bm25_logger, mock_query_parser_cls, mock_exists_in, mock_open_dir, mock_create_in):
        mock_exists_in.return_value = False
        mock_index_instance = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index_instance.schema = BM25Schema() # Mock schema attribute
        mock_create_in.return_value = mock_index_instance
        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance

        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)

        self.assertTrue((self.test_dir / "whoosh_bm25_index").exists())
        mock_exists_in.assert_called_once_with(str(self.test_dir / "whoosh_bm25_index"))
        mock_create_in.assert_called_once_with(str(self.test_dir / "whoosh_bm25_index"), retriever.schema)
        mock_open_dir.assert_not_called()
        self.assertIsNotNone(retriever.ix)
        self.assertIsNotNone(retriever.parser)
        mock_query_parser_cls.assert_called_once_with("content", schema=mock_index_instance.schema)
        mock_bm25_logger.info.assert_any_call(f"Creating new Whoosh index at: {retriever.index_path}")

    @patch('llamasearch.core.bm25.whoosh_index.create_in')
    @patch('llamasearch.core.bm25.whoosh_index.open_dir')
    @patch('llamasearch.core.bm25.whoosh_index.exists_in')
    @patch('llamasearch.core.bm25.QueryParser', spec=WhooshQueryParser)
    @patch('llamasearch.core.bm25.logger')
    def test_init_opens_existing_index(self, mock_bm25_logger, mock_query_parser_cls, mock_exists_in, mock_open_dir, mock_create_in):
        mock_exists_in.return_value = True
        mock_index_instance = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index_instance.schema = BM25Schema()
        mock_open_dir.return_value = mock_index_instance
        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance

        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)

        mock_exists_in.assert_called_once_with(str(self.test_dir / "whoosh_bm25_index"))
        mock_open_dir.assert_called_once_with(str(self.test_dir / "whoosh_bm25_index"), schema=retriever.schema)
        mock_create_in.assert_not_called()
        self.assertIsNotNone(retriever.ix)
        self.assertIsNotNone(retriever.parser)
        mock_bm25_logger.info.assert_any_call(f"Opening existing Whoosh index at: {retriever.index_path}")

    @patch('llamasearch.core.bm25.shutil.rmtree')
    @patch('llamasearch.core.bm25.whoosh_index.create_in')
    @patch('llamasearch.core.bm25.whoosh_index.open_dir')
    @patch('llamasearch.core.bm25.whoosh_index.exists_in')
    @patch('llamasearch.core.bm25.QueryParser', spec=WhooshQueryParser)
    @patch('llamasearch.core.bm25.logger')
    def test_init_recreates_empty_corrupt_index(self, mock_bm25_logger, mock_query_parser_cls, mock_exists_in, mock_open_dir, mock_create_in, mock_rmtree):
        mock_exists_in.return_value = True
        mock_open_dir.side_effect = EmptyIndexError("Test empty index")
        mock_index_instance = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index_instance.schema = BM25Schema()
        mock_create_in.return_value = mock_index_instance
        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance

        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
        index_path = self.test_dir / "whoosh_bm25_index"

        mock_rmtree.assert_called_once_with(index_path)
        self.assertTrue(index_path.exists()) # Recreated by code
        mock_create_in.assert_called_once_with(str(index_path), retriever.schema)
        self.assertIsNotNone(retriever.ix)
        mock_bm25_logger.warning.assert_any_call(f"Whoosh index at {index_path} exists but is empty/corrupt. Recreating.")


    @patch('llamasearch.core.bm25.shutil.rmtree')
    @patch('llamasearch.core.bm25.whoosh_index.create_in')
    @patch('llamasearch.core.bm25.whoosh_index.open_dir')
    @patch('llamasearch.core.bm25.whoosh_index.exists_in')
    @patch('llamasearch.core.bm25.QueryParser', spec=WhooshQueryParser)
    @patch('llamasearch.core.bm25.logger')
    def test_init_recreates_index_on_open_error(self, mock_bm25_logger, mock_query_parser_cls, mock_exists_in, mock_open_dir, mock_create_in, mock_rmtree):
        mock_exists_in.return_value = True
        mock_open_dir.side_effect = ValueError("Test open error") # Generic error
        mock_index_instance = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index_instance.schema = BM25Schema()
        mock_create_in.return_value = mock_index_instance
        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance
        
        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
        index_path = self.test_dir / "whoosh_bm25_index"

        mock_rmtree.assert_called_once_with(index_path)
        self.assertTrue(index_path.exists())
        mock_create_in.assert_called_once_with(str(index_path), retriever.schema)
        self.assertIsNotNone(retriever.ix)
        mock_bm25_logger.error.assert_any_call(f"Error opening existing Whoosh index {index_path}, attempting recreation: ValueError('Test open error')", exc_info=True)
        mock_bm25_logger.info.assert_any_call(f"Recreated Whoosh index at: {index_path}")

    @patch('llamasearch.core.bm25.shutil.rmtree')
    @patch('llamasearch.core.bm25.whoosh_index.create_in')
    @patch('llamasearch.core.bm25.whoosh_index.open_dir')
    @patch('llamasearch.core.bm25.whoosh_index.exists_in')
    @patch('llamasearch.core.bm25.logger')
    def test_init_handles_recreation_failure_after_open_error(self, mock_bm25_logger, mock_exists_in, mock_open_dir, mock_create_in, mock_rmtree):
        mock_exists_in.return_value = True
        mock_open_dir.side_effect = ValueError("Test open error")
        mock_create_in.side_effect = IOError("Test recreate error") # Error during create_in

        with self.assertRaisesRegex(RuntimeError, "Failed to open or recreate Whoosh index"):
            WhooshBM25Retriever(storage_dir=self.test_dir)
        
        index_path = self.test_dir / "whoosh_bm25_index"
        mock_rmtree.assert_called_once_with(index_path)
        mock_bm25_logger.critical.assert_any_call(f"FATAL: Could not recreate Whoosh index after open failure: IOError('Test recreate error')", exc_info=True)

    @patch('llamasearch.core.bm25.whoosh_index.create_in')
    @patch('llamasearch.core.bm25.whoosh_index.exists_in')
    @patch('llamasearch.core.bm25.logger')
    def test_init_handles_initial_create_failure(self, mock_bm25_logger, mock_exists_in, mock_create_in):
        mock_exists_in.return_value = False
        mock_create_in.side_effect = IOError("Test initial create error")

        with self.assertRaisesRegex(RuntimeError, "Failed to initialize Whoosh index"):
            WhooshBM25Retriever(storage_dir=self.test_dir)
        
        mock_bm25_logger.error.assert_any_call(f"Failed to open or create Whoosh index at {self.test_dir / 'whoosh_bm25_index'}: IOError('Test initial create error')", exc_info=True)


    @patch('llamasearch.core.bm25.whoosh_index.create_in')
    @patch('llamasearch.core.bm25.whoosh_index.exists_in')
    @patch('llamasearch.core.bm25.shutil.rmtree')
    @patch('llamasearch.core.bm25.logger')
    def test_init_handles_rmtree_failure_during_recreation(self, mock_bm25_logger, mock_rmtree, mock_exists_in, mock_create_in):
        mock_exists_in.return_value = True
        # Mock open_dir to raise an error that triggers recreation path
        with patch('llamasearch.core.bm25.whoosh_index.open_dir', side_effect=ValueError("Simulated open error")):
            mock_rmtree.side_effect = OSError("Simulated rmtree failure")

            with self.assertRaisesRegex(RuntimeError, "Failed to initialize Whoosh index"):
                WhooshBM25Retriever(storage_dir=self.test_dir)
        
        mock_bm25_logger.error.assert_any_call(
            f"Error opening existing Whoosh index {self.test_dir / 'whoosh_bm25_index'}, attempting recreation: ValueError('Simulated open error')",
            exc_info=True
        )
        # The OSError from rmtree should be caught by the outer try-except in _open_or_create_index
        mock_bm25_logger.error.assert_any_call(
            f"Failed to open or create Whoosh index at {self.test_dir / 'whoosh_bm25_index'}: OSError('Simulated rmtree failure')",
            exc_info=True
        )
        mock_create_in.assert_not_called() # Should fail before create_in

    def _get_initialized_retriever(self, mock_index=None, mock_parser=None):
        """Helper to get an initialized retriever, bypassing real index creation."""
        with patch.object(WhooshBM25Retriever, '_open_or_create_index') as mock_open_create:
            retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
            retriever.ix = mock_index if mock_index else MagicMock(spec=whoosh_real_index.FileIndex)
            if retriever.ix : retriever.ix.schema = BM25Schema()
            retriever.parser = mock_parser if mock_parser else MagicMock(spec=WhooshQueryParser)
            mock_open_create.assert_called_once() # Ensure __init__ tried to call it
        return retriever

    @patch('llamasearch.core.bm25.logger')
    def test_add_document_success(self, mock_bm25_logger):
        mock_writer_instance = MagicMock()
        mock_writer_cm = MagicMock()
        mock_writer_cm.__enter__.return_value = mock_writer_instance
        mock_writer_cm.__exit__.return_value = None
        
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer_cm
        
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        result = retriever.add_document("some text", "doc1")
        
        self.assertTrue(result)
        mock_index.writer.assert_called_once_with(timeout=DEFAULT_WRITER_TIMEOUT)
        mock_writer_instance.update_document.assert_called_once_with(chunk_id="doc1", content="some text")
        mock_bm25_logger.debug.assert_any_call("Added/Updated document chunk_id 'doc1' in Whoosh index.")

    @patch('llamasearch.core.bm25.logger')
    def test_add_document_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None # Force uninitialized state
        
        result = retriever.add_document("some text", "doc1")
        
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with("Cannot add document, Whoosh index is not initialized.")

    @patch('llamasearch.core.bm25.logger')
    def test_add_document_empty_text_or_id(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        
        result_empty_text = retriever.add_document("", "doc1")
        self.assertFalse(result_empty_text)
        mock_bm25_logger.warning.assert_any_call("Skipping add_document: Empty text or doc_id provided (ID: 'doc1').")
        
        result_empty_id = retriever.add_document("some text", "")
        self.assertFalse(result_empty_id)
        mock_bm25_logger.warning.assert_any_call("Skipping add_document: Empty text or doc_id provided (ID: '').")

    @patch('llamasearch.core.bm25.logger')
    def test_add_document_lock_error(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.side_effect = LockError("Test lock error")
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        result = retriever.add_document("some text", "doc1")
        
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with("Failed to acquire lock for adding document chunk_id 'doc1': Test lock error")

    @patch('llamasearch.core.bm25.logger')
    def test_add_document_generic_exception(self, mock_bm25_logger):
        mock_writer_instance = MagicMock()
        mock_writer_instance.update_document.side_effect = Exception("Test update error")
        mock_writer_cm = MagicMock()
        mock_writer_cm.__enter__.return_value = mock_writer_instance
        mock_writer_cm.__exit__.return_value = None # Simulate error handled by context manager

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer_cm
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        result = retriever.add_document("some text", "doc1")
        
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with("Failed to add document chunk_id 'doc1' to Whoosh index: Exception('Test update error')", exc_info=True)

    @patch('llamasearch.core.bm25.logger')
    def test_remove_document_success(self, mock_bm25_logger):
        mock_writer_instance = MagicMock()
        mock_writer_instance.delete_by_term.return_value = 1 # Simulate one doc deleted
        mock_writer_cm = MagicMock()
        mock_writer_cm.__enter__.return_value = mock_writer_instance
        mock_writer_cm.__exit__.return_value = None
        
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer_cm
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        result = retriever.remove_document("doc1")
        
        self.assertTrue(result)
        mock_index.writer.assert_called_once_with(timeout=DEFAULT_WRITER_TIMEOUT)
        mock_writer_instance.delete_by_term.assert_called_once_with("chunk_id", "doc1")
        mock_bm25_logger.debug.assert_any_call("Attempted removal of document chunk_id 'doc1' from Whoosh index (deleted 1 segment docs).")

    @patch('llamasearch.core.bm25.logger')
    def test_remove_document_non_existent(self, mock_bm25_logger):
        mock_writer_instance = MagicMock()
        mock_writer_instance.delete_by_term.return_value = 0 # Simulate zero docs deleted
        mock_writer_cm = MagicMock()
        mock_writer_cm.__enter__.return_value = mock_writer_instance
        mock_writer_cm.__exit__.return_value = None
        
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer_cm
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        result = retriever.remove_document("doc_not_exists")
        
        self.assertTrue(result) # Should still be true
        mock_writer_instance.delete_by_term.assert_called_once_with("chunk_id", "doc_not_exists")
        mock_bm25_logger.debug.assert_any_call("Attempted removal of document chunk_id 'doc_not_exists' from Whoosh index (deleted 0 segment docs).")


    @patch('llamasearch.core.bm25.logger')
    def test_remove_document_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None
        
        result = retriever.remove_document("doc1")
        
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with("Cannot remove document, Whoosh index is not initialized.")

    @patch('llamasearch.core.bm25.logger')
    def test_remove_document_empty_id(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        result = retriever.remove_document("")
        self.assertFalse(result)
        mock_bm25_logger.warning.assert_called_once_with("Skipping remove_document: Empty doc_id provided.")

    @patch('llamasearch.core.bm25.logger')
    def test_remove_document_lock_error(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.side_effect = LockError("Test lock error remove")
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        result = retriever.remove_document("doc1")
        
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with("Failed to acquire lock for removing document chunk_id 'doc1': Test lock error remove")

    @patch('llamasearch.core.bm25.logger')
    def test_remove_document_generic_exception(self, mock_bm25_logger):
        mock_writer_instance = MagicMock()
        mock_writer_instance.delete_by_term.side_effect = Exception("Test delete error")
        mock_writer_cm = MagicMock()
        mock_writer_cm.__enter__.return_value = mock_writer_instance
        mock_writer_cm.__exit__.return_value = None

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer_cm
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        result = retriever.remove_document("doc1")
        
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with("Failed to remove document chunk_id 'doc1' from Whoosh index: Exception('Test delete error')", exc_info=True)

    @patch('llamasearch.core.bm25.logger')
    def test_query_success(self, mock_bm25_logger):
        mock_hit1 = MagicMock()
        mock_hit1.get.side_effect = lambda key: "id1" if key == "chunk_id" else None
        mock_hit1.score = 0.9

        mock_hit2 = MagicMock()
        mock_hit2.get.side_effect = lambda key: "id2" if key == "chunk_id" else None
        mock_hit2.score = 0.8
        
        mock_results = [mock_hit1, mock_hit2]
        
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = mock_results
        mock_searcher_cm = MagicMock()
        mock_searcher_cm.__enter__.return_value = mock_searcher_instance
        mock_searcher_cm.__exit__.return_value = None

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.searcher.return_value = mock_searcher_cm
        
        mock_query_obj = MagicMock()
        mock_parser = MagicMock(spec=WhooshQueryParser)
        mock_parser.parse.return_value = mock_query_obj
        
        retriever = self._get_initialized_retriever(mock_index=mock_index, mock_parser=mock_parser)
        
        query_text = "test query"
        results = retriever.query(query_text, n_results=2)
        
        expected_results = {
            "query": query_text,
            "ids": ["id1", "id2"],
            "scores": [0.9, 0.8],
            "documents": [None, None]
        }
        self.assertEqual(results, expected_results)
        mock_parser.parse.assert_called_once_with(query_text)
        mock_index.searcher.assert_called_once_with(weighting=ANY) # Check BM25F is used
        self.assertIsInstance(mock_index.searcher.call_args[1]['weighting'], BM25F)
        mock_searcher_instance.search.assert_called_once_with(mock_query_obj, limit=2)

    @patch('llamasearch.core.bm25.logger')
    def test_query_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None # Force uninitialized
        
        query_text = "test query"
        results = retriever.query(query_text)
        
        expected_empty = {"query": query_text, "ids": [], "scores": [], "documents": []}
        self.assertEqual(results, expected_empty)
        mock_bm25_logger.error.assert_called_once_with("Cannot query, Whoosh index or parser not initialized.")

    @patch('llamasearch.core.bm25.logger')
    def test_query_empty_text(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        results = retriever.query("")
        expected_empty = {"query": "", "ids": [], "scores": [], "documents": []}
        self.assertEqual(results, expected_empty)
        mock_bm25_logger.debug.assert_called_once_with("BM25 query text is empty.")

    @patch('llamasearch.core.bm25.logger')
    def test_query_no_results_found(self, mock_bm25_logger):
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = [] # No results
        mock_searcher_cm = MagicMock()
        mock_searcher_cm.__enter__.return_value = mock_searcher_instance
        mock_searcher_cm.__exit__.return_value = None

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.searcher.return_value = mock_searcher_cm
        
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        query_text = "rare term"
        results = retriever.query(query_text)
        
        expected_empty = {"query": query_text, "ids": [], "scores": [], "documents": []}
        self.assertEqual(results, expected_empty)
        mock_bm25_logger.debug.assert_any_call(f"Whoosh BM25 query returned 0 results.")

    @patch('llamasearch.core.bm25.logger')
    def test_query_hit_missing_fields(self, mock_bm25_logger):
        mock_hit_valid = MagicMock()
        mock_hit_valid.get.side_effect = lambda key: "id1" if key == "chunk_id" else None
        mock_hit_valid.score = 0.9

        mock_hit_no_id = MagicMock()
        mock_hit_no_id.get.return_value = None # No chunk_id
        mock_hit_no_id.score = 0.8

        mock_hit_no_score = MagicMock()
        mock_hit_no_score.get.side_effect = lambda key: "id2" if key == "chunk_id" else None
        mock_hit_no_score.score = None # No score
        
        mock_results = [mock_hit_valid, mock_hit_no_id, mock_hit_no_score]
        
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = mock_results
        mock_searcher_cm = MagicMock()
        mock_searcher_cm.__enter__.return_value = mock_searcher_instance
        mock_searcher_cm.__exit__.return_value = None

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.searcher.return_value = mock_searcher_cm
        retriever = self._get_initialized_retriever(mock_index=mock_index)

        query_text = "test query"
        results = retriever.query(query_text)

        expected_results = {"query": query_text, "ids": ["id1"], "scores": [0.9], "documents": [None]}
        self.assertEqual(results, expected_results)
        mock_bm25_logger.warning.assert_any_call(f"Whoosh hit missing 'chunk_id' or 'score': {mock_hit_no_id}")
        mock_bm25_logger.warning.assert_any_call(f"Whoosh hit missing 'chunk_id' or 'score': {mock_hit_no_score}")


    @patch('llamasearch.core.bm25.logger')
    def test_query_generic_exception(self, mock_bm25_logger):
        mock_parser = MagicMock(spec=WhooshQueryParser)
        mock_parser.parse.side_effect = Exception("Test parse error")
        retriever = self._get_initialized_retriever(mock_parser=mock_parser)
        
        query_text = "test query"
        results = retriever.query(query_text)
        
        expected_empty = {"query": query_text, "ids": [], "scores": [], "documents": []}
        self.assertEqual(results, expected_empty)
        mock_bm25_logger.error.assert_called_once_with("Whoosh query failed: Exception('Test parse error')", exc_info=True)

    def test_get_doc_count_success(self):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.doc_count.return_value = 42
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        self.assertEqual(retriever.get_doc_count(), 42)
        mock_index.doc_count.assert_called_once()

    def test_get_doc_count_index_not_initialized(self):
        retriever = self._get_initialized_retriever()
        retriever.ix = None
        self.assertEqual(retriever.get_doc_count(), 0)
    
    @patch('llamasearch.core.bm25.logger')
    def test_get_doc_count_exception(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.doc_count.side_effect = Exception("Test doc_count error")
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        self.assertEqual(retriever.get_doc_count(), 0)
        mock_bm25_logger.error.assert_called_once_with("Failed to get Whoosh doc count: Exception('Test doc_count error')")

    @patch('llamasearch.core.bm25.logger')
    def test_save_is_noop(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.save()
        mock_bm25_logger.debug.assert_called_once_with("Whoosh index persistence is handled internally on add/remove.")

    @patch('llamasearch.core.bm25.logger')
    def test_close_success(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        retriever.close()
        
        mock_index.close.assert_called_once()
        self.assertIsNone(retriever.ix)
        self.assertIsNone(retriever.parser)
        mock_bm25_logger.info.assert_any_call("Whoosh index closed.")

    @patch('llamasearch.core.bm25.logger')
    def test_close_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None
        retriever.parser = None # already None if ix is None generally
        
        retriever.close() # Should not raise error
        mock_bm25_logger.info.assert_any_call("Closing Whoosh index...")
        # No call to ix.close, and no error if already None

    @patch('llamasearch.core.bm25.logger')
    def test_close_exception(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.close.side_effect = Exception("Test close error")
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        
        retriever.close() # Should not re-raise
        
        mock_index.close.assert_called_once()
        mock_bm25_logger.error.assert_called_once_with("Error closing Whoosh index: Exception('Test close error')")
        # ix and parser should still be set to None
        self.assertIsNone(retriever.ix)
        self.assertIsNone(retriever.parser)

    # --- Integration Test using Real Whoosh Index ---
    def test_integration_add_query_remove_real_index(self):
        # This test uses the real Whoosh library without mocks for core indexing
        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
        self.assertIsInstance(retriever.ix.schema, BM25Schema) # Ensure schema is BM25Schema
        
        # Test initial state
        self.assertEqual(retriever.get_doc_count(), 0)
        
        # Add documents
        self.assertTrue(retriever.add_document("The quick brown fox", "doc1"))
        self.assertTrue(retriever.add_document("jumps over the lazy dog", "doc2"))
        self.assertTrue(retriever.add_document("A quick test document", "doc3"))
        self.assertEqual(retriever.get_doc_count(), 3)

        # Query for "quick"
        results_quick = retriever.query("quick", n_results=2)
        self.assertEqual(results_quick["query"], "quick")
        self.assertEqual(len(results_quick["ids"]), 2)
        self.assertIn("doc1", results_quick["ids"])
        self.assertIn("doc3", results_quick["ids"])
        self.assertEqual(len(results_quick["scores"]), 2)
        self.assertEqual(results_quick["documents"], [None, None])

        # Query for "fox"
        results_fox = retriever.query("fox", n_results=1)
        self.assertEqual(results_fox["ids"], ["doc1"])
        self.assertEqual(len(results_fox["scores"]), 1)
        self.assertTrue(results_fox["scores"][0] > 0)

        # Remove a document
        self.assertTrue(retriever.remove_document("doc1"))
        self.assertEqual(retriever.get_doc_count(), 2) # One less

        # Query for "fox" again (should be gone)
        results_fox_after_remove = retriever.query("fox", n_results=1)
        self.assertEqual(results_fox_after_remove["ids"], [])

        # Query for "quick" again
        results_quick_after_remove = retriever.query("quick", n_results=2)
        self.assertEqual(results_quick_after_remove["ids"], ["doc3"]) # Only doc3 should match now

        # Test adding an existing document (update)
        self.assertTrue(retriever.add_document("new content for dog document", "doc2"))
        self.assertEqual(retriever.get_doc_count(), 2) # Count should not change
        
        # Query for "new content"
        results_new_content = retriever.query("new content", n_results=1)
        self.assertEqual(results_new_content["ids"], ["doc2"])

        # Close the retriever
        retriever.close()
        self.assertIsNone(retriever.ix)

        # Re-open and check persistence
        retriever2 = WhooshBM25Retriever(storage_dir=self.test_dir)
        self.assertEqual(retriever2.get_doc_count(), 2)
        results_dog_reopened = retriever2.query("dog document", n_results=1) # "dog" was in original doc2, "dog document" in updated
        self.assertEqual(results_dog_reopened["ids"], ["doc2"])
        retriever2.close()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)