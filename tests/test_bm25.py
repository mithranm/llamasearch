# tests/test_bm25.py
import tempfile 
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch 

from whoosh import index as whoosh_real_index  # For integration tests

# Import the base Schema class for isinstance check
from whoosh.fields import Schema as WhooshBaseSchema
from whoosh.index import EmptyIndexError, LockError 
from whoosh.qparser import QueryParser as WhooshQueryParser
from whoosh.scoring import BM25F

# Now import the module under test and other necessary components
from llamasearch.core.bm25 import (
    DEFAULT_WRITER_TIMEOUT,
    BM25Schema,
    WhooshBM25Retriever,
)
import logging


# IMPORTANT: Patch setup_logging at the very beginning,
# before importing the module under test.
MOCK_SETUP_LOGGING_TARGET = "llamasearch.utils.setup_logging"
mock_logger_instance = MagicMock(spec=logging.Logger) # Use spec=logging.Logger
# global_logger_patcher = patch(
#     MOCK_SETUP_LOGGING_TARGET, return_value=mock_logger_instance
# )

# This function will be called by unittest TestLoader or pytest
# def setUpModule():
#     global_logger_patcher.start()

# def tearDownModule():
#     global_logger_patcher.stop()


class TestWhooshBM25Retriever(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Start patch for all tests in this class
        cls.logger_patcher = patch(MOCK_SETUP_LOGGING_TARGET, return_value=mock_logger_instance)
        cls.logger_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.logger_patcher.stop()

    def setUp(self):
        # Reset the shared mock for each test method
        mock_logger_instance.reset_mock()
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir_obj.name)


    def tearDown(self):
        self.temp_dir_obj.cleanup()

    # Helper to get an initialized retriever, bypassing real index creation.
    def _get_initialized_retriever(self, mock_index=None, mock_parser=None):
        with patch.object(
            WhooshBM25Retriever, "_open_or_create_index"
        ) as mock_open_create:
            retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
            # If a mock_index is provided, use it. Ensure it has a schema.
            if mock_index:
                retriever.ix = mock_index
                if not hasattr(mock_index, 'schema') or mock_index.schema is None:
                    # Assign a schema if the provided mock doesn't have one
                    # This is important because QueryParser needs ix.schema
                    mock_index.schema = BM25Schema()
            else:
                # If no mock_index, create a default one with a schema
                retriever.ix = MagicMock(spec=whoosh_real_index.FileIndex)
                retriever.ix.schema = BM25Schema()

            retriever.parser = (
                mock_parser if mock_parser else MagicMock(spec=WhooshQueryParser)
            )
            mock_open_create.assert_called_once()
        return retriever

    # --- Test __init__ scenarios ---
    @patch("llamasearch.core.bm25.whoosh_index.create_in")
    @patch("llamasearch.core.bm25.whoosh_index.open_dir")
    @patch("llamasearch.core.bm25.whoosh_index.exists_in")
    @patch("llamasearch.core.bm25.QueryParser", spec=WhooshQueryParser)
    @patch("llamasearch.core.bm25.logger") 
    def test_init_creates_new_index(
        self,
        mock_bm25_logger, 
        mock_query_parser_cls,
        mock_exists_in,
        mock_open_dir,
        mock_create_in,
    ):
        mock_exists_in.return_value = False
        mock_index_instance = MagicMock(spec=whoosh_real_index.FileIndex)
        created_schema_instance = BM25Schema()
        mock_index_instance.schema = created_schema_instance
        mock_create_in.return_value = mock_index_instance

        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance

        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)

        self.assertTrue((self.test_dir / "whoosh_bm25_index").exists())
        mock_exists_in.assert_called_once_with(str(self.test_dir / "whoosh_bm25_index"))
        mock_create_in.assert_called_once_with(
            str(self.test_dir / "whoosh_bm25_index"), retriever.schema
        )
        mock_query_parser_cls.assert_called_once_with(
            "content", schema=mock_index_instance.schema
        )
        self.assertIs(mock_index_instance.schema, created_schema_instance) 

        mock_open_dir.assert_not_called()
        self.assertIsNotNone(retriever.ix)
        self.assertIsNotNone(retriever.parser)
        mock_bm25_logger.info.assert_any_call(
            f"Creating new Whoosh index at: {retriever.index_path}"
        )

    @patch("llamasearch.core.bm25.whoosh_index.create_in")
    @patch("llamasearch.core.bm25.whoosh_index.open_dir")
    @patch("llamasearch.core.bm25.whoosh_index.exists_in")
    @patch("llamasearch.core.bm25.QueryParser", spec=WhooshQueryParser)
    @patch("llamasearch.core.bm25.logger")
    def test_init_opens_existing_index(
        self,
        mock_bm25_logger,
        mock_query_parser_cls,
        mock_exists_in,
        mock_open_dir,
        mock_create_in,
    ):
        mock_exists_in.return_value = True
        mock_index_instance = MagicMock(spec=whoosh_real_index.FileIndex)
        opened_schema_instance = BM25Schema() 
        mock_index_instance.schema = opened_schema_instance
        mock_open_dir.return_value = mock_index_instance

        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance

        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)

        mock_exists_in.assert_called_once_with(str(self.test_dir / "whoosh_bm25_index"))
        mock_open_dir.assert_called_once_with(
            str(self.test_dir / "whoosh_bm25_index"), schema=retriever.schema
        )
        mock_query_parser_cls.assert_called_once_with(
            "content", schema=mock_index_instance.schema 
        )
        self.assertIs(mock_index_instance.schema, opened_schema_instance)

        mock_create_in.assert_not_called()
        self.assertIsNotNone(retriever.ix)
        self.assertIsNotNone(retriever.parser)
        mock_bm25_logger.info.assert_any_call(
            f"Opening existing Whoosh index at: {retriever.index_path}"
        )

    @patch("llamasearch.core.bm25.shutil.rmtree")
    @patch("llamasearch.core.bm25.whoosh_index.create_in")
    @patch("llamasearch.core.bm25.whoosh_index.open_dir")
    @patch("llamasearch.core.bm25.whoosh_index.exists_in")
    @patch("llamasearch.core.bm25.QueryParser", spec=WhooshQueryParser)
    @patch("llamasearch.core.bm25.logger")
    def test_init_recreates_empty_corrupt_index(
        self,
        mock_bm25_logger,
        mock_query_parser_cls,
        mock_exists_in,
        mock_open_dir,
        mock_create_in,
        mock_rmtree,
    ):
        mock_exists_in.return_value = True
        mock_open_dir.side_effect = EmptyIndexError("Test empty index")

        mock_index_instance_recreated = MagicMock(spec=whoosh_real_index.FileIndex)
        recreated_schema = BM25Schema()
        mock_index_instance_recreated.schema = recreated_schema
        mock_create_in.return_value = mock_index_instance_recreated

        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance

        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
        index_path = self.test_dir / "whoosh_bm25_index"

        mock_rmtree.assert_called_once_with(index_path)
        self.assertTrue(index_path.exists())
        mock_create_in.assert_called_once_with(str(index_path), retriever.schema)
        mock_query_parser_cls.assert_called_once_with("content", schema=mock_index_instance_recreated.schema)

        self.assertIsNotNone(retriever.ix)
        mock_bm25_logger.warning.assert_any_call(
            f"Whoosh index at {index_path} exists but is empty/corrupt. Recreating."
        )

    @patch("llamasearch.core.bm25.shutil.rmtree")
    @patch("llamasearch.core.bm25.whoosh_index.create_in")
    @patch("llamasearch.core.bm25.whoosh_index.open_dir")
    @patch("llamasearch.core.bm25.whoosh_index.exists_in")
    @patch("llamasearch.core.bm25.QueryParser", spec=WhooshQueryParser)
    @patch("llamasearch.core.bm25.logger")
    def test_init_recreates_index_on_open_error(
        self,
        mock_bm25_logger,
        mock_query_parser_cls,
        mock_exists_in,
        mock_open_dir,
        mock_create_in,
        mock_rmtree,
    ):
        mock_exists_in.return_value = True
        open_error = ValueError("Test open error")
        mock_open_dir.side_effect = open_error

        mock_index_instance_recreated = MagicMock(spec=whoosh_real_index.FileIndex)
        recreated_schema = BM25Schema()
        mock_index_instance_recreated.schema = recreated_schema
        mock_create_in.return_value = mock_index_instance_recreated

        mock_parser_instance = MagicMock()
        mock_query_parser_cls.return_value = mock_parser_instance

        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
        index_path = self.test_dir / "whoosh_bm25_index"

        mock_rmtree.assert_called_once_with(index_path)
        self.assertTrue(index_path.exists())
        mock_create_in.assert_called_once_with(str(index_path), retriever.schema)
        mock_query_parser_cls.assert_called_once_with("content", schema=mock_index_instance_recreated.schema)

        self.assertIsNotNone(retriever.ix)
        mock_bm25_logger.error.assert_any_call(
            f"Error opening existing Whoosh index {index_path}, attempting recreation: {open_error}", 
            exc_info=True,
        )
        mock_bm25_logger.info.assert_any_call(
            f"Recreated Whoosh index at: {index_path}"
        )

    @patch("llamasearch.core.bm25.shutil.rmtree")
    @patch("llamasearch.core.bm25.whoosh_index.create_in")
    @patch("llamasearch.core.bm25.whoosh_index.open_dir")
    @patch("llamasearch.core.bm25.whoosh_index.exists_in")
    @patch("llamasearch.core.bm25.logger")
    def test_init_handles_recreation_failure_after_open_error(
        self,
        mock_bm25_logger,
        mock_exists_in,
        mock_open_dir,
        mock_create_in,
        mock_rmtree,
    ):
        mock_exists_in.return_value = True
        open_error = ValueError("Test open error")
        mock_open_dir.side_effect = open_error
        recreate_error = IOError("Test recreate error")
        mock_create_in.side_effect = recreate_error

        with self.assertRaisesRegex(RuntimeError, "Failed to initialize Whoosh index"):
            WhooshBM25Retriever(storage_dir=self.test_dir)

        index_path = self.test_dir / "whoosh_bm25_index"
        mock_rmtree.assert_called_once_with(index_path)
        mock_bm25_logger.critical.assert_any_call(
            f"FATAL: Could not recreate Whoosh index after open failure: {recreate_error}", 
            exc_info=True,
        )
        expected_runtime_error = RuntimeError('Failed to open or recreate Whoosh index')
        # The final logged error uses the string representation of the raised RuntimeError
        mock_bm25_logger.error.assert_any_call(
            f"Failed to open or create Whoosh index at {self.test_dir / 'whoosh_bm25_index'}: {expected_runtime_error}",
            exc_info=True,
        )

    @patch("llamasearch.core.bm25.whoosh_index.create_in")
    @patch("llamasearch.core.bm25.whoosh_index.exists_in")
    @patch("llamasearch.core.bm25.logger")
    def test_init_handles_initial_create_failure(
        self, mock_bm25_logger, mock_exists_in, mock_create_in
    ):
        mock_exists_in.return_value = False
        initial_create_error = IOError("Test initial create error")
        mock_create_in.side_effect = initial_create_error

        with self.assertRaisesRegex(RuntimeError, "Failed to initialize Whoosh index"):
            WhooshBM25Retriever(storage_dir=self.test_dir)

        mock_bm25_logger.error.assert_any_call(
            f"Failed to open or create Whoosh index at {self.test_dir / 'whoosh_bm25_index'}: {initial_create_error}", 
            exc_info=True,
        )

    @patch("llamasearch.core.bm25.shutil.rmtree")
    @patch("llamasearch.core.bm25.whoosh_index.create_in")
    @patch("llamasearch.core.bm25.whoosh_index.open_dir")
    @patch("llamasearch.core.bm25.whoosh_index.exists_in")
    @patch("llamasearch.core.bm25.logger")
    def test_init_handles_rmtree_failure_during_recreation(
        self,
        mock_bm25_logger,
        mock_exists_in,
        mock_open_dir,
        mock_create_in,
        mock_rmtree,
    ):
        mock_exists_in.return_value = True
        open_error = ValueError("Simulated open error")
        mock_open_dir.side_effect = open_error
        rmtree_error = OSError("Simulated rmtree failure")
        mock_rmtree.side_effect = rmtree_error

        with self.assertRaisesRegex(RuntimeError, "Failed to initialize Whoosh index"):
            WhooshBM25Retriever(storage_dir=self.test_dir)

        mock_bm25_logger.error.assert_any_call(
            f"Error opening existing Whoosh index {self.test_dir / 'whoosh_bm25_index'}, attempting recreation: {open_error}", 
            exc_info=True,
        )
        # The rmtree_error (OSError) causes a RuntimeError ("Failed to open or recreate...") to be raised, 
        # which is then caught by the outermost except block.
        expected_logged_exception = RuntimeError("Failed to open or recreate Whoosh index")
        mock_bm25_logger.error.assert_any_call(
            f"Failed to open or create Whoosh index at {self.test_dir / 'whoosh_bm25_index'}: {expected_logged_exception}",
            exc_info=True,
        )
        mock_create_in.assert_not_called()

    # --- Test add_document ---
    @patch("llamasearch.core.bm25.logger")
    def test_add_document_success(self, mock_bm25_logger):
        mock_writer = MagicMock() 
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer

        retriever = self._get_initialized_retriever(mock_index=mock_index)
        result = retriever.add_document("some text", "doc1")

        self.assertTrue(result)
        mock_index.writer.assert_called_once_with(timeout=DEFAULT_WRITER_TIMEOUT)
        mock_writer.update_document.assert_called_once_with(
            chunk_id="doc1", content="some text"
        )
        mock_writer.__enter__.assert_called_once()
        mock_writer.__exit__.assert_called_once()
        mock_bm25_logger.debug.assert_any_call(
            "Added/Updated document chunk_id 'doc1' in Whoosh index."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_add_document_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None
        result = retriever.add_document("some text", "doc1")
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with(
            "Cannot add document, Whoosh index is not initialized."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_add_document_empty_text_or_id(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        result_empty_text = retriever.add_document("", "doc1")
        self.assertFalse(result_empty_text)
        mock_bm25_logger.warning.assert_any_call(
            "Skipping add_document: Empty text or doc_id provided (ID: 'doc1')."
        )
        result_empty_id = retriever.add_document("some text", "")
        self.assertFalse(result_empty_id)
        mock_bm25_logger.warning.assert_any_call(
            "Skipping add_document: Empty text or doc_id provided (ID: '')."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_add_document_lock_error(self, mock_bm25_logger):
        lock_err = LockError("Test lock error")
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.side_effect = lock_err
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        result = retriever.add_document("some text", "doc1")
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with(
            f"Failed to acquire lock for adding document chunk_id 'doc1': {lock_err}" 
        )

    @patch("llamasearch.core.bm25.logger")
    def test_add_document_generic_exception(self, mock_bm25_logger):
        mock_writer = MagicMock()
        update_error = Exception("Test update error")
        mock_writer.update_document.side_effect = update_error
        mock_writer.__exit__.side_effect = lambda exc_type, exc_val, exc_tb: False

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer
        retriever = self._get_initialized_retriever(mock_index=mock_index)

        result = retriever.add_document("some text", "doc1")

        self.assertFalse(result)
        mock_index.writer.assert_called_once_with(timeout=DEFAULT_WRITER_TIMEOUT)
        mock_writer.update_document.assert_called_once()
        mock_bm25_logger.error.assert_called_once_with(
            f"Failed to add document chunk_id 'doc1' to Whoosh index: {update_error}", 
            exc_info=True,
        )
        mock_writer.__exit__.assert_called_once()

    # --- Test remove_document ---
    @patch("llamasearch.core.bm25.logger")
    def test_remove_document_success(self, mock_bm25_logger):
        mock_writer = MagicMock()
        mock_writer.delete_by_term.return_value = 1

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer
        retriever = self._get_initialized_retriever(mock_index=mock_index)

        result = retriever.remove_document("doc1")

        self.assertTrue(result)
        mock_index.writer.assert_called_once_with(timeout=DEFAULT_WRITER_TIMEOUT)
        mock_writer.delete_by_term.assert_called_once_with("chunk_id", "doc1")
        mock_writer.__enter__.assert_called_once()
        mock_writer.__exit__.assert_called_once()
        mock_bm25_logger.debug.assert_any_call(
            "Attempted removal of document chunk_id 'doc1' from Whoosh index (deleted 1 segment docs)."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_remove_document_non_existent(self, mock_bm25_logger):
        mock_writer = MagicMock()
        mock_writer.delete_by_term.return_value = 0

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer
        retriever = self._get_initialized_retriever(mock_index=mock_index)

        result = retriever.remove_document("doc_not_exists")

        self.assertTrue(result)
        mock_writer.delete_by_term.assert_called_once_with(
            "chunk_id", "doc_not_exists"
        )
        mock_writer.__enter__.assert_called_once()
        mock_writer.__exit__.assert_called_once()
        mock_bm25_logger.debug.assert_any_call(
            "Attempted removal of document chunk_id 'doc_not_exists' from Whoosh index (deleted 0 segment docs)."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_remove_document_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None
        result = retriever.remove_document("doc1")
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with(
            "Cannot remove document, Whoosh index is not initialized."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_remove_document_empty_id(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        result = retriever.remove_document("")
        self.assertFalse(result) 
        mock_bm25_logger.warning.assert_called_once_with(
            "Skipping remove_document: Empty doc_id provided."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_remove_document_lock_error(self, mock_bm25_logger):
        lock_err = LockError("Test lock error remove")
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.side_effect = lock_err
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        result = retriever.remove_document("doc1")
        self.assertFalse(result)
        mock_bm25_logger.error.assert_called_once_with(
            f"Failed to acquire lock for removing document chunk_id 'doc1': {lock_err}" 
        )

    @patch("llamasearch.core.bm25.logger")
    def test_remove_document_generic_exception(self, mock_bm25_logger):
        mock_writer = MagicMock()
        delete_error = Exception("Test delete error")
        mock_writer.delete_by_term.side_effect = delete_error
        mock_writer.__exit__.side_effect = lambda exc_type, exc_val, exc_tb: False


        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.writer.return_value = mock_writer
        retriever = self._get_initialized_retriever(mock_index=mock_index)

        result = retriever.remove_document("doc1")

        self.assertFalse(result)
        mock_writer.delete_by_term.assert_called_once()
        mock_bm25_logger.error.assert_called_once_with(
            f"Failed to remove document chunk_id 'doc1' from Whoosh index: {delete_error}", 
            exc_info=True,
        )
        mock_writer.__exit__.assert_called_once()

    # --- Test query ---
    @patch("llamasearch.core.bm25.logger")
    def test_query_success(self, mock_bm25_logger):
        mock_hit1 = MagicMock()
        mock_hit1.get.return_value = "id1" 
        mock_hit1.score = 0.9
        mock_hit2 = MagicMock()
        mock_hit2.get.return_value = "id2" 
        mock_hit2.score = 0.8

        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = [mock_hit1, mock_hit2]

        mock_searcher_cm = MagicMock() 
        mock_searcher_cm.__enter__.return_value = mock_searcher_instance 
        mock_searcher_cm.__exit__.return_value = None 

        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        mock_index.searcher.return_value = mock_searcher_cm 

        mock_query_obj = MagicMock()
        mock_parser = MagicMock(spec=WhooshQueryParser)
        mock_parser.parse.return_value = mock_query_obj

        retriever = self._get_initialized_retriever(
            mock_index=mock_index, mock_parser=mock_parser
        )

        query_text = "test query"
        results = retriever.query(query_text, n_results=2)

        expected_results = {
            "query": query_text,
            "ids": ["id1", "id2"],
            "scores": [0.9, 0.8],
            "documents": [None, None],
        }
        self.assertEqual(results, expected_results)
        mock_parser.parse.assert_called_once_with(query_text)
        mock_index.searcher.assert_called_once_with(weighting=ANY)
        self.assertIsInstance(mock_index.searcher.call_args[1]["weighting"], BM25F)
        mock_searcher_instance.search.assert_called_once_with(mock_query_obj, limit=2)
        mock_searcher_cm.__exit__.assert_called_once()

    @patch("llamasearch.core.bm25.logger")
    def test_query_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None
        query_text = "test query"
        results = retriever.query(query_text)
        expected_empty = {"query": query_text, "ids": [], "scores": [], "documents": []}
        self.assertEqual(results, expected_empty)
        mock_bm25_logger.error.assert_called_once_with(
            "Cannot query, Whoosh index or parser not initialized."
        )

    @patch("llamasearch.core.bm25.logger")
    def test_query_empty_text(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        results = retriever.query("")
        expected_empty = {"query": "", "ids": [], "scores": [], "documents": []}
        self.assertEqual(results, expected_empty)
        mock_bm25_logger.debug.assert_any_call("BM25 query text is empty.")

    @patch("llamasearch.core.bm25.logger")
    def test_query_no_results_found(self, mock_bm25_logger):
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = [] 
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
        mock_bm25_logger.debug.assert_any_call("Whoosh BM25 query returned 0 results.")


    @patch("llamasearch.core.bm25.logger")
    def test_query_hit_missing_fields(self, mock_bm25_logger):
        mock_hit_valid = MagicMock()
        mock_hit_valid.get.side_effect = lambda key: "id1" if key == "chunk_id" else None
        mock_hit_valid.score = 0.9

        mock_hit_no_id = MagicMock()
        mock_hit_no_id.get.side_effect = lambda key: None 
        mock_hit_no_id.score = 0.8

        mock_hit_no_score = MagicMock()
        mock_hit_no_score.get.side_effect = lambda key: "id2" if key == "chunk_id" else None
        mock_hit_no_score.score = None 

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

        expected_results = {
            "query": query_text,
            "ids": ["id1"], 
            "scores": [0.9],
            "documents": [None],
        }
        self.assertEqual(results, expected_results)
        mock_hit_valid.get.assert_any_call("chunk_id")
        mock_hit_no_id.get.assert_any_call("chunk_id")
        mock_hit_no_score.get.assert_any_call("chunk_id")

        mock_bm25_logger.warning.assert_any_call(
            f"Whoosh hit missing 'chunk_id' or 'score': {mock_hit_no_id!r}"
        )
        mock_bm25_logger.warning.assert_any_call(
            f"Whoosh hit missing 'chunk_id' or 'score': {mock_hit_no_score!r}"
        )


    @patch("llamasearch.core.bm25.logger")
    def test_query_generic_exception(self, mock_bm25_logger):
        mock_parser = MagicMock(spec=WhooshQueryParser)
        parse_error = Exception("Test parse error")
        mock_parser.parse.side_effect = parse_error
        retriever = self._get_initialized_retriever(mock_parser=mock_parser)
        query_text = "test query"
        results = retriever.query(query_text)
        expected_empty = {"query": query_text, "ids": [], "scores": [], "documents": []}
        self.assertEqual(results, expected_empty)
        mock_bm25_logger.error.assert_called_once_with(
            f"Whoosh query failed: {parse_error}", exc_info=True 
        )

    # --- Test get_doc_count ---
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

    @patch("llamasearch.core.bm25.logger")
    def test_get_doc_count_exception(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        doc_count_error = Exception("Test doc_count error")
        mock_index.doc_count.side_effect = doc_count_error
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        self.assertEqual(retriever.get_doc_count(), 0)
        mock_bm25_logger.error.assert_called_once_with(
            f"Failed to get Whoosh doc count: {doc_count_error}" 
        )

    # --- Test save ---
    @patch("llamasearch.core.bm25.logger")
    def test_save_is_noop(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.save()
        mock_bm25_logger.debug.assert_any_call(
            "Whoosh index persistence is handled internally on add/remove."
        )

    # --- Test close ---
    @patch("llamasearch.core.bm25.logger")
    def test_close_success(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        retriever.close()
        mock_index.close.assert_called_once()
        self.assertIsNone(retriever.ix)
        self.assertIsNone(retriever.parser)
        mock_bm25_logger.info.assert_any_call("Whoosh index resources released/nulled.")

    @patch("llamasearch.core.bm25.logger")
    def test_close_index_not_initialized(self, mock_bm25_logger):
        retriever = self._get_initialized_retriever()
        retriever.ix = None
        retriever.parser = None 
        retriever.close()
        mock_bm25_logger.info.assert_any_call("Closing Whoosh index...")
        mock_bm25_logger.info.assert_any_call("Whoosh index was already None or closed.")
        mock_bm25_logger.error.assert_not_called()
        self.assertIsNone(retriever.ix)
        self.assertIsNone(retriever.parser)


    @patch("llamasearch.core.bm25.logger")
    def test_close_exception(self, mock_bm25_logger):
        mock_index = MagicMock(spec=whoosh_real_index.FileIndex)
        close_error = Exception("Test close error")
        mock_index.close.side_effect = close_error
        retriever = self._get_initialized_retriever(mock_index=mock_index)
        retriever.close()
        mock_index.close.assert_called_once()
        mock_bm25_logger.error.assert_called_once_with(
            f"Error closing Whoosh index: {close_error}" 
        )
        self.assertIsNone(retriever.ix)
        self.assertIsNone(retriever.parser)
        mock_bm25_logger.info.assert_any_call("Whoosh index resources released/nulled.")


    # --- Integration Test ---
    def test_integration_add_query_remove_real_index(self):
        retriever = WhooshBM25Retriever(storage_dir=self.test_dir)
        self.assertIsNotNone(retriever.ix)
        self.assertIsInstance(retriever.ix.schema, WhooshBaseSchema) # type: ignore
        self.assertEqual(set(retriever.ix.schema.names()), {"chunk_id", "content"}) # type: ignore

        self.assertEqual(retriever.get_doc_count(), 0)
        self.assertTrue(retriever.add_document("The quick brown fox", "doc1"))
        self.assertTrue(retriever.add_document("jumps over the lazy dog", "doc2"))
        self.assertTrue(retriever.add_document("A quick test document", "doc3"))
        self.assertEqual(retriever.get_doc_count(), 3)

        results_quick = retriever.query("quick", n_results=2)
        self.assertEqual(results_quick["query"], "quick")
        self.assertCountEqual(results_quick["ids"], ["doc1", "doc3"])
        self.assertEqual(len(results_quick["scores"]), 2)
        self.assertEqual(results_quick["documents"], [None, None])

        results_fox = retriever.query("fox", n_results=1)
        self.assertEqual(results_fox["ids"], ["doc1"])

        self.assertTrue(retriever.remove_document("doc1"))
        self.assertEqual(retriever.get_doc_count(), 2)

        results_fox_after_remove = retriever.query("fox", n_results=1)
        self.assertEqual(results_fox_after_remove["ids"], [])

        results_quick_after_remove = retriever.query("quick", n_results=2)
        self.assertEqual(results_quick_after_remove["ids"], ["doc3"])

        self.assertTrue(retriever.add_document("new content for dog document", "doc2"))
        self.assertEqual(retriever.get_doc_count(), 2)

        results_new_content = retriever.query("new content", n_results=1)
        self.assertEqual(results_new_content["ids"], ["doc2"])

        retriever.close()
        self.assertIsNone(retriever.ix)

        retriever2 = WhooshBM25Retriever(storage_dir=self.test_dir)
        self.assertEqual(retriever2.get_doc_count(), 2)
        results_dog_reopened = retriever2.query("dog document", n_results=1)
        self.assertEqual(results_dog_reopened["ids"], ["doc2"])
        retriever2.close()


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)