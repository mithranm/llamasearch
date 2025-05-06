# tests/test_utils.py
import unittest
import json
import logging
import logging.handlers
import sys
import numpy as np
from pathlib import Path
import tempfile # For creating temporary directories
from unittest.mock import patch, MagicMock, mock_open

# Import components under test or needed for patching
from llamasearch.utils import (
    get_llamasearch_dir,
    NumpyEncoder,
    log_query,
    setup_logging,
)

# Import module itself for patching module-level variables/functions if needed
import llamasearch.utils  # For _qt_log_handler_instance


@patch("llamasearch.utils.Path")  # Mock Path class
class TestGetLlamasearchDir(unittest.TestCase):
    def test_get_dir_always_home(self, MockPath):
        mock_home_path_instance = MagicMock(spec=Path)
        MockPath.home.return_value = mock_home_path_instance
        mock_final_dir_instance = MagicMock(spec=Path)
        mock_home_path_instance.__truediv__.return_value = mock_final_dir_instance

        result_path = get_llamasearch_dir()

        MockPath.home.assert_called_once()
        mock_home_path_instance.__truediv__.assert_called_once_with(".llamasearch")
        mock_final_dir_instance.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )
        self.assertIs(result_path, mock_final_dir_instance)


class TestNumpyEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = NumpyEncoder()

    def test_numpy_float(self):
        data = {"val": np.float32(1.23)}
        result = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(result)
        self.assertAlmostEqual(loaded["val"], 1.23, places=5)

    def test_numpy_int(self):
        data = {"val": np.int64(123)}
        result = json.dumps(data, cls=NumpyEncoder)
        self.assertEqual(result, '{"val": 123}')

    def test_numpy_array(self):
        data = {"arr": np.array([[1, 2], [3, 4]], dtype=np.int32)}
        result = json.dumps(data, cls=NumpyEncoder)
        self.assertEqual(result, '{"arr": [[1, 2], [3, 4]]}')

    def test_set(self):
        data = {"items": {1, "a", 3.0}}
        result = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(result)
        self.assertIsInstance(loaded["items"], list)
        self.assertCountEqual(loaded["items"], [1, "a", 3.0])

    def test_frozenset(self):
        data = {"items": frozenset([True, False])}
        result = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(result)
        self.assertIsInstance(loaded["items"], list)
        self.assertCountEqual(loaded["items"], [True, False])

    def test_path(self):
        p = Path("tmp") / "test" / "file.txt"
        data = {"path": p}
        result = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(result)
        self.assertEqual(loaded["path"], p.as_posix())

    def test_bytes(self):
        data = {"data": b"hello \xc3\xa9 world"}
        result = json.dumps(data, cls=NumpyEncoder)
        self.assertEqual(result, '{"data": "hello \\u00e9 world"}')

    def test_unhandled_type(self):
        class Unhandled:
            pass

        data = {"obj": Unhandled()}
        with self.assertRaises(TypeError):
            json.dumps(data, cls=NumpyEncoder)


@patch("llamasearch.utils.get_llamasearch_dir")
@patch("llamasearch.utils.setup_logging")
@patch("builtins.open", new_callable=mock_open)
@patch("llamasearch.utils.json.dump")
@patch("llamasearch.utils.time.strftime")
class TestLogQuery(unittest.TestCase):
    def setUp(self):
        self.mock_logger_for_log_query = MagicMock(spec=logging.Logger)
        self.temp_log_dir_obj = tempfile.TemporaryDirectory(prefix="test_log_query_")
        self.temp_log_dir = Path(self.temp_log_dir_obj.name)

    def tearDown(self):
        self.temp_log_dir_obj.cleanup()

    def test_log_query_full(
        self,
        mock_strftime,
        mock_json_dump,
        mock_open_func,
        mock_setup_logging_in_log_query,
        mock_get_dir,
    ):
        mock_setup_logging_in_log_query.return_value = self.mock_logger_for_log_query
        mock_get_dir.return_value = self.temp_log_dir # Use real temp path
        # SUT constructs paths: logs_dir = self.temp_log_dir / "logs", log_file = logs_dir / "query_log.jsonl"
        # So, mock_open_func will be called with self.temp_log_dir / "logs" / "query_log.jsonl"

        mock_strftime.return_value = "FAKE_TIMESTAMP"

        query = "Full query"
        chunks = [{"id": "f1", "document": np.array([1, 2])}] 
        response = "Full response"
        debug_info = {
            "retrieval_time": 0.2,
            "llm_generation_time": 0.5,
            "total_query_processing_time": 0.7,
            "vector_initial_results": 10,
            "bm25_initial_results": 5,
            "final_selected_chunk_count": 3,
            "query_embedding_time": 0.1,
            "final_context_content_token_count": 500,
            "estimated_full_prompt_tokens": 550,
            "extra_debug": "value",
            "raw_llm_output": {"details": "..."},
            "final_selected_chunk_details": ["detail1"],
        }
        log_query(query, chunks, response, debug_info, full_logging=True)

        self.assertEqual(mock_json_dump.call_count, 1)
        logged_data = mock_json_dump.call_args[0][0]
        self.assertEqual(logged_data["timestamp"], "FAKE_TIMESTAMP")
        self.assertListEqual(logged_data["chunks_retrieved_details"], chunks)
        expected_debug_keys = {
            "retrieval_time",
            "llm_generation_time",
            "total_query_processing_time",
            "vector_initial_results",
            "bm25_initial_results",
            "final_selected_chunk_count",
            "query_embedding_time",
            "final_context_content_token_count",
            "estimated_full_prompt_tokens",
            "extra_debug",
        }
        self.assertEqual(set(logged_data["debug_info"].keys()), expected_debug_keys)
        self.assertNotIn("raw_llm_output", logged_data["debug_info"])
        self.assertNotIn("final_selected_chunk_details", logged_data["debug_info"])
        # Assert that open was called with the correct constructed path
        mock_open_func.assert_called_once_with(self.temp_log_dir / "logs" / "query_log.jsonl", "a", encoding="utf-8")

    def test_log_query_simplified(
        self,
        mock_strftime,
        mock_json_dump,
        mock_open_func,
        mock_setup_logging_in_log_query,
        mock_get_dir,
    ):
        mock_setup_logging_in_log_query.return_value = self.mock_logger_for_log_query
        mock_get_dir.return_value = self.temp_log_dir # Use real temp path
        mock_strftime.return_value = "FAKE_TIMESTAMP_SIMPLE"

        query = "Simple query"
        long_doc = "This is a very long document text " * 10  
        chunks = [
            {"id": "s1", "score": 0.8, "document": "Short doc 1", "source_path": "/path/a", "filename": "a.txt", "original_chunk_index": 0,},
            {"id": "s2", "score": 0.7, "document": long_doc, "source_path": "/path/b", "filename": "b.md", "original_chunk_index": 5,},
            {"id": "s3", "score": 0.6, "document": None, "source_path": "/path/c", "filename": "c.html",},
            {"id": "s4", "score": 0.5, "source_path": "/path/d", "filename": "d.txt", "original_chunk_index": 2, "text_preview": "Existing preview",},
            "just a string chunk",
        ]
        response = "Simple response"
        debug_info = {
            "retrieval_time": 0.1, "llm_generation_time": 0.3, "total_query_processing_time": 0.45,
            "vector_initial_results": 8, "bm25_initial_results": 4, "final_selected_chunk_count": 2,
            "final_context_content_token_count": 300, "estimated_full_prompt_tokens": 350,
            "query_embedding_time": 0.05, "other_info": "ignore this",
        }
        log_query(query, chunks, response, debug_info, full_logging=False)
        self.assertEqual(mock_json_dump.call_count, 1)
        logged_data = mock_json_dump.call_args[0][0]

        expected_simplified_chunks = [
            {"id": "s1", "score": 0.8, "source_path": "/path/a", "filename": "a.txt", "original_chunk_index": 0, "text_preview": "Short doc 1",},
            {"id": "s2", "score": 0.7, "source_path": "/path/b", "filename": "b.md", "original_chunk_index": 5, "text_preview": long_doc[:150] + "...",},
            {"id": "s3", "score": 0.6, "source_path": "/path/c", "filename": "c.html",},
            {"id": "s4", "score": 0.5, "source_path": "/path/d", "filename": "d.txt", "original_chunk_index": 2, "text_preview": "Existing preview",},
            "just a string chunk",
        ]
        self.assertEqual(logged_data["chunks_retrieved_details"], expected_simplified_chunks)
        essential_debug_keys = {
            "retrieval_time", "llm_generation_time", "total_query_processing_time",
            "vector_initial_results", "bm25_initial_results", "final_selected_chunk_count",
            "final_context_content_token_count", "estimated_full_prompt_tokens", "query_embedding_time",
        }
        self.assertTrue(essential_debug_keys.issubset(logged_data["debug_info"].keys()))
        self.assertNotIn("other_info", logged_data["debug_info"])
        mock_open_func.assert_called_once_with(self.temp_log_dir / "logs" / "query_log.jsonl", "a", encoding="utf-8")

    def test_log_query_mkdir_fail(
        self,
        mock_strftime,
        mock_json_dump,
        mock_open_func,
        mock_setup_logging_in_log_query,
        mock_get_dir,
    ):
        mock_setup_logging_in_log_query.return_value = self.mock_logger_for_log_query
        # Simulate get_llamasearch_dir() returning a path, but logs_dir.mkdir() failing
        mock_real_base_dir = self.temp_log_dir 
        mock_get_dir.return_value = mock_real_base_dir
        
        # To simulate logs_dir.mkdir failing, we need to patch Path.mkdir for the specific instance
        with patch.object(Path, 'mkdir', side_effect=OSError("Permission denied")) as mock_mkdir_method:
            result_path = log_query("q", [], "r", {})
            # Assert that mkdir was called on the logs_dir path
            mock_mkdir_method.assert_any_call(parents=True, exist_ok=True) # get_llamasearch_dir and logs_dir creation attempt


        self.mock_logger_for_log_query.error.assert_called_once_with(
            "Cannot create/access logs directory: Permission denied. Skipping query log."
        )
        mock_open_func.assert_not_called()
        self.assertEqual(result_path, "")

    def test_log_query_file_write_fail(
        self,
        mock_strftime,
        mock_json_dump,
        mock_open_func,
        mock_setup_logging_in_log_query,
        mock_get_dir,
    ):
        mock_setup_logging_in_log_query.return_value = self.mock_logger_for_log_query
        mock_get_dir.return_value = self.temp_log_dir # Use real temp path
        mock_open_func.side_effect = IOError("Disk full")  

        result_path = log_query("q", [], "r", {})
        mock_json_dump.assert_not_called()
        self.mock_logger_for_log_query.error.assert_called_once_with(
            f"Error saving query log to {self.temp_log_dir / 'logs' / 'query_log.jsonl'}: Disk full"
        )
        self.assertEqual(result_path, "")


@patch("llamasearch.utils.get_llamasearch_dir")
class TestSetupLogging(unittest.TestCase):
    def setUp(self):
        self.logger_name = "test_logger_for_setup"
        # Reset module-level state for Qt handler for each test
        llamasearch.utils._qt_log_handler_instance = None 
        llamasearch.utils._qt_logging_available = False

        # Thoroughly clean up logging state before each test
        logging.shutdown() 
        # Remove handlers from the root logger and any specific test loggers
        loggers_to_clear = [logging.getLogger(name) for name in ["llamasearch", "", self.logger_name, "other_logger", "third_logger"]]
        for logger_obj in loggers_to_clear:
            for handler in list(logger_obj.handlers): # Iterate over a copy
                logger_obj.removeHandler(handler)
                if hasattr(handler, "close"):
                    handler.close()
            logger_obj.setLevel(logging.NOTSET) # Reset level
            logger_obj.propagate = True # Reset propagate
            if hasattr(logger_obj, '_noisy_libs_silenced'): # Reset custom flag
                delattr(logger_obj, '_noisy_libs_silenced')
        
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_utils_log_setup_")
        self.temp_dir = Path(self.temp_dir_obj.name)

    def tearDown(self):
        logging.shutdown() # Ensure all handlers are closed
        self.temp_dir_obj.cleanup()
        # Reset module state again after test
        llamasearch.utils._qt_log_handler_instance = None
        llamasearch.utils._qt_logging_available = False


    def test_setup_logging_basic(self, mock_get_dir):
        mock_get_dir.return_value = self.temp_dir

        llamasearch_root_logger = logging.getLogger("llamasearch")
        
        logger_returned = setup_logging(
            self.logger_name, level=logging.INFO, use_qt_handler=False
        )

        self.assertEqual(logger_returned.name, self.logger_name)
        self.assertEqual(logger_returned.level, logging.INFO)
        self.assertEqual(llamasearch_root_logger.level, logging.DEBUG)
        
        # After setup, we expect 2 handlers on the root logger: File and Console.
        self.assertEqual(len(llamasearch_root_logger.handlers), 2)

        file_handler_found = False
        console_handler_found = False
        log_file_path = self.temp_dir / "logs" / "llamasearch.log"

        for handler in llamasearch_root_logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                if Path(handler.baseFilename).resolve() == log_file_path.resolve():
                    file_handler_found = True
                    self.assertEqual(handler.level, logging.DEBUG)
            elif isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                console_handler_found = True
                self.assertEqual(handler.level, logging.INFO)
        
        self.assertTrue(file_handler_found, "File handler not found or misconfigured")
        self.assertTrue(console_handler_found, "Console handler not found or misconfigured")
        self.assertTrue(getattr(llamasearch_root_logger, '_noisy_libs_silenced', False))

    @patch("llamasearch.utils.qt_log_emitter") 
    def test_setup_logging_with_qt(self, mock_qt_emitter_global_obj, mock_get_dir):
        mock_get_dir.return_value = self.temp_dir
        llamasearch.utils._qt_logging_available = True
        llamasearch.utils.qt_log_emitter = mock_qt_emitter_global_obj

        llamasearch_root_logger = logging.getLogger("llamasearch")
        
        # --- First Call (DEBUG level, use_qt_handler=True) ---
        logger1 = setup_logging(self.logger_name, level=logging.DEBUG, use_qt_handler=True)
        self.assertEqual(logger1.level, logging.DEBUG)
        self.assertEqual(len(llamasearch_root_logger.handlers), 3) # File, Console, Qt
        
        qt_handler_instance = llamasearch.utils._qt_log_handler_instance
        self.assertIsNotNone(qt_handler_instance)
        self.assertIsInstance(qt_handler_instance, llamasearch.utils.QtLogHandler) # type: ignore
        self.assertEqual(qt_handler_instance.level, logging.DEBUG) # type: ignore

        for h in llamasearch_root_logger.handlers:
            if isinstance(h, logging.handlers.RotatingFileHandler):
                self.assertEqual(h.level, logging.DEBUG)
            elif isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
                self.assertEqual(h.level, logging.DEBUG)

        # --- Second Call (INFO level, use_qt_handler=True) ---
        logger2 = setup_logging("other_logger", level=logging.INFO, use_qt_handler=True)
        self.assertEqual(logger2.level, logging.INFO)
        self.assertEqual(len(llamasearch_root_logger.handlers), 3) 
        self.assertEqual(qt_handler_instance.level, logging.INFO) # type: ignore
        for h in llamasearch_root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
                self.assertEqual(h.level, logging.INFO)

        # --- Third Call (DEBUG level, use_qt_handler=False) ---
        logger3 = setup_logging("third_logger", level=logging.DEBUG, use_qt_handler=False)
        self.assertEqual(logger3.level, logging.DEBUG)
        self.assertEqual(len(llamasearch_root_logger.handlers), 3)
        self.assertEqual(qt_handler_instance.level, logging.INFO) # type: ignore
        for h in llamasearch_root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
                self.assertEqual(h.level, logging.DEBUG)

    @patch("llamasearch.utils.logging.basicConfig") # Mock basicConfig to check its call
    def test_setup_logging_exception_logs_error(self, mock_basic_config, mock_get_dir):
        mock_get_dir.side_effect = Exception("Another config error")
        
        # We use assertLogs to capture messages logged to the root logger.
        # The `setup_logging` function, in its except block, calls `logging.error`.
        # This `logging.error` will use the handlers configured by `logging.basicConfig`
        # (which is called just before `logging.error` in the except block).
        with self.assertLogs(level="ERROR") as cm: # Captures from root logger
            setup_logging("another_logger", level=logging.WARNING)
            
        self.assertTrue(any("Failed custom logging setup: Another config error" in msg for msg in cm.output))
        # `basicConfig` in the SUT's `except` block is hardcoded to `logging.INFO`.
        mock_basic_config.assert_called_once_with(level=logging.INFO)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)