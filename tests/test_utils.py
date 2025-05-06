# tests/test_utils.py

import unittest
import os
import sys
import json
import logging
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open, ANY, call

# External library mocks needed for type hints or direct patching
# import torch # Not actually used in this file's tests
import numpy as np # Need numpy to test the encoder

# --- Project-specific imports ---
# Import the module itself to access module-level variables if needed
import llamasearch.utils

# Import specific components under test or needed for patching
from llamasearch.utils import (
    is_dev_mode,
    get_llamasearch_dir,
    setup_logging,
    NumpyEncoder,
    log_query,
    _qt_logging_available, # Import to allow patching
    QtLogHandler,          # Import to allow patching/checking type
    qt_log_emitter         # Import to allow patching
)

# (No need to import data_manager here, it will be patched at its source)


class TestIsDevMode(unittest.TestCase):
    # --- No changes needed here ---
    @patch.dict(os.environ, {}, clear=True)
    def test_dev_mode_unset(self):
        self.assertFalse(is_dev_mode())

    @patch.dict(os.environ, {"LLAMASEARCH_DEV_MODE": "1"}, clear=True)
    def test_dev_mode_1(self):
        self.assertTrue(is_dev_mode())

    @patch.dict(os.environ, {"LLAMASEARCH_DEV_MODE": "true"}, clear=True)
    def test_dev_mode_true(self):
        self.assertTrue(is_dev_mode())

    @patch.dict(os.environ, {"LLAMASEARCH_DEV_MODE": "YES"}, clear=True)
    def test_dev_mode_yes_case_insensitive(self):
        self.assertTrue(is_dev_mode())

    @patch.dict(os.environ, {"LLAMASEARCH_DEV_MODE": "0"}, clear=True)
    def test_dev_mode_0(self):
        self.assertFalse(is_dev_mode())

    @patch.dict(os.environ, {"LLAMASEARCH_DEV_MODE": "false"}, clear=True)
    def test_dev_mode_false(self):
        self.assertFalse(is_dev_mode())

    @patch.dict(os.environ, {"LLAMASEARCH_DEV_MODE": "no"}, clear=True)
    def test_dev_mode_no(self):
        self.assertFalse(is_dev_mode())

    @patch.dict(os.environ, {"LLAMASEARCH_DEV_MODE": "other"}, clear=True)
    def test_dev_mode_other(self):
        self.assertFalse(is_dev_mode())


# --- FIX 1: Patch the actual data_manager singleton instance ---
@patch('llamasearch.data_manager.data_manager')
@patch('llamasearch.utils.Path')         # Mock the Path class used inside
class TestGetLlamasearchDir(unittest.TestCase):

    def test_dir_from_data_manager(self, MockPath, mock_data_manager_instance):
        """Test getting the directory from data_manager."""
        mock_base_path_str = "/fake/data/manager/base"
        mock_data_manager_instance.get_data_paths.return_value = {"base": mock_base_path_str}

        mock_path_instance = MagicMock(spec=Path)
        MockPath.return_value = mock_path_instance

        result_path = get_llamasearch_dir()

        mock_data_manager_instance.get_data_paths.assert_called_once()
        MockPath.assert_called_once_with(mock_base_path_str)
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertIs(result_path, mock_path_instance)

    # --- FIX 1: Removed inner patch, configure home on MockPath ---
    # @patch('llamasearch.utils.Path.home') # REMOVED
    def test_dir_fallback_to_home(self, MockPath, mock_data_manager_instance): # mock_home removed from args
        """Test fallback to ~/.llamasearch when data_manager lacks 'base'."""
        mock_data_manager_instance.get_data_paths.return_value = {"logs": "/some/logs"} # No 'base'

        # Configure MockPath.home directly
        mock_home_path_instance = MagicMock(spec=Path)
        MockPath.home.return_value = mock_home_path_instance # Configure the class mock

        mock_fallback_dir_instance = MagicMock(spec=Path)
        mock_home_path_instance.__truediv__.return_value = mock_fallback_dir_instance

        result_path = get_llamasearch_dir()

        mock_data_manager_instance.get_data_paths.assert_called_once()
        MockPath.home.assert_called_once() # Assert the mocked class method was called
        mock_home_path_instance.__truediv__.assert_called_once_with(".llamasearch")
        mock_fallback_dir_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertIs(result_path, mock_fallback_dir_instance)
        # Ensure the regular Path constructor wasn't called with a string
        # Check calls to MockPath excluding the .home() call
        path_constructor_calls = [c for c in MockPath.call_args_list if c != call.home()]
        self.assertEqual(len(path_constructor_calls), 0)



class TestNumpyEncoder(unittest.TestCase):
    # --- No changes needed here ---
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
        p = Path("/tmp/test/file.txt")
        data = {"path": p}
        result = json.dumps(data, cls=NumpyEncoder)
        self.assertEqual(result, f'{{"path": "{str(p)}"}}')

    def test_bytes(self):
        data = {"data": b"hello \xc3\xa9 world"}
        result = json.dumps(data, cls=NumpyEncoder)
        self.assertEqual(result, '{"data": "hello \\u00e9 world"}')

    def test_unhandled_type(self):
        class Unhandled: pass
        data = {"obj": Unhandled()}
        with self.assertRaises(TypeError):
            json.dumps(data, cls=NumpyEncoder)


# --- FIX 1: Patch the actual data_manager singleton instance ---
@patch('llamasearch.data_manager.data_manager')
@patch('llamasearch.utils.Path')
@patch('llamasearch.utils.setup_logging')
@patch('builtins.open', new_callable=mock_open)
@patch('llamasearch.utils.json.dump')
class TestLogQuery(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock(spec=logging.Logger)

    def test_log_query_full(self, mock_json_dump, mock_open_func, mock_setup_logging, MockPath, mock_data_manager_instance):
        """Test query logging with full_logging=True."""
        mock_setup_logging.return_value = self.mock_logger
        mock_data_manager_instance.get_data_paths.return_value = {"logs": "/fake/logs"}
        mock_log_dir_instance = MagicMock(spec=Path); mock_log_file_instance = MagicMock(spec=Path)
        MockPath.return_value = mock_log_dir_instance; mock_log_dir_instance.__truediv__.return_value = mock_log_file_instance

        query = "Full query"; chunks = [{"id": "f1", "document": np.array([1,2])}]; response = "Full response"
        debug_info = {"retrieval_time": 0.2, "extra_debug": "value", "raw_llm_output": {"details": "..."}}

        log_query(query, chunks, response, debug_info, full_logging=True)

        self.assertEqual(mock_json_dump.call_count, 1)
        logged_data = mock_json_dump.call_args[0][0]
        self.assertListEqual(logged_data['chunks_retrieved_details'], chunks)
        expected_debug = {"retrieval_time": 0.2, "extra_debug": "value"}
        self.assertDictEqual(logged_data['debug_info'], expected_debug)

    def test_log_query_mkdir_fail(self, mock_json_dump, mock_open_func, mock_setup_logging, MockPath, mock_data_manager_instance):
        """Test log_query when creating the logs directory fails."""
        mock_setup_logging.return_value = self.mock_logger
        mock_data_manager_instance.get_data_paths.return_value = {"logs": "/restricted/logs"}
        mock_log_dir_instance = MagicMock(spec=Path); MockPath.return_value = mock_log_dir_instance
        mock_log_dir_instance.mkdir.side_effect = OSError("Permission denied")

        result_path = log_query("q", [], "r", {})

        self.mock_logger.error.assert_called_once_with("Cannot create/access logs directory: Permission denied. Skipping query log.")
        mock_open_func.assert_not_called(); mock_json_dump.assert_not_called()
        self.assertEqual(result_path, "")

    def test_log_query_file_write_fail(self, mock_json_dump, mock_open_func, mock_setup_logging, MockPath, mock_data_manager_instance):
        """Test log_query when writing to the log file fails."""
        mock_setup_logging.return_value = self.mock_logger
        mock_data_manager_instance.get_data_paths.return_value = {"logs": "/fake/logs"}
        mock_log_dir_instance = MagicMock(spec=Path); mock_log_file_instance = MagicMock(spec=Path)
        MockPath.return_value = mock_log_dir_instance; mock_log_dir_instance.__truediv__.return_value = mock_log_file_instance
        mock_json_dump.side_effect = IOError("Disk full")

        result_path = log_query("q", [], "r", {})

        mock_open_func.assert_called_once()
        self.mock_logger.error.assert_called_once_with(f"Error saving query log to {mock_log_file_instance}: Disk full")
        self.assertEqual(result_path, "")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)