# tests/test_app_logic.py
import unittest
from unittest.mock import MagicMock, patch, ANY, AsyncMock 
import asyncio
import tempfile
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from PySide6.QtCore import Signal, QObject # Import QObject for AppLogicSignals

from llamasearch.ui.app_logic import LlamaSearchApp, AppLogicSignals
from llamasearch.core.search_engine import LLMSearch
from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.exceptions import ModelNotFoundError


# Path for data_manager used by AppLogic
MOCK_APP_LOGIC_DATAMANAGER_PATH = "llamasearch.ui.app_logic.data_manager"

# Patch setup_logging for this test module
MOCK_APP_LOGIC_SETUP_LOGGING_TARGET = "llamasearch.ui.app_logic.setup_logging"
mock_app_logic_logger_global_instance = MagicMock(spec=logging.Logger)


class TestAppLogicModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logger_patcher = patch(MOCK_APP_LOGIC_SETUP_LOGGING_TARGET, return_value=mock_app_logic_logger_global_instance)
        cls.logger_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.logger_patcher.stop()

    def setUp(self):
        mock_app_logic_logger_global_instance.reset_mock()

        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="test_app_logic_")
        self.index_dir = Path(self.temp_dir_obj.name) / "index"
        self.crawl_dir = Path(self.temp_dir_obj.name) / "crawl_data"
        
        self.data_manager_patcher = patch(MOCK_APP_LOGIC_DATAMANAGER_PATH)
        self.mock_data_manager = self.data_manager_patcher.start()
        self.mock_data_manager.get_data_paths.return_value = {
            "index": str(self.index_dir),
            "crawl_data": str(self.crawl_dir),
            "base": str(self.temp_dir_obj.name)
        }

        self.mock_executor = MagicMock(spec=ThreadPoolExecutor)
        
        self.MockLLMSearch = patch("llamasearch.ui.app_logic.LLMSearch").start()
        self.mock_llmsearch_instance = MagicMock(spec=LLMSearch)
        self.mock_llmsearch_instance.model = MagicMock() 
        self.mock_llmsearch_instance.model.model_info.model_id = "test-model"
        self.mock_llmsearch_instance.model.model_info.model_engine = "test-engine"
        self.mock_llmsearch_instance.model.model_info.context_length = 1024
        self.MockLLMSearch.return_value = self.mock_llmsearch_instance
        
        self.MockCrawl4AICrawler = patch("llamasearch.ui.app_logic.Crawl4AICrawler").start()
        self.mock_crawler_instance = MagicMock(spec=Crawl4AICrawler)
        self.MockCrawl4AICrawler.return_value = self.mock_crawler_instance

        self.MockQTimer = patch("llamasearch.ui.app_logic.QTimer").start()
        
        # Store the original QTimer.singleShot for restoration if needed
        self.original_qtimer_singleShot = self.MockQTimer.singleShot

        # Define the immediate call behavior as an instance method
        def _qtimer_singleshot_immediate_call_impl(delay, func_to_call, *args, **kwargs):
            if not hasattr(self.MockQTimer.singleShot, 'custom_side_effect') or \
               self.MockQTimer.singleShot.custom_side_effect is None:
                if delay == 0 or delay == 100 or delay == 150: 
                    func_to_call()
            elif callable(self.MockQTimer.singleShot.custom_side_effect):
                 self.MockQTimer.singleShot.custom_side_effect(delay, func_to_call, *args, **kwargs)
        
        self._qtimer_singleshot_immediate_call_impl = _qtimer_singleshot_immediate_call_impl
        self.MockQTimer.singleShot.side_effect = self._qtimer_singleshot_immediate_call_impl
        self.MockQTimer.singleShot.custom_side_effect = None

        # Patch AppLogicSignals class to return a MagicMock instance for its constructor
        self.app_logic_signals_patcher = patch("llamasearch.ui.app_logic.AppLogicSignals")
        self.MockAppLogicSignals_cls = self.app_logic_signals_patcher.start()
        self.mock_app_logic_signals_instance = MagicMock(spec=AppLogicSignals)
        # Ensure the mocked signals object has the necessary signals as mocks
        for signal_name in AppLogicSignals.__dict__:
            if isinstance(getattr(AppLogicSignals, signal_name), Signal):
                setattr(self.mock_app_logic_signals_instance, signal_name, MagicMock(spec=Signal))
        
        self.MockAppLogicSignals_cls.return_value = self.mock_app_logic_signals_instance
        
        self.app_logic: LlamaSearchApp 


    def tearDown(self):
        # Restore original QTimer.singleShot if it was changed
        if hasattr(self, 'original_qtimer_singleShot'):
            self.MockQTimer.singleShot = self.original_qtimer_singleShot

        if hasattr(self, 'app_logic') and self.app_logic: 
            self.app_logic.close()
        patch.stopall() # Stops all patchers started with patch()
        self.temp_dir_obj.cleanup()
        # No need to individually stop patchers started with .start() if using patch.stopall()
        # self.data_manager_patcher.stop()
        # self.app_logic_signals_patcher.stop()


    def _create_app_logic(self, debug=False, init_llmsearch_success=True):
        if not init_llmsearch_success:
            self.MockLLMSearch.side_effect = ModelNotFoundError("Test init fail")
            
        self.app_logic = LlamaSearchApp(executor=self.mock_executor, debug=debug)
        # self.app_logic.signals is already the mock_app_logic_signals_instance due to class patching
        # We need to ensure that the internal signal connection is also to a mock if we want to test that path
        # Or, allow the real connection and mock the slot it connects to.
        # For `_internal_task_completed`, AppLogic connects it to `self._final_gui_callback`.
        # So, `self.mock_app_logic_signals_instance._internal_task_completed.connect` would have been called.
        # To test `_final_gui_callback`, we can call it directly or mock `_internal_task_completed.emit`.


    def test_app_logic_init_success(self):
        self._create_app_logic(init_llmsearch_success=True)
        self.MockLLMSearch.assert_called_once_with(
            storage_dir=self.index_dir, 
            shutdown_event=ANY, 
            debug=False,
            verbose=False,
            max_results=3
        )
        self.assertIsNotNone(self.app_logic.llm_search)
        self.assertEqual(self.app_logic._current_config["model_id"], "test-model")
        # The signals object is now self.mock_app_logic_signals_instance
        self.mock_app_logic_signals_instance.refresh_needed.emit.assert_called_once()


    def test_app_logic_init_llm_search_fail_model_not_found(self):
        self.MockLLMSearch.side_effect = ModelNotFoundError("LLM init error")
        
        # AppLogic constructor will use the mocked AppLogicSignals class,
        # so self.mock_app_logic_signals_instance will be self.app_logic.signals
        self._create_app_logic(init_llmsearch_success=False)
        
        self.assertIsNone(self.app_logic.llm_search)
        mock_app_logic_logger_global_instance.error.assert_any_call(
            "Model setup required: LLM init error. Run 'llamasearch-setup'."
        )
        self.mock_app_logic_signals_instance.status_updated.emit.assert_called_once_with(
            "Backend initialization failed. Run setup.", "error"
        )


    def test_run_in_background_success(self):
        self._create_app_logic()
        mock_task_func = MagicMock(return_value=("Task success message", True))
        mock_task_func.__name__ = "mock_task_func_success"
        
        # Create a real Signal instance for completion_signal to test _final_gui_callback's emit
        class TestSignals(QObject):
            test_completion_signal = Signal(str, bool)
        real_test_signals = TestSignals()
        mock_slot_for_completion = MagicMock()
        real_test_signals.test_completion_signal.connect(mock_slot_for_completion)

        mock_future = MagicMock(spec=Future)
        self.mock_executor.submit.return_value = mock_future

        # Temporarily use original QTimer.singleShot for this specific test path
        self.MockQTimer.singleShot.side_effect = self.original_qtimer_singleShot.side_effect

        self.app_logic._run_in_background(mock_task_func, "arg1", completion_signal=real_test_signals.test_completion_signal)
        
        # Restore mocked QTimer behavior
        self.MockQTimer.singleShot.side_effect = self._qtimer_singleshot_immediate_call_impl

        self.mock_executor.submit.assert_called_once_with(mock_task_func, "arg1")
        mock_future.add_done_callback.assert_called_once()
        
        done_callback = mock_future.add_done_callback.call_args[0][0]
        mock_future.cancelled.return_value = False
        mock_future.exception.return_value = None
        mock_future.result.return_value = ("Task success message", True)
        
        # Mock the _internal_task_completed.emit which is on our mocked signals object
        self.mock_app_logic_signals_instance._internal_task_completed.emit.reset_mock()
        done_callback(mock_future) 

        self.mock_app_logic_signals_instance._internal_task_completed.emit.assert_called_once_with(
            ("Task success message", True), None, False, real_test_signals.test_completion_signal
        )

        # Test _final_gui_callback directly, which will use the real signal
        self.app_logic._final_gui_callback(
            result=("Task success message", True),
            exception=None,
            cancelled=False,
            completion_signal=real_test_signals.test_completion_signal # Pass the real signal
        )
        mock_slot_for_completion.assert_called_once_with("Task success message", True)
        self.mock_app_logic_signals_instance.actions_should_reenable.emit.assert_called()


    def test_run_in_background_exception_in_task(self):
        self._create_app_logic()
        mock_task_func = MagicMock(side_effect=ValueError("Task error"))
        mock_task_func.__name__ = "mock_task_func_exception"

        class TestSignals(QObject):
            test_completion_signal = Signal(str, bool)
        real_test_signals = TestSignals()
        mock_slot_for_completion = MagicMock()
        real_test_signals.test_completion_signal.connect(mock_slot_for_completion)

        mock_future = MagicMock(spec=Future)
        self.mock_executor.submit.return_value = mock_future

        self.MockQTimer.singleShot.side_effect = self.original_qtimer_singleShot.side_effect
        self.app_logic._run_in_background(mock_task_func, completion_signal=real_test_signals.test_completion_signal)
        self.MockQTimer.singleShot.side_effect = self._qtimer_singleshot_immediate_call_impl
        
        done_callback = mock_future.add_done_callback.call_args[0][0]
        mock_future.cancelled.return_value = False
        mock_future.exception.return_value = ValueError("Task error") 
        
        self.mock_app_logic_signals_instance._internal_task_completed.emit.reset_mock()
        done_callback(mock_future)
        self.mock_app_logic_signals_instance._internal_task_completed.emit.assert_called_once_with(
            None, ANY, False, real_test_signals.test_completion_signal 
        )
        self.assertIsInstance(self.mock_app_logic_signals_instance._internal_task_completed.emit.call_args[0][1], ValueError)

        self.app_logic._final_gui_callback(
            result=None,
            exception=ValueError("Task error"),
            cancelled=False,
            completion_signal=real_test_signals.test_completion_signal
        )
        mock_slot_for_completion.assert_called_once_with("Task Error: Task error", False)
        self.mock_app_logic_signals_instance.actions_should_reenable.emit.assert_called()


    def test_run_in_background_task_cancelled(self):
        self._create_app_logic()
        mock_task_func = MagicMock()
        mock_task_func.__name__ = "mock_task_func_cancelled"
        class TestSignals(QObject):
            test_completion_signal = Signal(str, bool)
        real_test_signals = TestSignals()
        mock_slot_for_completion = MagicMock()
        real_test_signals.test_completion_signal.connect(mock_slot_for_completion)

        mock_future = MagicMock(spec=Future)
        self.mock_executor.submit.return_value = mock_future

        self.MockQTimer.singleShot.side_effect = self.original_qtimer_singleShot.side_effect
        self.app_logic._run_in_background(mock_task_func, completion_signal=real_test_signals.test_completion_signal)
        self.MockQTimer.singleShot.side_effect = self._qtimer_singleshot_immediate_call_impl
        
        done_callback = mock_future.add_done_callback.call_args[0][0]
        mock_future.cancelled.return_value = True 

        self.mock_app_logic_signals_instance._internal_task_completed.emit.reset_mock()
        done_callback(mock_future)
        self.mock_app_logic_signals_instance._internal_task_completed.emit.assert_called_once_with(
            None, None, True, real_test_signals.test_completion_signal
        )
        self.app_logic._final_gui_callback(
            result=None,
            exception=None,
            cancelled=True,
            completion_signal=real_test_signals.test_completion_signal
        )
        mock_slot_for_completion.assert_called_once_with("Task cancelled during execution.", False)
        self.mock_app_logic_signals_instance.actions_should_reenable.emit.assert_called()


    def test_submit_search_success(self):
        self._create_app_logic()
        with patch.object(self.app_logic._thread_pool, "submit") as mock_submit_to_pool:
            mock_future = MagicMock(spec=Future)
            mock_submit_to_pool.return_value = mock_future
            
            self.app_logic.submit_search("test query")
        
        mock_submit_to_pool.assert_called_once_with(
            self.app_logic._execute_search_task, "test query"
        )
        mock_future.add_done_callback.assert_called_once()
        self.mock_app_logic_signals_instance.status_updated.emit.assert_called_with("Searching 'test query...'","info")


    def test_submit_crawl_and_index_success(self):
        self._create_app_logic()
        urls = ["http://url1.com"]
        target_links = 10
        max_depth = 2
        keywords = ["kw1"]
        with patch.object(self.app_logic._thread_pool, "submit") as mock_submit_to_pool:
            mock_future = MagicMock(spec=Future)
            mock_submit_to_pool.return_value = mock_future
            self.app_logic.submit_crawl_and_index(urls, target_links, max_depth, keywords)

        mock_submit_to_pool.assert_called_once_with(
            self.app_logic._execute_crawl_and_index_task,
            urls, target_links, max_depth, keywords
        )
        mock_future.add_done_callback.assert_called_once()
        self.mock_app_logic_signals_instance.status_updated.emit.assert_called_with("Starting crawl & index for 1 URL(s)...", "info")


    def test_execute_search_task_success(self):
        self._create_app_logic()
        self.mock_llmsearch_instance.llm_query.return_value = {"formatted_response": "HTML response"}
        
        message, success = self.app_logic._execute_search_task("a query")
        
        self.assertTrue(success)
        self.assertEqual(message, "HTML response")
        self.mock_llmsearch_instance.llm_query.assert_called_once_with("a query", debug_mode=False)


    def test_execute_search_task_llm_search_not_ready(self):
        self._create_app_logic()
        self.app_logic.llm_search = None 
        
        message, success = self.app_logic._execute_search_task("a query")
        
        self.assertFalse(success)
        self.assertEqual(message, "Search Error: LLMSearch instance not available.")


    @patch('asyncio.get_event_loop_policy')
    @patch('llamasearch.ui.app_logic.Path') 
    def test_execute_crawl_and_index_task_success(self, MockSUTPath, mock_get_policy):
        self._create_app_logic()
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_loop.is_running.return_value = False # Assume loop is not running for close tests
        mock_loop.is_closed.return_value = False

        mock_asyncio_future = asyncio.Future() # type: ignore
        mock_asyncio_future.set_result(["http://url1/crawled"])
        mock_loop.run_until_complete.return_value = ["http://url1/crawled"]
        
        mock_policy_obj = MagicMock()
        mock_policy_obj.new_event_loop.return_value = mock_loop
        mock_policy_obj.get_event_loop.return_value = mock_loop
        mock_get_policy.return_value = mock_policy_obj

        self.mock_crawler_instance.run_crawl = AsyncMock(return_value=["http://url1/crawled"])
        self.mock_crawler_instance.close = AsyncMock()
        self.mock_llmsearch_instance.add_source.return_value = (3, False)


        mock_crawl_dir_base = MagicMock(spec=Path)
        mock_raw_output_dir = MagicMock(spec=Path)
        mock_md_file_path_obj = MagicMock(spec=Path)
        
        # Configure side effect for Path mock used within the SUT
        def path_side_effect_for_crawl_task(path_arg):
            if str(path_arg) == str(self.crawl_dir):
                return mock_crawl_dir_base
            elif str(path_arg) == str(self.crawl_dir / "raw"):
                return mock_raw_output_dir
            # Allow specific file paths to be created if needed
            elif Path(str(path_arg)).name == "some_crawled_file.md":
                 mock_md_file_path_obj.name = "some_crawled_file.md" # Ensure name is set
                 mock_md_file_path_obj.__str__ = MagicMock(return_value=str(self.crawl_dir / "raw" / "some_crawled_file.md"))
                 return mock_md_file_path_obj
            return MagicMock(spec=Path, name=f"GenericPathMock({path_arg})")


        MockSUTPath.side_effect = path_side_effect_for_crawl_task
        
        mock_crawl_dir_base.is_dir.return_value = True
        mock_crawl_dir_base.__truediv__.return_value = mock_raw_output_dir
        
        mock_raw_output_dir.is_dir.return_value = True
        # Ensure glob returns the specifically configured mock_md_file_path_obj
        mock_raw_output_dir.glob.return_value = [mock_md_file_path_obj] 

        # Mock _load_reverse_lookup if it reads files
        with patch.object(self.app_logic.llm_search, '_load_reverse_lookup', MagicMock()):
            urls = ["http://url1.com"]
            message, success = self.app_logic._execute_crawl_and_index_task(urls, 5, 1, None)
        
        self.assertTrue(success)
        self.assertIn("Crawl OK", message)
        self.assertIn("Index OK", message)
        self.assertIn("3 chunks added", message)
        self.MockCrawl4AICrawler.assert_called_once()
        self.mock_crawler_instance.run_crawl.assert_called_once()
        # Assert add_source was called with the string representation of the path
        self.mock_llmsearch_instance.add_source.assert_called_with(
            str(self.crawl_dir / "raw" / "some_crawled_file.md"), internal_call=True
        )
        self.mock_app_logic_signals_instance.refresh_needed.emit.assert_called_once()


    def test_apply_settings_debug_mode_change(self):
        self._create_app_logic(debug=False)
        self.assertFalse(self.app_logic.debug)
        
        mock_root_logger = MagicMock(spec=logging.Logger)
        mock_file_handler = MagicMock(spec=logging.FileHandler)
        mock_stream_handler = MagicMock(spec=logging.StreamHandler)
        mock_root_logger.handlers = [mock_file_handler, mock_stream_handler]
        
        with patch('logging.getLogger', return_value=mock_root_logger):
            self.app_logic.apply_settings({"debug_mode": True, "max_results": 5})

        self.assertTrue(self.app_logic.debug)
        mock_root_logger.setLevel.assert_called_with(logging.DEBUG)
        mock_file_handler.setLevel.assert_called_with(logging.DEBUG) 
        mock_stream_handler.setLevel.assert_called_with(logging.DEBUG) 
        self.assertEqual(self.app_logic._current_config["max_results"], 5)
        self.assertEqual(self.mock_llmsearch_instance.max_results, 5)
        self.mock_app_logic_signals_instance.settings_applied.emit.assert_called_with("Settings applied successfully.", "success")


    def test_close_logic(self):
        self._create_app_logic()
        self.app_logic._active_crawler = self.mock_crawler_instance
        
        self.app_logic.close()
        
        self.assertTrue(self.app_logic._shutdown_event.is_set())
        self.mock_crawler_instance.abort.assert_called_once()
        self.mock_llmsearch_instance.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()