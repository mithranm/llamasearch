# tests/test_app_logic.py
import unittest
from unittest.mock import MagicMock, patch, ANY
import tempfile
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from PySide6.QtCore import Signal, QObject # Import QObject for signals

from llamasearch.ui.app_logic import LlamaSearchApp
from llamasearch.core.search_engine import LLMSearch
from llamasearch.core.crawler import Crawl4AICrawler
from llamasearch.exceptions import ModelNotFoundError


MOCK_APP_LOGIC_DATAMANAGER_PATH = "llamasearch.ui.app_logic.data_manager"
# Patch the logger instance directly in the app_logic module
MOCK_APP_LOGIC_LOGGER_INSTANCE_TARGET = "llamasearch.ui.app_logic.logger"


class TestSignals(QObject):
    status_updated = Signal(str, str)
    search_completed = Signal(str, bool)
    crawl_index_completed = Signal(str, bool)
    manual_index_completed = Signal(str, bool)
    removal_completed = Signal(str, bool)
    refresh_needed = Signal()
    settings_applied = Signal(str, str)
    actions_should_reenable = Signal()
    _internal_task_completed = Signal(object, object, bool, Signal)


class TestAppLogicModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Patch the logger instance directly in the app_logic module
        cls.module_logger_patcher = patch(MOCK_APP_LOGIC_LOGGER_INSTANCE_TARGET, spec=logging.Logger)
        cls.mock_app_logic_module_logger_direct = cls.module_logger_patcher.start()


    @classmethod
    def tearDownClass(cls):
        cls.module_logger_patcher.stop()


    def setUp(self):
        # Reset the shared mock logger for each test method
        self.mock_app_logic_module_logger_direct.reset_mock()
        # Specifically reset call lists for methods like 'error', 'info', etc.
        for method_name in ['error', 'info', 'warning', 'debug', 'critical']:
            if hasattr(self.mock_app_logic_module_logger_direct, method_name):
                getattr(self.mock_app_logic_module_logger_direct, method_name).reset_mock()

        # Add handlers attribute to the mock logger for test_apply_settings_debug_mode_change
        self.mock_app_logic_module_logger_direct.handlers = []


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
        
        self.qtimer_lambdas_map = {} 
        def store_qtimer_lambda(delay, func_to_call, *args, **kwargs):
            if delay not in self.qtimer_lambdas_map:
                self.qtimer_lambdas_map[delay] = []
            self.qtimer_lambdas_map[delay].append(func_to_call)

        self.MockQTimer.singleShot.side_effect = store_qtimer_lambda
        
        self.app_logic: LlamaSearchApp

    def tearDown(self):
        if hasattr(self, 'app_logic') and self.app_logic:
            self.app_logic.close()
        # patch.stopall() # This can interfere if not managed carefully per test
        self.data_manager_patcher.stop()
        self.MockLLMSearch.stop()
        self.MockCrawl4AICrawler.stop()
        self.MockQTimer.stop()
        self.temp_dir_obj.cleanup()
        self.qtimer_lambdas_map.clear()

    def _create_app_logic(self, debug=False, init_llmsearch_success=True):
        if not init_llmsearch_success:
            self.MockLLMSearch.side_effect = ModelNotFoundError("Test init fail")
        self.app_logic = LlamaSearchApp(executor=self.mock_executor, debug=debug)


    def test_app_logic_init_success(self):
        self._create_app_logic(init_llmsearch_success=True)

        mock_refresh_slot = MagicMock()
        self.app_logic.signals.refresh_needed.connect(mock_refresh_slot)

        self.assertTrue(150 in self.qtimer_lambdas_map and self.qtimer_lambdas_map[150])
        refresh_lambda = self.qtimer_lambdas_map[150][0] 
        refresh_lambda() 

        mock_refresh_slot.assert_called_once()

        self.MockLLMSearch.assert_called_once_with(
            storage_dir=self.index_dir,
            shutdown_event=ANY,
            debug=False,
            verbose=False,
            max_results=3
        )
        self.assertIsNotNone(self.app_logic.llm_search)
        self.assertEqual(self.app_logic._current_config["model_id"], "test-model")

    def test_app_logic_init_llm_search_fail_model_not_found(self):
        self.MockLLMSearch.side_effect = ModelNotFoundError("LLM init error")

        app_logic_instance = LlamaSearchApp(executor=self.mock_executor, debug=False)

        mock_status_slot = MagicMock()
        app_logic_instance.signals.status_updated.connect(mock_status_slot)

        self.assertTrue(100 in self.qtimer_lambdas_map and self.qtimer_lambdas_map[100])
        status_update_lambda = self.qtimer_lambdas_map[100][0]
        status_update_lambda()

        mock_status_slot.assert_called_once_with("Backend initialization failed. Run setup.", "error")

        self.assertIsNone(app_logic_instance.llm_search)
        
        self.mock_app_logic_module_logger_direct.error.assert_any_call(
            "Model setup required: LLM init error. Run 'llamasearch-setup'."
        )
        app_logic_instance.close()


    def test_run_in_background_success(self):
        self._create_app_logic()
        mock_task_func = MagicMock(return_value=("Task success message", True))
        mock_task_func.__name__ = "mock_task_func_success"

        test_completion_signal_emitter = TestSignals()
        mock_completion_slot = MagicMock()
        test_completion_signal_emitter.manual_index_completed.connect(mock_completion_slot)
        completion_signal_instance = test_completion_signal_emitter.manual_index_completed

        mock_future = MagicMock(spec=Future)
        self.mock_executor.submit.return_value = mock_future

        mock_internal_slot = MagicMock()
        self.app_logic.signals._internal_task_completed.connect(mock_internal_slot)

        self.app_logic._run_in_background(mock_task_func, "arg1", completion_signal=completion_signal_instance)

        self.mock_executor.submit.assert_called_once_with(mock_task_func, "arg1")
        mock_future.add_done_callback.assert_called_once()

        done_callback = mock_future.add_done_callback.call_args[0][0]

        mock_future.cancelled.return_value = False
        mock_future.exception.return_value = None
        mock_future.result.return_value = ("Task success message", True)

        done_callback(mock_future)
        mock_internal_slot.assert_called_once_with(
            ("Task success message", True), None, False, completion_signal_instance
        )

        mock_completion_slot.assert_called_once_with("Task success message", True)

        mock_reenable_slot = MagicMock()
        self.app_logic.signals.actions_should_reenable.connect(mock_reenable_slot)
        
        if 0 in self.qtimer_lambdas_map and self.qtimer_lambdas_map[0]:
            found_and_called_reenable = False
            for lmbd in self.qtimer_lambdas_map[0]:
                lmbd() 
                if mock_reenable_slot.called:
                    found_and_called_reenable = True
                    break
            self.assertTrue(found_and_called_reenable, "Re-enable lambda not found or not effective")
            mock_reenable_slot.assert_called_once()


    def test_run_in_background_exception_in_task(self):
        self._create_app_logic()
        mock_task_func = MagicMock(side_effect=ValueError("Task error"))
        mock_task_func.__name__ = "mock_task_func_exception"

        dummy_emitter = TestSignals()
        mock_completion_slot = MagicMock()
        dummy_emitter.search_completed.connect(mock_completion_slot)
        test_completion_signal = dummy_emitter.search_completed

        mock_future = MagicMock(spec=Future)
        self.mock_executor.submit.return_value = mock_future

        mock_internal_slot = MagicMock()
        self.app_logic.signals._internal_task_completed.connect(mock_internal_slot)

        self.app_logic._run_in_background(mock_task_func, completion_signal=test_completion_signal)

        done_callback = mock_future.add_done_callback.call_args[0][0]

        mock_future.cancelled.return_value = False
        task_exception = ValueError("Task error")
        mock_future.exception.return_value = task_exception

        done_callback(mock_future)
        mock_internal_slot.assert_called_once_with(
            None, task_exception, False, test_completion_signal
        )
        mock_completion_slot.assert_called_once_with("Task Error: Task error", False)

    def test_run_in_background_task_cancelled(self):
        self._create_app_logic()
        mock_task_func = MagicMock()
        mock_task_func.__name__ = "mock_task_func_cancelled"

        dummy_emitter = TestSignals()
        mock_completion_slot = MagicMock()
        dummy_emitter.removal_completed.connect(mock_completion_slot)
        test_completion_signal = dummy_emitter.removal_completed

        mock_future = MagicMock(spec=Future)
        self.mock_executor.submit.return_value = mock_future

        mock_internal_slot = MagicMock()
        self.app_logic.signals._internal_task_completed.connect(mock_internal_slot)

        self.app_logic._run_in_background(mock_task_func, completion_signal=test_completion_signal)

        done_callback = mock_future.add_done_callback.call_args[0][0]
        mock_future.cancelled.return_value = True

        done_callback(mock_future)
        mock_internal_slot.assert_called_once_with(
            None, None, True, test_completion_signal
        )
        mock_completion_slot.assert_called_once_with("Task cancelled during execution.", False)


    def test_submit_search_success(self):
        self._create_app_logic()

        mock_status_slot = MagicMock()
        self.app_logic.signals.status_updated.connect(mock_status_slot)

        with patch.object(self.app_logic, '_run_in_background') as mock_run_bg:
            self.app_logic.submit_search("test query")
            
            self.assertTrue(0 in self.qtimer_lambdas_map and self.qtimer_lambdas_map[0])
            submit_lambda = self.qtimer_lambdas_map[0].pop(0) 
            submit_lambda()


            mock_run_bg.assert_called_once_with(
                self.app_logic._execute_search_task, "test query",
                completion_signal=self.app_logic.signals.search_completed
            )
        mock_status_slot.assert_called_with("Searching 'test query...'", "info")


    def test_submit_crawl_and_index_success(self):
        self._create_app_logic()
        urls = ["http://url1.com"]
        target_links = 10
        max_depth = 2
        keywords = ["kw1"]

        mock_status_slot = MagicMock()
        self.app_logic.signals.status_updated.connect(mock_status_slot)

        with patch.object(self.app_logic, '_run_in_background') as mock_run_bg:
            self.app_logic.submit_crawl_and_index(urls, target_links, max_depth, keywords)

            self.assertTrue(0 in self.qtimer_lambdas_map and self.qtimer_lambdas_map[0])
            submit_lambda = self.qtimer_lambdas_map[0].pop(0)
            submit_lambda()

            mock_run_bg.assert_called_once_with(
                self.app_logic._execute_crawl_and_index_task,
                urls, target_links, max_depth, keywords,
                completion_signal=self.app_logic.signals.crawl_index_completed
            )
        mock_status_slot.assert_called_with("Starting crawl & index for 1 URL(s)...", "info")


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

    def test_apply_settings_debug_mode_change(self):
        self._create_app_logic(debug=False)
        self.assertFalse(self.app_logic.debug)

        mock_file_handler = MagicMock(spec=logging.FileHandler)
        mock_stream_handler = MagicMock(spec=logging.StreamHandler)
        
        mock_root_logger_for_apply = MagicMock(spec=logging.Logger)
        mock_root_logger_for_apply.handlers = [mock_file_handler, mock_stream_handler]
        mock_root_logger_for_apply.level = logging.INFO 

        mock_settings_applied_slot = MagicMock()
        self.app_logic.signals.settings_applied.connect(mock_settings_applied_slot)

        with patch('logging.getLogger') as mock_get_logger_global:
            mock_get_logger_global.return_value = mock_root_logger_for_apply 
            self.app_logic.apply_settings({"debug_mode": True, "max_results": 5})

        self.assertTrue(self.app_logic.debug)
        mock_root_logger_for_apply.setLevel.assert_any_call(logging.DEBUG)
        mock_file_handler.setLevel.assert_called_with(logging.DEBUG) 
        mock_stream_handler.setLevel.assert_called_with(logging.DEBUG) 
        self.assertEqual(self.app_logic._current_config["max_results"], 5)
        if self.app_logic.llm_search:
            self.assertEqual(self.app_logic.llm_search.max_results, 5)
        mock_settings_applied_slot.assert_called_with("Settings applied successfully.", "success")
        

    def test_close_logic(self):
        self._create_app_logic()
        self.app_logic._active_crawler = self.mock_crawler_instance

        self.app_logic.close()

        self.assertTrue(self.app_logic._shutdown_event.is_set())
        self.mock_crawler_instance.abort.assert_called_once()
        self.mock_llmsearch_instance.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()