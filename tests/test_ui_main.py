import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import logging
import signal

# Import classes from the module to be tested
from llamasearch.ui.main import main as ui_main_entry_point
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, QTimer # Added QTimer here

# Import the SUT module to access its globals and allow patching its members
from llamasearch.ui import main as ui_main_sut_module


# Define paths for mocking
MOCK_UI_MAIN_SETUP_LOGGING = "llamasearch.ui.main.setup_logging"
MOCK_QAPPLICATION = "llamasearch.ui.main.QApplication"
MOCK_MAINWINDOW_CLS_FOR_ENTRY = f"{ui_main_sut_module.__name__}.MainWindow"
MOCK_SIGNAL_SIGNAL = "llamasearch.ui.main.signal.signal"
MOCK_QTIMER_GLOBAL = "llamasearch.ui.main.QTimer" 

MOCK_APP_LOGIC_CLS = "llamasearch.ui.main.LlamaSearchApp"
MOCK_HEADER_COMPONENT = "llamasearch.ui.main.header_component"
MOCK_SEARCH_VIEW_CLS = "llamasearch.ui.main.SearchAndIndexView"
MOCK_SETTINGS_VIEW_CLS = "llamasearch.ui.main.SettingsView"
MOCK_TERMINAL_VIEW_CLS = "llamasearch.ui.main.TerminalView"


class TestUIMain(unittest.TestCase):

    def setUp(self):
        self.mock_setup_logging = patch(MOCK_UI_MAIN_SETUP_LOGGING).start()
        self.mock_qapplication_cls = patch(MOCK_QAPPLICATION).start()
        self.mock_mainwindow_cls_for_entry = patch(MOCK_MAINWINDOW_CLS_FOR_ENTRY).start()
        self.mock_signal_dot_signal = patch(MOCK_SIGNAL_SIGNAL).start()
        self.mock_qtimer_global_cls = patch(MOCK_QTIMER_GLOBAL).start() # Patch the class
        
        self.mock_qapplication_instance = MagicMock(spec=QApplication)
        self.mock_qapplication_cls.return_value = self.mock_qapplication_instance
        self.mock_qapplication_instance.exec.return_value = 0 

        self.mock_logger_instance = MagicMock(spec=logging.Logger)
        self.mock_setup_logging.return_value = self.mock_logger_instance
        
        # Patch QTimer instance methods for the global SIGINT timer
        self.mock_sigint_qtimer_instance = MagicMock(spec=QTimer)
        self.mock_qtimer_global_cls.return_value = self.mock_sigint_qtimer_instance # Constructor returns this
        
        self.original_argv = sys.argv
        self.original_exit = sys.exit 
        sys.exit = MagicMock() 

        self.mock_app_logic_cls_mw = patch(MOCK_APP_LOGIC_CLS).start()
        self.mock_header_comp_func_mw = patch(MOCK_HEADER_COMPONENT).start()
        self.mock_search_view_cls_mw = patch(MOCK_SEARCH_VIEW_CLS).start()
        self.mock_settings_view_cls_mw = patch(MOCK_SETTINGS_VIEW_CLS).start()
        self.mock_terminal_view_cls_mw = patch(MOCK_TERMINAL_VIEW_CLS).start()

        self.mock_app_logic_instance_mw = MagicMock()
        self.mock_app_logic_instance_mw.data_paths = {"base": "/fake/path"}
        self.mock_app_logic_cls_mw.return_value = self.mock_app_logic_instance_mw

        self.mock_header_widget_mw = MagicMock(spec=QWidget)
        self.mock_header_comp_func_mw.return_value = self.mock_header_widget_mw

        ui_main_sut_module._app_instance = None
        ui_main_sut_module._main_window = None


    def tearDown(self):
        patch.stopall()
        sys.argv = self.original_argv
        sys.exit = self.original_exit 

    def test_ui_main_entry_point_success(self):
        sys.argv = ["main.py"] 
        
        ui_main_entry_point()

        self.mock_qapplication_cls.setAttribute.assert_any_call(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
        self.mock_qapplication_cls.setAttribute.assert_any_call(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
        self.mock_qapplication_cls.setApplicationName.assert_called_with("LlamaSearch")
        
        self.mock_setup_logging.assert_called_once_with("llamasearch.ui.main", level=logging.INFO, use_qt_handler=True)
        
        self.mock_qapplication_cls.assert_called_once_with(sys.argv)
        self.mock_qapplication_instance.exec.assert_called_once()
        
        self.mock_mainwindow_cls_for_entry.assert_called_once()
        self.mock_mainwindow_cls_for_entry.return_value.show.assert_called_once()
        
        self.mock_signal_dot_signal.assert_called_once_with(signal.SIGINT, ANY)
        
        # Check calls on the QTimer instance used for SIGINT
        self.mock_qtimer_global_cls.assert_called_once() # Check constructor was called
        self.mock_sigint_qtimer_instance.start.assert_called_once_with(500)
        self.mock_sigint_qtimer_instance.timeout.connect.assert_called_once()

        sys.exit.assert_called_once_with(0) # type: ignore


    def test_ui_main_entry_point_debug_mode(self):
        sys.argv = ["main.py", "--debug"]
        ui_main_entry_point()
        self.mock_setup_logging.assert_called_once_with("llamasearch.ui.main", level=logging.DEBUG, use_qt_handler=True)

if __name__ == "__main__":
    unittest.main()