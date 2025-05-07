# tests/test_ui_components.py
import unittest
from unittest.mock import ANY, MagicMock, patch

from PySide6.QtGui import QPixmap, Qt
from PySide6.QtWidgets import QLabel

from llamasearch.ui.components import header_component


class TestHeaderComponent(unittest.TestCase):

    @patch("llamasearch.ui.components.logger")
    @patch("llamasearch.ui.components.importlib.resources")
    @patch("llamasearch.ui.components.QWidget")
    @patch("llamasearch.ui.components.QHBoxLayout")
    @patch("llamasearch.ui.components.QLabel")
    @patch("llamasearch.ui.components.QPixmap")
    def test_header_component_logo_found(
        self,
        MockQPixmap,
        MockQLabelCls,
        MockQHBoxLayout,
        MockQWidget,
        mock_resources,
        mock_logger,
    ):
        mock_logo_resource_path = MagicMock()
        mock_resources.files.return_value.joinpath.return_value = (
            mock_logo_resource_path
        )

        mock_logo_file_path_obj = MagicMock()
        mock_logo_file_path_obj.exists.return_value = True

        mock_as_file_context_manager = MagicMock()
        mock_as_file_context_manager.__enter__.return_value = mock_logo_file_path_obj
        mock_as_file_context_manager.__exit__.return_value = None
        mock_resources.as_file.return_value = mock_as_file_context_manager

        mock_pixmap_instance = MockQPixmap.return_value
        mock_pixmap_instance.isNull.return_value = False
        mock_scaled_pixmap = MagicMock(spec=QPixmap)
        mock_pixmap_instance.scaled.return_value = mock_scaled_pixmap

        mock_logo_label = MagicMock(spec=QLabel)
        mock_title_label = MagicMock(spec=QLabel)

        # Use a list to control the return values for QLabel constructor
        # First call to QLabel() is for logo_label (no args)
        # Second call is for title_label (with "LlamaSearch" arg)
        def qlabel_side_effect(*args, **kwargs):
            if not args and not kwargs:  # For logo_label
                return mock_logo_label
            elif args and args[0] == "LlamaSearch":  # For title_label
                return mock_title_label
            return MagicMock(spec=QLabel)  # Default for unexpected calls

        MockQLabelCls.side_effect = qlabel_side_effect

        data_paths_for_log = {"base": "/fake/base"}
        header_widget_instance = header_component(data_paths_for_log)

        MockQWidget.assert_called_once()
        self.assertIs(header_widget_instance, MockQWidget.return_value)

        MockQHBoxLayout.assert_called_once_with(MockQWidget.return_value)
        mock_layout_instance = MockQHBoxLayout.return_value
        mock_layout_instance.setContentsMargins.assert_called_once_with(0, 0, 0, 0)

        mock_resources.files.assert_called_once_with("llamasearch.ui")
        mock_resources.files.return_value.joinpath.assert_called_once_with(
            "assets/llamasearch.png"
        )
        mock_resources.as_file.assert_called_once_with(mock_logo_resource_path)

        MockQPixmap.assert_called_once_with(str(mock_logo_file_path_obj))
        mock_pixmap_instance.scaled.assert_called_once_with(
            60,
            60,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        mock_logo_label.setPixmap.assert_called_once_with(mock_scaled_pixmap)
        mock_logo_label.setFixedSize.assert_called_once_with(60, 60)

        mock_title_label.setStyleSheet.assert_called_once_with(
            "font-size: 20px; font-weight: bold; margin-left: 10px;"
        )

        self.assertEqual(mock_layout_instance.addWidget.call_count, 2)
        mock_layout_instance.addWidget.assert_any_call(mock_logo_label)
        mock_layout_instance.addWidget.assert_any_call(mock_title_label)
        mock_layout_instance.addStretch.assert_called_once()
        MockQWidget.return_value.setLayout.assert_called_once_with(mock_layout_instance)
        mock_logger.debug.assert_any_call(
            f"Loaded logo resource: {mock_logo_resource_path}"
        )

    @patch("llamasearch.ui.components.logger")
    @patch("llamasearch.ui.components.importlib.resources")
    @patch("llamasearch.ui.components.QWidget")
    @patch("llamasearch.ui.components.QHBoxLayout")
    @patch("llamasearch.ui.components.QLabel")
    @patch("llamasearch.ui.components.QPixmap")
    def test_header_component_logo_not_found_fallback(
        self,
        MockQPixmap,
        MockQLabelCls,
        MockQHBoxLayout,
        MockQWidget,
        mock_resources,
        mock_logger,
    ):
        mock_logo_resource_path = MagicMock()
        mock_resources.files.return_value.joinpath.return_value = (
            mock_logo_resource_path
        )

        mock_logo_file_path_obj = MagicMock()
        mock_logo_file_path_obj.exists.return_value = False

        mock_as_file_context_manager = MagicMock()
        mock_as_file_context_manager.__enter__.return_value = mock_logo_file_path_obj
        mock_as_file_context_manager.__exit__.return_value = None
        mock_resources.as_file.return_value = mock_as_file_context_manager

        mock_logo_label = MagicMock(spec=QLabel)
        MockQLabelCls.side_effect = [mock_logo_label, MagicMock(spec=QLabel)]

        data_paths_for_log = {"base": "/fake/base/no_logo"}
        header_component(data_paths_for_log)

        MockQPixmap.assert_not_called()
        mock_logo_label.setPixmap.assert_not_called()

        mock_logo_label.setText.assert_called_once_with("LS")
        mock_logo_label.setFixedSize.assert_called_once_with(60, 60)
        mock_logo_label.setAlignment.assert_called_once_with(
            Qt.AlignmentFlag.AlignCenter
        )
        mock_logo_label.setStyleSheet.assert_called_once_with(ANY)
        mock_logger.warning.assert_any_call("Using placeholder text for logo.")

    @patch("llamasearch.ui.components.logger")
    @patch("llamasearch.ui.components.importlib.resources")
    @patch("llamasearch.ui.components.QWidget")
    @patch("llamasearch.ui.components.QHBoxLayout")
    @patch("llamasearch.ui.components.QLabel")
    @patch("llamasearch.ui.components.QPixmap")
    def test_header_component_logo_resource_load_exception(
        self,
        MockQPixmap,
        MockQLabelCls,
        MockQHBoxLayout,
        MockQWidget,
        mock_resources,
        mock_logger,
    ):
        mock_resources.files.side_effect = FileNotFoundError(
            "Mocked resource system error"
        )

        mock_logo_label = MagicMock(spec=QLabel)
        MockQLabelCls.side_effect = [mock_logo_label, MagicMock(spec=QLabel)]

        data_paths_for_log = {"base": "/fake/base/exc"}
        header_component(data_paths_for_log)

        mock_logo_label.setText.assert_called_once_with("LS")
        mock_logo_label.setFixedSize.assert_called_once_with(60, 60)
        mock_logger.error.assert_any_call(
            "Error loading logo resource: Mocked resource system error", exc_info=True
        )


if __name__ == "__main__":
    unittest.main()
