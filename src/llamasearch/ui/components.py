# src/llamasearch/ui/components.py
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import logging
import importlib.resources  # Use importlib.resources for package data

logger = logging.getLogger(__name__)


def header_component(data_paths: dict):  # data_paths for logging context
    """
    Creates a header widget with a logo and title.
    Uses importlib.resources to find the logo asset bundled with the package.
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)

    # --- Logo Loading using importlib.resources ---
    logo_label = QLabel()
    logo_found = False
    try:
        # Access the 'assets' directory within the 'llamasearch.ui' package
        logo_resource_path = importlib.resources.files("llamasearch.ui").joinpath(
            "assets/llamasearch.png"
        )

        # Use 'as_file' context manager for compatibility
        with importlib.resources.as_file(logo_resource_path) as logo_file_path:
            if logo_file_path.exists():
                pixmap = QPixmap(str(logo_file_path))
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        60,
                        60,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    logo_label.setPixmap(pixmap)
                    logo_label.setFixedSize(60, 60)
                    logger.debug(f"Loaded logo resource: {logo_resource_path}")
                    logo_found = True
                else:
                    logger.warning(f"Logo resource is invalid: {logo_resource_path}")
            else:
                logger.warning(f"Logo resource path DNE: {logo_file_path}")

    except (FileNotFoundError, ModuleNotFoundError, Exception) as e:
        logger.error(f"Error loading logo resource: {e}", exc_info=True)
        base_data_path_str = data_paths.get("base")
        if base_data_path_str:
            logger.error(f"(Data base path: {base_data_path_str})")

    if not logo_found:
        logger.warning("Using placeholder text for logo.")
        logo_label.setText("LS")
        logo_label.setFixedSize(60, 60)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet(
            "QLabel { border: 1px solid #cccccc; background-color: #f0f0f0; font-weight: bold; font-size: 18px; color: #555555; }"
        )

    # --- Title ---
    title_label = QLabel("LlamaSearch")
    title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-left: 10px;")

    layout.addWidget(logo_label)
    layout.addWidget(title_label)
    layout.addStretch()

    widget.setLayout(layout)
    return widget
