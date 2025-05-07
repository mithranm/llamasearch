#!/usr/bin/env python3
"""
__main__.py - Entry point for LlamaSearch GUI.
"""

import argparse
import logging
import logging.handlers
import signal
import sys
from pathlib import Path
# from typing import Optional # No longer needed

from llamasearch.data_manager import data_manager
from llamasearch.exceptions import ModelNotFoundError, SetupError
from llamasearch.utils import setup_logging

logger = setup_logging()

shutdown_requested = False

def handle_signal(sig, frame):
    """Gracefully handle SIGINT and SIGTERM for the GUI."""
    global shutdown_requested
    if not shutdown_requested:
        logger.warning(f"Received signal {sig}. Initiating graceful shutdown...")
        shutdown_requested = True

        try:
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication

            app_instance = QApplication.instance()
            if app_instance:
                logger.info("Requesting Qt Application quit.")
                QTimer.singleShot(0, app_instance.quit)
        except ImportError:
            logger.warning("GUI not available (PySide6 not imported) for signal handling.")
        except Exception as e:
            logger.warning(f"Error requesting Qt Application quit: {e}")
    else:
        logger.warning("Shutdown already requested. Force exiting...")
        sys.exit(1)

def main():
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(
        description="LlamaSearch GUI: Index and search your documents."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging level",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    root_logger = logging.getLogger("llamasearch")
    # Set root logger level. setup_logging now ensures it's at least DEBUG.
    # We adjust the effective level for handlers here.
    if root_logger.level > log_level : # only set if more verbose is requested
        root_logger.setLevel(log_level)


    for handler in root_logger.handlers:
        if isinstance(
            handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)
        ):
            handler.setLevel(logging.DEBUG) # File handler always logs DEBUG and up
        elif isinstance(handler, logging.StreamHandler): # Console handler
            handler.setLevel(log_level)
        else: # Other handlers like QtLogHandler
            handler.setLevel(log_level)
        try: # Ensure QtLogHandler also respects the debug flag
            from llamasearch.ui.qt_logging import QtLogHandler
            if QtLogHandler and isinstance(handler, QtLogHandler):
                handler.setLevel(log_level)
        except ImportError:
            pass # Qt not available

    logger.info("LlamaSearch GUI starting...")
    if args.debug:
        logger.debug("Debug logging enabled.")

    try:
        logger.info("Verifying essential models before starting GUI...")
        models_dir_path_str = data_manager.get_data_paths().get("models")
        if not models_dir_path_str:
            raise SetupError("Models directory path not configured in settings.")
        models_dir = Path(models_dir_path_str)
        active_dir = models_dir / "active_model"
        onnx_dir = active_dir / "onnx"
        if not active_dir.is_dir() or not onnx_dir.is_dir():
            raise ModelNotFoundError(
                "Active model directory not found or incomplete. Run 'llamasearch-setup'."
            )
        logger.info("Basic model structure check passed.")

        from llamasearch.ui.main import main as gui_main

        logger.info("Starting GUI...")
        gui_main()
        logger.info("GUI finished.")

    except ModelNotFoundError as e:
        logger.error(f"GUI Error: {e}")
        logger.error(
            "Essential models not found. Please run 'llamasearch-setup' first."
        )
        sys.exit(1)
    except SetupError as e:
        logger.error(f"GUI Error: {e}")
        logger.error(
            "There was a setup issue. Please run 'llamasearch-setup' again or check settings."
        )
        sys.exit(1)
    except ImportError as e:
        if "PySide6" in str(e):
            logger.error("GUI Error: PySide6 (Qt bindings) not installed.")
            logger.error(
                "To use the GUI, please install LlamaSearch with GUI extras:"
            )
            logger.error('  pip install "llamasearch[gui]"')
            sys.exit(1)
        else:
            logger.error(
                f"GUI Error: Failed to import UI components: {e}",
                exc_info=args.debug,
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start GUI: {e}", exc_info=args.debug)
        sys.exit(1)

if __name__ == "__main__":
    main()