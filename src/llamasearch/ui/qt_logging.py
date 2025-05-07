# src/llamasearch/ui/qt_logging.py

import logging

from PySide6.QtCore import QObject, Signal


class QtLogSignalEmitter(QObject):
    """Contains a signal to emit log messages."""

    # Signal emits: level_name (str), message (str)
    log_message = Signal(str, str)


class QtLogHandler(logging.Handler):
    """
    A logging handler that emits Qt signals for each log record.
    """

    def __init__(self, signal_emitter: QtLogSignalEmitter, level=logging.NOTSET):
        super().__init__(level=level)
        self.signal_emitter = signal_emitter

    def emit(self, record: logging.LogRecord):
        """
        Formats the log record and emits it via the Qt signal.
        This method can be called from any thread.
        """
        try:
            msg = self.format(record)
            # Emit the signal - Qt handles cross-thread delivery if connected correctly
            self.signal_emitter.log_message.emit(record.levelname, msg)
        except Exception:
            self.handleError(record)  # Default error handling


# Global instance of the emitter
# IMPORTANT: This emitter should be created *before* any loggers that use QtLogHandler
# It's often convenient to create it early in the GUI application setup.
qt_log_emitter = QtLogSignalEmitter()
