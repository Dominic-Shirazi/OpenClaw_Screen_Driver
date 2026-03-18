"""Thread-safe signal bridge for recording pipeline results.

PipelineBridge is a QObject whose pyqtSignals provide the sole mechanism
for background pipeline threads (detection, VLM, execution, save) to
deliver results back to the main Qt thread.

IMPORTANT: Background threads MUST emit signals on this bridge rather than
calling QTimer.singleShot (which is not thread-safe from non-Qt threads).
Qt's AutoConnection mode guarantees delivery on the receiver's thread.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class PipelineBridge(QObject):
    """Thread-safe signal bridge for recording pipeline results.

    Signals:
        detection_ready: Detection finished with result dict.
            Payload: ``{bbox: {x, y, w, h}, candidates: [...], type_guess: str}``
        vlm_ready: VLM analysis finished with result dict.
            Payload: ``{element_type, label_guess, confidence, ocr_text, caption}``
        vlm_failed: VLM analysis failed with error message.
        execution_complete: Dry-run action finished (no payload).
        save_complete: Routine saved successfully with directory path.
        save_failed: Routine save failed with error message.

    Args:
        parent: Optional QObject parent for Qt ownership.
    """

    detection_ready = pyqtSignal(dict)
    """Detection finished: {bbox: {x,y,w,h}, candidates: [...], type_guess: str}."""

    vlm_ready = pyqtSignal(dict)
    """VLM analysis finished: {element_type, label_guess, confidence, ocr_text, caption}."""

    vlm_failed = pyqtSignal(str)
    """VLM analysis failed: error message string."""

    execution_complete = pyqtSignal()
    """Dry-run action finished (emitted from background thread, received on main thread)."""

    save_complete = pyqtSignal(str)
    """Routine saved successfully: directory path string."""

    save_failed = pyqtSignal(str)
    """Routine save failed: error message string."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        logger.debug("PipelineBridge created with 6 signals")
