"""Context manager for the hide-flush-capture-show lifecycle.

Before every screenshot capture the overlay must be hidden and the
compositor must be flushed so that overlay pixels do not appear in
the captured image.  This module wraps that sequence into a single
context manager.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Generator

from PyQt6.QtWidgets import QApplication

from core.config import get_config
from recorder.overlay.platform_win32 import dwm_flush

logger = logging.getLogger(__name__)


@contextmanager
def capture_guard(
    view: Any,
    flush_ms: int | None = None,
) -> Generator[None, None, None]:
    """Hide the overlay, flush the compositor, yield for capture, restore.

    Args:
        view: The overlay ``QWidget`` (must have ``.hide()`` / ``.show()``).
        flush_ms: Override for compositor flush delay in milliseconds.
            When *None*, reads ``overlay.capture_delay_ms`` from
            ``config.yaml`` (default 100).

    Yields:
        Control to the caller, who should take the screenshot.
    """
    if flush_ms is None:
        cfg = get_config()
        flush_ms = cfg.get("overlay", {}).get("capture_delay_ms", 100)

    view.hide()
    QApplication.processEvents()

    if sys.platform == "win32":
        dwm_flush()
    else:
        time.sleep(flush_ms / 1000.0)

    try:
        yield
    finally:
        view.show()
        QApplication.processEvents()
