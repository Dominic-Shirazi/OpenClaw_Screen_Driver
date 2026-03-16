"""Linux X11/XCB platform helpers for the overlay.

Handles Wayland session detection (forcing the XCB backend) and
click-through toggling via Qt widget attributes on Linux.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def ensure_xcb_platform() -> None:
    """Force the XCB Qt platform backend when running under Wayland.

    Wayland does not support ``WindowStaysOnTopHint`` or arbitrary
    window positioning.  Most Ubuntu Wayland sessions also run
    XWayland, so the xcb backend works transparently.

    **Must be called before** ``QApplication`` is created.
    """
    if sys.platform == "win32":
        return

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "")

    if session_type == "wayland" or wayland_display:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        logger.info(
            "Wayland detected (XDG_SESSION_TYPE=%s, WAYLAND_DISPLAY=%s); "
            "forcing QT_QPA_PLATFORM=xcb",
            session_type,
            wayland_display,
        )


def set_click_through_linux(widget: Any, passthrough: bool) -> None:
    """Toggle click-through on a Linux X11/XCB widget.

    Args:
        widget: A ``QWidget`` instance.
        passthrough: True to let mouse events pass through, False to capture.
    """
    # Import Qt inside function to avoid top-level PyQt6 import
    # when this module is loaded on non-Linux platforms.
    from PyQt6.QtCore import Qt  # noqa: PLC0415

    widget.setAttribute(
        Qt.WidgetAttribute.WA_TransparentForMouseEvents, passthrough
    )
