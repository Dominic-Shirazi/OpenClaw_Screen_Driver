"""DPI coordinate conversion utilities.

Centralizes conversion between Qt logical (device-independent) pixels
and mss physical pixels.  All coordinate exchanges between the Qt
overlay and the screenshot system must go through these helpers.
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


def _get_dpr() -> float:
    """Return the device-pixel-ratio for the primary screen.

    Returns:
        The ratio (e.g. 2.0 on a Retina/HiDPI display), or 1.0 if
        no screen is available.
    """
    screen = QApplication.primaryScreen()
    if screen is None:
        logger.debug("No primary screen found; assuming DPR 1.0")
        return 1.0
    return screen.devicePixelRatio()


def logical_to_physical(x: int, y: int) -> tuple[int, int]:
    """Convert Qt logical coordinates to mss physical pixels.

    Args:
        x: Logical x coordinate.
        y: Logical y coordinate.

    Returns:
        Physical (x, y) tuple.
    """
    dpr = _get_dpr()
    return int(x * dpr), int(y * dpr)


def physical_to_logical(x: int, y: int) -> tuple[int, int]:
    """Convert mss physical pixels to Qt logical coordinates.

    Args:
        x: Physical x coordinate.
        y: Physical y coordinate.

    Returns:
        Logical (x, y) tuple.
    """
    dpr = _get_dpr()
    return int(x / dpr), int(y / dpr)


def get_physical_screen_size() -> tuple[int, int]:
    """Return the screen size in physical pixels (what mss sees).

    Returns:
        (width, height) in physical pixels, or (1920, 1080) as fallback.
    """
    screen = QApplication.primaryScreen()
    if screen is None:
        logger.debug("No primary screen; returning default 1920x1080")
        return 1920, 1080
    size = screen.size()  # logical
    dpr = screen.devicePixelRatio()
    return int(size.width() * dpr), int(size.height() * dpr)
