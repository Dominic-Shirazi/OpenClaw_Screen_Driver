"""Cross-platform utility functions for recording flow."""
from __future__ import annotations

import logging
import sys
import time

logger = logging.getLogger(__name__)


def minimize_all_windows() -> bool:
    """Minimize all windows to show desktop (Win+D equivalent).

    Uses Win32 keybd_event to simulate Win+D on Windows.
    Other platforms are not yet supported.

    Returns:
        True if successful, False if platform not supported or failed.
    """
    if sys.platform != "win32":
        logger.warning("Window minimize only supported on Windows currently")
        return False
    try:
        import ctypes

        VK_LWIN = 0x5B
        VK_D = 0x44
        KEYEVENTF_KEYUP = 0x0002
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_LWIN, 0, 0, 0)
        user32.keybd_event(VK_D, 0, 0, 0)
        user32.keybd_event(VK_D, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.5)  # Wait for minimize animation
        return True
    except Exception as e:
        logger.warning("Failed to minimize windows: %s", e)
        return False
