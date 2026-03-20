"""Cross-platform terminal minimize and restore utilities.

Provides functions to programmatically minimize and restore the
console window. Currently implemented for Windows; other platforms
silently no-op.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def minimize_terminal() -> None:
    """Minimize the console/terminal window.

    On Windows, uses Win32 API via ctypes to minimize the console
    window. On other platforms, silently does nothing.
    """
    if sys.platform != "win32":
        return

    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()  # type: ignore[attr-defined]
        if hwnd != 0:
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # type: ignore[attr-defined]  # SW_MINIMIZE
    except OSError as exc:
        logger.debug("Could not minimize terminal: %s", exc)


def restore_terminal() -> None:
    """Restore the console/terminal window from minimized state.

    On Windows, uses Win32 API via ctypes to restore the console
    window. On other platforms, silently does nothing.
    """
    if sys.platform != "win32":
        return

    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()  # type: ignore[attr-defined]
        if hwnd != 0:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # type: ignore[attr-defined]  # SW_RESTORE
    except OSError as exc:
        logger.debug("Could not restore terminal: %s", exc)
