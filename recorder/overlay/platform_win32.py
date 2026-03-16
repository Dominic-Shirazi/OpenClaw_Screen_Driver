"""Win32-specific window flag management and DwmFlush.

Provides helpers for applying layered window styles, toggling
click-through via WS_EX_TRANSPARENT, and synchronising with the
Desktop Window Manager compositor before screenshot capture.
"""

from __future__ import annotations

import ctypes
import logging

logger = logging.getLogger(__name__)

# Win32 extended window style constants
GWL_EXSTYLE: int = -20
WS_EX_LAYERED: int = 0x00080000
WS_EX_TRANSPARENT: int = 0x00000020
WS_EX_TOOLWINDOW: int = 0x00000080
WS_EX_NOACTIVATE: int = 0x08000000


def setup_win32_layered(hwnd: int) -> None:
    """Apply WS_EX_LAYERED, WS_EX_TOOLWINDOW, and WS_EX_NOACTIVATE flags.

    Args:
        hwnd: Native window handle obtained from ``int(widget.winId())``.
    """
    try:
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        new_style = style | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        logger.debug("Win32 layered flags applied (hwnd=%d)", hwnd)
    except (AttributeError, OSError) as exc:
        logger.warning("Failed to apply Win32 layered flags: %s", exc)


def set_click_through_win32(hwnd: int, passthrough: bool) -> None:
    """Toggle click-through on a Win32 window.

    Args:
        hwnd: Native window handle.
        passthrough: True to pass mouse events through, False to capture them.
    """
    try:
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if passthrough:
            style |= WS_EX_TRANSPARENT
        else:
            style &= ~WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
    except (AttributeError, OSError) as exc:
        logger.warning("Failed to set click-through: %s", exc)


def dwm_flush() -> None:
    """Block until the DWM has composited the current frame.

    Safe no-op when DWM is unavailable (pre-Vista or Server Core).
    """
    try:
        ctypes.windll.dwmapi.DwmFlush()
    except (AttributeError, OSError):
        pass
