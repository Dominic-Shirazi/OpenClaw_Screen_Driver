"""Cross-platform utility functions for recording flow."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def _minimize_windows_win32() -> bool:
    """Minimize all windows on Windows via Win+D keybd_event."""
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
        logger.warning("Failed to minimize windows (Win32): %s", e)
        return False


def _minimize_windows_darwin() -> bool:
    """Minimize all windows on macOS via AppleScript (Cmd+Option+M)."""
    try:
        subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to keystroke "m" '
                'using {command down, option down}',
            ],
            check=True,
            capture_output=True,
            timeout=5,
        )
        time.sleep(0.5)
        return True
    except Exception as e:
        logger.warning("Failed to minimize windows (macOS): %s", e)
        return False


def _minimize_windows_linux() -> bool:
    """Minimize all windows on Linux via wmctrl or xdotool."""
    if shutil.which("wmctrl"):
        try:
            subprocess.run(
                ["wmctrl", "-k", "on"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.warning("wmctrl minimize failed: %s", e)
            return False

    if shutil.which("xdotool"):
        try:
            subprocess.run(
                ["xdotool", "key", "super+d"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.warning("xdotool minimize failed: %s", e)
            return False

    logger.warning(
        "No supported minimize tool found on Linux. "
        "Install wmctrl or xdotool for window-minimize support."
    )
    return False


def minimize_all_windows() -> bool:
    """Minimize all windows to show desktop (Win+D equivalent).

    Platform-guarded implementations:
    - Windows: Win32 keybd_event (Win+D).
    - macOS: AppleScript (Cmd+Option+M via System Events).
    - Linux: wmctrl or xdotool (falls back gracefully with a warning).

    Returns:
        True if successful, False if platform not supported or failed.
    """
    if sys.platform == "win32":
        return _minimize_windows_win32()
    if sys.platform == "darwin":
        return _minimize_windows_darwin()
    if sys.platform.startswith("linux"):
        return _minimize_windows_linux()

    logger.warning("minimize_all_windows: unsupported platform %s", sys.platform)
    return False
