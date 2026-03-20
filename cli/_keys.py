"""Cross-platform single-keypress reader for TUI menu navigation.

Provides a unified ``read_key()`` function that works on Windows (msvcrt)
and Unix (tty/termios) systems, returning normalised key names for
arrow keys, enter, escape, and backspace.
"""

from __future__ import annotations

import sys


def read_key() -> str:
    """Read a single keypress and return a normalised key name.

    Returns one of the following strings:
    - ``"up"``, ``"down"``, ``"left"``, ``"right"`` for arrow keys
    - ``"enter"`` for Return/Enter
    - ``"escape"`` for Escape
    - ``"backspace"`` for Backspace/Delete-back
    - A single printable character for all other keys

    Returns:
        Normalised key name as a string.
    """
    if sys.platform == "win32":
        return _read_key_windows()
    return _read_key_unix()


def _read_key_windows() -> str:
    """Windows keypress reader using msvcrt.

    Returns:
        Normalised key name.
    """
    import msvcrt  # noqa: PLC0415

    ch = msvcrt.getwch()

    # Special key prefix: arrow keys etc. send two-char sequences
    if ch in ("\x00", "\xe0"):
        second = msvcrt.getwch()
        mapping = {
            "H": "up",
            "P": "down",
            "K": "left",
            "M": "right",
        }
        return mapping.get(second, "")

    if ch == "\r":
        return "enter"
    if ch == "\x1b":
        return "escape"
    if ch == "\x08":
        return "backspace"

    return ch


def _read_key_unix() -> str:
    """Unix keypress reader using tty/termios raw mode.

    Returns:
        Normalised key name.
    """
    import termios  # noqa: PLC0415
    import tty  # noqa: PLC0415

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        if ch == "\x1b":
            # Could be an escape sequence or standalone Escape
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                mapping = {
                    "A": "up",
                    "B": "down",
                    "C": "right",
                    "D": "left",
                }
                return mapping.get(ch3, "")
            # Standalone escape (no bracket followed)
            return "escape"

        if ch == "\r":
            return "enter"
        if ch == "\x7f":
            return "backspace"

        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
