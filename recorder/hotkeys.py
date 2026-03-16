"""Global hotkey listeners for the recording overlay.

Win32: Polls GetAsyncKeyState on a QTimer (only approach that works
alongside PyQt6's event loop — RegisterHotKey and pynput both fail).
Non-Windows: pynput GlobalHotKeys fallback.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from PyQt6.QtCore import QTimer

logger = logging.getLogger(__name__)


class _Win32PollingHotkeyListener:
    """Polls keyboard state via GetAsyncKeyState on a QTimer.

    Both RegisterHotKey (hooks on a background thread) and pynput
    (SetWindowsHookEx) fail to receive key events when PyQt6's event
    loop is running. GetAsyncKeyState reads raw key state from the OS
    regardless of focus or hooks — and since QTimer fires on the Qt
    main thread, no cross-thread marshaling is needed.

    Supported shortcuts:
        Ctrl+R / F2  → toggle mode
        Ctrl+Q / ESC → close overlay
    """

    # Virtual key codes
    VK_CONTROL = 0x11
    VK_R = 0x52
    VK_Q = 0x51
    VK_F2 = 0x71
    VK_ESCAPE = 0x1B

    _POLL_MS = 80  # ~12 Hz — responsive without burning CPU

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_close: Callable[[], None],
    ) -> None:
        self._on_toggle = on_toggle
        self._on_close = on_close
        self._timer: QTimer | None = None
        self._prev_r = False
        self._prev_q = False
        self._prev_f2 = False
        self._prev_esc = False

    def start(self) -> None:
        """Starts the polling timer."""
        import ctypes
        ctypes.windll.user32.GetAsyncKeyState  # quick sanity check

        self._timer = QTimer()
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        logger.info(
            "Win32 polling hotkeys started (%d ms): Ctrl+R, Ctrl+Q, F2, ESC",
            self._POLL_MS,
        )

    def stop(self) -> None:
        """Stops the polling timer."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
            logger.debug("Win32 polling hotkeys stopped")

    def _poll(self) -> None:
        """Called every _POLL_MS to check key states."""
        import ctypes
        user32 = ctypes.windll.user32
        get = user32.GetAsyncKeyState

        ctrl = bool(get(self.VK_CONTROL) & 0x8000)
        r_down = bool(get(self.VK_R) & 0x8000)
        q_down = bool(get(self.VK_Q) & 0x8000)
        f2_down = bool(get(self.VK_F2) & 0x8000)
        esc_down = bool(get(self.VK_ESCAPE) & 0x8000)

        if ctrl and r_down and not self._prev_r:
            self._on_toggle()
        if f2_down and not self._prev_f2:
            self._on_toggle()

        if ctrl and q_down and not self._prev_q:
            self._on_close()
        if esc_down and not self._prev_esc:
            self._on_close()

        self._prev_r = ctrl and r_down
        self._prev_q = ctrl and q_down
        self._prev_f2 = f2_down
        self._prev_esc = esc_down


class _PynputHotkeyListener:
    """Global hotkey listener using pynput (macOS / Linux fallback).

    Note: pynput does NOT work alongside PyQt6 on Windows — the
    low-level keyboard hooks receive zero events when Qt's event loop
    is running. Use _Win32PollingHotkeyListener on Windows instead.
    """

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_close: Callable[[], None],
    ) -> None:
        self._on_toggle = on_toggle
        self._on_close = on_close
        self._listener: Any = None
        self._key_listener: Any = None

    def start(self) -> None:
        """Starts the pynput key listener."""
        try:
            from pynput import keyboard

            def on_activate_toggle() -> None:
                QTimer.singleShot(0, self._on_toggle)

            def on_activate_close() -> None:
                QTimer.singleShot(0, self._on_close)

            hotkeys = keyboard.GlobalHotKeys({
                "<ctrl>+r": on_activate_toggle,
                "<ctrl>+q": on_activate_close,
            })
            hotkeys.start()
            self._listener = hotkeys

            def _on_press(key: Any) -> None:
                try:
                    if key == keyboard.Key.f2:
                        QTimer.singleShot(0, self._on_toggle)
                    elif key == keyboard.Key.esc:
                        QTimer.singleShot(0, self._on_close)
                except Exception:
                    pass

            self._key_listener = keyboard.Listener(on_press=_on_press)
            self._key_listener.start()

            logger.info("pynput global hotkeys registered: Ctrl+R, Ctrl+Q, F2, ESC")
        except ImportError:
            logger.warning(
                "pynput not installed — global hotkeys unavailable. "
                "Ctrl+R / Ctrl+Q will only work when the overlay has focus."
            )
        except Exception as e:
            logger.warning("Failed to start pynput hotkeys: %s", e)

    def stop(self) -> None:
        """Stops the pynput listeners."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._key_listener is not None:
            self._key_listener.stop()
            self._key_listener = None


def _create_hotkey_listener(
    on_toggle: Callable[[], None],
    on_close: Callable[[], None],
) -> _Win32PollingHotkeyListener | _PynputHotkeyListener:
    """Creates the appropriate global hotkey listener for the platform."""
    if sys.platform == "win32":
        return _Win32PollingHotkeyListener(on_toggle, on_close)
    return _PynputHotkeyListener(on_toggle, on_close)
