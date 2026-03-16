"""Tests for platform-specific overlay helpers (Win32 and Linux)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestEnsureXcbPlatform:
    """Tests for ensure_xcb_platform() Wayland detection."""

    @patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}, clear=False)
    @patch("recorder.overlay.platform_linux.sys")
    def test_wayland_forces_xcb(self, mock_sys: MagicMock) -> None:
        """When XDG_SESSION_TYPE=wayland, QT_QPA_PLATFORM is set to xcb."""
        mock_sys.platform = "linux"
        # Remove any pre-existing value
        os.environ.pop("QT_QPA_PLATFORM", None)

        from recorder.overlay.platform_linux import ensure_xcb_platform

        ensure_xcb_platform()
        assert os.environ.get("QT_QPA_PLATFORM") == "xcb"

    @patch.dict(
        os.environ,
        {"WAYLAND_DISPLAY": "wayland-0", "XDG_SESSION_TYPE": ""},
        clear=False,
    )
    @patch("recorder.overlay.platform_linux.sys")
    def test_wayland_display_forces_xcb(self, mock_sys: MagicMock) -> None:
        """When WAYLAND_DISPLAY is set, QT_QPA_PLATFORM is set to xcb."""
        mock_sys.platform = "linux"
        os.environ.pop("QT_QPA_PLATFORM", None)

        from recorder.overlay.platform_linux import ensure_xcb_platform

        ensure_xcb_platform()
        assert os.environ.get("QT_QPA_PLATFORM") == "xcb"

    @patch.dict(
        os.environ,
        {"XDG_SESSION_TYPE": "x11", "WAYLAND_DISPLAY": ""},
        clear=False,
    )
    @patch("recorder.overlay.platform_linux.sys")
    def test_x11_no_change(self, mock_sys: MagicMock) -> None:
        """When XDG_SESSION_TYPE=x11, QT_QPA_PLATFORM is NOT set."""
        mock_sys.platform = "linux"
        os.environ.pop("QT_QPA_PLATFORM", None)

        from recorder.overlay.platform_linux import ensure_xcb_platform

        ensure_xcb_platform()
        assert "QT_QPA_PLATFORM" not in os.environ

    @patch("recorder.overlay.platform_linux.sys")
    def test_windows_skips(self, mock_sys: MagicMock) -> None:
        """On Windows, ensure_xcb_platform() does nothing."""
        mock_sys.platform = "win32"
        os.environ.pop("QT_QPA_PLATFORM", None)

        from recorder.overlay.platform_linux import ensure_xcb_platform

        ensure_xcb_platform()
        assert "QT_QPA_PLATFORM" not in os.environ


class TestSetClickThroughWin32:
    """Tests for Win32 click-through flag manipulation."""

    @patch("recorder.overlay.platform_win32.ctypes")
    def test_set_click_through_win32_passthrough(
        self, mock_ctypes: MagicMock
    ) -> None:
        """Passthrough=True ORs WS_EX_TRANSPARENT into the window style."""
        mock_ctypes.windll.user32.GetWindowLongW.return_value = 0x00080000
        from recorder.overlay.platform_win32 import (
            WS_EX_TRANSPARENT,
            set_click_through_win32,
        )

        set_click_through_win32(12345, passthrough=True)

        mock_ctypes.windll.user32.SetWindowLongW.assert_called_once()
        args = mock_ctypes.windll.user32.SetWindowLongW.call_args[0]
        assert args[2] & WS_EX_TRANSPARENT  # transparent bit set

    @patch("recorder.overlay.platform_win32.ctypes")
    def test_set_click_through_win32_capture(
        self, mock_ctypes: MagicMock
    ) -> None:
        """Passthrough=False clears WS_EX_TRANSPARENT from the window style."""
        from recorder.overlay.platform_win32 import WS_EX_TRANSPARENT

        mock_ctypes.windll.user32.GetWindowLongW.return_value = (
            0x00080000 | WS_EX_TRANSPARENT
        )
        from recorder.overlay.platform_win32 import set_click_through_win32

        set_click_through_win32(12345, passthrough=False)

        args = mock_ctypes.windll.user32.SetWindowLongW.call_args[0]
        assert not (args[2] & WS_EX_TRANSPARENT)  # transparent bit cleared


class TestDwmFlush:
    """Tests for dwm_flush() compositor synchronization."""

    @patch("recorder.overlay.platform_win32.ctypes")
    def test_dwm_flush_calls_api(self, mock_ctypes: MagicMock) -> None:
        """dwm_flush() calls ctypes.windll.dwmapi.DwmFlush."""
        from recorder.overlay.platform_win32 import dwm_flush

        dwm_flush()
        mock_ctypes.windll.dwmapi.DwmFlush.assert_called_once()

    @patch("recorder.overlay.platform_win32.ctypes")
    def test_dwm_flush_handles_missing(self, mock_ctypes: MagicMock) -> None:
        """dwm_flush() does not raise when DwmFlush is unavailable."""
        mock_ctypes.windll.dwmapi.DwmFlush.side_effect = AttributeError(
            "no DwmFlush"
        )
        from recorder.overlay.platform_win32 import dwm_flush

        # Should not raise
        dwm_flush()
