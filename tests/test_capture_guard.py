"""Tests for capture guard hide-flush-capture-show lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch


class TestCaptureGuard:
    """Tests for the capture_guard context manager."""

    @patch("recorder.overlay.capture_guard.get_config", return_value={"overlay": {"capture_delay_ms": 100}})
    @patch("recorder.overlay.capture_guard.QApplication")
    @patch("recorder.overlay.capture_guard.sys")
    def test_hide_before_capture(
        self,
        mock_sys: MagicMock,
        mock_qapp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """view.hide() is called before the yield."""
        mock_sys.platform = "linux"
        view = MagicMock()
        call_order: list[str] = []
        view.hide.side_effect = lambda: call_order.append("hide")
        view.show.side_effect = lambda: call_order.append("show")

        from recorder.overlay.capture_guard import capture_guard

        with patch("recorder.overlay.capture_guard.time") as mock_time:
            with capture_guard(view) as _:
                call_order.append("yield")

        assert call_order[0] == "hide"
        assert call_order.index("hide") < call_order.index("yield")

    @patch("recorder.overlay.capture_guard.get_config", return_value={"overlay": {"capture_delay_ms": 100}})
    @patch("recorder.overlay.capture_guard.QApplication")
    @patch("recorder.overlay.capture_guard.sys")
    def test_show_after_capture(
        self,
        mock_sys: MagicMock,
        mock_qapp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """view.show() is called after the yield."""
        mock_sys.platform = "linux"
        view = MagicMock()
        call_order: list[str] = []
        view.hide.side_effect = lambda: call_order.append("hide")
        view.show.side_effect = lambda: call_order.append("show")

        from recorder.overlay.capture_guard import capture_guard

        with patch("recorder.overlay.capture_guard.time") as mock_time:
            with capture_guard(view) as _:
                call_order.append("yield")

        assert call_order.index("show") > call_order.index("yield")

    @patch("recorder.overlay.capture_guard.get_config", return_value={"overlay": {"capture_delay_ms": 100}})
    @patch("recorder.overlay.capture_guard.QApplication")
    @patch("recorder.overlay.capture_guard.sys")
    def test_process_events_called(
        self,
        mock_sys: MagicMock,
        mock_qapp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """QApplication.processEvents() is called during guard lifecycle."""
        mock_sys.platform = "linux"
        view = MagicMock()

        from recorder.overlay.capture_guard import capture_guard

        with patch("recorder.overlay.capture_guard.time"):
            with capture_guard(view):
                pass

        assert mock_qapp.processEvents.call_count >= 1

    @patch("recorder.overlay.capture_guard.get_config", return_value={"overlay": {"capture_delay_ms": 100}})
    @patch("recorder.overlay.capture_guard.QApplication")
    @patch("recorder.overlay.capture_guard.sys")
    @patch("recorder.overlay.capture_guard.time")
    def test_dwm_flush_called_on_windows(
        self,
        mock_time: MagicMock,
        mock_sys: MagicMock,
        mock_qapp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """DwmFlush is called on Windows platform."""
        mock_sys.platform = "win32"
        view = MagicMock()

        with patch(
            "recorder.overlay.capture_guard.dwm_flush"
        ) as mock_dwm:
            from recorder.overlay.capture_guard import capture_guard

            with capture_guard(view):
                pass

            mock_dwm.assert_called_once()

    @patch("recorder.overlay.capture_guard.get_config", return_value={"overlay": {"capture_delay_ms": 100}})
    @patch("recorder.overlay.capture_guard.QApplication")
    @patch("recorder.overlay.capture_guard.sys")
    @patch("recorder.overlay.capture_guard.time")
    def test_sleep_fallback_on_linux(
        self,
        mock_time: MagicMock,
        mock_sys: MagicMock,
        mock_qapp: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """time.sleep is called on non-Windows platforms."""
        mock_sys.platform = "linux"
        view = MagicMock()

        from recorder.overlay.capture_guard import capture_guard

        with capture_guard(view):
            pass

        mock_time.sleep.assert_called_once_with(0.1)  # 100ms / 1000

    @patch("recorder.overlay.capture_guard.QApplication")
    @patch("recorder.overlay.capture_guard.sys")
    @patch("recorder.overlay.capture_guard.time")
    def test_config_delay_used(
        self,
        mock_time: MagicMock,
        mock_sys: MagicMock,
        mock_qapp: MagicMock,
    ) -> None:
        """capture_delay_ms from config is used as sleep duration."""
        mock_sys.platform = "linux"
        view = MagicMock()

        with patch(
            "recorder.overlay.capture_guard.get_config",
            return_value={"overlay": {"capture_delay_ms": 200}},
        ):
            from recorder.overlay.capture_guard import capture_guard

            with capture_guard(view):
                pass

        mock_time.sleep.assert_called_once_with(0.2)  # 200ms / 1000
