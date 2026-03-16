"""Tests for DPI coordinate conversion utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestLogicalToPhysical:
    """Tests for logical_to_physical conversion."""

    @patch("recorder.overlay.dpi.QApplication")
    def test_logical_to_physical_1x(self, mock_qapp: MagicMock) -> None:
        """At 1x DPI, coordinates are unchanged."""
        mock_screen = MagicMock()
        mock_screen.devicePixelRatio.return_value = 1.0
        mock_qapp.primaryScreen.return_value = mock_screen

        from recorder.overlay.dpi import logical_to_physical

        assert logical_to_physical(100, 200) == (100, 200)

    @patch("recorder.overlay.dpi.QApplication")
    def test_logical_to_physical_2x(self, mock_qapp: MagicMock) -> None:
        """At 2x DPI, coordinates are doubled."""
        mock_screen = MagicMock()
        mock_screen.devicePixelRatio.return_value = 2.0
        mock_qapp.primaryScreen.return_value = mock_screen

        from recorder.overlay.dpi import logical_to_physical

        assert logical_to_physical(100, 200) == (200, 400)


class TestPhysicalToLogical:
    """Tests for physical_to_logical conversion."""

    @patch("recorder.overlay.dpi.QApplication")
    def test_physical_to_logical_2x(self, mock_qapp: MagicMock) -> None:
        """At 2x DPI, physical coords are halved to logical."""
        mock_screen = MagicMock()
        mock_screen.devicePixelRatio.return_value = 2.0
        mock_qapp.primaryScreen.return_value = mock_screen

        from recorder.overlay.dpi import physical_to_logical

        assert physical_to_logical(200, 400) == (100, 200)


class TestRoundTrip:
    """Tests for DPI conversion round-trip accuracy."""

    @patch("recorder.overlay.dpi.QApplication")
    def test_round_trip(self, mock_qapp: MagicMock) -> None:
        """physical_to_logical(logical_to_physical(x,y)) == (x,y)."""
        mock_screen = MagicMock()
        mock_screen.devicePixelRatio.return_value = 2.0
        mock_qapp.primaryScreen.return_value = mock_screen

        from recorder.overlay.dpi import logical_to_physical, physical_to_logical

        x, y = 100, 200
        px, py = logical_to_physical(x, y)
        rx, ry = physical_to_logical(px, py)
        assert (rx, ry) == (x, y)

    @patch("recorder.overlay.dpi.QApplication")
    def test_no_screen_returns_identity(self, mock_qapp: MagicMock) -> None:
        """When no primary screen, return input unchanged."""
        mock_qapp.primaryScreen.return_value = None

        from recorder.overlay.dpi import logical_to_physical

        assert logical_to_physical(100, 200) == (100, 200)


class TestGetPhysicalScreenSize:
    """Tests for get_physical_screen_size."""

    @patch("recorder.overlay.dpi.QApplication")
    def test_get_physical_screen_size_2x(self, mock_qapp: MagicMock) -> None:
        """At 2x DPI, logical 1920x1080 becomes 3840x2160."""
        mock_screen = MagicMock()
        mock_screen.devicePixelRatio.return_value = 2.0
        mock_size = MagicMock()
        mock_size.width.return_value = 1920
        mock_size.height.return_value = 1080
        mock_screen.size.return_value = mock_size
        mock_qapp.primaryScreen.return_value = mock_screen

        from recorder.overlay.dpi import get_physical_screen_size

        assert get_physical_screen_size() == (3840, 2160)

    @patch("recorder.overlay.dpi.QApplication")
    def test_no_screen_returns_default(self, mock_qapp: MagicMock) -> None:
        """When no primary screen, return default 1920x1080."""
        mock_qapp.primaryScreen.return_value = None

        from recorder.overlay.dpi import get_physical_screen_size

        assert get_physical_screen_size() == (1920, 1080)
