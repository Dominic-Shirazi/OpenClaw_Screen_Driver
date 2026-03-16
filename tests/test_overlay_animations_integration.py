"""Integration tests for animation wiring in OverlayView and OverlayController.

Verifies that AnimationClock, ShimmerLayer, ScanLayer, DonutCloudLayer,
and bbox morph are properly wired into the overlay view and controller.

Uses QT_QPA_PLATFORM=offscreen so tests run without a display.
"""

from __future__ import annotations

import os

# MUST be set before any PyQt6 import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from unittest.mock import MagicMock, patch
from PyQt6.QtWidgets import QApplication

from recorder.overlay.animation_clock import AnimationClock
from recorder.overlay.bbox_layer import BboxLayer
from recorder.overlay.donut_cloud_layer import DonutCloudLayer
from recorder.overlay.scan_layer import ScanLayer
from recorder.overlay.shimmer_layer import ShimmerLayer
from recorder.overlay.state import OverlayState
from recorder.overlay.view import OverlayView


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Provide a QApplication instance for the test module."""
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture()
def view(qapp: QApplication) -> OverlayView:
    """Create an OverlayView instance in offscreen mode."""
    return OverlayView()


class TestViewAnimationSetup:
    """Tests for animation infrastructure in OverlayView."""

    def test_view_creates_shimmer_layer(self, view: OverlayView) -> None:
        """OverlayView creates a ShimmerLayer instead of BorderLayer."""
        assert hasattr(view, "_shimmer")
        assert isinstance(view._shimmer, ShimmerLayer)

    def test_view_creates_animation_clock(self, view: OverlayView) -> None:
        """OverlayView creates and owns an AnimationClock."""
        assert hasattr(view, "_clock")
        assert isinstance(view._clock, AnimationClock)

    def test_shimmer_in_scene(self, view: OverlayView) -> None:
        """ShimmerLayer is present in the scene items."""
        items = view.scene().items()
        shimmer_items = [i for i in items if isinstance(i, ShimmerLayer)]
        assert len(shimmer_items) >= 1

    def test_mouse_tracking_enabled(self, view: OverlayView) -> None:
        """OverlayView has mouse tracking enabled for shimmer retreat."""
        assert view.hasMouseTracking() is True


class TestApplyStateShimmer:
    """Tests for shimmer state integration via apply_state()."""

    def test_apply_state_calls_shimmer_set_state(self, view: OverlayView) -> None:
        """apply_state(RECORDING) delegates to shimmer.set_state()."""
        with patch.object(view._shimmer, "set_state") as mock_set:
            view.apply_state(OverlayState.RECORDING)
            mock_set.assert_called_once_with(OverlayState.RECORDING)


class TestCaptureLifecycle:
    """Tests for clock stop/start during capture."""

    def test_hide_for_capture_stops_clock(self, view: OverlayView) -> None:
        """hide_for_capture() stops the animation clock."""
        with patch.object(view._clock, "stop") as mock_stop:
            view.hide_for_capture()
            mock_stop.assert_called_once()

    def test_show_after_capture_starts_clock(self, view: OverlayView) -> None:
        """show_after_capture() restarts the animation clock."""
        with patch.object(view._clock, "start") as mock_start:
            view.show_after_capture()
            mock_start.assert_called_once()


class TestScanLayerWiring:
    """Tests for scan layer lifecycle in view."""

    def test_start_scan_creates_scan_layer(self, view: OverlayView) -> None:
        """start_scan() creates a ScanLayer and adds it to the scene."""
        view.start_scan(100, 100, 200, 200)
        assert view._scan_layer is not None
        assert isinstance(view._scan_layer, ScanLayer)

        items = view.scene().items()
        scan_items = [i for i in items if isinstance(i, ScanLayer)]
        assert len(scan_items) >= 1

        # Cleanup
        view.remove_scan()

    def test_finish_scan_triggers_bbox_morph(self, view: OverlayView) -> None:
        """finish_scan() calls morph_to() on the active bbox."""
        bbox = BboxLayer(100, 100, 200, 200, (255, 50, 50, 200))
        view.start_scan(100, 100, 200, 200, bbox=bbox)

        with patch.object(bbox, "morph_to") as mock_morph:
            view.finish_scan(110, 105, 190, 195)
            mock_morph.assert_called_once_with(110, 105, 190, 195)

        # Cleanup
        view.remove_scan()


class TestDonutCloudWiring:
    """Tests for donut cloud lifecycle in view."""

    def test_show_donut_cloud_creates_layer(self, view: OverlayView) -> None:
        """show_donut_cloud() creates a DonutCloudLayer in the scene."""
        view.show_donut_cloud(300.0, 300.0)
        assert view._donut_cloud is not None
        assert isinstance(view._donut_cloud, DonutCloudLayer)

        items = view.scene().items()
        cloud_items = [i for i in items if isinstance(i, DonutCloudLayer)]
        assert len(cloud_items) >= 1

        # Cleanup
        view.remove_donut_cloud()


class TestControllerDelegation:
    """Tests for controller delegation to view."""

    def test_controller_start_scan_delegates(self, qapp: QApplication) -> None:
        """Controller.start_scan() delegates to view.start_scan()."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        ctrl.start_scan(100, 100, 200, 200)
        mock_view.start_scan.assert_called_once_with(100, 100, 200, 200, bbox=None)

    def test_controller_show_donut_cloud_delegates(self, qapp: QApplication) -> None:
        """Controller.show_donut_cloud() delegates to view.show_donut_cloud()."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        ctrl.show_donut_cloud(300.0, 300.0)
        mock_view.show_donut_cloud.assert_called_once_with(300.0, 300.0, 60.0)

    def test_controller_accept_donut_cloud_delegates(self, qapp: QApplication) -> None:
        """Controller.accept_donut_cloud() delegates to view."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        ctrl.accept_donut_cloud()
        mock_view.accept_donut_cloud.assert_called_once()

    def test_controller_finish_scan_delegates(self, qapp: QApplication) -> None:
        """Controller.finish_scan() delegates to view.finish_scan()."""
        from recorder.overlay.controller import OverlayController

        ctrl = OverlayController()
        mock_view = MagicMock()
        ctrl._view = mock_view

        ctrl.finish_scan(110, 105, 190, 195)
        mock_view.finish_scan.assert_called_once_with(110, 105, 190, 195)
