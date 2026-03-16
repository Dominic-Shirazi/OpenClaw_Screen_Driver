"""Integration tests for OverlayView layer management.

Uses QT_QPA_PLATFORM=offscreen so tests run without a display.
"""

from __future__ import annotations

import os

# MUST be set before any PyQt6 import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PyQt6.QtWidgets import QApplication

from recorder.overlay.bbox_layer import BboxLayer
from recorder.overlay.click_catcher_layer import ClickCatcherLayer
from recorder.overlay.mode_indicator_layer import ModeIndicatorLayer
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
    v = OverlayView()
    return v


class TestViewSetup:
    """Tests for view initialization and scene creation."""

    def test_view_creates_scene(self, view: OverlayView) -> None:
        """OverlayView has a QGraphicsScene after construction."""
        scene = view.scene()
        assert scene is not None

    def test_view_has_shimmer_layer(self, view: OverlayView) -> None:
        """The scene contains a ShimmerLayer item (replaced BorderLayer)."""
        items = view.scene().items()
        shimmer_items = [i for i in items if isinstance(i, ShimmerLayer)]
        assert len(shimmer_items) >= 1

    def test_view_has_mode_indicator(self, view: OverlayView) -> None:
        """The scene contains a ModeIndicatorLayer item."""
        items = view.scene().items()
        mode_items = [i for i in items if isinstance(i, ModeIndicatorLayer)]
        assert len(mode_items) >= 1


class TestApplyState:
    """Tests for apply_state() visual updates."""

    def test_apply_state_updates_shimmer(self, view: OverlayView) -> None:
        """apply_state(RECORDING) updates the shimmer layer state."""
        view.apply_state(OverlayState.RECORDING)

        # ShimmerLayer should have updated its base color to red
        items = view.scene().items()
        shimmer_layers = [i for i in items if isinstance(i, ShimmerLayer)]
        assert len(shimmer_layers) >= 1
        shimmer = shimmer_layers[0]

        # After set_state(RECORDING), base color should be red
        assert shimmer._base_color.red() > 200

    def test_apply_state_adds_click_catcher(self, view: OverlayView) -> None:
        """apply_state(RECORDING) adds a ClickCatcherLayer to the scene."""
        view.apply_state(OverlayState.RECORDING)

        items = view.scene().items()
        catchers = [i for i in items if isinstance(i, ClickCatcherLayer)]
        assert len(catchers) >= 1

    def test_apply_state_removes_click_catcher(self, view: OverlayView) -> None:
        """apply_state(READY) removes the ClickCatcherLayer from the scene."""
        # First add it
        view.apply_state(OverlayState.RECORDING)
        # Then remove it
        view.apply_state(OverlayState.READY)

        items = view.scene().items()
        catchers = [i for i in items if isinstance(i, ClickCatcherLayer)]
        assert len(catchers) == 0

    def test_mode_indicator_updates(self, view: OverlayView) -> None:
        """apply_state(RECORDING) updates the mode indicator text."""
        view.apply_state(OverlayState.RECORDING)

        items = view.scene().items()
        indicators = [i for i in items if isinstance(i, ModeIndicatorLayer)]
        assert len(indicators) >= 1

        # The mode indicator's child text item should contain "RECORDING"
        indicator = indicators[0]
        found_recording = False
        for child in indicator.childItems():
            if hasattr(child, "text") and "RECORDING" in child.text():
                found_recording = True
                break
        assert found_recording, "Mode indicator should show RECORDING text"


class TestBboxManagement:
    """Tests for render_bboxes() and clear_bboxes()."""

    def test_render_bboxes_adds_items(self, view: OverlayView) -> None:
        """render_bboxes with 2 boxes adds 2 BboxLayer items."""
        boxes = [
            {
                "x": 10,
                "y": 20,
                "w": 100,
                "h": 50,
                "color": (255, 0, 0, 200),
                "label": "test1",
                "confidence": 0.9,
            },
            {
                "x": 200,
                "y": 300,
                "w": 80,
                "h": 40,
                "color": (0, 255, 0, 200),
                "label": "test2",
                "confidence": 0.8,
            },
        ]
        view.render_bboxes(boxes)

        items = view.scene().items()
        bbox_items = [i for i in items if isinstance(i, BboxLayer)]
        assert len(bbox_items) == 2

    def test_clear_bboxes_removes_items(self, view: OverlayView) -> None:
        """clear_bboxes leaves 0 BboxLayer items in the scene."""
        boxes = [
            {
                "x": 10,
                "y": 20,
                "w": 100,
                "h": 50,
                "color": (255, 0, 0, 200),
                "label": "test",
                "confidence": 0.9,
            },
        ]
        view.render_bboxes(boxes)
        view.clear_bboxes()

        items = view.scene().items()
        bbox_items = [i for i in items if isinstance(i, BboxLayer)]
        assert len(bbox_items) == 0
