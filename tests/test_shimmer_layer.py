"""Unit tests for ShimmerLayer -- animated border shimmer glow.

Tests cover instantiation, bounding rect, state-driven color/speed/glow,
mouse position storage, tick phase advancement, retreat factor near/far
from mouse, and glow radius per state.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

_app: QApplication | None = None


@pytest.fixture(autouse=True)
def qapp() -> QApplication:
    """Provide a QApplication instance for all tests."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


def _make_shimmer(w: int = 1920, h: int = 1080):  # noqa: ANN202
    """Import and instantiate ShimmerLayer."""
    from recorder.overlay.shimmer_layer import ShimmerLayer

    return ShimmerLayer(w, h)


class TestShimmerInstantiation:
    """Test 1: ShimmerLayer can be instantiated with screen dimensions."""

    def test_create(self) -> None:
        layer = _make_shimmer(1920, 1080)
        assert layer is not None

    def test_z_value(self) -> None:
        layer = _make_shimmer()
        assert layer.zValue() == 10


class TestBoundingRect:
    """Test 2: boundingRect returns QRectF(0, 0, screen_w, screen_h)."""

    def test_bounding_rect(self) -> None:
        layer = _make_shimmer(1920, 1080)
        rect = layer.boundingRect()
        assert rect == QRectF(0, 0, 1920, 1080)

    def test_bounding_rect_custom(self) -> None:
        layer = _make_shimmer(3840, 2160)
        rect = layer.boundingRect()
        assert rect == QRectF(0, 0, 3840, 2160)


class TestSetStateReady:
    """Test 3: set_state(READY) sets green color and 4-6s loop duration."""

    def test_ready_color(self) -> None:
        from recorder.overlay.state import OverlayState

        layer = _make_shimmer()
        layer.set_state(OverlayState.READY)
        assert layer._base_color.red() == 50
        assert layer._base_color.green() == 200
        assert layer._base_color.blue() == 50

    def test_ready_loop_duration(self) -> None:
        from recorder.overlay.state import OverlayState

        layer = _make_shimmer()
        layer.set_state(OverlayState.READY)
        assert 4.0 <= layer._loop_duration <= 6.0


class TestSetStateRecording:
    """Test 4: set_state(RECORDING) sets red color and ~2s loop duration."""

    def test_recording_color(self) -> None:
        from recorder.overlay.state import OverlayState

        layer = _make_shimmer()
        layer.set_state(OverlayState.RECORDING)
        assert layer._base_color.red() == 255
        assert layer._base_color.green() == 50
        assert layer._base_color.blue() == 50

    def test_recording_loop_duration(self) -> None:
        from recorder.overlay.state import OverlayState

        layer = _make_shimmer()
        layer.set_state(OverlayState.RECORDING)
        assert layer._loop_duration == 2.0


class TestMousePos:
    """Test 5: set_mouse_pos stores the position for retreat calculation."""

    def test_set_mouse_pos(self) -> None:
        layer = _make_shimmer()
        layer.set_mouse_pos(500.0, 300.0)
        assert layer._mouse_pos.x() == 500.0
        assert layer._mouse_pos.y() == 300.0


class TestTickPhase:
    """Test 6: tick(dt) advances _phase by dt/loop_duration and wraps at 1.0."""

    def test_phase_advances(self) -> None:
        layer = _make_shimmer()
        layer._loop_duration = 5.0
        layer._phase = 0.0
        layer.tick(1.0)
        assert abs(layer._phase - 0.2) < 0.001  # 1.0 / 5.0 = 0.2

    def test_phase_wraps(self) -> None:
        layer = _make_shimmer()
        layer._loop_duration = 5.0
        layer._phase = 0.9
        layer.tick(1.0)  # 0.9 + 0.2 = 1.1 -> wraps to 0.1
        assert abs(layer._phase - 0.1) < 0.01


class TestRetreatFactor:
    """Test 7: retreat_factor returns ~0 near mouse, ~1 far away."""

    def test_retreat_far_from_mouse(self) -> None:
        layer = _make_shimmer()
        layer.set_mouse_pos(-1000.0, -1000.0)  # Far offscreen
        factor = layer._retreat_factor(960.0, 0.0)
        assert factor > 0.95

    def test_retreat_at_mouse(self) -> None:
        layer = _make_shimmer()
        layer.set_mouse_pos(960.0, 0.0)
        factor = layer._retreat_factor(960.0, 0.0)
        assert factor < 0.05

    def test_retreat_smoothstep_midpoint(self) -> None:
        """At half the retreat radius, factor should be ~0.5 (smoothstep)."""
        layer = _make_shimmer()
        # _RETREAT_RADIUS is 250.0
        layer.set_mouse_pos(0.0, 0.0)
        factor = layer._retreat_factor(125.0, 0.0)
        assert 0.3 < factor < 0.7  # smoothstep at t=0.5 => 0.5


class TestGlowRadius:
    """Test 8: glow_radius is 150 for READY/PAUSED and 100 for RECORDING."""

    def test_ready_glow(self) -> None:
        from recorder.overlay.state import OverlayState

        layer = _make_shimmer()
        layer.set_state(OverlayState.READY)
        assert layer._target_glow_radius == 140.0

    def test_recording_glow(self) -> None:
        from recorder.overlay.state import OverlayState

        layer = _make_shimmer()
        layer.set_state(OverlayState.RECORDING)
        assert layer._target_glow_radius == 120.0

    def test_paused_glow(self) -> None:
        from recorder.overlay.state import OverlayState

        layer = _make_shimmer()
        layer.set_state(OverlayState.PAUSED)
        assert layer._target_glow_radius == 140.0
