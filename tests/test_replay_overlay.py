"""Tests for replay overlay widgets and state.

Verifies REPLAYING state, shimmer handling, StatusBadge, TargetHighlight,
CameraFlash lifecycle, and replay config defaults.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


def test_replaying_state_exists() -> None:
    """REPLAYING state exists in OverlayState with purple color tuple."""
    from recorder.overlay.state import STATE_COLORS, OverlayState

    assert hasattr(OverlayState, "REPLAYING")
    rgba = STATE_COLORS[OverlayState.REPLAYING]
    assert rgba == (160, 80, 220, 180), f"Expected purple RGBA, got {rgba}"


def test_shimmer_replaying_state() -> None:
    """ShimmerLayer.set_state(REPLAYING) sets purple shimmer params."""
    from recorder.overlay.shimmer_layer import ShimmerLayer
    from recorder.overlay.state import OverlayState

    layer = ShimmerLayer(1920, 1080)
    layer.set_state(OverlayState.REPLAYING)

    assert layer._loop_duration == 3.0
    assert layer._alpha_mult == 0.45


# ---------------------------------------------------------------------------
# StatusBadge tests
# ---------------------------------------------------------------------------


def test_status_badge_set_text() -> None:
    """StatusBadge.set_text stores text attribute."""
    from recorder.overlay.status_badge import StatusBadge

    clock = MagicMock()
    badge = StatusBadge(clock, 1920, 1080)
    badge.set_text("Step 3/8: Click Submit")
    assert badge._text == "Step 3/8: Click Submit"


def test_status_badge_position() -> None:
    """StatusBadge is positioned at top-center."""
    from recorder.overlay.status_badge import StatusBadge

    clock = MagicMock()
    badge = StatusBadge(clock, 1920, 1080)
    # Should be centered horizontally: (1920/2 - 200) = 760
    assert badge.pos().x() == 760.0
    assert badge.pos().y() == 20.0


# ---------------------------------------------------------------------------
# TargetHighlight tests
# ---------------------------------------------------------------------------


def test_target_highlight_lifecycle() -> None:
    """TargetHighlight shows for 300ms then fades out."""
    from recorder.overlay.target_highlight import TargetHighlight

    clock = MagicMock()
    hl = TargetHighlight(clock)

    # Initially no rect
    assert hl._rect is None

    # Highlight a region
    hl.highlight(100, 200, 50, 30)
    assert hl._rect is not None
    assert hl._opacity == 1.0

    # Tick past duration (0.35s > 0.3s)
    hl.tick(0.35)
    assert hl._rect is None
    assert hl._opacity == 0.0


def test_target_highlight_partial_fade() -> None:
    """TargetHighlight opacity decreases linearly during fade."""
    from recorder.overlay.target_highlight import TargetHighlight

    clock = MagicMock()
    hl = TargetHighlight(clock)
    hl.highlight(10, 20, 30, 40)

    hl.tick(0.15)  # Half of 0.3s duration
    assert hl._rect is not None
    assert math.isclose(hl._opacity, 0.5, abs_tol=0.01)


def test_target_highlight_hide() -> None:
    """TargetHighlight.hide() immediately clears."""
    from recorder.overlay.target_highlight import TargetHighlight

    clock = MagicMock()
    hl = TargetHighlight(clock)
    hl.highlight(10, 20, 30, 40)
    hl.hide()
    assert hl._rect is None
    assert hl._opacity == 0.0


# ---------------------------------------------------------------------------
# CameraFlash tests
# ---------------------------------------------------------------------------


def test_camera_flash_lifecycle() -> None:
    """CameraFlash triggers then fades over 200ms."""
    from recorder.overlay.camera_flash import CameraFlash

    clock = MagicMock()
    flash = CameraFlash(clock, 1920, 1080)

    # Initially inactive
    assert flash._active is False
    assert flash._opacity == 0.0

    # Flash
    flash.flash()
    assert flash._active is True
    assert flash._opacity == 1.0

    # Tick halfway
    flash.tick(0.1)
    assert flash._active is True
    assert math.isclose(flash._opacity, 0.5, abs_tol=0.01)

    # Tick past duration
    flash.tick(0.15)
    assert flash._active is False
    assert flash._opacity == 0.0


def test_camera_flash_border_width() -> None:
    """CameraFlash border width is 15px."""
    from recorder.overlay.camera_flash import CameraFlash

    clock = MagicMock()
    flash = CameraFlash(clock, 1920, 1080)
    assert flash._border_width == 15


def test_camera_flash_hide() -> None:
    """CameraFlash.hide() immediately deactivates."""
    from recorder.overlay.camera_flash import CameraFlash

    clock = MagicMock()
    flash = CameraFlash(clock, 1920, 1080)
    flash.flash()
    assert flash._active is True
    flash.hide()
    assert flash._active is False
    assert flash._opacity == 0.0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_replay_config_defaults() -> None:
    """Replay config section exists with correct defaults."""
    from core.config import _DEFAULTS

    replay = _DEFAULTS["replay"]
    assert replay["show_overlay"] is True
    assert replay["keep_successful_runs"] == 5
    assert replay["keep_failed_runs"] == 10
    assert replay["poll_interval"] == 2.0
    assert replay["max_cascade_retries"] == 2
