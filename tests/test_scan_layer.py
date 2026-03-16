"""Unit tests for ScanLayer multi-phase animation state machine.

Tests cover phase transitions, timing, laser loop, AI bbox reception,
and reset behavior.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ensure offscreen rendering for headless CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

# Create QApplication once for all tests in this module
_app: QApplication | None = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from recorder.overlay.scan_layer import ScanLayer, ScanPhase


class TestScanLayerConstruction:
    """Test ScanLayer instantiation."""

    def test_instantiates_with_rect(self) -> None:
        """ScanLayer instantiates with bounding rect coordinates."""
        layer = ScanLayer(100, 200, 300, 400)
        assert layer is not None

    def test_initial_phase_is_idle(self) -> None:
        """ScanLayer starts in IDLE phase."""
        layer = ScanLayer(10, 20, 100, 100)
        assert layer.phase == ScanPhase.IDLE


class TestScanLayerPhaseTransitions:
    """Test phase transition logic."""

    def test_start_scan_transitions_to_corner_glow(self) -> None:
        """start_scan() transitions from IDLE to CORNER_GLOW."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        assert layer.phase == ScanPhase.CORNER_GLOW

    def test_tick_advances_corner_glow_to_line_draw(self) -> None:
        """Ticking past CORNER_GLOW duration advances to LINE_DRAW."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        # CORNER_GLOW duration is 0.3s
        layer.tick(0.31)
        assert layer.phase == ScanPhase.LINE_DRAW

    def test_tick_advances_through_all_initial_phases(self) -> None:
        """Ticking advances CORNER_GLOW -> LINE_DRAW -> FILL_INWARD -> LASER_V -> LASER_H."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()

        # CORNER_GLOW (0.3s)
        layer.tick(0.31)
        assert layer.phase == ScanPhase.LINE_DRAW

        # LINE_DRAW (0.8s)
        layer.tick(0.81)
        assert layer.phase == ScanPhase.FILL_INWARD

        # FILL_INWARD (0.4s)
        layer.tick(0.41)
        assert layer.phase == ScanPhase.LASER_VERTICAL

        # LASER_VERTICAL (1.0s)
        layer.tick(1.01)
        assert layer.phase == ScanPhase.LASER_HORIZONTAL

    def test_laser_horizontal_transitions_to_waiting_ai(self) -> None:
        """After LASER_HORIZONTAL completes without AI result, enters WAITING_AI."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        # Advance through all phases to LASER_HORIZONTAL
        layer.tick(0.31)  # -> LINE_DRAW
        layer.tick(0.81)  # -> FILL_INWARD
        layer.tick(0.41)  # -> LASER_VERTICAL
        layer.tick(1.01)  # -> LASER_HORIZONTAL
        layer.tick(1.01)  # -> WAITING_AI
        assert layer.phase == ScanPhase.WAITING_AI

    def test_waiting_ai_loops_back_to_laser_vertical(self) -> None:
        """In WAITING_AI, loops back to LASER_VERTICAL for repeated scanning."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        # Advance to WAITING_AI
        layer.tick(0.31)  # -> LINE_DRAW
        layer.tick(0.81)  # -> FILL_INWARD
        layer.tick(0.41)  # -> LASER_VERTICAL
        layer.tick(1.01)  # -> LASER_HORIZONTAL
        layer.tick(1.01)  # -> WAITING_AI
        assert layer.phase == ScanPhase.WAITING_AI

        # WAITING_AI transitions immediately to LASER_VERTICAL on next tick
        layer.tick(0.01)
        assert layer.phase == ScanPhase.LASER_VERTICAL

    def test_receive_fitted_bbox_transitions_to_snap(self) -> None:
        """receive_fitted_bbox() transitions to SNAP_TO_FITTED when in WAITING_AI."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        # Advance to WAITING_AI
        layer.tick(0.31)
        layer.tick(0.81)
        layer.tick(0.41)
        layer.tick(1.01)
        layer.tick(1.01)
        assert layer.phase == ScanPhase.WAITING_AI

        layer.receive_fitted_bbox(10, 15, 180, 190)
        assert layer.phase == ScanPhase.SNAP_TO_FITTED

    def test_snap_to_fitted_transitions_to_done(self) -> None:
        """SNAP_TO_FITTED phase completes and transitions to DONE."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        # Advance to WAITING_AI and receive bbox
        layer.tick(0.31)
        layer.tick(0.81)
        layer.tick(0.41)
        layer.tick(1.01)
        layer.tick(1.01)
        layer.receive_fitted_bbox(10, 15, 180, 190)
        assert layer.phase == ScanPhase.SNAP_TO_FITTED

        # SNAP_TO_FITTED duration is 0.5s
        layer.tick(0.51)
        assert layer.phase == ScanPhase.DONE

    def test_reset_returns_to_idle(self) -> None:
        """reset() returns to IDLE from any phase."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        layer.tick(0.31)
        assert layer.phase != ScanPhase.IDLE

        layer.reset()
        assert layer.phase == ScanPhase.IDLE


class TestScanLayerPhaseDurations:
    """Test that phase durations are respected."""

    def test_corner_glow_duration(self) -> None:
        """CORNER_GLOW lasts 0.3 seconds."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        # Not enough time
        layer.tick(0.2)
        assert layer.phase == ScanPhase.CORNER_GLOW
        # Enough time
        layer.tick(0.11)
        assert layer.phase == ScanPhase.LINE_DRAW

    def test_line_draw_duration(self) -> None:
        """LINE_DRAW lasts 0.8 seconds."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        layer.tick(0.31)  # -> LINE_DRAW
        layer.tick(0.7)   # still LINE_DRAW
        assert layer.phase == ScanPhase.LINE_DRAW
        layer.tick(0.11)  # -> FILL_INWARD
        assert layer.phase == ScanPhase.FILL_INWARD

    def test_idle_and_done_ignore_tick(self) -> None:
        """tick() does nothing in IDLE and DONE phases."""
        layer = ScanLayer(0, 0, 200, 200)
        # IDLE
        layer.tick(10.0)
        assert layer.phase == ScanPhase.IDLE

    def test_receive_bbox_during_laser_phases(self) -> None:
        """receive_fitted_bbox() during laser phases triggers SNAP_TO_FITTED."""
        layer = ScanLayer(0, 0, 200, 200)
        layer.start_scan()
        layer.tick(0.31)  # -> LINE_DRAW
        layer.tick(0.81)  # -> FILL_INWARD
        layer.tick(0.41)  # -> LASER_VERTICAL
        assert layer.phase == ScanPhase.LASER_VERTICAL

        layer.receive_fitted_bbox(5, 5, 190, 195)
        assert layer.phase == ScanPhase.SNAP_TO_FITTED
