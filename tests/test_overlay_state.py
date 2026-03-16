"""Tests for overlay state machine transitions and color map."""

from __future__ import annotations

from recorder.overlay.state import TRANSITIONS, STATE_COLORS, OverlayState, transition


class TestOverlayState:
    """State machine enum and transition table tests."""

    def test_initial_state(self) -> None:
        """READY is the expected default starting state."""
        assert OverlayState.READY is not None

    def test_f2_ready_to_recording(self) -> None:
        """F2 while READY transitions to RECORDING."""
        assert TRANSITIONS[(OverlayState.READY, "f2")] == OverlayState.RECORDING

    def test_f2_recording_to_ready(self) -> None:
        """F2 while RECORDING transitions to READY."""
        assert TRANSITIONS[(OverlayState.RECORDING, "f2")] == OverlayState.READY

    def test_f2_paused_to_recording(self) -> None:
        """F2 while PAUSED transitions to RECORDING."""
        assert TRANSITIONS[(OverlayState.PAUSED, "f2")] == OverlayState.RECORDING

    def test_invalid_transition(self) -> None:
        """Unknown trigger for READY is not in the transition table."""
        assert (OverlayState.READY, "unknown") not in TRANSITIONS

    def test_transition_helper_returns_state(self) -> None:
        """transition() helper returns the correct new state."""
        result = transition(OverlayState.READY, "f2")
        assert result == OverlayState.RECORDING

    def test_transition_helper_returns_none_for_invalid(self) -> None:
        """transition() helper returns None for unknown triggers."""
        result = transition(OverlayState.READY, "unknown")
        assert result is None


class TestStateColors:
    """Color map tests for overlay states."""

    def test_state_colors_green_for_ready(self) -> None:
        """READY state is green."""
        assert STATE_COLORS[OverlayState.READY] == (50, 200, 50, 150)

    def test_state_colors_red_for_recording(self) -> None:
        """RECORDING state is red."""
        assert STATE_COLORS[OverlayState.RECORDING] == (255, 50, 50, 200)

    def test_paused_same_color_as_ready(self) -> None:
        """PAUSED state has the same color as READY."""
        assert STATE_COLORS[OverlayState.PAUSED] == STATE_COLORS[OverlayState.READY]
