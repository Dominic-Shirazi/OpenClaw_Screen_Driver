"""Overlay state machine: states, transitions, and color map.

Defines the three overlay states (READY, RECORDING, PAUSED), the
transition table driven by hotkey triggers, and the RGBA color map
used for border rendering in each state.
"""

from __future__ import annotations

from enum import Enum, auto


class OverlayState(Enum):
    """Overlay operational state."""

    READY = auto()       # Green shimmer, click-through ON
    RECORDING = auto()   # Red shimmer, click-through OFF (captures clicks)
    PAUSED = auto()      # Green shimmer, click-through ON (same visual as READY)


# Transition table: (current_state, trigger) -> new_state
TRANSITIONS: dict[tuple[OverlayState, str], OverlayState] = {
    (OverlayState.READY, "f2"): OverlayState.RECORDING,
    (OverlayState.RECORDING, "f2"): OverlayState.READY,
    (OverlayState.PAUSED, "f2"): OverlayState.RECORDING,
}

# RGBA color map per state (used by border glow rendering)
STATE_COLORS: dict[OverlayState, tuple[int, int, int, int]] = {
    OverlayState.READY: (50, 200, 50, 150),       # Green
    OverlayState.RECORDING: (255, 50, 50, 200),    # Red
    OverlayState.PAUSED: (50, 200, 50, 150),       # Green (same as READY)
}


def transition(current: OverlayState, trigger: str) -> OverlayState | None:
    """Look up the next state for a given trigger.

    Args:
        current: The current overlay state.
        trigger: The trigger string (e.g. "f2").

    Returns:
        The new state if the transition is valid, or None.
    """
    return TRANSITIONS.get((current, trigger))
