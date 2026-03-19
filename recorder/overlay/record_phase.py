"""Recording pipeline sub-states.

Defines the 10 sub-states of the recording flow.  Each RecordPhase
describes a visual + interaction mode of the overlay during recording.
RecordSession (future plan) will own the transition table between phases.
"""
from __future__ import annotations

from enum import Enum, auto


class RecordPhase(Enum):
    """Sub-states of the recording pipeline.

    Each member corresponds to a distinct visual and interaction behavior
    in the overlay during active recording.
    """

    AWAITING_CLICK = auto()
    """Red shimmer, crosshair cursor, waiting for click/drag."""

    CAPTURING = auto()
    """Overlay hidden, screenshot in progress."""

    DETECTING = auto()
    """Background OmniParser running, card glow pulsing."""

    BBOX_EDITING = auto()
    """User editing/accepting AI bbox proposal."""

    VLM_ANALYZING = auto()
    """VLM running, card glow pulsing."""

    TAG_DIALOG = auto()
    """Tag dialog open, user reviewing/editing."""

    COUNTDOWN = auto()
    """3-2-1 countdown before dry-run."""

    EXECUTING = auto()
    """Dry-run action in progress, overlay click-through."""

    VALIDATING = auto()
    """Post-execution, toolbar shows Yes/No/Retry."""

    SUCCESS_FLASH = auto()
    """Brief green flash, then back to AWAITING_CLICK."""

    AWAITING_REGION_DRAG = auto()
    """After 'Look Here' button, waiting for user to drag a region."""

    AWAITING_DRAG_TARGET = auto()
    """After click_drag tag confirm, waiting for second click on drag target."""

    WAIT_CONFIGURING = auto()
    """Wait condition mini-dialog open."""

    PROMPT_CONFIGURING = auto()
    """Prompt question mini-dialog open."""

    LOOP_DEFINING = auto()
    """Loop definition dialog open, user selecting step range and condition."""
