"""Shared data types for the OCSD framework.

Defines the data classes and exceptions used across all modules.
These types form the contract between modules — changes here affect everything.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Value objects (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    """Screen coordinate."""
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    """Bounding rectangle."""
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Point:
        return Point(self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LocateResult:
    """Result of attempting to locate an element on screen."""
    point: Point
    method: str          # "ocr" | "embedding" | "vlm" | "direct"
    confidence: float
    rect: Rect | None = None


@dataclass
class VisionResult:
    """Result of VLM element analysis."""
    element_type: str
    label_guess: str
    confidence: float
    ocr_text: str | None = None


@dataclass
class ConfirmResult:
    """Result of VLM action confirmation."""
    success: bool
    confidence: float
    notes: str


@dataclass
class CandidateElement:
    """A candidate element detected by first_pass_map."""
    rect: Rect
    type_guess: str
    label_guess: str
    confidence: float


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class WatcherEvent:
    """Event emitted by the background watcher."""
    event_type: str      # "window_changed" | "url_changed" | "pixel_diff_exceeded"
    old_value: str | None
    new_value: str | None
    diff_pct: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Replay logging
# ---------------------------------------------------------------------------

@dataclass
class ReplayStep:
    """A single step in a replay log."""
    node_id: str
    located_at: Point | None
    locate_method: str   # "ocr" | "embedding" | "vlm" | "failed"
    vlm_confidence: float
    pixel_diff_pct: float
    success: bool
    error: str | None = None


@dataclass
class ReplayLog:
    """Complete log of a skill execution."""
    replay_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skill_id: str = ""
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    steps: list[ReplayStep] = field(default_factory=list)
    overall_success: bool = True
    duration_ms: int = 0

    def append_step(self, step: ReplayStep) -> None:
        self.steps.append(step)
        if not step.success:
            self.overall_success = False

    def to_dict(self) -> dict:
        return {
            "replay_id": self.replay_id,
            "skill_id": self.skill_id,
            "executed_at": self.executed_at,
            "steps": [
                {
                    "node_id": s.node_id,
                    "located_at": {"x": s.located_at.x, "y": s.located_at.y} if s.located_at else None,
                    "locate_method": s.locate_method,
                    "vlm_confidence": s.vlm_confidence,
                    "pixel_diff_pct": s.pixel_diff_pct,
                    "success": s.success,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "overall_success": self.overall_success,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ElementNotFoundError(Exception):
    """Raised when an element cannot be located on screen after all fallbacks."""

    def __init__(self, node_id: str, message: str = ""):
        self.node_id = node_id
        super().__init__(message or f"Element not found: {node_id}")


class LowConfidenceError(Exception):
    """Raised when VLM confidence is below the threshold."""

    def __init__(self, node_id: str, confidence: float, threshold: float):
        self.node_id = node_id
        self.confidence = confidence
        self.threshold = threshold
        super().__init__(
            f"Low confidence for {node_id}: {confidence:.2f} < {threshold:.2f}"
        )


class PathNotFoundError(Exception):
    """Raised when no path exists between two nodes in the graph."""

    def __init__(self, start_id: str, end_id: str):
        self.start_id = start_id
        self.end_id = end_id
        super().__init__(f"No path from {start_id} to {end_id}")
