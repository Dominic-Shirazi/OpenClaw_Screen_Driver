"""YOLOE targeted visual finder for element re-location during replay.

Uses YOLOE's visual prompt mode (SAVPE) to perform one-shot grounding:
give it a saved snippet from recording time and it finds the matching
element on the current screen, returning a confidence-scored bbox.

This is NOT a general "detect everything then filter" approach.  It is
a prompted, targeted search: "here's what it looked like → find it now."

Requires:
    pip install ultralytics>=8.3
    Model weights are downloaded on first use (~50 MB for yoloe-26s-seg).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import get_config
from core.types import LocateResult, Point, Rect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singleton
# ---------------------------------------------------------------------------
_model: Any | None = None
_model_path: str = ""


@dataclass
class YOLOEMatch:
    """A single match returned by the targeted finder."""

    bbox: Rect          # absolute pixel bbox on the target image
    confidence: float   # 0-1 detection confidence
    center: Point       # convenience: bbox center


def _load_model() -> Any:
    """Lazily loads the YOLOE model (downloads weights on first run)."""
    global _model, _model_path

    if _model is not None:
        return _model

    try:
        from ultralytics import YOLOE
    except ImportError:
        raise ImportError(
            "YOLOE requires ultralytics>=8.3.  "
            "Install with:  pip install ultralytics"
        )

    cfg = get_config()
    _model_path = cfg.get("models", {}).get("yoloe", "yoloe-26s-seg.pt")
    logger.info("Loading YOLOE model: %s", _model_path)

    _model = YOLOE(_model_path)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_element(
    snippet: np.ndarray,
    screen: np.ndarray,
    *,
    hint_x: int | None = None,
    hint_y: int | None = None,
    search_radius: int = 400,
    conf_threshold: float = 0.25,
) -> list[YOLOEMatch]:
    """One-shot targeted finder: locate *snippet* on *screen*.

    Uses YOLOE visual prompts — the snippet is the reference image and
    the full bbox of the snippet is the visual prompt (meaning "the
    whole reference image is the thing I want you to find").

    If *hint_x*/*hint_y* are provided the search is spatially narrowed
    to a region of *search_radius* pixels around that point, which is
    both faster and more precise (avoids false positives from ads etc.).

    Args:
        snippet: BGR image of the element saved during recording.
        screen:  BGR full-screen capture of the current display.
        hint_x:  Optional expected X pixel position on *screen*.
        hint_y:  Optional expected Y pixel position on *screen*.
        search_radius: Pixel radius around the hint to crop (default 400).
        conf_threshold: Minimum detection confidence to keep.

    Returns:
        List of :class:`YOLOEMatch` sorted by confidence (best first).
        Empty list if nothing above *conf_threshold* was found.
    """
    model = _load_model()

    # Build the search region — either the full screen or a crop
    offset_x, offset_y = 0, 0
    target = screen

    if hint_x is not None and hint_y is not None:
        sh, sw = screen.shape[:2]
        x1 = max(0, hint_x - search_radius)
        y1 = max(0, hint_y - search_radius)
        x2 = min(sw, hint_x + search_radius)
        y2 = min(sh, hint_y + search_radius)
        target = screen[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1

    # Visual prompt: the entire snippet IS the thing to find.
    # bbox covers the full snippet image.
    snip_h, snip_w = snippet.shape[:2]
    visual_prompts = {
        "bboxes": np.array([[0, 0, snip_w, snip_h]], dtype=np.float32),
        "cls": np.array([0]),
    }

    try:
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        results = model.predict(
            target,
            refer_image=snippet,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=conf_threshold,
            verbose=False,
        )
    except Exception as e:
        logger.warning("YOLOE prediction failed: %s", e)
        return []

    # Parse results into YOLOEMatch objects
    matches: list[YOLOEMatch] = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            conf_val = float(boxes.conf[i])
            if conf_val < conf_threshold:
                continue

            # xyxy format: [x1, y1, x2, y2]
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            bx1, by1, bx2, by2 = xyxy

            # Offset back to full-screen coordinates
            abs_x1 = bx1 + offset_x
            abs_y1 = by1 + offset_y
            abs_x2 = bx2 + offset_x
            abs_y2 = by2 + offset_y

            rect = Rect(
                x=abs_x1,
                y=abs_y1,
                w=abs_x2 - abs_x1,
                h=abs_y2 - abs_y1,
            )
            center = Point(
                x=abs_x1 + (abs_x2 - abs_x1) // 2,
                y=abs_y1 + (abs_y2 - abs_y1) // 2,
            )
            matches.append(YOLOEMatch(
                bbox=rect,
                confidence=conf_val,
                center=center,
            ))

    # Sort best-first
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches


def find_element_locate(
    snippet: np.ndarray,
    screen: np.ndarray,
    hint_x: int | None = None,
    hint_y: int | None = None,
    search_radius: int = 400,
    conf_threshold: float = 0.35,
) -> LocateResult | None:
    """Convenience wrapper that returns a :class:`LocateResult` or *None*.

    This is the function called directly by the locate cascade in
    :mod:`mapper.runner`.
    """
    matches = find_element(
        snippet,
        screen,
        hint_x=hint_x,
        hint_y=hint_y,
        search_radius=search_radius,
        conf_threshold=conf_threshold,
    )
    if not matches:
        return None

    best = matches[0]
    return LocateResult(
        point=best.center,
        confidence=best.confidence,
        method="yoloe",
        rect=best.bbox,
    )
