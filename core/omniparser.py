"""OmniParser YOLOv8 icon_detect wrapper.

Wraps Microsoft OmniParser's fine-tuned YOLOv8 model for UI element
detection. Implements the DetectionProvider protocol from core.detection.

The model weights are auto-downloaded from HuggingFace on first use
via core.model_cache.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from core import model_cache
from core.config import get_config
from core.types import LocateResult, Point, Rect

logger = logging.getLogger(__name__)


# Lazy import — only needed when model actually runs
def _import_yolo():
    from ultralytics import YOLO
    return YOLO

YOLO = None  # set on first use


def generate_embedding(img: np.ndarray) -> np.ndarray:
    """Import and call core.embeddings.generate_embedding.

    Separated for easy mocking in tests.

    Args:
        img: RGB image as numpy array.

    Returns:
        L2-normalized embedding vector of shape (1, D).
    """
    from core.embeddings import generate_embedding as _gen
    return _gen(img)


class OmniParserProvider:
    """DetectionProvider implementation using OmniParser's YOLOv8 icon_detect.

    Attributes:
        _model: The loaded YOLO model (None until first detect() call).
        _confidence_threshold: Minimum detection confidence.
        _lock: Thread lock for lazy model loading.
    """

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        """Initialize. Model loads lazily on first detect() call.

        Args:
            confidence_threshold: Minimum confidence for detections.
        """
        self._model: Any = None
        self._confidence_threshold = confidence_threshold
        self._lock = threading.Lock()

    def _load_model(self) -> None:
        """Download and load the OmniParser YOLOv8 model."""
        global YOLO
        if YOLO is None:
            YOLO = _import_yolo()

        cfg = get_config()
        repo_id = cfg.get("models", {}).get(
            "omniparser_weights", "microsoft/OmniParser-v2.0"
        )

        model_path = model_cache.ensure_model(repo_id, "icon_detect/model.pt")
        logger.info("Loading OmniParser model from %s", model_path)

        # Determine device
        device = "cpu"
        try:
            import torch
            gpu_id = cfg.get("hardware", {}).get("gpu_vlm", 0)
            if torch.cuda.is_available():
                device = f"cuda:{gpu_id}"
        except ImportError:
            pass

        self._model = YOLO(str(model_path))
        self._model.to(device)
        logger.info("OmniParser model loaded on %s", device)

    def _ensure_model(self) -> None:
        """Thread-safe lazy model initialization."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                self._load_model()

    def detect(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLOv8 icon_detect on screenshot.

        Args:
            screenshot: BGR full-screen image as numpy array.

        Returns:
            List of candidate dicts matching the smart_detect format:
              - rect: {"x": int, "y": int, "w": int, "h": int}
              - type_guess: YOLOv8 class name
              - label_guess: "" (Florence-2 fills this later)
              - confidence: detection confidence 0.0-1.0
        """
        self._ensure_model()

        results = self._model.predict(
            screenshot,
            conf=self._confidence_threshold,
            verbose=False,
        )

        candidates: list[dict[str, Any]] = []
        if not results:
            return candidates

        result = results[0]
        boxes = result.boxes
        names = result.names

        for i in range(len(boxes.xyxy)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            # Filter by confidence (YOLO may return some below threshold)
            if conf < self._confidence_threshold:
                continue

            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)

            candidates.append({
                "rect": {"x": x, "y": y, "w": w, "h": h},
                "type_guess": names.get(cls_id, "unknown"),
                "label_guess": "",
                "confidence": conf,
            })

        logger.info("OmniParser detected %d UI elements", len(candidates))
        return candidates

    def detect_and_match(
        self,
        screenshot: np.ndarray,
        saved_snippet: np.ndarray,
        hint_x: int,
        hint_y: int,
        match_threshold: float = 0.7,
        search_radius: int = 400,
    ) -> LocateResult | None:
        """Detect all elements, CLIP-match saved snippet against crops.

        Pipeline:
        1. detect(screenshot) → all bboxes
        2. Filter to bboxes within search_radius of (hint_x, hint_y)
        3. Generate CLIP embedding of saved_snippet
        4. Generate CLIP embedding of each candidate crop
        5. Return best match above match_threshold as LocateResult

        Args:
            screenshot: BGR full-screen image.
            saved_snippet: BGR image of the element from recording.
            hint_x: Expected X position from recording.
            hint_y: Expected Y position from recording.
            match_threshold: Minimum CLIP cosine similarity.
            search_radius: Pixel radius around hint to search.

        Returns:
            LocateResult if match found, None otherwise.
        """
        import cv2

        candidates = self.detect(screenshot)
        if not candidates:
            return None

        # Filter by search radius
        nearby: list[dict[str, Any]] = []
        for c in candidates:
            r = c["rect"]
            cx = r["x"] + r["w"] // 2
            cy = r["y"] + r["h"] // 2
            dist = ((cx - hint_x) ** 2 + (cy - hint_y) ** 2) ** 0.5
            if dist <= search_radius:
                nearby.append(c)

        if not nearby:
            logger.debug("No detections within %dpx of hint", search_radius)
            return None

        # CLIP embedding of the saved snippet
        snippet_rgb = cv2.cvtColor(saved_snippet, cv2.COLOR_BGR2RGB)
        snippet_emb = generate_embedding(snippet_rgb)

        # Compare against each candidate crop
        best_score = -1.0
        best_candidate = None

        sh, sw = screenshot.shape[:2]
        for c in nearby:
            r = c["rect"]
            x1 = max(0, r["x"])
            y1 = max(0, r["y"])
            x2 = min(sw, r["x"] + r["w"])
            y2 = min(sh, r["y"] + r["h"])

            crop = screenshot[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_emb = generate_embedding(crop_rgb)

            # Cosine similarity (both L2-normalized → dot product)
            score = float(np.dot(snippet_emb, crop_emb.T).item())

            if score > best_score:
                best_score = score
                best_candidate = c

        if best_candidate is None or best_score < match_threshold:
            logger.debug(
                "Best CLIP score %.3f below threshold %.3f",
                best_score, match_threshold,
            )
            return None

        r = best_candidate["rect"]
        center = Point(r["x"] + r["w"] // 2, r["y"] + r["h"] // 2)
        rect = Rect(r["x"], r["y"], r["w"], r["h"])

        logger.info(
            "OmniParser matched at (%d, %d) CLIP=%.3f",
            center.x, center.y, best_score,
        )

        return LocateResult(
            point=center,
            method="omniparser",
            confidence=best_score,
            rect=rect,
        )
