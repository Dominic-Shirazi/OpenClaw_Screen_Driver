# Wave 1: YOLOE Detection Foundation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix "nothing highlighted" — users see YOLOE-detected UI elements on screen during recording, replacing VLM-based detection with fast, local YOLOE text-prompt detection.

**Architecture:** YOLOE text-prompt mode detects all UI elements in a single pass on the full screenshot. VLM is removed from the detection path entirely (it only runs on individual crops when the user clicks an element in workflow mode). OCR remains as the fallback when YOLOE returns zero detections. Text embeddings are cached at startup so subsequent `predict()` calls are fast.

**Tech Stack:** ultralytics (YOLOE), CLIP text embeddings (via ultralytics internals), PyQt6 overlay, pytest + mocked ultralytics for tests.

**Spec:** `docs/superpowers/specs/2026-03-13-detection-pipeline-redesign.md`

**YOLOE Best Practices (from research):**
- Use `conf=0.01` — open-vocabulary YOLOE produces valid detections at very low confidence
- Add null/disambiguation classes ("text label", "image") to reduce false positives
- Use short concrete nouns for class names — no spatial/relational terms
- Filter out bboxes covering >40% of image area (YOLOE sometimes detects "groups")
- Call `set_classes()` + `get_text_pe()` once, reuse for all predictions
- Per-class confidence thresholds in post-processing, not a single global threshold

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `core/yoloe.py` | Modify | Add `detect_all_elements()`, text-prompt caching, class maps |
| `recorder/smart_detect.py` | Rewrite | Rewire pipeline: YOLOE first, OCR fallback, VLM removed from detection |
| `recorder/overlay.py` | No change needed | Already renders candidates correctly (w>0, h>0 filter is correct) |
| `main.py` | Modify | Update `_trigger_smart_detect` to pass VLM-on-click flag |
| `tests/test_yoloe_text_prompt.py` | Create | Tests for text-prompt detection, embedding cache, class mapping |
| `tests/test_smart_detect.py` | Modify | Update tests for new YOLOE-first pipeline |
| `scripts/benchmark_yoloe_ui.py` | Create | Go/no-go benchmark script |
| `assets/screenshots/` | Create | Directory for benchmark screenshots |

---

## Chunk 1: YOLOE Text-Prompt Benchmark (Go/No-Go Gate)

### Task 1: Benchmark Script

**Files:**
- Create: `scripts/benchmark_yoloe_ui.py`
- Create: `assets/screenshots/` (directory for test screenshots)

The benchmark is the first deliverable. It tests whether YOLOE text-prompt mode can detect UI elements on real screenshots before any pipeline changes are made.

- [ ] **Step 1: Create benchmark script**

```python
"""YOLOE text-prompt benchmark for UI element detection.

Go/no-go gate: If YOLOE cannot detect >50% of obvious UI elements
on 3-5 real screenshots, the text-prompt approach is not viable and
we fall back to the spec's Fallback A/B/C plans.

Usage:
    python scripts/benchmark_yoloe_ui.py [--screenshots-dir assets/screenshots]

Place 3-5 PNG screenshots in assets/screenshots/ before running.
Screenshots should include: Windows desktop, browser page, form/dialog.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Element classes to benchmark — short concrete nouns per YOLOE best practices
ELEMENT_CLASSES = [
    "button", "icon", "text field", "checkbox", "dropdown",
    "scrollbar", "tab", "link", "toggle", "slider", "menu item",
    # Null/disambiguation classes (reduce false positives)
    "text label", "image",
]

# Per-class minimum confidence thresholds for post-filtering
# UI-novel classes need lower thresholds than COCO-adjacent ones
CLASS_CONF_THRESHOLDS = {
    "button": 0.03,
    "icon": 0.05,
    "text field": 0.02,
    "checkbox": 0.03,
    "dropdown": 0.02,
    "scrollbar": 0.05,
    "tab": 0.03,
    "link": 0.02,
    "toggle": 0.03,
    "slider": 0.05,
    "menu item": 0.02,
    "text label": 0.03,
    "image": 0.05,
}

# Maximum bbox area as fraction of image — filter out "group" detections
MAX_BBOX_AREA_FRACTION = 0.40


def load_model(model_path: str = "yoloe-26s-seg.pt"):
    """Load YOLOE model and cache text embeddings."""
    from ultralytics import YOLOE

    logger.info("Loading YOLOE model: %s", model_path)
    model = YOLOE(model_path)

    logger.info("Computing text embeddings for %d classes...", len(ELEMENT_CLASSES))
    t0 = time.perf_counter()
    tpe = model.model.get_text_pe(ELEMENT_CLASSES)
    model.model.set_classes(ELEMENT_CLASSES, tpe)
    embed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Text embeddings cached in %.0f ms", embed_ms)

    return model


def run_detection(model, image: np.ndarray, image_name: str) -> list[dict]:
    """Run YOLOE text-prompt detection on a single image."""
    h, w = image.shape[:2]
    image_area = h * w

    t0 = time.perf_counter()
    results = model.predict(image, conf=0.01, iou=0.4, verbose=False)
    detect_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = ELEMENT_CLASSES[cls_id] if cls_id < len(ELEMENT_CLASSES) else "unknown"
            conf = float(boxes.conf[i])

            # Per-class confidence filtering
            min_conf = CLASS_CONF_THRESHOLDS.get(cls_name, 0.03)
            if conf < min_conf:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            bx1, by1, bx2, by2 = xyxy
            bw, bh = bx2 - bx1, by2 - by1

            # Filter out oversized detections
            if (bw * bh) / image_area > MAX_BBOX_AREA_FRACTION:
                continue

            # Filter out tiny detections (noise)
            if bw < 5 or bh < 5:
                continue

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 4),
                "bbox": [int(bx1), int(by1), int(bw), int(bh)],
            })

    logger.info(
        "  %s: %d detections in %.0f ms",
        image_name, len(detections), detect_ms,
    )

    # Print per-class breakdown
    class_counts: dict[str, int] = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        logger.info("    %-15s %d", cls, count)

    return detections


def draw_detections(image: np.ndarray, detections: list[dict], output_path: Path) -> None:
    """Draw detection boxes on image and save for visual review."""
    vis = image.copy()
    colors = {
        "button": (0, 255, 0), "icon": (255, 165, 0), "text field": (255, 0, 0),
        "checkbox": (0, 255, 255), "dropdown": (255, 0, 255), "scrollbar": (128, 128, 0),
        "tab": (0, 128, 255), "link": (255, 255, 0), "toggle": (128, 0, 255),
        "slider": (0, 255, 128), "menu item": (255, 128, 0),
        "text label": (180, 180, 180), "image": (100, 100, 100),
    }
    for det in detections:
        x, y, w, h = det["bbox"]
        color = colors.get(det["class"], (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{det['class']} {det['confidence']:.0%}"
        cv2.putText(vis, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(output_path), vis)
    logger.info("  Visualization saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOE UI detection benchmark")
    parser.add_argument("--screenshots-dir", default="assets/screenshots",
                        help="Directory containing PNG screenshots to benchmark")
    parser.add_argument("--model", default="yoloe-26s-seg.pt",
                        help="YOLOE model weights path")
    parser.add_argument("--output-dir", default="assets/screenshots/benchmark_results",
                        help="Directory for visualization outputs")
    args = parser.parse_args()

    screenshots_dir = Path(args.screenshots_dir)
    if not screenshots_dir.exists():
        logger.error("Screenshots directory not found: %s", screenshots_dir)
        logger.error("Place 3-5 PNG screenshots in %s and re-run.", screenshots_dir)
        sys.exit(1)

    images = sorted(screenshots_dir.glob("*.png"))
    if len(images) < 3:
        logger.error("Need at least 3 screenshots, found %d", len(images))
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    logger.info("\n=== YOLOE UI Detection Benchmark ===\n")

    all_detections = {}
    total_elements = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read %s, skipping", img_path)
            continue

        detections = run_detection(model, image, img_path.name)
        all_detections[img_path.name] = detections
        total_elements += len(detections)

        # Save visualization
        vis_path = output_dir / f"bench_{img_path.name}"
        draw_detections(image, detections, vis_path)

    # Summary
    logger.info("\n=== BENCHMARK SUMMARY ===")
    logger.info("Screenshots tested: %d", len(all_detections))
    logger.info("Total detections:   %d", total_elements)
    avg = total_elements / max(1, len(all_detections))
    logger.info("Average per image:  %.1f", avg)

    # Save JSON report
    report = {
        "model": args.model,
        "classes": ELEMENT_CLASSES,
        "conf_thresholds": CLASS_CONF_THRESHOLDS,
        "results": {
            name: {"count": len(dets), "detections": dets}
            for name, dets in all_detections.items()
        },
        "summary": {
            "total_images": len(all_detections),
            "total_detections": total_elements,
            "avg_per_image": round(avg, 1),
        },
    }
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)

    # Go/no-go verdict
    logger.info("\n=== GO / NO-GO VERDICT ===")
    if avg >= 5:
        logger.info("PASS: YOLOE text-prompt finds %.1f elements/image on average.", avg)
        logger.info("Proceed with Wave 1 pipeline integration.")
    else:
        logger.warning("FAIL: YOLOE text-prompt finds only %.1f elements/image.", avg)
        logger.warning("Execute fallback plan (see spec: Fallback A/B/C).")
        sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create screenshots directory**

```bash
mkdir -p assets/screenshots
```

Add a `.gitkeep` file:
```bash
touch assets/screenshots/.gitkeep
```

- [ ] **Step 3: Capture 3-5 test screenshots**

Take PNG screenshots of real UI:
1. Windows desktop (taskbar, icons, system tray)
2. Browser with a web page (buttons, links, form fields)
3. A dialog/form (textboxes, dropdowns, checkboxes)
4. (Optional) IDE or complex app
5. (Optional) Settings page with toggles/sliders

Save as `assets/screenshots/01_desktop.png`, `02_browser.png`, etc.

- [ ] **Step 4: Run benchmark**

```bash
python scripts/benchmark_yoloe_ui.py
```

Expected output: detection counts per image, per-class breakdown, visualizations in `assets/screenshots/benchmark_results/`, and a GO/PASS or FAIL verdict.

**Decision point:**
- If PASS (avg >= 5 elements/image): proceed to Task 2
- If FAIL: review visualizations, tune class names and thresholds, re-run
- If still FAIL after tuning: execute Fallback A/B/C from spec

**Note:** The spec's go/no-go gate uses "< 50% recall on obvious elements" which requires
ground-truth annotations. This benchmark uses avg elements/image as a simpler proxy.
If YOLOE finds 5+ elements on a typical screenshot, it is detecting obvious UI elements.
Review the visualization PNGs to confirm the detections are real UI elements, not false positives.

**STOP GATE: Do NOT proceed to Task 2 if the benchmark fails.** If YOLOE text-prompt cannot
reliably detect UI elements after class name tuning, the entire Wave 1 approach must be
reconsidered. Execute the spec's Fallback A (YOLOE standard + VLM classification),
Fallback B (fine-tuned UI YOLO model), or Fallback C (Windows accessibility tree).

- [ ] **Step 5: Commit benchmark**

```bash
git add scripts/benchmark_yoloe_ui.py assets/screenshots/.gitkeep
git commit -m "feat(benchmark): YOLOE text-prompt UI detection go/no-go gate"
```

---

## Chunk 2: YOLOE Text-Prompt Detection API

### Task 2: Add text-prompt detection to yoloe.py

**Files:**
- Modify: `core/yoloe.py`
- Create: `tests/test_yoloe_text_prompt.py`

Add `detect_all_elements()` and text-embedding caching to the existing YOLOE module.

- [ ] **Step 1: Write failing tests for text-prompt detection**

Create `tests/test_yoloe_text_prompt.py`:

```python
"""Tests for YOLOE text-prompt detection (Wave 1).

Tests the detect_all_elements() function and text embedding caching.
All ultralytics internals are mocked — no GPU or model weights needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_screenshot() -> np.ndarray:
    """1920x1080 BGR image."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def mock_yoloe_model():
    """Mock YOLOE model with text-prompt support."""
    model = MagicMock()
    model.model = MagicMock()
    model.model.get_text_pe = MagicMock(return_value="fake_embeddings")
    model.model.set_classes = MagicMock()
    return model


def _fake_tensor(values):
    """Create a mock object that behaves like a torch tensor for testing."""
    t = MagicMock()
    t.cpu.return_value = t
    t.numpy.return_value = np.array(values, dtype=np.int32)
    t.__getitem__ = lambda self, idx: values[idx]
    return t


@pytest.fixture
def sample_boxes():
    """Mock ultralytics Boxes result with 3 detections."""
    boxes = MagicMock()
    boxes.__len__ = lambda self: 3
    boxes.conf = MagicMock()
    boxes.conf.__getitem__ = lambda self, i: [0.15, 0.08, 0.60][i]
    boxes.cls = MagicMock()
    boxes.cls.__getitem__ = lambda self, i: [0, 2, 0][i]  # button, text field, button
    boxes.xyxy = MagicMock()
    # button at (100,200)-(200,240), text field at (300,400)-(600,430), button at (0,0)-(1920,1080)
    boxes.xyxy.__getitem__ = lambda self, i: [
        _fake_tensor([100, 200, 200, 240]),
        _fake_tensor([300, 400, 600, 430]),
        _fake_tensor([0, 0, 1920, 1080]),  # oversized — should be filtered
    ][i]
    return boxes


# ---------------------------------------------------------------------------
# Tests: Embedding Cache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    """Tests for YOLOE text embedding initialization."""

    @patch("core.yoloe._load_model")
    def test_init_text_embeddings_calls_set_classes(self, mock_load, mock_yoloe_model):
        """init_text_embeddings caches embeddings via set_classes."""
        mock_load.return_value = mock_yoloe_model

        from core.yoloe import init_text_embeddings
        init_text_embeddings()

        mock_yoloe_model.model.get_text_pe.assert_called()
        mock_yoloe_model.model.set_classes.assert_called()

    @patch("core.yoloe._load_model")
    def test_init_text_embeddings_is_idempotent(self, mock_load, mock_yoloe_model):
        """Calling init_text_embeddings twice only computes embeddings once."""
        mock_load.return_value = mock_yoloe_model

        from core.yoloe import init_text_embeddings
        # Reset cache state
        import core.yoloe as yoloe_mod
        yoloe_mod._element_embeddings = None

        init_text_embeddings()
        init_text_embeddings()

        # get_text_pe called once for element classes
        assert mock_yoloe_model.model.get_text_pe.call_count == 1


# ---------------------------------------------------------------------------
# Tests: detect_all_elements
# ---------------------------------------------------------------------------

class TestDetectAllElements:
    """Tests for full-screen YOLOE text-prompt detection."""

    @patch("core.yoloe._load_model")
    def test_returns_candidate_dicts(self, mock_load, mock_yoloe_model, fake_screenshot, sample_boxes):
        """detect_all_elements returns list of candidate dicts."""
        mock_load.return_value = mock_yoloe_model

        result_obj = MagicMock()
        result_obj.boxes = sample_boxes
        mock_yoloe_model.predict.return_value = [result_obj]

        from core.yoloe import detect_all_elements, ELEMENT_CLASSES
        import core.yoloe as yoloe_mod
        yoloe_mod._element_embeddings = "cached"

        candidates = detect_all_elements(fake_screenshot)

        # Should filter out the oversized bbox (1920x1080 = 100% of image)
        # Should keep the two normal detections
        assert isinstance(candidates, list)
        for c in candidates:
            assert "rect" in c
            assert "type_guess" in c
            assert "confidence" in c
            assert c["rect"]["w"] > 0
            assert c["rect"]["h"] > 0

    @patch("core.yoloe._load_model")
    def test_returns_empty_on_no_detections(self, mock_load, mock_yoloe_model, fake_screenshot):
        """Returns empty list when YOLOE finds nothing."""
        mock_load.return_value = mock_yoloe_model

        result_obj = MagicMock()
        result_obj.boxes = None
        mock_yoloe_model.predict.return_value = [result_obj]

        from core.yoloe import detect_all_elements
        import core.yoloe as yoloe_mod
        yoloe_mod._element_embeddings = "cached"

        candidates = detect_all_elements(fake_screenshot)
        assert candidates == []

    @patch("core.yoloe._load_model")
    def test_filters_oversized_bboxes(self, mock_load, mock_yoloe_model, fake_screenshot):
        """Bboxes covering >40% of image are filtered out."""
        mock_load.return_value = mock_yoloe_model

        boxes = MagicMock()
        boxes.__len__ = lambda self: 1
        boxes.conf = MagicMock()
        boxes.conf.__getitem__ = lambda self, i: 0.9
        boxes.cls = MagicMock()
        boxes.cls.__getitem__ = lambda self, i: 0
        boxes.xyxy = MagicMock()
        boxes.xyxy.__getitem__ = lambda self, i: _fake_tensor([0, 0, 1920, 1080])

        result_obj = MagicMock()
        result_obj.boxes = boxes
        mock_yoloe_model.predict.return_value = [result_obj]

        from core.yoloe import detect_all_elements
        import core.yoloe as yoloe_mod
        yoloe_mod._element_embeddings = "cached"

        candidates = detect_all_elements(fake_screenshot)
        assert len(candidates) == 0

    @patch("core.yoloe._load_model")
    def test_exception_returns_empty(self, mock_load, mock_yoloe_model, fake_screenshot):
        """Prediction exceptions are caught and return empty list."""
        mock_load.return_value = mock_yoloe_model
        mock_yoloe_model.predict.side_effect = RuntimeError("CUDA OOM")

        from core.yoloe import detect_all_elements
        import core.yoloe as yoloe_mod
        yoloe_mod._element_embeddings = "cached"

        candidates = detect_all_elements(fake_screenshot)
        assert candidates == []


# ---------------------------------------------------------------------------
# Tests: Class mapping
# ---------------------------------------------------------------------------

class TestClassMapping:
    """Tests for YOLOE class name → ElementType mapping."""

    def test_yoloe_to_element_type_mapping_complete(self):
        """Every YOLOE class has a mapping to an ElementType value."""
        from core.yoloe import ELEMENT_CLASSES, YOLOE_TO_ELEMENT_TYPE

        for cls in ELEMENT_CLASSES:
            assert cls in YOLOE_TO_ELEMENT_TYPE, f"Missing mapping for '{cls}'"

    def test_null_classes_map_to_non_interactive(self):
        """Null classes map to static/non-interactive ElementType values."""
        from core.yoloe import YOLOE_TO_ELEMENT_TYPE

        assert YOLOE_TO_ELEMENT_TYPE["text label"] == "read_here"
        assert YOLOE_TO_ELEMENT_TYPE["image"] == "image"


# ---------------------------------------------------------------------------
# Tests: _detect_via_yoloe wrapper (error handling)
# ---------------------------------------------------------------------------

class TestDetectViaYOLOE:
    """Tests for the YOLOE detection wrapper in smart_detect."""

    def test_import_error_returns_empty(self, fake_screenshot):
        """ImportError from core.yoloe returns empty list."""
        from recorder.smart_detect import _detect_via_yoloe

        with patch.dict("sys.modules", {"core.yoloe": None}):
            result = _detect_via_yoloe(fake_screenshot)
            assert result == []

    def test_exception_returns_empty(self, fake_screenshot):
        """Runtime exception from YOLOE returns empty list."""
        from recorder.smart_detect import _detect_via_yoloe

        mock_module = MagicMock()
        mock_module.detect_all_elements.side_effect = RuntimeError("CUDA OOM")
        with patch.dict("sys.modules", {"core.yoloe": mock_module}):
            result = _detect_via_yoloe(fake_screenshot)
            assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /c/Users/persi/Documents/OpenClaw_Screen_Driver
.venv/Scripts/python -m pytest tests/test_yoloe_text_prompt.py -v
```

Expected: ImportError or AttributeError — `detect_all_elements`, `init_text_embeddings`, `ELEMENT_CLASSES`, `YOLOE_TO_ELEMENT_TYPE` don't exist yet.

- [ ] **Step 3: Implement text-prompt detection in yoloe.py**

Add the following to `core/yoloe.py` after the existing imports and before `_load_model()`:

```python
# ---------------------------------------------------------------------------
# Text-prompt class definitions (Wave 1)
# ---------------------------------------------------------------------------

# Short concrete nouns — aligned with CLIP/LVIS vocabulary for best accuracy.
# Null classes (text_label, image) reduce false positives on similar-looking elements.
ELEMENT_CLASSES: list[str] = [
    "button", "icon", "text field", "checkbox", "dropdown",
    "scrollbar", "tab", "link", "toggle", "slider", "menu item",
    # Null / disambiguation classes
    "text label", "image",
]

# Per-class minimum confidence thresholds.
# Open-vocabulary YOLOE produces valid detections at very low confidence.
# UI-novel classes need lower thresholds than COCO-adjacent ones.
CLASS_CONF_THRESHOLDS: dict[str, float] = {
    "button": 0.03,
    "icon": 0.05,
    "text field": 0.02,
    "checkbox": 0.03,
    "dropdown": 0.02,
    "scrollbar": 0.05,
    "tab": 0.03,
    "link": 0.02,
    "toggle": 0.03,
    "slider": 0.05,
    "menu item": 0.02,
    "text label": 0.03,
    "image": 0.05,
}

# YOLOE class name → ElementType enum value string.
# Null classes map to non-interactive types.
YOLOE_TO_ELEMENT_TYPE: dict[str, str] = {
    "button": "button",
    "icon": "icon",
    "text field": "textbox",
    "checkbox": "toggle",
    "dropdown": "dropdown",
    "scrollbar": "scrollbar",
    "tab": "tab",
    "link": "link",
    "toggle": "toggle",
    "slider": "scrollbar",
    "menu item": "button",
    "text label": "read_here",
    "image": "image",
}

# Maximum bbox area as fraction of image — filter out "group" detections
MAX_BBOX_AREA_FRACTION = 0.40

# Minimum bbox dimensions in pixels — filter noise
MIN_BBOX_W = 5
MIN_BBOX_H = 5

# Cached text embeddings (computed once, reused for all predictions)
_element_embeddings: Any | None = None
```

Then add these functions after the `_load_model()` function:

```python
# ---------------------------------------------------------------------------
# Text-prompt embedding cache
# ---------------------------------------------------------------------------

def init_text_embeddings() -> None:
    """Pre-compute and cache CLIP text embeddings for element classes.

    Call this once at app startup. After this, YOLOE can run
    text-prompt detection without loading CLIP on every predict() call.

    Safe to call multiple times — embeddings are computed only once.
    """
    global _element_embeddings

    if _element_embeddings is not None:
        return

    model = _load_model()

    try:
        logger.info("Computing YOLOE text embeddings for %d element classes...", len(ELEMENT_CLASSES))
        _element_embeddings = model.model.get_text_pe(ELEMENT_CLASSES)
        model.model.set_classes(ELEMENT_CLASSES, _element_embeddings)
        logger.info("YOLOE text embeddings cached successfully")
    except Exception as e:
        logger.warning("Failed to cache text embeddings: %s", e)
        _element_embeddings = None


def _ensure_text_embeddings() -> bool:
    """Ensures text embeddings are loaded. Returns True if ready."""
    if _element_embeddings is None:
        init_text_embeddings()
    return _element_embeddings is not None


# ---------------------------------------------------------------------------
# Text-prompt detection (Wave 1 core feature)
# ---------------------------------------------------------------------------

def detect_all_elements(
    screen: np.ndarray,
    *,
    conf: float = 0.01,
    iou: float = 0.4,
) -> list[dict[str, Any]]:
    """Detect all UI elements on screen using YOLOE text-prompt mode.

    Uses pre-cached CLIP text embeddings to find buttons, icons,
    textboxes, checkboxes, dropdowns, and other interactive UI elements
    in a single inference pass.

    Args:
        screen: BGR full-screen capture as numpy array.
        conf: Raw confidence threshold for YOLOE predict(). Use low
              values (0.01) — per-class filtering happens in post-processing.
        iou: IoU threshold for NMS.

    Returns:
        List of candidate dicts compatible with OverlayController.set_candidates():
        - rect: {x, y, w, h} in pixels
        - type_guess: ElementType string value
        - label_guess: YOLOE class name (VLM provides real labels later)
        - confidence: 0.0-1.0
        - ocr_text: None (OCR not run here)
    """
    if not _ensure_text_embeddings():
        logger.warning("Text embeddings not available, cannot run text-prompt detection")
        return []

    model = _load_model()
    sh, sw = screen.shape[:2]
    image_area = sh * sw

    try:
        results = model.predict(screen, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        logger.warning("YOLOE text-prompt detection failed: %s", e)
        return []

    candidates: list[dict[str, Any]] = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            if cls_id >= len(ELEMENT_CLASSES):
                continue

            cls_name = ELEMENT_CLASSES[cls_id]
            conf_val = float(boxes.conf[i])

            # Per-class confidence filtering
            min_conf = CLASS_CONF_THRESHOLDS.get(cls_name, 0.03)
            if conf_val < min_conf:
                continue

            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            bx1, by1, bx2, by2 = xyxy
            bw = bx2 - bx1
            bh = by2 - by1

            # Filter oversized detections (YOLOE sometimes detects "groups")
            if image_area > 0 and (bw * bh) / image_area > MAX_BBOX_AREA_FRACTION:
                continue

            # Filter tiny detections (noise)
            if bw < MIN_BBOX_W or bh < MIN_BBOX_H:
                continue

            element_type = YOLOE_TO_ELEMENT_TYPE.get(cls_name, "unknown")

            candidates.append({
                "rect": {"x": int(bx1), "y": int(by1), "w": int(bw), "h": int(bh)},
                "type_guess": element_type,
                "label_guess": cls_name,
                "confidence": round(conf_val, 4),
                "ocr_text": None,
            })

    # Sort by confidence descending
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    logger.info("YOLOE text-prompt detected %d UI elements", len(candidates))
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python -m pytest tests/test_yoloe_text_prompt.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/yoloe.py tests/test_yoloe_text_prompt.py
git commit -m "feat(yoloe): add text-prompt detection with embedding cache"
```

---

## Chunk 3: Rewire Smart Detection Pipeline

### Task 3: Update smart_detect.py to use YOLOE first

**Files:**
- Modify: `recorder/smart_detect.py`
- Modify: `tests/test_smart_detect.py`

Replace the VLM-first pipeline with YOLOE-first. VLM is removed from the detection path entirely. OCR remains as the fallback when YOLOE returns zero elements.

- [ ] **Step 1: Write failing tests for new pipeline**

Add to `tests/test_smart_detect.py` (new test class, keep existing tests for backward compat reference):

```python
class TestYOLOEFirstPipeline:
    """Tests for the YOLOE-first detection pipeline (Wave 1)."""

    @patch("recorder.smart_detect._detect_via_yoloe")
    def test_yoloe_returns_results(
        self, mock_yoloe: MagicMock, fake_screenshot: np.ndarray
    ) -> None:
        """When YOLOE succeeds, its results are returned directly."""
        mock_yoloe.return_value = [
            {
                "rect": {"x": 10, "y": 20, "w": 80, "h": 30},
                "type_guess": "button",
                "label_guess": "button",
                "confidence": 0.15,
                "ocr_text": None,
            },
        ]

        results = detect_ui_elements(fake_screenshot)

        assert len(results) == 1
        assert results[0]["type_guess"] == "button"
        mock_yoloe.assert_called_once()

    @patch("recorder.smart_detect._detect_via_ocr")
    @patch("recorder.smart_detect._detect_via_yoloe")
    def test_yoloe_empty_falls_back_to_ocr(
        self,
        mock_yoloe: MagicMock,
        mock_ocr: MagicMock,
        fake_screenshot: np.ndarray,
    ) -> None:
        """When YOLOE returns empty, OCR fallback is used."""
        mock_yoloe.return_value = []
        mock_ocr.return_value = [
            {
                "rect": {"x": 5, "y": 5, "w": 40, "h": 15},
                "type_guess": "unknown",
                "label_guess": "Login",
                "confidence": 0.7,
                "ocr_text": "Login",
            }
        ]

        results = detect_ui_elements(fake_screenshot)

        assert len(results) == 1
        assert results[0]["ocr_text"] == "Login"
        mock_yoloe.assert_called_once()
        mock_ocr.assert_called_once()

    @patch("recorder.smart_detect._detect_via_ocr")
    @patch("recorder.smart_detect._detect_via_yoloe")
    def test_yoloe_disabled_uses_ocr(
        self,
        mock_yoloe: MagicMock,
        mock_ocr: MagicMock,
        fake_screenshot: np.ndarray,
    ) -> None:
        """When use_yoloe=False, falls through to OCR."""
        mock_ocr.return_value = [
            {
                "rect": {"x": 0, "y": 0, "w": 50, "h": 20},
                "type_guess": "unknown",
                "label_guess": "OK",
                "confidence": 0.8,
                "ocr_text": "OK",
            }
        ]

        results = detect_ui_elements(fake_screenshot, use_yoloe=False)

        assert len(results) == 1
        mock_yoloe.assert_not_called()
        mock_ocr.assert_called_once()

    @patch("recorder.smart_detect._detect_via_yoloe")
    def test_vlm_flag_has_no_effect_on_detection(
        self, mock_yoloe: MagicMock, fake_screenshot: np.ndarray
    ) -> None:
        """use_vlm flag no longer affects detection (VLM removed from pipeline)."""
        mock_yoloe.return_value = [
            {
                "rect": {"x": 10, "y": 20, "w": 80, "h": 30},
                "type_guess": "button",
                "label_guess": "button",
                "confidence": 0.15,
                "ocr_text": None,
            },
        ]

        results_vlm_on = detect_ui_elements(fake_screenshot, use_vlm=True)
        results_vlm_off = detect_ui_elements(fake_screenshot, use_vlm=False)

        # Both should use YOLOE, VLM flag is kept for API compat but ignored
        assert len(results_vlm_on) == 1
        assert len(results_vlm_off) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python -m pytest tests/test_smart_detect.py::TestYOLOEFirstPipeline -v
```

Expected: FAIL — `_detect_via_yoloe` doesn't exist yet.

- [ ] **Step 3: Rewrite smart_detect.py**

Replace the full content of `recorder/smart_detect.py`:

```python
"""Smart element detection for recording sessions.

During --record mode, automatically detects UI elements on screen using
YOLOE text-prompt mode (fast, local object detection). VLM is NOT used
for detection — only for labeling individual crops on user click.

Detection pipeline (Wave 1):
1. Capture screenshot
2. Run YOLOE text-prompt detection → bounding boxes + element types
3. If YOLOE returns 0 results, fall back to OCR
4. Return candidates for OverlayController.set_candidates()

All AI calls are guarded with try/except so recording works even without
GPU libs installed — candidates will just be empty.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def detect_ui_elements(
    screenshot: np.ndarray,
    *,
    use_vlm: bool = True,
    use_yoloe: bool = True,
) -> list[dict[str, Any]]:
    """Detects UI elements on a screenshot using YOLOE text-prompt mode.

    Pipeline: YOLOE text-prompt (primary) → OCR (fallback).
    VLM is NOT used for detection. The use_vlm parameter is kept for
    API compatibility but has no effect on the detection pipeline.

    Args:
        screenshot: BGR full-screen image as numpy array.
        use_vlm: Kept for API compatibility. Has no effect (VLM removed
                 from detection pipeline in Wave 1).
        use_yoloe: Whether to use YOLOE for detection. If False,
                   falls through directly to OCR.

    Returns:
        List of candidate dicts with keys:
        - rect: {x, y, w, h} in pixels
        - type_guess: element type string
        - label_guess: human-readable label
        - confidence: 0.0-1.0
        - ocr_text: visible text (if detected via OCR)
    """
    candidates: list[dict[str, Any]] = []

    # Primary: YOLOE text-prompt detection (fast, local)
    if use_yoloe:
        yoloe_candidates = _detect_via_yoloe(screenshot)
        if yoloe_candidates:
            return yoloe_candidates

    # Fallback: OCR-based detection for text elements
    ocr_candidates = _detect_via_ocr(screenshot)
    if ocr_candidates:
        candidates.extend(ocr_candidates)

    return candidates


def detect_ui_elements_async(
    screenshot: np.ndarray,
    callback: Any,
    *,
    use_vlm: bool = True,
    use_yoloe: bool = True,
) -> threading.Thread:
    """Runs detection in a background thread and calls callback with results.

    Args:
        screenshot: BGR screenshot to analyze.
        callback: Callable that receives list[dict] of candidates.
                  Called on the background thread — use QTimer.singleShot
                  to marshal to Qt main thread if needed.
        use_vlm: Kept for API compatibility. Has no effect.
        use_yoloe: Whether to use YOLOE for detection.

    Returns:
        The started Thread object (for joining if needed).
    """
    def _worker() -> None:
        try:
            results = detect_ui_elements(
                screenshot, use_vlm=use_vlm, use_yoloe=use_yoloe,
            )
            callback(results)
        except Exception as e:
            logger.error("Async detection failed: %s", e)
            callback([])

    thread = threading.Thread(target=_worker, daemon=True, name="smart-detect")
    thread.start()
    return thread


def _detect_via_yoloe(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses YOLOE text-prompt mode to detect all UI elements.

    Calls detect_all_elements() from core.yoloe which uses pre-cached
    CLIP text embeddings for fast inference.

    Args:
        screenshot: BGR full-screen image.

    Returns:
        List of candidate dicts, or empty list on failure.
    """
    try:
        from core.yoloe import detect_all_elements

        candidates = detect_all_elements(screenshot)
        logger.info("YOLOE detected %d UI elements", len(candidates))
        return candidates
    except ImportError:
        logger.debug("YOLOE module not available for smart detection")
        return []
    except Exception as e:
        logger.warning("YOLOE detection failed: %s", e)
        return []


def _detect_via_ocr(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses OCR to find text regions as potential UI elements.

    This is a lightweight fallback when YOLOE is unavailable or returns
    zero results. It finds text on screen and creates candidate boxes
    for each text region.

    Args:
        screenshot: BGR full-screen image.

    Returns:
        List of candidate dicts, or empty list on failure.
    """
    try:
        import cv2
        import pytesseract

        # Convert to grayscale for OCR
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Get bounding boxes for all detected text
        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT
        )

        candidates: list[dict[str, Any]] = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Skip empty or low-confidence text
            if not text or conf < 40:
                continue

            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            # Skip tiny boxes (noise)
            if w < 10 or h < 8:
                continue

            candidates.append({
                "rect": {"x": x, "y": y, "w": w, "h": h},
                "type_guess": "unknown",
                "label_guess": text[:30],
                "confidence": conf / 100.0,
                "ocr_text": text,
            })

        logger.info("OCR detected %d text regions", len(candidates))
        return candidates
    except ImportError:
        logger.debug("pytesseract not available for OCR detection")
        return []
    except Exception as e:
        logger.warning("OCR detection failed: %s", e)
        return []
```

- [ ] **Step 4: Run all smart_detect tests**

```bash
.venv/Scripts/python -m pytest tests/test_smart_detect.py -v
```

Expected: New `TestYOLOEFirstPipeline` tests PASS. Some old `TestDetectUIElements` tests may need updating since `_detect_via_vlm` no longer exists — update them to use `_detect_via_yoloe` mock instead.

- [ ] **Step 5: Replace old tests to match new pipeline**

The rewrite removes `_detect_via_vlm`, so existing tests that reference it will break.

**Delete these test classes** (they test the old VLM-first pipeline):
- `TestDetectUIElements` — replaced by `TestYOLOEFirstPipeline` (added in Step 1)
- `TestDetectViaVLM` — replaced by `TestDetectViaYOLOE` (added in test_yoloe_text_prompt.py)

**Keep these test classes** (they test code that still exists):
- `TestDetectUIElementsAsync` — still valid, just update the mock:
  - `@patch("recorder.smart_detect.detect_ui_elements")` → no change needed (function name preserved)
- `TestDetectViaOCR` — still valid, `_detect_via_ocr` is unchanged

- [ ] **Step 6: Run full test suite**

```bash
.venv/Scripts/python -m pytest tests/test_smart_detect.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add recorder/smart_detect.py tests/test_smart_detect.py
git commit -m "feat(smart_detect): rewire pipeline — YOLOE first, OCR fallback, VLM removed"
```

---

## Chunk 4: Integration — Main.py and Startup

### Task 4: Wire up text embedding init and update detection trigger

**Files:**
- Modify: `main.py`

Two changes:
1. Call `init_text_embeddings()` at startup (after model load, before recording)
2. `_trigger_smart_detect` already calls `detect_ui_elements_async` which now uses YOLOE

- [ ] **Step 1: Add embedding init to cmd_record()**

In `main.py`, inside `cmd_record()` (around line 215-220), add text embedding initialization before the overlay starts:

```python
    # Initialize YOLOE text embeddings for fast detection during recording
    try:
        from core.yoloe import init_text_embeddings
        init_text_embeddings()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("YOLOE text embeddings not available: %s", e)
```

Add this **before** the `overlay = OverlayController(...)` line. This ensures embeddings are cached before the first detection trigger.

- [ ] **Step 2: Verify _trigger_smart_detect needs no changes**

Read `main.py:187-212`. The function calls `detect_ui_elements_async(screenshot, _on_results)` which now routes through the new YOLOE-first pipeline automatically. No changes needed.

- [ ] **Step 3: Verify overlay rendering needs no changes**

Read `recorder/overlay.py:628-670`. The `render_candidates()` method draws boxes for any candidate with `w > 0` and `h > 0`. YOLOE text-prompt produces proper bboxes (unlike VLM which returned bad coordinates), so candidates will now actually render on screen. No changes needed.

- [ ] **Step 4: Manual smoke test**

```bash
.venv/Scripts/python main.py --record test_wave1
```

1. Overlay appears in PASSTHROUGH mode (green border)
2. Press Ctrl+R → switches to RECORD mode (red border)
3. YOLOE detection runs in background
4. Bounding boxes appear on screen for detected UI elements
5. Click an element → TagDialog opens (pre-populated with YOLOE type_guess)
6. Press Ctrl+Q → save

If no boxes appear: check logs for YOLOE errors, verify GPU/CUDA is available, check `assets/screenshots/benchmark_results/` for reference.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(main): init YOLOE text embeddings at recording startup"
```

---

### Task 5: VLM labeling on click (workflow mode prep)

**Files:**
- Modify: `main.py`

In workflow mode, when the user clicks an element, crop the bbox + 30% buffer and send to VLM `analyze_crop()` for labeling. This pre-populates the TagDialog with a real label instead of just the YOLOE class name.

- [ ] **Step 1: Update on_element_clicked to call VLM on crop**

In `main.py`, find the `on_element_clicked` callback inside `cmd_record()`. After YOLOE refinement and before TagDialog, add VLM crop labeling:

```python
        # VLM labeling: crop element + 30% buffer, ask VLM for label
        vlm_label = ""
        vlm_type = ""
        if matched_candidate and matched_candidate.get("rect"):
            try:
                from core.vision import analyze_crop_array
                from core.capture import screenshot_full

                screen = screenshot_full()
                r = matched_candidate["rect"]
                buf_x = int(r["w"] * 0.30)
                buf_y = int(r["h"] * 0.30)
                sh, sw = screen.shape[:2]
                crop_x1 = max(0, r["x"] - buf_x)
                crop_y1 = max(0, r["y"] - buf_y)
                crop_x2 = min(sw, r["x"] + r["w"] + buf_x)
                crop_y2 = min(sh, r["y"] + r["h"] + buf_y)
                crop = screen[crop_y1:crop_y2, crop_x1:crop_x2]

                if crop.size > 0:
                    # analyze_crop_array returns a dict with keys:
                    # element_type, label_guess, confidence, ocr_text
                    vision_result = analyze_crop_array(
                        crop, "Identify this UI element type and label"
                    )
                    if vision_result:
                        vlm_label = vision_result.get("label_guess", "")
                        vlm_type = vision_result.get("element_type", "")
                        logger.info("VLM labeled element: %s (%s)", vlm_label, vlm_type)
            except ImportError:
                logger.debug("VLM module not available for labeling")
            except RuntimeError as e:
                logger.debug("VLM labeling failed: %s", e)
```

Then use `vlm_label` as `label_guess` in the TagDialog if available, falling back to the YOLOE class name.

- [ ] **Step 2: Write test for VLM crop labeling**

Add to `tests/test_yoloe_text_prompt.py` (or create `tests/test_vlm_on_click.py`):

```python
class TestVLMCropLabeling:
    """Tests for VLM labeling of YOLOE-detected element crops."""

    def test_crop_buffer_calculation(self):
        """Crop buffer is 30% of element dimensions."""
        rect = {"x": 100, "y": 200, "w": 80, "h": 30}
        buf_x = int(rect["w"] * 0.30)
        buf_y = int(rect["h"] * 0.30)

        assert buf_x == 24  # 80 * 0.30
        assert buf_y == 9   # 30 * 0.30

        # Crop region (clamped to screen)
        sw, sh = 1920, 1080
        crop_x1 = max(0, rect["x"] - buf_x)
        crop_y1 = max(0, rect["y"] - buf_y)
        crop_x2 = min(sw, rect["x"] + rect["w"] + buf_x)
        crop_y2 = min(sh, rect["y"] + rect["h"] + buf_y)

        assert crop_x1 == 76
        assert crop_y1 == 191
        assert crop_x2 == 204
        assert crop_y2 == 239

    @patch("core.vision.analyze_crop_array")
    def test_vlm_result_is_dict_access(self, mock_analyze):
        """VLM results are accessed as dict, not attributes."""
        mock_analyze.return_value = {
            "element_type": "button",
            "label_guess": "Submit Order",
            "confidence": 0.9,
            "ocr_text": "Submit",
        }

        result = mock_analyze(MagicMock(), "test prompt")
        assert result.get("label_guess") == "Submit Order"
        assert result.get("element_type") == "button"
```

- [ ] **Step 3: Run test**

```bash
.venv/Scripts/python -m pytest tests/test_yoloe_text_prompt.py::TestVLMCropLabeling -v
```

Expected: PASS.

- [ ] **Step 4: Manual smoke test**

Record a workflow. Click an element. Verify:
1. YOLOE bbox highlights appear first (fast)
2. After clicking, VLM labels the crop (may take 1-3s)
3. TagDialog opens with VLM label (or YOLOE class name if VLM unavailable)

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_yoloe_text_prompt.py
git commit -m "feat(main): VLM labels element crops on click in workflow mode"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite**

```bash
.venv/Scripts/python -m pytest tests/ -v --tb=short
```

Expected: All tests PASS. No regressions.

- [ ] **Step 2: Verify no import cycles**

```bash
.venv/Scripts/python -c "from core.yoloe import detect_all_elements, init_text_embeddings; print('OK')"
.venv/Scripts/python -c "from recorder.smart_detect import detect_ui_elements; print('OK')"
```

- [ ] **Step 3: Performance check (if GPU available)**

```bash
.venv/Scripts/python -c "
import time, numpy as np
from core.yoloe import init_text_embeddings, detect_all_elements

screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

t0 = time.perf_counter()
init_text_embeddings()
print(f'Embedding init: {(time.perf_counter()-t0)*1000:.0f} ms')

t0 = time.perf_counter()
result = detect_all_elements(screen)
print(f'Detection: {(time.perf_counter()-t0)*1000:.0f} ms, {len(result)} elements')
"
```

Target: embedding init < 2s, detection < 500ms on GPU.

- [ ] **Step 4: Commit any final fixes**

```bash
git add -u
git commit -m "fix(wave1): final adjustments from verification"
```

---

## Summary of Changes

| File | What Changed |
|------|-------------|
| `core/yoloe.py` | Added `ELEMENT_CLASSES`, `YOLOE_TO_ELEMENT_TYPE`, `CLASS_CONF_THRESHOLDS`, `init_text_embeddings()`, `detect_all_elements()` |
| `recorder/smart_detect.py` | Replaced VLM-first with YOLOE-first pipeline. Added `_detect_via_yoloe()`. Removed `_detect_via_vlm()`. |
| `main.py` | Added `init_text_embeddings()` call at recording startup. Added VLM crop labeling on element click. |
| `tests/test_yoloe_text_prompt.py` | New test file for text-prompt detection, embedding cache, class mapping |
| `tests/test_smart_detect.py` | Updated tests for YOLOE-first pipeline |
| `scripts/benchmark_yoloe_ui.py` | New benchmark script for go/no-go gate |

## What's NOT in Wave 1

- No area hierarchy (Wave 2)
- No batch VLM labeling (Wave 2)
- No action type dropdown in TagDialog (Wave 2)
- No annotation reuse (Wave 2)
- No area-scoped replay cascade (Wave 3)
- No global_areas.json (Wave 3)
- TagDialog unchanged — no new fields
- Overlay rendering unchanged — same color scheme and layout
