# OmniParser + Florence-2 Integration Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace YOLOE detection with OmniParser YOLOv8 (UI-trained) + Florence-2 captioning, preserving the existing candidate dict contract and Qwen3-VL deep reasoning.

**Architecture:** DetectionProvider protocol abstracts detection backends. OmniParser is the first provider. Florence-2 handles fast labeling. All existing YOLOE imports become dormant (code stays, imports removed from active paths).

**Tech Stack:** ultralytics (YOLOv8), transformers (Florence-2), huggingface-hub (weight download), existing CLIP/FAISS for replay matching.

**Spec:** `docs/superpowers/specs/2026-03-13-omniparser-integration-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `core/model_cache.py` | CREATE | HuggingFace weight download + cache utility |
| `core/detection.py` | CREATE | DetectionProvider protocol + get_detector() factory |
| `core/omniparser.py` | CREATE | OmniParser YOLOv8 icon_detect wrapper |
| `core/florence.py` | CREATE | Florence-2 captioning (caption_crop, caption_batch, describe_element) |
| `tests/test_model_cache.py` | CREATE | Tests for model_cache |
| `tests/test_detection_provider.py` | CREATE | Tests for protocol + factory |
| `tests/test_omniparser.py` | CREATE | Tests for OmniParserProvider |
| `tests/test_florence.py` | CREATE | Tests for Florence-2 module |
| `core/config.py` | MODIFY | Update `_DEFAULTS` for new config keys |
| `config.yaml` | MODIFY | Add OmniParser/Florence-2 config, remove YOLOE keys |
| `pyproject.toml` | MODIFY | Add omniparser, florence, model-cache, detection dep groups |
| `tests/conftest.py` | MODIFY | Add huggingface_hub mock |
| `recorder/smart_detect.py` | MODIFY | Replace YOLOE with OmniParser + Florence-2 |
| `tests/test_smart_detect.py` | MODIFY | Update mocks for new pipeline |
| `mapper/runner.py` | MODIFY | Replace YOLOE Stage 1 with OmniParser detect_and_match |
| `tests/test_cascade.py` | MODIFY | Update Stage 1 mocks |
| `main.py` | MODIFY | Replace 3 YOLOE touchpoints |

---

## Chunk 1: Foundation Modules

### Task 1: Model Cache Module

**Files:**
- Create: `core/model_cache.py`
- Create: `tests/test_model_cache.py`
- Modify: `tests/conftest.py` (add `huggingface_hub` mock)

- [ ] **Step 1: Add huggingface_hub mock to conftest.py**

In `tests/conftest.py`, add `"huggingface_hub"` to the `_MOCK_MODULES` list so tests can import `core.model_cache` without installing the package.

```python
_MOCK_MODULES = [
    "faiss",
    "torch",
    "transformers",
    "transformers.CLIPModel",
    "transformers.CLIPProcessor",
    "ultralytics",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.yoloe",
    "pywinauto",
    "pywinauto.application",
    "pytesseract",
    "huggingface_hub",
]
```

- [ ] **Step 2: Write failing tests for model_cache**

Create `tests/test_model_cache.py`:

```python
"""Tests for core.model_cache — HuggingFace weight download utility."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_returns_default_when_no_config(self):
        from core.model_cache import get_cache_dir

        with patch("core.model_cache.get_config", return_value={"paths": {}}):
            result = get_cache_dir()
            assert result == Path.home() / ".cache" / "ocsd" / "models"

    def test_returns_config_value_expanded(self):
        from core.model_cache import get_cache_dir

        with patch(
            "core.model_cache.get_config",
            return_value={"paths": {"model_cache": "~/.custom/models"}},
        ):
            result = get_cache_dir()
            assert result == Path.home() / ".custom" / "models"


class TestEnsureModel:
    """Tests for ensure_model()."""

    def test_downloads_single_file(self):
        from core.model_cache import ensure_model

        mock_download = MagicMock(return_value="/fake/path/model.pt")
        # Patch at the source module since ensure_model() does
        # `from huggingface_hub import hf_hub_download` inside the function body.
        with patch("huggingface_hub.hf_hub_download", mock_download), \
             patch("core.model_cache.get_cache_dir", return_value=Path("/cache")):
            result = ensure_model("microsoft/OmniParser-v2.0", "icon_detect/model.pt")

        assert result == Path("/fake/path/model.pt")
        mock_download.assert_called_once_with(
            repo_id="microsoft/OmniParser-v2.0",
            filename="icon_detect/model.pt",
            cache_dir=Path("/cache"),
        )

    def test_downloads_full_repo_when_no_filename(self):
        from core.model_cache import ensure_model

        mock_snapshot = MagicMock(return_value="/fake/repo/snapshot")
        with patch("huggingface_hub.snapshot_download", mock_snapshot), \
             patch("core.model_cache.get_cache_dir", return_value=Path("/cache")):
            result = ensure_model("microsoft/Florence-2-large")

        assert result == Path("/fake/repo/snapshot")
        mock_snapshot.assert_called_once_with(
            repo_id="microsoft/Florence-2-large",
            cache_dir=Path("/cache"),
        )

    def test_custom_cache_dir_overrides_default(self):
        from core.model_cache import ensure_model

        custom = Path("/my/cache")
        mock_download = MagicMock(return_value="/my/cache/model.pt")
        with patch("huggingface_hub.hf_hub_download", mock_download):
            ensure_model("repo/name", "file.pt", cache_dir=custom)

        mock_download.assert_called_once_with(
            repo_id="repo/name",
            filename="file.pt",
            cache_dir=custom,
        )


class TestClearCache:
    """Tests for clear_cache()."""

    def test_clear_specific_repo(self, tmp_path):
        from core.model_cache import clear_cache

        repo_dir = tmp_path / "models--microsoft--OmniParser-v2.0"
        repo_dir.mkdir(parents=True)
        (repo_dir / "model.pt").touch()

        with patch("core.model_cache.get_cache_dir", return_value=tmp_path):
            clear_cache("microsoft/OmniParser-v2.0")

        assert not repo_dir.exists()

    def test_clear_all(self, tmp_path):
        from core.model_cache import clear_cache

        repo1 = tmp_path / "models--org--repo1"
        repo1.mkdir(parents=True)
        repo2 = tmp_path / "models--org--repo2"
        repo2.mkdir(parents=True)

        with patch("core.model_cache.get_cache_dir", return_value=tmp_path):
            clear_cache()

        assert not repo1.exists()
        assert not repo2.exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_model_cache.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'core.model_cache'`

- [ ] **Step 4: Implement core/model_cache.py**

```python
"""HuggingFace model weight download and cache management.

Provides a single utility for downloading model weights from HuggingFace Hub
and caching them locally. Used by OmniParser and Florence-2 modules.

No GPU, no torch — pure download utility.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from core.config import get_config

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Return model cache directory from config or default ~/.cache/ocsd/models/."""
    cfg = get_config()
    custom = cfg.get("paths", {}).get("model_cache")
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".cache" / "ocsd" / "models"


def ensure_model(
    repo_id: str,
    filename: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download model file from HuggingFace if not cached. Return local path.

    Args:
        repo_id: HuggingFace repo (e.g., "microsoft/OmniParser-v2.0").
        filename: Specific file within repo (e.g., "icon_detect/model.pt").
                  If None, downloads entire repo snapshot.
        cache_dir: Override default cache directory.

    Returns:
        Path to local cached file or directory.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    if cache_dir is None:
        cache_dir = get_cache_dir()

    if filename is not None:
        logger.info("Ensuring model file: %s/%s", repo_id, filename)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        return Path(local_path)
    else:
        logger.info("Ensuring model repo: %s", repo_id)
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
        )
        return Path(local_dir)


def clear_cache(repo_id: str | None = None) -> None:
    """Remove cached weights. If repo_id given, only that repo.

    Args:
        repo_id: HuggingFace repo to clear (e.g., "microsoft/OmniParser-v2.0").
                 If None, removes all cached model directories.
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return

    if repo_id is not None:
        # HuggingFace cache uses models--org--repo format
        dir_name = f"models--{repo_id.replace('/', '--')}"
        target = cache_dir / dir_name
        if target.exists():
            shutil.rmtree(target)
            logger.info("Cleared cache for %s", repo_id)
    else:
        for child in cache_dir.iterdir():
            if child.is_dir() and child.name.startswith("models--"):
                shutil.rmtree(child)
        logger.info("Cleared all model caches")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_model_cache.py -v`

Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py core/model_cache.py tests/test_model_cache.py
git commit -m "feat(model_cache): HuggingFace weight download + cache utility"
```

---

### Task 2: Detection Provider Protocol + Factory

**Files:**
- Create: `core/detection.py`
- Create: `tests/test_detection_provider.py`

- [ ] **Step 1: Write failing tests for detection provider**

Create `tests/test_detection_provider.py`:

```python
"""Tests for core.detection — DetectionProvider protocol + factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGetDetector:
    """Tests for get_detector() factory."""

    def test_returns_omniparser_provider_by_default(self):
        from core.detection import get_detector
        import core.detection as det_mod

        # Reset singleton
        det_mod._detector_instance = None

        with patch("core.detection.OmniParserProvider") as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            result = get_detector({"models": {"detector": "omniparser"}})

        assert result is mock_instance
        # Reset for other tests
        det_mod._detector_instance = None

    def test_returns_singleton(self):
        from core.detection import get_detector
        import core.detection as det_mod

        det_mod._detector_instance = None

        with patch("core.detection.OmniParserProvider") as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            first = get_detector({"models": {"detector": "omniparser"}})
            second = get_detector({"models": {"detector": "omniparser"}})

        assert first is second
        MockProvider.assert_called_once()
        det_mod._detector_instance = None

    def test_raises_for_unknown_detector(self):
        from core.detection import get_detector
        import core.detection as det_mod

        det_mod._detector_instance = None

        with pytest.raises(ValueError, match="Unknown detector"):
            get_detector({"models": {"detector": "nonexistent"}})

        det_mod._detector_instance = None

    def test_uses_config_when_no_arg(self):
        from core.detection import get_detector
        import core.detection as det_mod

        det_mod._detector_instance = None

        with patch("core.detection.OmniParserProvider") as MockProvider, \
             patch("core.detection.get_config",
                   return_value={"models": {"detector": "omniparser"}}):
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            result = get_detector()

        assert result is mock_instance
        det_mod._detector_instance = None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_detection_provider.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'core.detection'`

- [ ] **Step 3: Implement core/detection.py**

```python
"""Detection provider protocol and factory.

Abstracts detection backends behind a common protocol so they can be
swapped via config. OmniParser is the first (and currently only) provider.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

import numpy as np

from core.config import get_config
from core.types import LocateResult

logger = logging.getLogger(__name__)

_detector_instance: Any = None
_detector_lock = threading.Lock()


class DetectionProvider(Protocol):
    """Protocol for UI element detection backends."""

    def detect(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Detect all UI elements in screenshot.

        Returns list of candidate dicts:
            {
                "rect": {"x": int, "y": int, "w": int, "h": int},
                "type_guess": str,
                "label_guess": str,
                "confidence": float,
            }
        """
        ...

    def detect_and_match(
        self,
        screenshot: np.ndarray,
        saved_snippet: np.ndarray,
        hint_x: int,
        hint_y: int,
        match_threshold: float = 0.7,
        search_radius: int = 400,
    ) -> LocateResult | None:
        """Detect all elements, then find best CLIP match to saved_snippet.

        Returns LocateResult if match above threshold, else None.
        """
        ...


def get_detector(config: dict | None = None) -> DetectionProvider:
    """Factory that returns the configured detection provider.

    Reads config['models']['detector'] to select backend.
    Returns a singleton instance (lazy-loaded, thread-safe).

    Args:
        config: Optional config dict override. If None, reads from get_config().

    Returns:
        DetectionProvider instance.

    Raises:
        ValueError: If detector name is not recognized.
    """
    global _detector_instance

    if _detector_instance is not None:
        return _detector_instance

    with _detector_lock:
        # Double-check after acquiring lock
        if _detector_instance is not None:
            return _detector_instance

        if config is None:
            config = get_config()

        detector_name = config.get("models", {}).get("detector", "omniparser")

        if detector_name == "omniparser":
            from core.omniparser import OmniParserProvider

            conf_threshold = config.get("detection", {}).get(
                "confidence_threshold", 0.3
            )
            _detector_instance = OmniParserProvider(
                confidence_threshold=conf_threshold
            )
        else:
            raise ValueError(
                f"Unknown detector: {detector_name!r}. "
                f"Supported: 'omniparser'"
            )

        logger.info("Initialized detector: %s", detector_name)
        return _detector_instance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_detection_provider.py -v`

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/detection.py tests/test_detection_provider.py
git commit -m "feat(detection): DetectionProvider protocol + get_detector() factory"
```

---

### Task 3: OmniParser Provider

**Files:**
- Create: `core/omniparser.py`
- Create: `tests/test_omniparser.py`

- [ ] **Step 1: Write failing tests for OmniParserProvider**

Create `tests/test_omniparser.py`:

```python
"""Tests for core.omniparser — OmniParser YOLOv8 icon_detect wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.types import LocateResult, Point


def _make_yolo_result(boxes: list[dict]) -> MagicMock:
    """Build a mock YOLO prediction result.

    Args:
        boxes: List of dicts with keys: xyxy (4-tuple), cls (int), conf (float).
    """
    result = MagicMock()
    result_boxes = MagicMock()

    xyxy_list = []
    cls_list = []
    conf_list = []
    for b in boxes:
        xyxy_list.append(b["xyxy"])
        cls_list.append(b["cls"])
        conf_list.append(b["conf"])

    result_boxes.xyxy = np.array(xyxy_list) if xyxy_list else np.empty((0, 4))
    result_boxes.cls = np.array(cls_list) if cls_list else np.empty(0)
    result_boxes.conf = np.array(conf_list) if conf_list else np.empty(0)
    result.boxes = result_boxes
    # names: class index → class name
    result.names = {0: "icon", 1: "text", 2: "button", 3: "input_field"}
    return result


class TestOmniParserDetect:
    """Tests for OmniParserProvider.detect()."""

    def test_returns_candidates_list(self):
        from core.omniparser import OmniParserProvider

        mock_model = MagicMock()
        yolo_result = _make_yolo_result([
            {"xyxy": [100, 200, 150, 230], "cls": 0, "conf": 0.85},
            {"xyxy": [300, 400, 400, 430], "cls": 1, "conf": 0.72},
        ])
        mock_model.predict.return_value = [yolo_result]

        provider = OmniParserProvider(confidence_threshold=0.3)

        with patch.object(provider, "_load_model") as mock_load:
            provider._model = mock_model
            candidates = provider.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

        assert len(candidates) == 2
        assert candidates[0]["rect"] == {"x": 100, "y": 200, "w": 50, "h": 30}
        assert candidates[0]["type_guess"] == "icon"
        assert candidates[0]["confidence"] == pytest.approx(0.85, abs=0.01)
        assert candidates[0]["label_guess"] == ""

    def test_filters_by_confidence(self):
        from core.omniparser import OmniParserProvider

        mock_model = MagicMock()
        yolo_result = _make_yolo_result([
            {"xyxy": [10, 20, 50, 60], "cls": 0, "conf": 0.9},
            {"xyxy": [100, 200, 150, 250], "cls": 1, "conf": 0.2},  # below threshold
        ])
        mock_model.predict.return_value = [yolo_result]

        provider = OmniParserProvider(confidence_threshold=0.3)
        provider._model = mock_model

        candidates = provider.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

        assert len(candidates) == 1
        assert candidates[0]["confidence"] == pytest.approx(0.9, abs=0.01)

    def test_returns_empty_on_no_detections(self):
        from core.omniparser import OmniParserProvider

        mock_model = MagicMock()
        yolo_result = _make_yolo_result([])
        mock_model.predict.return_value = [yolo_result]

        provider = OmniParserProvider()
        provider._model = mock_model

        candidates = provider.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
        assert candidates == []

    def test_lazy_model_loading(self):
        from core.omniparser import OmniParserProvider

        provider = OmniParserProvider()
        assert provider._model is None  # not loaded yet

        mock_model = MagicMock()
        yolo_result = _make_yolo_result([])
        mock_model.predict.return_value = [yolo_result]

        with patch("core.omniparser.model_cache") as mock_cache, \
             patch("core.omniparser.YOLO", return_value=mock_model):
            mock_cache.ensure_model.return_value = Path("/fake/model.pt")
            provider.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        assert provider._model is mock_model


class TestOmniParserDetectAndMatch:
    """Tests for OmniParserProvider.detect_and_match()."""

    def test_returns_best_clip_match(self):
        from core.omniparser import OmniParserProvider

        mock_model = MagicMock()
        # Two detections near the hint point
        yolo_result = _make_yolo_result([
            {"xyxy": [90, 190, 140, 220], "cls": 0, "conf": 0.8},   # near hint
            {"xyxy": [95, 195, 145, 225], "cls": 2, "conf": 0.75},  # near hint
        ])
        mock_model.predict.return_value = [yolo_result]

        provider = OmniParserProvider(confidence_threshold=0.3)
        provider._model = mock_model

        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        snippet = np.zeros((30, 50, 3), dtype=np.uint8)

        # Mock CLIP with L2-normalized embeddings (as CLIP produces)
        # Snippet and crop1 are similar (high cosine), crop2 is different (low cosine)
        rng = np.random.RandomState(42)
        base_vec = rng.randn(1, 512).astype(np.float32)
        base_vec /= np.linalg.norm(base_vec)

        # Crop1: very similar to snippet (add small noise)
        crop1_vec = base_vec + rng.randn(1, 512).astype(np.float32) * 0.05
        crop1_vec /= np.linalg.norm(crop1_vec)

        # Crop2: quite different
        crop2_vec = rng.randn(1, 512).astype(np.float32)
        crop2_vec /= np.linalg.norm(crop2_vec)

        call_count = [0]
        def mock_gen_emb(img):
            call_count[0] += 1
            if call_count[0] == 1:
                return base_vec      # snippet embedding
            elif call_count[0] == 2:
                return crop1_vec     # crop1: high similarity
            else:
                return crop2_vec     # crop2: low similarity

        with patch("core.omniparser.generate_embedding", side_effect=mock_gen_emb):
            result = provider.detect_and_match(
                screenshot, snippet, hint_x=115, hint_y=210,
                match_threshold=0.5, search_radius=400,
            )

        assert result is not None
        assert isinstance(result, LocateResult)
        assert result.method == "omniparser"

    def test_returns_none_below_threshold(self):
        from core.omniparser import OmniParserProvider

        mock_model = MagicMock()
        yolo_result = _make_yolo_result([
            {"xyxy": [90, 190, 140, 220], "cls": 0, "conf": 0.8},
        ])
        mock_model.predict.return_value = [yolo_result]

        provider = OmniParserProvider(confidence_threshold=0.3)
        provider._model = mock_model

        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        snippet = np.zeros((30, 50, 3), dtype=np.uint8)

        # Return orthogonal L2-normalized embeddings → cosine similarity ≈ 0
        rng = np.random.RandomState(99)
        call_count = [0]
        def mock_gen_emb(img):
            call_count[0] += 1
            vec = rng.randn(1, 512).astype(np.float32)
            vec /= np.linalg.norm(vec)
            return vec

        with patch("core.omniparser.generate_embedding", side_effect=mock_gen_emb):
            result = provider.detect_and_match(
                screenshot, snippet, hint_x=115, hint_y=210,
                match_threshold=0.7, search_radius=400,
            )

        assert result is None

    def test_filters_by_search_radius(self):
        from core.omniparser import OmniParserProvider

        mock_model = MagicMock()
        yolo_result = _make_yolo_result([
            {"xyxy": [90, 190, 140, 220], "cls": 0, "conf": 0.8},     # near hint
            {"xyxy": [1500, 1500, 1550, 1530], "cls": 1, "conf": 0.9},  # far away
        ])
        mock_model.predict.return_value = [yolo_result]

        provider = OmniParserProvider(confidence_threshold=0.3)
        provider._model = mock_model

        screenshot = np.zeros((1920, 1920, 3), dtype=np.uint8)
        snippet = np.zeros((30, 50, 3), dtype=np.uint8)

        # High CLIP score for both, but only near one should be considered
        mock_emb = np.ones((1, 512), dtype=np.float32) * 0.9

        with patch("core.omniparser.generate_embedding", return_value=mock_emb):
            result = provider.detect_and_match(
                screenshot, snippet, hint_x=115, hint_y=210,
                match_threshold=0.5, search_radius=100,
            )

        # Should only match the near detection
        if result is not None:
            assert abs(result.point.x - 115) < 100
            assert abs(result.point.y - 205) < 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_omniparser.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'core.omniparser'`

- [ ] **Step 3: Implement core/omniparser.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_omniparser.py -v`

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/omniparser.py tests/test_omniparser.py
git commit -m "feat(omniparser): OmniParser YOLOv8 icon_detect provider"
```

---

### Task 4: Florence-2 Captioning Module

**Files:**
- Create: `core/florence.py`
- Create: `tests/test_florence.py`

- [ ] **Step 1: Write failing tests for Florence-2**

Create `tests/test_florence.py`:

```python
"""Tests for core.florence — Florence-2 captioning module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


def _reset_florence():
    """Reset Florence-2 module state between tests."""
    import core.florence as fl
    fl._model = None
    fl._processor = None


class TestLoadModel:
    """Tests for load_model()."""

    def setup_method(self):
        _reset_florence()

    def teardown_method(self):
        _reset_florence()

    def test_loads_default_variant(self):
        from core.florence import load_model
        import core.florence as fl

        mock_model = MagicMock()
        mock_proc = MagicMock()

        with patch("core.florence.model_cache") as mock_cache, \
             patch("core.florence.AutoModelForCausalLM") as MockModel, \
             patch("core.florence.AutoProcessor") as MockProc, \
             patch("core.florence.get_config", return_value={
                 "models": {"florence": "microsoft/Florence-2-large"},
                 "hardware": {"gpu_embeddings": 1},
             }):
            mock_cache.ensure_model.return_value = Path("/fake/florence")
            MockModel.from_pretrained.return_value = mock_model
            MockProc.from_pretrained.return_value = mock_proc

            load_model()

        assert fl._model is mock_model
        assert fl._processor is mock_proc


class TestCaptionCrop:
    """Tests for caption_crop()."""

    def setup_method(self):
        _reset_florence()

    def teardown_method(self):
        _reset_florence()

    def test_returns_string_caption(self):
        from core.florence import caption_crop
        import core.florence as fl

        # Pre-load mock model
        fl._model = MagicMock()
        fl._processor = MagicMock()
        fl._device = "cpu"

        # Mock processor call returns input_ids tensor
        mock_inputs = {"input_ids": MagicMock(), "pixel_values": MagicMock()}
        for v in mock_inputs.values():
            v.to = MagicMock(return_value=v)
        fl._processor.return_value = mock_inputs

        # Mock model.generate returns token ids
        fl._model.generate.return_value = MagicMock()

        # Mock processor.batch_decode returns caption text
        fl._processor.batch_decode.return_value = ["a blue submit button"]

        # Mock post_process to return caption
        fl._processor.post_process_generation.return_value = {
            "<CAPTION>": "a blue submit button"
        }

        img = np.zeros((30, 50, 3), dtype=np.uint8)
        result = caption_crop(img)

        assert isinstance(result, str)
        assert len(result) > 0


class TestCaptionBatch:
    """Tests for caption_batch()."""

    def setup_method(self):
        _reset_florence()

    def teardown_method(self):
        _reset_florence()

    def test_returns_list_of_captions(self):
        from core.florence import caption_batch
        import core.florence as fl

        # Pre-set state to avoid loading
        fl._model = MagicMock()
        fl._processor = MagicMock()
        fl._device = "cpu"

        with patch("core.florence.caption_crop", side_effect=["button", "text field"]):
            results = caption_batch([
                np.zeros((30, 50, 3), dtype=np.uint8),
                np.zeros((20, 80, 3), dtype=np.uint8),
            ])

        assert results == ["button", "text field"]

    def test_handles_degenerate_crops(self):
        from core.florence import caption_batch
        import core.florence as fl

        fl._model = MagicMock()
        fl._processor = MagicMock()
        fl._device = "cpu"

        with patch("core.florence.caption_crop", return_value="ok"):
            results = caption_batch([
                np.zeros((0, 0, 3), dtype=np.uint8),  # degenerate
                np.zeros((30, 50, 3), dtype=np.uint8),  # valid
            ])

        assert results[0] == ""  # degenerate → empty
        assert results[1] == "ok"


class TestDescribeElement:
    """Tests for describe_element()."""

    def setup_method(self):
        _reset_florence()

    def teardown_method(self):
        _reset_florence()

    def test_returns_structured_dict(self):
        from core.florence import describe_element
        import core.florence as fl

        fl._model = MagicMock()
        fl._processor = MagicMock()
        fl._device = "cpu"

        with patch(
            "core.florence.caption_crop",
            return_value="a blue rectangular submit button with white text",
        ):
            result = describe_element(np.zeros((30, 50, 3), dtype=np.uint8))

        assert "type_guess" in result
        assert "label_guess" in result
        assert "description" in result
        assert result["type_guess"] == "button"
        assert len(result["label_guess"]) > 0
        assert result["description"] == "a blue rectangular submit button with white text"

    def test_returns_unknown_for_unrecognized(self):
        from core.florence import describe_element
        import core.florence as fl

        fl._model = MagicMock()
        fl._processor = MagicMock()
        fl._device = "cpu"

        with patch(
            "core.florence.caption_crop",
            return_value="a colorful shape with gradient borders",
        ):
            result = describe_element(np.zeros((30, 50, 3), dtype=np.uint8))

        assert result["type_guess"] == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_florence.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'core.florence'`

- [ ] **Step 3: Implement core/florence.py**

```python
"""Florence-2 captioning module.

Fast element labeling using Microsoft Florence-2 vision model.
Provides caption_crop(), caption_batch(), and describe_element()
for UI element identification.

Model weights auto-download from HuggingFace on first use.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from core import model_cache
from core.config import get_config

logger = logging.getLogger(__name__)

# Module-level state (lazy-loaded)
_model: Any = None
_processor: Any = None
_device: str = "cpu"
_lock = threading.Lock()

# Lazy imports — only needed when model loads
AutoModelForCausalLM: Any = None
AutoProcessor: Any = None

# Known UI element type keywords for describe_element()
_TYPE_KEYWORDS = [
    "button", "textbox", "text field", "input field", "input box",
    "checkbox", "radio", "toggle", "switch",
    "dropdown", "select", "combobox",
    "slider", "scrollbar", "scroll bar",
    "tab", "menu", "menubar", "toolbar",
    "link", "hyperlink", "anchor",
    "icon", "image", "logo",
    "label", "heading", "title",
    "dialog", "modal", "popup",
    "list", "table", "grid",
]


def _import_transformers():
    """Lazy-import transformers classes."""
    global AutoModelForCausalLM, AutoProcessor
    if AutoModelForCausalLM is None:
        from transformers import AutoModelForCausalLM as _M, AutoProcessor as _P
        AutoModelForCausalLM = _M
        AutoProcessor = _P


def load_model(variant: str | None = None) -> None:
    """Pre-load Florence-2 model. Called at startup for warm cache.

    Args:
        variant: Model name override. Default from config
                 (florence-2-large or florence-2-base).
    """
    global _model, _processor, _device

    with _lock:
        if _model is not None:
            return

        _import_transformers()

        cfg = get_config()
        if variant is None:
            variant = cfg.get("models", {}).get(
                "florence", "microsoft/Florence-2-large"
            )

        model_dir = model_cache.ensure_model(variant)
        logger.info("Loading Florence-2 from %s", model_dir)

        # Determine device
        _device = "cpu"
        try:
            import torch
            gpu_id = cfg.get("hardware", {}).get("gpu_embeddings", 1)
            if torch.cuda.is_available():
                _device = f"cuda:{gpu_id}"
        except ImportError:
            pass

        _model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )
        _processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )

        if _device != "cpu":
            _model = _model.to(_device)

        logger.info("Florence-2 loaded on %s", _device)


def _ensure_model() -> None:
    """Ensure model is loaded before use."""
    if _model is None:
        load_model()


def caption_crop(image: np.ndarray, task: str = "<CAPTION>") -> str:
    """Caption a single image crop.

    Args:
        image: RGB numpy array of element crop.
        task: Florence-2 task token. Options:
              "<CAPTION>" — short caption
              "<DETAILED_CAPTION>" — detailed description
              "<MORE_DETAILED_CAPTION>" — very detailed

    Returns:
        Caption string.
    """
    _ensure_model()

    from PIL import Image
    pil_img = Image.fromarray(image)

    inputs = _processor(text=task, images=pil_img, return_tensors="pt")
    # Move to device
    for k in inputs:
        if hasattr(inputs[k], "to"):
            inputs[k] = inputs[k].to(_device)

    generated_ids = _model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        num_beams=3,
    )

    generated_text = _processor.batch_decode(
        generated_ids, skip_special_tokens=False,
    )[0]

    parsed = _processor.post_process_generation(
        generated_text, task=task,
        image_size=(pil_img.width, pil_img.height),
    )

    caption = parsed.get(task, generated_text)
    if isinstance(caption, dict):
        caption = str(caption)

    return caption.strip()


def caption_batch(
    images: list[np.ndarray], task: str = "<CAPTION>"
) -> list[str]:
    """Caption multiple crops. Returns list of caption strings.

    If a crop is degenerate (empty, 0-dim), returns empty string for that
    position to maintain 1:1 correspondence with input list.

    Args:
        images: List of RGB numpy arrays.
        task: Florence-2 task token.

    Returns:
        List of caption strings, one per input image.
    """
    _ensure_model()

    results: list[str] = []
    for img in images:
        if img.size == 0 or img.ndim < 2:
            results.append("")
            continue
        try:
            results.append(caption_crop(img, task))
        except Exception as e:
            logger.warning("Florence-2 caption failed for crop: %s", e)
            results.append("")

    return results


def describe_element(image: np.ndarray) -> dict[str, str]:
    """Structured element description for UI labeling.

    Uses <DETAILED_CAPTION> internally, then parses the response
    to extract type and label.

    Args:
        image: RGB numpy array of element crop.

    Returns:
        Dict with keys: type_guess, label_guess, description.
    """
    caption = caption_crop(image, "<DETAILED_CAPTION>")

    type_guess = "unknown"
    label_guess = caption

    caption_lower = caption.lower()
    for keyword in _TYPE_KEYWORDS:
        if keyword in caption_lower:
            # Map multi-word keywords to canonical types
            if keyword in ("text field", "input field", "input box"):
                type_guess = "textbox"
            elif keyword in ("dropdown", "select", "combobox"):
                type_guess = "dropdown"
            elif keyword in ("link", "hyperlink", "anchor"):
                type_guess = "link"
            elif keyword in ("scroll bar",):
                type_guess = "scrollbar"
            elif keyword in ("switch",):
                type_guess = "toggle"
            else:
                type_guess = keyword

            # Extract label: text before or around the keyword
            # Try to find quoted text first
            quoted = re.findall(r'["\']([^"\']+)["\']', caption)
            if quoted:
                label_guess = quoted[0]
            else:
                # Use the caption but shortened
                label_guess = caption[:50]
            break

    return {
        "type_guess": type_guess,
        "label_guess": label_guess,
        "description": caption,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_florence.py -v`

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/florence.py tests/test_florence.py
git commit -m "feat(florence): Florence-2 captioning module"
```

---

## Chunk 2: Configuration & Wiring

### Task 5: Config + Dependency Updates

**Files:**
- Modify: `config.yaml`
- Modify: `core/config.py` (lines 27-64, `_DEFAULTS` dict)
- Modify: `pyproject.toml` (lines 20-57, optional-dependencies)

- [ ] **Step 1: Update config.yaml**

Replace the `models:` and `detection:` sections, and add `model_cache` path:

In `config.yaml`, change the `models:` section — remove `yoloe` key, add `detector`, `florence`, `omniparser_weights`:

```yaml
models:
  vlm: "vision"
  quick: "quick"
  planning: "planning"
  coding: "coding"
  clip: "openai/clip-vit-base-patch32"
  whisper_model: "small"
  # Detection backend selector
  detector: "omniparser"
  # Florence-2 captioning model (or "microsoft/Florence-2-base" for lighter)
  florence: "microsoft/Florence-2-large"
  # OmniParser YOLOv8 weights repo
  omniparser_weights: "microsoft/OmniParser-v2.0"
```

In `config.yaml`, update the `detection:` section:

```yaml
detection:
  # OmniParser confidence threshold (lower than YOLOE — UI-trained model is more precise)
  confidence_threshold: 0.3
  # CLIP cosine similarity threshold for replay matching
  match_threshold: 0.7
  # Extra padding around detected elements when cropping for VLM
  crop_buffer_pct: 0.10
  # Search radius (pixels) around expected position
  search_radius: 400
```

In `config.yaml`, add `model_cache` under `paths:`:

```yaml
paths:
  skills_dir: "./skills"
  snippets_dir: "./assets/snippets"
  faiss_index: "./assets/faiss.index"
  replay_logs: "./logs/replays"
  model_cache: "~/.cache/ocsd/models"
```

- [ ] **Step 2: Update core/config.py _DEFAULTS**

In `core/config.py`, update the `"models"` and `"detection"` sections of `_DEFAULTS`.
**Remove:** `models.yoloe` key (was `"yoloe-26s-seg.pt"`), `detection.model` key (was `"yolo-e"`).
**Add:** `models.detector`, `models.florence`, `models.omniparser_weights`, `detection.match_threshold`.
**Change:** `detection.confidence_threshold` from 0.4 → 0.3, `detection.crop_buffer_pct` from 0.30 → 0.10.

New sections:

```python
    "models": {
        "vlm": os.environ.get("OCSD_VLM_MODEL", "vision"),
        "quick": os.environ.get("OCSD_QUICK_MODEL", "quick"),
        "planning": os.environ.get("OCSD_PLANNING_MODEL", "planning"),
        "coding": os.environ.get("OCSD_CODING_MODEL", "coding"),
        "clip": "openai/clip-vit-base-patch32",
        "whisper_model": "small",
        "detector": "omniparser",
        "florence": "microsoft/Florence-2-large",
        "omniparser_weights": "microsoft/OmniParser-v2.0",
    },
    "detection": {
        "confidence_threshold": 0.3,
        "match_threshold": 0.7,
        "crop_buffer_pct": 0.10,
        "search_radius": 400,
    },
```

And add `"model_cache"` to the `"paths"` section:

```python
    "paths": {
        "skills_dir": "./skills",
        "snippets_dir": "./assets/snippets",
        "faiss_index": "./assets/faiss.index",
        "replay_logs": "./logs/replays",
        "model_cache": "~/.cache/ocsd/models",
    },
```

- [ ] **Step 3: Update pyproject.toml optional-dependencies**

Add new dependency groups and update `all`:

```toml
[project.optional-dependencies]
# Core AI features (CLIP embeddings + FAISS similarity search)
embeddings = [
    "transformers>=4.36",
    "faiss-cpu>=1.7",
    "torch>=2.1",
]
# YOLO-E visual grounding (dormant — kept for reference)
yoloe = [
    "ultralytics>=8.3",
]
# VLM vision analysis via LiteLLM proxy
vlm = [
    "openai>=1.0.0",
]
# Windows-only accessibility bridge
windows = [
    "pywinauto>=0.6.8",
]
# REST API server
api = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
]
# TUI launcher (rich-based interactive menu)
tui = [
    "rich>=13.0",
]
# OmniParser YOLOv8 icon detection
omniparser = [
    "ultralytics>=8.3",
]
# Florence-2 captioning
florence = [
    "transformers>=4.36",
    "torch>=2.1",
]
# Model weight management
model-cache = [
    "huggingface-hub>=0.20",
]
# Full detection stack (OmniParser + Florence-2 + model cache)
detection = [
    "openclaw-screen-driver[omniparser,florence,model-cache]",
]
# Everything (install all optional features)
all = [
    "openclaw-screen-driver[detection,embeddings,vlm,windows,api,tui]",
]
# Development tools
dev = [
    "pytest>=7.0",
    "pytest-mock>=3.0",
    "ruff>=0.1",
]
```

- [ ] **Step 4: Run existing tests to verify nothing broke**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/ -v --tb=short`

Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add config.yaml core/config.py pyproject.toml
git commit -m "feat(config): update config for OmniParser + Florence-2 integration"
```

---

### Task 6: Rewire recorder/smart_detect.py

**Files:**
- Modify: `recorder/smart_detect.py`
- Modify: `tests/test_smart_detect.py`

- [ ] **Step 1: Update test_smart_detect.py mocks**

Update `tests/test_smart_detect.py` to mock the new OmniParser + Florence-2 pipeline instead of YOLOE.

Specific changes to existing tests:

1. **Rename** `TestYOLOEFirstPipeline` → `TestOmniParserPipeline`

2. **`test_yoloe_returns_results`** → `test_omniparser_returns_results`:
   - Change `@patch("recorder.smart_detect._detect_via_yoloe")` → `@patch("recorder.smart_detect._detect_via_omniparser")`
   - Change `mock_yoloe` param → `mock_omni`
   - Add `@patch("recorder.smart_detect._enrich_with_florence")` that passes through candidates unchanged
   - Remove `use_yoloe` kwarg from call

3. **`test_yoloe_empty_falls_back_to_ocr`** → `test_omniparser_empty_falls_back_to_ocr`:
   - Change `@patch("recorder.smart_detect._detect_via_yoloe")` → `@patch("recorder.smart_detect._detect_via_omniparser")`
   - Add `@patch("recorder.smart_detect._enrich_with_florence")` (identity passthrough)

4. **Delete** `test_yoloe_disabled_uses_ocr` — `use_yoloe` param no longer exists

5. **Delete** `test_vlm_flag_has_no_effect_on_detection` — `use_vlm` param no longer exists

6. **`TestDetectUIElementsAsync`**: Remove `use_yoloe`/`use_vlm` kwargs if any test passes them. Currently none do — tests already call `detect_ui_elements_async(fake_screenshot, callback)` without extra kwargs. No changes needed.

7. **Delete** `TestDetectViaYOLOE` class entirely (tests `_detect_via_yoloe` which no longer exists)

8. **Replace** with new `TestDetectViaOmniParser` class (mirrors deleted YOLOE tests):

```python
class TestDetectViaOmniParser:
    """Tests for the OmniParser detection wrapper in smart_detect."""

    def test_import_error_returns_empty(self, fake_screenshot: np.ndarray) -> None:
        """ImportError from core.detection returns empty list."""
        from recorder.smart_detect import _detect_via_omniparser

        with patch.dict("sys.modules", {"core.detection": None}):
            result = _detect_via_omniparser(fake_screenshot)
            assert result == []

    def test_exception_returns_empty(self, fake_screenshot: np.ndarray) -> None:
        """Runtime exception from OmniParser returns empty list."""
        from recorder.smart_detect import _detect_via_omniparser

        mock_detection = MagicMock()
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("CUDA OOM")
        mock_detection.get_detector.return_value = mock_detector
        with patch.dict("sys.modules", {"core.detection": mock_detection}):
            result = _detect_via_omniparser(fake_screenshot)
            assert result == []

    @patch("core.detection.get_detector")
    def test_returns_candidates(
        self, mock_get_detector: MagicMock, fake_screenshot: np.ndarray
    ) -> None:
        """OmniParser returns detection candidates."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {
                "rect": {"x": 10, "y": 20, "w": 80, "h": 30},
                "type_guess": "button",
                "label_guess": "button",
                "confidence": 0.85,
                "ocr_text": None,
            },
        ]
        mock_get_detector.return_value = mock_detector

        from recorder.smart_detect import _detect_via_omniparser

        result = _detect_via_omniparser(fake_screenshot)
        assert len(result) == 1
        assert result[0]["type_guess"] == "button"
```

9. **Keep** `TestDetectViaOCR` class unchanged (OCR fallback is preserved)

9. **Add** new test class `TestEnrichWithFlorence`:

```python
class TestEnrichWithFlorence:
    """Tests for Florence-2 caption enrichment."""

    def test_enriches_label_guess(self, fake_screenshot):
        from recorder.smart_detect import _enrich_with_florence

        candidates = [
            {"rect": {"x": 10, "y": 20, "w": 80, "h": 30}, "label_guess": "", "confidence": 0.9},
        ]

        with patch("core.florence.caption_batch", return_value=["Submit button"]), \
             patch("cv2.cvtColor", return_value=np.zeros((30, 80, 3), dtype=np.uint8)):
            result = _enrich_with_florence(candidates, fake_screenshot)

        assert result[0]["label_guess"] == "Submit button"
        assert result[0]["florence_caption"] == "Submit button"

    def test_handles_import_error(self, fake_screenshot):
        from recorder.smart_detect import _enrich_with_florence

        candidates = [
            {"rect": {"x": 10, "y": 20, "w": 80, "h": 30}, "label_guess": "", "confidence": 0.9},
        ]

        with patch.dict("sys.modules", {"core.florence": None}):
            result = _enrich_with_florence(candidates, fake_screenshot)

        assert result[0]["label_guess"] == ""  # unchanged
```

- [ ] **Step 2: Run tests to verify they fail (mocks point to new code)**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_smart_detect.py -v`

Expected: FAIL — old code doesn't match new test expectations

- [ ] **Step 3: Update recorder/smart_detect.py**

Key changes:
1. Remove `use_yoloe` parameter from `detect_ui_elements()` and `detect_ui_elements_async()`
2. Remove `use_vlm` parameter (already no-op)
3. Remove `_detect_via_yoloe()` function
4. Add `_enrich_with_florence()` function
5. Main pipeline: `get_detector().detect()` → `_enrich_with_florence()` → fallback OCR

New `detect_ui_elements()` signature:

```python
def detect_ui_elements(
    screenshot: np.ndarray,
) -> list[dict[str, Any]]:
```

New `detect_ui_elements_async()` signature:

```python
def detect_ui_elements_async(
    screenshot: np.ndarray,
    callback: Any,
) -> threading.Thread:
```

New `_enrich_with_florence()`:

```python
def _enrich_with_florence(
    candidates: list[dict[str, Any]],
    screenshot: np.ndarray,
) -> list[dict[str, Any]]:
    """Add Florence-2 captions to each candidate's label_guess.

    Crops each candidate bbox from the screenshot, runs Florence-2
    caption_batch, and updates label_guess fields.
    """
    try:
        import cv2
        from core.florence import caption_batch

        crops = []
        for c in candidates:
            r = c["rect"]
            x1 = max(0, r["x"])
            y1 = max(0, r["y"])
            x2 = r["x"] + r["w"]
            y2 = r["y"] + r["h"]
            crop = screenshot[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            else:
                crops.append(np.zeros((1, 1, 3), dtype=np.uint8))

        captions = caption_batch(crops)
        for c, cap in zip(candidates, captions):
            if cap:
                c["label_guess"] = cap
                c["florence_caption"] = cap

    except ImportError:
        logger.debug("Florence-2 not available, skipping captioning")
    except Exception as e:
        logger.warning("Florence-2 captioning failed: %s", e)

    return candidates
```

New primary detection using OmniParser:

```python
def _detect_via_omniparser(screenshot: np.ndarray) -> list[dict[str, Any]]:
    """Uses OmniParser YOLOv8 to detect all UI elements."""
    try:
        from core.detection import get_detector

        detector = get_detector()
        candidates = detector.detect(screenshot)
        logger.info("OmniParser detected %d UI elements", len(candidates))
        return candidates
    except ImportError:
        logger.debug("OmniParser not available for detection")
        return []
    except Exception as e:
        logger.warning("OmniParser detection failed: %s", e)
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_smart_detect.py -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add recorder/smart_detect.py tests/test_smart_detect.py
git commit -m "feat(smart_detect): replace YOLOE with OmniParser + Florence-2"
```

---

### Task 7: Rewire mapper/runner.py Cascade

**Files:**
- Modify: `mapper/runner.py` (lines 114-148, Stage 1)
- Modify: `tests/test_cascade.py`

- [ ] **Step 1: Update test_cascade.py Stage 1 mocks**

Update Stage 1 tests to mock `core.detection.get_detector` instead of `core.yoloe.find_element_locate`. The test should verify that `locate_element()` calls `detector.detect_and_match()` with the correct arguments.

**Concrete changes to `tests/test_cascade.py`:**

1. In `test_yoloe_succeeds_returns_early` (line ~537), rename to `test_omniparser_succeeds_returns_early` and change:

```python
    def test_omniparser_succeeds_returns_early(self, mocker: Any) -> None:
        """When OmniParser matches, cascade returns immediately without OCR/VLM."""
        omni_result = LocateResult(
            point=Point(960, 540), method="omniparser", confidence=0.9
        )
        mocker.patch(
            "core.capture.load_snippet", return_value=_bgr_image(50, 50)
        )
        # Mock get_detector to return a mock provider whose detect_and_match
        # returns our result
        mock_detector = MagicMock()
        mock_detector.detect_and_match.return_value = omni_result
        mocker.patch(
            "core.detection.get_detector", return_value=mock_detector
        )
        mock_ocr = mocker.patch("core.ocr.find_text_on_screen")

        from mapper.runner import locate_element

        result = locate_element(_make_mock_graph(), "node-abc", skill_id="s1")

        assert result.method == "omniparser"
        assert result.confidence == pytest.approx(0.9)
        mock_detector.detect_and_match.assert_called_once()
        mock_ocr.assert_not_called()
```

2. In `test_yoloe_fails_clip_succeeds` (line ~561), rename to `test_omniparser_fails_clip_succeeds` and change:

```python
    def test_omniparser_fails_clip_succeeds(self, mocker: Any) -> None:
        """When OmniParser finds nothing but CLIP score is high, returns clip."""
        mocker.patch(
            "core.capture.load_snippet", return_value=_bgr_image(50, 50)
        )
        # OmniParser returns None (no match)
        mock_detector = MagicMock()
        mock_detector.detect_and_match.return_value = None
        mocker.patch(
            "core.detection.get_detector", return_value=mock_detector
        )

        # CLIP returns high similarity (keep existing CLIP mocks unchanged)
        saved_emb = np.random.randn(1, 512).astype("float32")
        saved_emb /= np.linalg.norm(saved_emb)
        current_emb = saved_emb.copy()

        mocker.patch(
            "core.embeddings.get_embedding_by_id", return_value=saved_emb
        )
        mocker.patch(
            "core.embeddings.generate_embedding", return_value=current_emb
        )
        mocker.patch(
            "core.capture.screenshot_region", return_value=_bgr_image(600, 600)
        )
        mocker.patch("cv2.cvtColor", return_value=_bgr_image(600, 600))
        mock_ocr = mocker.patch("core.ocr.find_text_on_screen")

        from mapper.runner import locate_element

        result = locate_element(_make_mock_graph(), "node-xyz", skill_id="s1")

        assert result.method == "clip"
        assert result.confidence > 0.5
        mock_ocr.assert_not_called()
```

3. In `test_yoloe_clip_fail_ocr_succeeds` (line ~598), rename to `test_omniparser_clip_fail_ocr_succeeds` and change Stage 1 mock:

```python
    def test_omniparser_clip_fail_ocr_succeeds(self, mocker: Any) -> None:
        """When OmniParser and CLIP both fail, OCR takes over."""
        # OmniParser — no snippet on disk
        mocker.patch("core.capture.load_snippet", return_value=None)

        # CLIP — no saved embedding (keep as-is)
        mocker.patch(
            "core.embeddings.get_embedding_by_id", return_value=None
        )

        # OCR succeeds (keep as-is)
        ocr_result = LocateResult(
            point=Point(960, 540), method="ocr", confidence=0.88
        )
        mocker.patch(
            "mapper.runner.find_text_on_screen", return_value=ocr_result
        )

        from mapper.runner import locate_element

        result = locate_element(_make_mock_graph(), "node-ocr", skill_id="s1")

        assert result.method == "ocr"
        assert result.confidence == pytest.approx(0.88)
```

Note: `test_all_visual_fail_position_fallback` and `test_nothing_available_raises_error` don't mock `core.yoloe` directly — they set `load_snippet` to `None` so Stage 1 is skipped entirely. These tests need NO changes.

4. Remove `core.yoloe` from any remaining import mocks in `test_cascade.py` if present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_cascade.py -v`

Expected: FAIL — mocks point at new imports

- [ ] **Step 3: Update mapper/runner.py Stage 1**

Replace the Stage 1 block (lines ~114-148) in `locate_element()`:

Old:
```python
    try:
        from core.capture import load_snippet
        from core.yoloe import find_element_locate
        # ...
```

New:
```python
    # ------------------------------------------------------------------
    # Stage 1: OmniParser detect + CLIP match
    # Detect all UI elements, find best CLIP match to saved snippet.
    # ------------------------------------------------------------------
    try:
        from core.capture import load_snippet
        from core.detection import get_detector

        snippet = load_snippet(skill_id, node_id[:12])
        if snippet is not None:
            screen = screenshot_full()
            cfg = get_config()
            detector = get_detector()
            result = detector.detect_and_match(
                screen, snippet, hint_x or 0, hint_y or 0,
                match_threshold=cfg.get("detection", {}).get("match_threshold", 0.7),
                search_radius=cfg.get("detection", {}).get("search_radius", 400),
            )
            if result is not None:
                logger.info(
                    "Located [%s] via OmniParser at (%d, %d) conf=%.2f",
                    node_id[:8],
                    result.point.x,
                    result.point.y,
                    result.confidence,
                )
                return result
            logger.debug("OmniParser found no match for [%s]", node_id[:8])
        else:
            logger.debug("No snippet on disk for [%s], skipping Stage 1", node_id[:8])
    except ImportError:
        logger.debug("Detection module not available, skipping Stage 1")
    except Exception as e:
        logger.debug("OmniParser locate error: %s", e)
```

Also update the module docstring (line 8) to say "OmniParser" instead of "YOLO-E".

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/test_cascade.py -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add mapper/runner.py tests/test_cascade.py
git commit -m "feat(runner): replace YOLOE Stage 1 with OmniParser detect+match"
```

---

### Task 8: Update main.py YOLOE Touchpoints

**Files:**
- Modify: `main.py` (3 YOLOE touchpoints)

- [ ] **Step 1: Replace YOLOE init (line ~352-357)**

Remove:
```python
    try:
        from core.yoloe import init_text_embeddings
        init_text_embeddings()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("YOLOE text embeddings not available: %s", e)
```

Replace with:
```python
    # Pre-load OmniParser and Florence-2 for fast detection during recording
    try:
        from core.detection import get_detector
        get_detector()  # triggers lazy model load
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("Detection models not available: %s", e)

    try:
        from core.florence import load_model as load_florence
        load_florence()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("Florence-2 not available: %s", e)
```

- [ ] **Step 2: Replace _try_refine_bbox YOLOE import (line ~108-114)**

Change the `_try_refine_bbox()` function to use OmniParser detection overlap instead of YOLOE's `infer_bbox_at_point` and `refine_bbox`. Keep the existing function signature unchanged — `matched_candidate` is already a parameter at line 86.

Remove:
```python
    try:
        from core.capture import screenshot_full
        from core.types import Rect
        from core.yoloe import infer_bbox_at_point, refine_bbox
    except (ImportError, OSError) as e:
        logger.debug("YOLOE not available, skipping bbox refinement: %s", e)
        return None
```

Replace with:
```python
    try:
        from core.capture import screenshot_full
        from core.detection import get_detector
        from core.types import Rect
    except (ImportError, OSError) as e:
        logger.debug("Detection module not available, skipping bbox refinement: %s", e)
        return None
```

Then replace the YOLOE refinement logic. Instead of `refine_bbox()` / `infer_bbox_at_point()`, detect all elements and find the detection that best overlaps the user's click point or drawn rect:

```python
    try:
        screen = screenshot_full()
    except Exception as e:
        logger.warning("Could not capture screen for refinement: %s", e)
        return None

    # Detect all elements on screen
    try:
        detector = get_detector()
        candidates = detector.detect(screen)
    except Exception as e:
        logger.debug("Detection failed during refinement: %s", e)
        return None

    if not candidates:
        return None

    # Check if smart_detect already has a bbox for point clicks
    if w == 0 and h == 0:
        if matched_candidate and "rect" in matched_candidate:
            r = matched_candidate["rect"]
            if r.get("w", 0) > 0 and r.get("h", 0) > 0:
                logger.info("Using smart_detect bbox for point click")
                return (r["x"], r["y"], r["w"], r["h"])

    # Find best overlapping detection
    best_overlap = 0.0
    best_rect = None

    for c in candidates:
        r = c["rect"]
        if w > 0 and h > 0:
            # Drawn bbox — compute IoU
            ix1 = max(x, r["x"])
            iy1 = max(y, r["y"])
            ix2 = min(x + w, r["x"] + r["w"])
            iy2 = min(y + h, r["y"] + r["h"])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = w * h + r["w"] * r["h"] - inter
                iou = inter / union if union > 0 else 0
                if iou > best_overlap:
                    best_overlap = iou
                    best_rect = r
        else:
            # Point click — find detection containing (x, y)
            if (r["x"] <= x <= r["x"] + r["w"]
                    and r["y"] <= y <= r["y"] + r["h"]):
                area = r["w"] * r["h"]
                # Prefer smallest containing bbox
                if best_rect is None or area < best_rect["w"] * best_rect["h"]:
                    best_overlap = 1.0
                    best_rect = r

    if best_rect is None:
        logger.debug("No detection overlaps the selection")
        return None
```

Then keep the existing review_mode logic (auto vs review dialog), but use `best_rect` instead of `yoloe_match.bbox`:

```python
    refined = Rect(best_rect["x"], best_rect["y"], best_rect["w"], best_rect["h"])

    if review_mode == "auto":
        logger.info(
            "Auto-refined bbox: (%d,%d) %dx%d → (%d,%d) %dx%d",
            x, y, w, h, refined.x, refined.y, refined.w, refined.h,
        )
        return (refined.x, refined.y, refined.w, refined.h)

    # review_mode == "review" — show the RefineDialog (existing code)
    ...
```

- [ ] **Step 3: Update _trigger_smart_detect call (no change needed)**

Verify that `_trigger_smart_detect()` at line ~187-212 doesn't pass `use_yoloe` to `detect_ui_elements_async()`. If the current code passes keyword args, remove them. The new `detect_ui_elements_async` takes only `(screenshot, callback)`.

Check line 211:
```python
    detect_ui_elements_async(screenshot, _on_results)
```
This is already correct — no `use_yoloe` kwarg. No change needed.

- [ ] **Step 4: Update the Florence-2 context in click handler (line ~271-276)**

In `on_element_clicked()` inside `cmd_record()`, update the VLM labeling section to include Florence-2 context in the `context_prompt`:

Find the `analyze_crop_array` call around line 274:
```python
                    vision_result = analyze_crop_array(
                        crop, "Identify this UI element type and label"
                    )
```

Replace with:
```python
                    florence_label = candidate.get("florence_caption", "")
                    if florence_label:
                        context_prompt = (
                            f'A fast vision model identified this element as: '
                            f'"{florence_label}". Confirm or correct the '
                            f'identification. Return JSON: '
                            f'{{"element_type": ..., "label_guess": ..., '
                            f'"confidence": 0.0-1.0, "ocr_text": ...}}'
                        )
                    else:
                        context_prompt = "Identify this UI element type and label"

                    vision_result = analyze_crop_array(crop, context_prompt)
```

- [ ] **Step 5: Add tests for _try_refine_bbox OmniParser logic**

Create or update tests in `tests/test_main_refine.py` (or inline in an existing test module) for the new `_try_refine_bbox` logic. The key scenarios:

```python
"""Tests for _try_refine_bbox OmniParser overlap logic in main.py."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest


class TestTryRefineBbox:
    """Tests for bbox refinement using OmniParser detections."""

    @patch("core.detection.get_detector")
    @patch("core.capture.screenshot_full")
    def test_point_click_uses_matched_candidate(
        self, mock_screenshot, mock_get_detector
    ):
        """Point click with matched_candidate returns its rect directly."""
        mock_screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"rect": {"x": 100, "y": 200, "w": 80, "h": 30}}
        ]
        mock_get_detector.return_value = mock_detector

        from main import _try_refine_bbox

        result = _try_refine_bbox(
            x=120, y=215, w=0, h=0,
            review_mode="auto",
            matched_candidate={"rect": {"x": 100, "y": 200, "w": 80, "h": 30}},
        )

        assert result == (100, 200, 80, 30)

    @patch("core.detection.get_detector")
    @patch("core.capture.screenshot_full")
    def test_point_click_finds_smallest_containing_detection(
        self, mock_screenshot, mock_get_detector
    ):
        """Point click finds smallest detection containing the click point."""
        mock_screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"rect": {"x": 50, "y": 50, "w": 500, "h": 500}},  # large
            {"rect": {"x": 100, "y": 200, "w": 80, "h": 30}},  # small, contains click
        ]
        mock_get_detector.return_value = mock_detector

        from main import _try_refine_bbox

        result = _try_refine_bbox(
            x=120, y=215, w=0, h=0,
            review_mode="auto",
            matched_candidate=None,
        )

        assert result == (100, 200, 80, 30)  # smallest containing

    @patch("core.detection.get_detector")
    @patch("core.capture.screenshot_full")
    def test_drawn_bbox_finds_best_iou_overlap(
        self, mock_screenshot, mock_get_detector
    ):
        """Drawn bbox finds detection with highest IoU overlap."""
        mock_screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"rect": {"x": 100, "y": 200, "w": 80, "h": 30}},  # good overlap
            {"rect": {"x": 500, "y": 500, "w": 80, "h": 30}},  # no overlap
        ]
        mock_get_detector.return_value = mock_detector

        from main import _try_refine_bbox

        result = _try_refine_bbox(
            x=95, y=195, w=90, h=40,
            review_mode="auto",
            matched_candidate=None,
        )

        assert result == (100, 200, 80, 30)

    @patch("core.detection.get_detector")
    @patch("core.capture.screenshot_full")
    def test_no_overlapping_detection_returns_none(
        self, mock_screenshot, mock_get_detector
    ):
        """When no detection overlaps selection, returns None."""
        mock_screenshot.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"rect": {"x": 800, "y": 800, "w": 50, "h": 50}},
        ]
        mock_get_detector.return_value = mock_detector

        from main import _try_refine_bbox

        result = _try_refine_bbox(
            x=100, y=100, w=0, h=0,
            review_mode="auto",
            matched_candidate=None,
        )

        assert result is None
```

- [ ] **Step 6: Run all tests to verify nothing broke**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/ -v --tb=short`

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_main_refine.py
git commit -m "feat(main): replace YOLOE touchpoints with OmniParser + Florence-2"
```

---

### Task 9: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -m pytest tests/ -v --tb=short`

Expected: All tests PASS (new + existing)

- [ ] **Step 2: Verify no remaining YOLOE imports in active code**

Run: `grep -rn "from core.yoloe" recorder/ mapper/ main.py core/detection.py core/omniparser.py core/florence.py`

Expected: No output (zero matches). YOLOE imports should only remain in `core/yoloe.py` itself and old test files.

- [ ] **Step 3: Verify config loads correctly**

Run: `cd /c/Users/persi/Documents/OpenClaw_Screen_Driver/.claude/worktrees/nifty-ritchie && .venv/Scripts/python -c "from core.config import load_config; c = load_config(); print(c['models']['detector']); print(c['detection']['match_threshold']); print(c['paths']['model_cache'])"`

Expected:
```
omniparser
0.7
~/.cache/ocsd/models
```

- [ ] **Step 4: Commit any remaining fixes and tag**

```bash
git add -A
git commit -m "chore: final verification pass for OmniParser integration"
```
