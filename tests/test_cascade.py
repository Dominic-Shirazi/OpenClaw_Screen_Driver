"""Tests for the locate cascade and related modules.

Covers YOLOE targeted finder, OCR region scoping, CLIP embedding
retrieval, snippet loading, and the full cascade in mapper.runner.

All external dependencies (ultralytics, torch, transformers, faiss,
pyautogui, tesseract, mss) are mocked — no GPU or model weights needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.types import ElementNotFoundError, LocateResult, Point, Rect


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _bgr_image(h: int = 100, w: int = 200, channels: int = 3) -> np.ndarray:
    """Creates a dummy BGR image filled with random pixel values."""
    return np.random.randint(0, 255, (h, w, channels), dtype=np.uint8)


def _full_screen_image() -> np.ndarray:
    """Creates a 1080p dummy BGR screenshot."""
    return _bgr_image(h=1080, w=1920)


# -----------------------------------------------------------------------
# 1. YOLOE targeted finder  (core/yoloe.py)
# -----------------------------------------------------------------------


class TestYOLOEFindElement:
    """Tests for core.yoloe.find_element and find_element_locate."""

    @pytest.fixture(autouse=True)
    def _reset_model(self) -> None:
        """Reset module-level singleton before each test."""
        import core.yoloe as yoloe_mod

        yoloe_mod._model = None
        yoloe_mod._model_path = ""

    def _make_fake_result(
        self, xyxy: list[int], conf: float
    ) -> MagicMock:
        """Builds a mock YOLOE result with one detection box."""
        box_mock = MagicMock()
        box_mock.conf = MagicMock()
        box_mock.conf.__getitem__ = lambda _self, i: conf
        box_mock.conf.__len__ = lambda _self: 1

        tensor = MagicMock()
        tensor.cpu.return_value.numpy.return_value.astype.return_value = np.array(
            xyxy, dtype=int
        )
        box_mock.xyxy = MagicMock()
        box_mock.xyxy.__getitem__ = lambda _self, i: tensor
        box_mock.__len__ = lambda _self: 1

        result = MagicMock()
        result.boxes = box_mock
        return result

    def test_find_element_passes_snippet_as_refer_image(
        self, mocker: Any
    ) -> None:
        """Verify snippet is passed as refer_image with visual_prompts."""
        fake_model = MagicMock()
        fake_model.predict.return_value = []

        mocker.patch("core.yoloe._load_model", return_value=fake_model)
        # Mock the ultralytics predictor import inside find_element
        fake_predictor = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "ultralytics": MagicMock(),
                "ultralytics.models": MagicMock(),
                "ultralytics.models.yolo": MagicMock(),
                "ultralytics.models.yolo.yoloe": MagicMock(
                    YOLOEVPSegPredictor=fake_predictor
                ),
            },
        )

        from core.yoloe import find_element

        snippet = _bgr_image(h=50, w=60)
        screen = _full_screen_image()
        find_element(snippet, screen)

        fake_model.predict.assert_called_once()
        call_kwargs = fake_model.predict.call_args
        assert call_kwargs.kwargs["refer_image"] is snippet
        assert "visual_prompts" in call_kwargs.kwargs
        vp = call_kwargs.kwargs["visual_prompts"]
        np.testing.assert_array_equal(vp["bboxes"], [[0, 0, 60, 50]])

    def test_find_element_crops_when_hint_provided(
        self, mocker: Any
    ) -> None:
        """With hint_x/hint_y the search region should be spatially cropped."""
        fake_model = MagicMock()
        fake_model.predict.return_value = []

        mocker.patch("core.yoloe._load_model", return_value=fake_model)
        mocker.patch.dict(
            "sys.modules",
            {
                "ultralytics": MagicMock(),
                "ultralytics.models": MagicMock(),
                "ultralytics.models.yolo": MagicMock(),
                "ultralytics.models.yolo.yoloe": MagicMock(
                    YOLOEVPSegPredictor=MagicMock()
                ),
            },
        )

        from core.yoloe import find_element

        screen = _full_screen_image()
        find_element(
            _bgr_image(50, 50),
            screen,
            hint_x=960,
            hint_y=540,
            search_radius=200,
        )

        # The first positional arg to predict should be the cropped region
        target_arg = fake_model.predict.call_args.args[0]
        assert target_arg.shape[0] == 400  # 200*2 radius
        assert target_arg.shape[1] == 400

    def test_find_element_offsets_back_to_full_screen(
        self, mocker: Any
    ) -> None:
        """Detections in a crop must be offset back to full-screen coords."""
        # Detection at (10, 20, 50, 60) inside the crop
        fake_result = self._make_fake_result([10, 20, 50, 60], 0.9)
        fake_model = MagicMock()
        fake_model.predict.return_value = [fake_result]

        mocker.patch("core.yoloe._load_model", return_value=fake_model)
        mocker.patch.dict(
            "sys.modules",
            {
                "ultralytics": MagicMock(),
                "ultralytics.models": MagicMock(),
                "ultralytics.models.yolo": MagicMock(),
                "ultralytics.models.yolo.yoloe": MagicMock(
                    YOLOEVPSegPredictor=MagicMock()
                ),
            },
        )

        from core.yoloe import find_element

        screen = _full_screen_image()
        matches = find_element(
            _bgr_image(50, 50),
            screen,
            hint_x=500,
            hint_y=400,
            search_radius=200,
        )

        assert len(matches) == 1
        m = matches[0]
        # Offset should be (500-200, 400-200) = (300, 200)
        assert m.bbox.x == 10 + 300
        assert m.bbox.y == 20 + 200

    def test_find_element_locate_returns_locate_result(
        self, mocker: Any
    ) -> None:
        """find_element_locate wraps the best match into a LocateResult."""
        fake_result = self._make_fake_result([100, 200, 150, 260], 0.85)
        fake_model = MagicMock()
        fake_model.predict.return_value = [fake_result]

        mocker.patch("core.yoloe._load_model", return_value=fake_model)
        mocker.patch.dict(
            "sys.modules",
            {
                "ultralytics": MagicMock(),
                "ultralytics.models": MagicMock(),
                "ultralytics.models.yolo": MagicMock(),
                "ultralytics.models.yolo.yoloe": MagicMock(
                    YOLOEVPSegPredictor=MagicMock()
                ),
            },
        )

        from core.yoloe import find_element_locate

        result = find_element_locate(
            _bgr_image(50, 50), _full_screen_image()
        )

        assert result is not None
        assert isinstance(result, LocateResult)
        assert result.method == "yoloe"
        assert result.confidence == pytest.approx(0.85)

    def test_find_element_locate_returns_none_on_no_match(
        self, mocker: Any
    ) -> None:
        """find_element_locate returns None when nothing is detected."""
        fake_model = MagicMock()
        fake_model.predict.return_value = []

        mocker.patch("core.yoloe._load_model", return_value=fake_model)
        mocker.patch.dict(
            "sys.modules",
            {
                "ultralytics": MagicMock(),
                "ultralytics.models": MagicMock(),
                "ultralytics.models.yolo": MagicMock(),
                "ultralytics.models.yolo.yoloe": MagicMock(
                    YOLOEVPSegPredictor=MagicMock()
                ),
            },
        )

        from core.yoloe import find_element_locate

        result = find_element_locate(
            _bgr_image(50, 50), _full_screen_image()
        )
        assert result is None


# -----------------------------------------------------------------------
# 2. OCR region scoping  (core/ocr.py)
# -----------------------------------------------------------------------


class TestOCRRegionScoping:
    """Tests for core.ocr.find_text_in_region and find_text_on_screen."""

    @pytest.fixture(autouse=True)
    def _patch_tesseract(self, mocker: Any) -> None:
        """Disable the real Tesseract binary check for every test."""
        import core.ocr  # noqa: F401 — force module load before patching
        mocker.patch("core.ocr._ensure_tesseract_installed")

    def test_find_text_in_region_searches_within_region(
        self, mocker: Any
    ) -> None:
        """find_text_in_region captures only the given region."""
        region_img = _bgr_image(200, 300)
        mock_screenshot_region = mocker.patch(
            "core.ocr.capture.screenshot_region", return_value=region_img
        )
        mocker.patch(
            "core.ocr.ocr_with_boxes",
            return_value=[
                {
                    "text": "Submit",
                    "rect": Rect(x=10, y=20, w=60, h=20),
                    "confidence": 0.92,
                }
            ],
        )

        from core.ocr import find_text_in_region

        result = find_text_in_region("submit", 100, 200, 300, 200)

        mock_screenshot_region.assert_called_once_with(100, 200, 300, 200)
        assert result is not None
        assert result.method == "ocr"

    def test_find_text_in_region_offsets_to_full_screen(
        self, mocker: Any
    ) -> None:
        """Returned coordinates must be offset to full-screen space."""
        mocker.patch(
            "core.ocr.capture.screenshot_region", return_value=_bgr_image()
        )
        mocker.patch(
            "core.ocr.ocr_with_boxes",
            return_value=[
                {
                    "text": "Login",
                    "rect": Rect(x=10, y=15, w=50, h=18),
                    "confidence": 0.88,
                }
            ],
        )

        from core.ocr import find_text_in_region

        result = find_text_in_region("login", 400, 300, 200, 200)

        assert result is not None
        assert result.rect is not None
        # rect should be offset by the region origin
        assert result.rect.x == 10 + 400
        assert result.rect.y == 15 + 300
        # center should also be offset
        assert result.point.x == result.rect.center.x
        assert result.point.y == result.rect.center.y

    def test_find_text_on_screen_with_hints_scopes_region(
        self, mocker: Any
    ) -> None:
        """When hints are given, search first scopes to a region."""
        mocker.patch("pyautogui.size", return_value=(1920, 1080))
        mock_find_region = mocker.patch(
            "core.ocr.find_text_in_region",
            return_value=LocateResult(
                point=Point(500, 400),
                method="ocr",
                confidence=0.9,
                rect=Rect(x=480, y=390, w=40, h=20),
            ),
        )

        from core.ocr import find_text_on_screen

        result = find_text_on_screen(
            "Login", hint_x=500, hint_y=400, search_radius=400
        )

        assert result is not None
        assert result.point == Point(500, 400)
        mock_find_region.assert_called_once()

    def test_find_text_on_screen_no_hints_falls_back_to_full(
        self, mocker: Any
    ) -> None:
        """Without hints, OCR scans the full screen."""
        mocker.patch(
            "core.ocr.capture.screenshot_full",
            return_value=_full_screen_image(),
        )
        mocker.patch(
            "core.ocr.ocr_with_boxes",
            return_value=[
                {
                    "text": "OK",
                    "rect": Rect(x=800, y=600, w=30, h=18),
                    "confidence": 0.95,
                }
            ],
        )

        from core.ocr import find_text_on_screen

        result = find_text_on_screen("ok")

        assert result is not None
        assert result.method == "ocr"
        # No offset applied when scanning full screen
        assert result.rect is not None
        assert result.rect.x == 800

    def test_find_text_in_region_returns_none_on_no_match(
        self, mocker: Any
    ) -> None:
        """Returns None when the target text isn't found in the region."""
        mocker.patch(
            "core.ocr.capture.screenshot_region", return_value=_bgr_image()
        )
        mocker.patch("core.ocr.ocr_with_boxes", return_value=[])

        from core.ocr import find_text_in_region

        result = find_text_in_region("nonexistent", 0, 0, 200, 200)
        assert result is None


# -----------------------------------------------------------------------
# 3. CLIP embedding direction  (core/embeddings.py)
# -----------------------------------------------------------------------


class TestCLIPEmbeddingRetrieval:
    """Tests for core.embeddings.get_embedding_by_id."""

    @pytest.fixture(autouse=True)
    def _reset_globals(self) -> None:
        """Reset module-level state so each test starts clean."""
        import core.embeddings as emb_mod

        emb_mod._clip_model = None
        emb_mod._clip_processor = None
        emb_mod._faiss_index = None
        emb_mod._metadata_map = {}

    def test_get_embedding_by_id_returns_correct_vector(
        self, mocker: Any
    ) -> None:
        """Should reconstruct and return the saved embedding vector."""
        import core.embeddings as emb_mod

        # Create a fake FAISS index that stores one vector
        expected_vec = np.random.randn(512).astype("float32")
        expected_vec /= np.linalg.norm(expected_vec)

        fake_index = MagicMock()
        fake_index.ntotal = 1

        def fake_reconstruct(idx: int, buf: np.ndarray) -> None:
            buf[:] = expected_vec

        fake_index.reconstruct = fake_reconstruct

        emb_mod._faiss_index = fake_index
        emb_mod._metadata_map = {0: {"element_id": "abc123", "x_pct": 0.5, "y_pct": 0.5}}

        # Patch load_index to be a no-op (state already set)
        mocker.patch("core.embeddings.load_index", return_value=fake_index)

        result = emb_mod.get_embedding_by_id("abc123")

        assert result is not None
        assert result.shape == (1, 512)
        np.testing.assert_allclose(result[0], expected_vec, atol=1e-6)

    def test_get_embedding_by_id_returns_none_for_missing(
        self, mocker: Any
    ) -> None:
        """Returns None when the element_id is not in the metadata map."""
        import core.embeddings as emb_mod

        fake_index = MagicMock()
        fake_index.ntotal = 0
        emb_mod._faiss_index = fake_index
        emb_mod._metadata_map = {}

        mocker.patch("core.embeddings.load_index", return_value=fake_index)

        result = emb_mod.get_embedding_by_id("not_there")
        assert result is None


# -----------------------------------------------------------------------
# 4. Snippet loading  (core/capture.py)
# -----------------------------------------------------------------------


class TestSnippetLoading:
    """Tests for core.capture.load_snippet."""

    def test_load_snippet_returns_bgr_image(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """When the snippet file exists, returns a BGR numpy array."""
        import cv2

        # Write a real tiny PNG to the tmp dir
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        png_path = skill_dir / "elem12345678.png"
        dummy = _bgr_image(30, 40)
        cv2.imwrite(str(png_path), dummy)

        mocker.patch(
            "core.capture.get_config",
            return_value={"paths": {"snippets_dir": str(tmp_path)}},
        )

        from core.capture import load_snippet

        result = load_snippet("my_skill", "elem12345678")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 3  # BGR

    def test_load_snippet_returns_none_when_missing(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Returns None when the snippet file doesn't exist."""
        mocker.patch(
            "core.capture.get_config",
            return_value={"paths": {"snippets_dir": str(tmp_path)}},
        )

        from core.capture import load_snippet

        result = load_snippet("no_skill", "no_element")
        assert result is None


# -----------------------------------------------------------------------
# 5. Full cascade  (mapper/runner.py — locate_element)
# -----------------------------------------------------------------------


def _make_mock_graph(
    node_data: dict[str, Any] | None = None,
) -> MagicMock:
    """Creates a mock OCSDGraph with one node that returns *node_data*."""
    data = {
        "element_type": "button",
        "label": "Submit",
        "ocr_text": "Submit",
        "relative_position": {"x_pct": 0.5, "y_pct": 0.5},
    }
    if node_data:
        data.update(node_data)

    graph = MagicMock()
    graph.get_node.return_value = data
    return graph


class TestFullCascade:
    """Tests for mapper.runner.locate_element cascade behaviour."""

    @pytest.fixture(autouse=True)
    def _patch_pyautogui_size(self, mocker: Any) -> None:
        """Mock pyautogui.size() to 1920x1080 for all cascade tests."""
        mocker.patch("pyautogui.size", return_value=(1920, 1080))
        mocker.patch(
            "mapper.runner.screenshot_full",
            return_value=_full_screen_image(),
        )

    # -- Stage 1 success: YOLOE finds it, no later stages called ----------

    def test_yoloe_succeeds_returns_early(self, mocker: Any) -> None:
        """When YOLOE matches, cascade returns immediately without OCR/VLM."""
        yoloe_result = LocateResult(
            point=Point(960, 540), method="yoloe", confidence=0.9
        )
        mocker.patch(
            "core.capture.load_snippet", return_value=_bgr_image(50, 50)
        )
        mocker.patch(
            "core.yoloe.find_element_locate", return_value=yoloe_result
        )
        mock_ocr = mocker.patch("core.ocr.find_text_on_screen")

        from mapper.runner import locate_element

        result = locate_element(_make_mock_graph(), "node-abc", skill_id="s1")

        assert result.method == "yoloe"
        assert result.confidence == pytest.approx(0.9)
        # OCR should NOT have been called
        mock_ocr.assert_not_called()

    # -- Stage 1 fails, Stage 2 (CLIP) succeeds ---------------------------

    def test_yoloe_fails_clip_succeeds(self, mocker: Any) -> None:
        """When YOLOE finds nothing but CLIP score is high, returns clip."""
        # YOLOE returns None
        mocker.patch(
            "core.capture.load_snippet", return_value=_bgr_image(50, 50)
        )
        mocker.patch("core.yoloe.find_element_locate", return_value=None)

        # CLIP returns high similarity
        saved_emb = np.random.randn(1, 512).astype("float32")
        saved_emb /= np.linalg.norm(saved_emb)
        current_emb = saved_emb.copy()  # identical → score ~1.0

        mocker.patch(
            "core.embeddings.get_embedding_by_id", return_value=saved_emb
        )
        mocker.patch(
            "core.embeddings.generate_embedding", return_value=current_emb
        )
        mocker.patch(
            "core.capture.screenshot_region", return_value=_bgr_image(600, 600)
        )
        # Mock cv2.cvtColor used inside the CLIP branch
        mocker.patch("cv2.cvtColor", return_value=_bgr_image(600, 600))

        mock_ocr = mocker.patch("core.ocr.find_text_on_screen")

        from mapper.runner import locate_element

        result = locate_element(_make_mock_graph(), "node-xyz", skill_id="s1")

        assert result.method == "clip"
        assert result.confidence > 0.5
        mock_ocr.assert_not_called()

    # -- Stages 1+2 fail, Stage 3 (OCR) succeeds --------------------------

    def test_yoloe_clip_fail_ocr_succeeds(self, mocker: Any) -> None:
        """When YOLOE and CLIP both fail, OCR takes over."""
        # YOLOE — no snippet on disk
        mocker.patch("core.capture.load_snippet", return_value=None)

        # CLIP — no saved embedding
        mocker.patch(
            "core.embeddings.get_embedding_by_id", return_value=None
        )

        # OCR succeeds — patch on the runner module since it's imported
        # at the top of mapper/runner.py (from core.ocr import find_text_on_screen)
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

    # -- Stages 1-3 fail, Stage 5 position fallback ------------------------

    def test_all_visual_fail_position_fallback(self, mocker: Any) -> None:
        """When all visual stages fail, returns position fallback."""
        mocker.patch("core.capture.load_snippet", return_value=None)
        mocker.patch(
            "core.embeddings.get_embedding_by_id", return_value=None
        )
        mocker.patch("core.ocr.find_text_on_screen", return_value=None)

        # VLM also fails (ImportError path)
        mocker.patch.dict("sys.modules", {"core.vision": None})

        from mapper.runner import locate_element

        graph = _make_mock_graph({"ocr_text": None, "label": ""})
        result = locate_element(graph, "node-fallback", skill_id="s1")

        assert result.method == "direct"
        assert result.confidence == pytest.approx(0.3)
        assert result.point == Point(960, 540)  # 0.5 * 1920, 0.5 * 1080

    # -- Nothing available at all → ElementNotFoundError -------------------

    def test_nothing_available_raises_error(self, mocker: Any) -> None:
        """When there's no position data and all stages fail, raises."""
        mocker.patch("core.capture.load_snippet", return_value=None)
        mocker.patch(
            "core.embeddings.get_embedding_by_id", return_value=None
        )
        mocker.patch("core.ocr.find_text_on_screen", return_value=None)
        mocker.patch.dict("sys.modules", {"core.vision": None})

        from mapper.runner import locate_element

        # Node with NO position data at all
        graph = _make_mock_graph(
            {
                "ocr_text": None,
                "label": "",
                "relative_position": {},
            }
        )

        with pytest.raises(ElementNotFoundError):
            locate_element(graph, "node-dead", skill_id="s1")
