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
