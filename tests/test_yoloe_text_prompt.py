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
