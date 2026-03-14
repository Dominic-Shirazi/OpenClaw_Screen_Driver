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
