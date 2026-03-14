"""Tests for recorder.smart_detect — smart UI element detection."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from recorder.smart_detect import (
    detect_ui_elements,
    detect_ui_elements_async,
)


@pytest.fixture
def fake_screenshot() -> np.ndarray:
    """Returns a small BGR image for testing."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_candidates() -> list[dict]:
    """Sample detection candidates for testing."""
    return [
        {
            "rect": {"x": 10, "y": 20, "w": 80, "h": 30},
            "type_guess": "button",
            "label_guess": "Submit",
            "confidence": 0.92,
            "ocr_text": "Submit",
        },
        {
            "rect": {"x": 10, "y": 60, "w": 150, "h": 25},
            "type_guess": "textbox",
            "label_guess": "Username",
            "confidence": 0.85,
            "ocr_text": "",
        },
    ]


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


class TestDetectUIElementsAsync:
    """Tests for the async (threaded) detection wrapper."""

    @patch("recorder.smart_detect.detect_ui_elements")
    def test_async_calls_callback_with_results(
        self, mock_detect: MagicMock, fake_screenshot: np.ndarray, sample_candidates: list[dict]
    ) -> None:
        """Async wrapper calls callback with detection results."""
        mock_detect.return_value = sample_candidates
        callback = MagicMock()

        thread = detect_ui_elements_async(fake_screenshot, callback)
        thread.join(timeout=5)

        callback.assert_called_once_with(sample_candidates)

    @patch("recorder.smart_detect.detect_ui_elements")
    def test_async_calls_callback_empty_on_error(
        self, mock_detect: MagicMock, fake_screenshot: np.ndarray
    ) -> None:
        """Async wrapper calls callback with [] on exception."""
        mock_detect.side_effect = RuntimeError("VLM crashed")
        callback = MagicMock()

        thread = detect_ui_elements_async(fake_screenshot, callback)
        thread.join(timeout=5)

        callback.assert_called_once_with([])

    @patch("recorder.smart_detect.detect_ui_elements")
    def test_async_returns_daemon_thread(
        self, mock_detect: MagicMock, fake_screenshot: np.ndarray
    ) -> None:
        """Returned thread is a daemon thread."""
        mock_detect.return_value = []
        callback = MagicMock()

        thread = detect_ui_elements_async(fake_screenshot, callback)
        assert thread.daemon is True
        thread.join(timeout=5)


class TestDetectViaYOLOE:
    """Tests for the YOLOE detection wrapper in smart_detect."""

    def test_import_error_returns_empty(self, fake_screenshot: np.ndarray) -> None:
        """ImportError from core.yoloe returns empty list."""
        from recorder.smart_detect import _detect_via_yoloe

        with patch.dict("sys.modules", {"core.yoloe": None}):
            result = _detect_via_yoloe(fake_screenshot)
            assert result == []

    def test_exception_returns_empty(self, fake_screenshot: np.ndarray) -> None:
        """Runtime exception from YOLOE returns empty list."""
        from recorder.smart_detect import _detect_via_yoloe

        mock_module = MagicMock()
        mock_module.detect_all_elements.side_effect = RuntimeError("CUDA OOM")
        with patch.dict("sys.modules", {"core.yoloe": mock_module}):
            result = _detect_via_yoloe(fake_screenshot)
            assert result == []


class TestDetectViaOCR:
    """Tests for the OCR detection backend."""

    def test_ocr_import_error_returns_empty(self, fake_screenshot: np.ndarray) -> None:
        """Missing pytesseract returns empty list."""
        from recorder.smart_detect import _detect_via_ocr

        with patch.dict("sys.modules", {"pytesseract": None}):
            result = _detect_via_ocr(fake_screenshot)
            assert result == []

    def test_ocr_filters_low_confidence(self, fake_screenshot: np.ndarray) -> None:
        """OCR skips text with confidence below 40."""
        from recorder.smart_detect import _detect_via_ocr

        mock_pytesseract = MagicMock()
        mock_pytesseract.Output.DICT = "dict"
        mock_pytesseract.image_to_data.return_value = {
            "text": ["Good", "Bad", ""],
            "conf": [80, 20, -1],
            "left": [10, 50, 0],
            "top": [10, 50, 0],
            "width": [40, 30, 0],
            "height": [15, 12, 0],
        }

        mock_cv2 = MagicMock()
        mock_cv2.cvtColor.return_value = fake_screenshot[:, :, 0]
        mock_cv2.COLOR_BGR2GRAY = 6

        with patch.dict(
            "sys.modules",
            {"pytesseract": mock_pytesseract, "cv2": mock_cv2},
        ):
            result = _detect_via_ocr(fake_screenshot)

        assert len(result) == 1
        assert result[0]["ocr_text"] == "Good"
        assert result[0]["confidence"] == 0.8
