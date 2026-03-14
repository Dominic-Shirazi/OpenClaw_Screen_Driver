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


class TestOmniParserPipeline:
    """Tests for the OmniParser-first detection pipeline."""

    @patch("recorder.smart_detect._enrich_with_florence")
    @patch("recorder.smart_detect._detect_via_omniparser")
    def test_omniparser_returns_results(
        self,
        mock_omni: MagicMock,
        mock_enrich: MagicMock,
        fake_screenshot: np.ndarray,
    ) -> None:
        """When OmniParser succeeds, its results are returned (after Florence enrichment)."""
        candidates = [
            {
                "rect": {"x": 10, "y": 20, "w": 80, "h": 30},
                "type_guess": "button",
                "label_guess": "button",
                "confidence": 0.15,
                "ocr_text": None,
            },
        ]
        mock_omni.return_value = candidates
        mock_enrich.side_effect = lambda c, s: c  # pass through unchanged

        results = detect_ui_elements(fake_screenshot)

        assert len(results) == 1
        assert results[0]["type_guess"] == "button"
        mock_omni.assert_called_once()

    @patch("recorder.smart_detect._detect_via_ocr")
    @patch("recorder.smart_detect._enrich_with_florence")
    @patch("recorder.smart_detect._detect_via_omniparser")
    def test_omniparser_empty_falls_back_to_ocr(
        self,
        mock_omni: MagicMock,
        mock_enrich: MagicMock,
        mock_ocr: MagicMock,
        fake_screenshot: np.ndarray,
    ) -> None:
        """When OmniParser returns empty, OCR fallback is used."""
        mock_omni.return_value = []
        mock_enrich.side_effect = lambda c, s: c  # pass through unchanged
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
        mock_omni.assert_called_once()
        mock_ocr.assert_called_once()


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


class TestEnrichWithFlorence:
    """Tests for Florence-2 caption enrichment."""

    def test_enriches_label_guess(self, fake_screenshot: np.ndarray) -> None:
        """Florence-2 captions update label_guess and add florence_caption."""
        from recorder.smart_detect import _enrich_with_florence

        candidates = [
            {"rect": {"x": 10, "y": 20, "w": 80, "h": 30}, "label_guess": "", "confidence": 0.9},
        ]

        mock_cv2 = MagicMock()
        mock_cv2.cvtColor.return_value = np.zeros((30, 80, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 4

        with patch("core.florence.caption_batch", return_value=["Submit button"]), \
             patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = _enrich_with_florence(candidates, fake_screenshot)

        assert result[0]["label_guess"] == "Submit button"
        assert result[0]["florence_caption"] == "Submit button"

    def test_handles_import_error(self, fake_screenshot: np.ndarray) -> None:
        """ImportError from core.florence leaves candidates unchanged."""
        from recorder.smart_detect import _enrich_with_florence

        candidates = [
            {"rect": {"x": 10, "y": 20, "w": 80, "h": 30}, "label_guess": "", "confidence": 0.9},
        ]

        with patch.dict("sys.modules", {"core.florence": None}):
            result = _enrich_with_florence(candidates, fake_screenshot)

        assert result[0]["label_guess"] == ""  # unchanged
