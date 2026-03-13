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
def vlm_candidates() -> list[dict]:
    """Sample VLM detection results."""
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


class TestDetectUIElements:
    """Tests for the synchronous detect_ui_elements function."""

    @patch("recorder.smart_detect._detect_via_vlm")
    def test_vlm_returns_results(
        self, mock_vlm: MagicMock, fake_screenshot: np.ndarray, vlm_candidates: list[dict]
    ) -> None:
        """When VLM succeeds, its results are returned directly."""
        mock_vlm.return_value = vlm_candidates

        results = detect_ui_elements(fake_screenshot, use_vlm=True)

        assert len(results) == 2
        assert results[0]["type_guess"] == "button"
        assert results[1]["type_guess"] == "textbox"
        mock_vlm.assert_called_once_with(fake_screenshot)

    @patch("recorder.smart_detect._detect_via_ocr")
    @patch("recorder.smart_detect._detect_via_vlm")
    def test_vlm_empty_falls_back_to_ocr(
        self,
        mock_vlm: MagicMock,
        mock_ocr: MagicMock,
        fake_screenshot: np.ndarray,
    ) -> None:
        """When VLM returns empty, OCR fallback is used."""
        mock_vlm.return_value = []
        mock_ocr.return_value = [
            {
                "rect": {"x": 5, "y": 5, "w": 40, "h": 15},
                "type_guess": "unknown",
                "label_guess": "Login",
                "confidence": 0.7,
                "ocr_text": "Login",
            }
        ]

        results = detect_ui_elements(fake_screenshot, use_vlm=True)

        assert len(results) == 1
        assert results[0]["ocr_text"] == "Login"
        mock_vlm.assert_called_once()
        mock_ocr.assert_called_once()

    @patch("recorder.smart_detect._detect_via_ocr")
    def test_vlm_disabled_uses_ocr(
        self, mock_ocr: MagicMock, fake_screenshot: np.ndarray
    ) -> None:
        """When use_vlm=False, only OCR is used."""
        mock_ocr.return_value = [
            {
                "rect": {"x": 0, "y": 0, "w": 50, "h": 20},
                "type_guess": "unknown",
                "label_guess": "OK",
                "confidence": 0.8,
                "ocr_text": "OK",
            }
        ]

        results = detect_ui_elements(fake_screenshot, use_vlm=False)

        assert len(results) == 1
        mock_ocr.assert_called_once()

    @patch("recorder.smart_detect._detect_via_ocr")
    @patch("recorder.smart_detect._detect_via_vlm")
    def test_both_empty_returns_empty(
        self,
        mock_vlm: MagicMock,
        mock_ocr: MagicMock,
        fake_screenshot: np.ndarray,
    ) -> None:
        """When both backends return empty, result is empty."""
        mock_vlm.return_value = []
        mock_ocr.return_value = []

        results = detect_ui_elements(fake_screenshot)

        assert results == []


class TestDetectUIElementsAsync:
    """Tests for the async (threaded) detection wrapper."""

    @patch("recorder.smart_detect.detect_ui_elements")
    def test_async_calls_callback_with_results(
        self, mock_detect: MagicMock, fake_screenshot: np.ndarray, vlm_candidates: list[dict]
    ) -> None:
        """Async wrapper calls callback with detection results."""
        mock_detect.return_value = vlm_candidates
        callback = MagicMock()

        thread = detect_ui_elements_async(fake_screenshot, callback)
        thread.join(timeout=5)

        callback.assert_called_once_with(vlm_candidates)

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


class TestDetectViaVLM:
    """Tests for the VLM detection backend."""

    @patch("recorder.smart_detect.first_pass_map_array", create=True)
    def test_vlm_import_error_returns_empty(self, fake_screenshot: np.ndarray) -> None:
        """ImportError from core.vision returns empty list."""
        # We test this by directly calling _detect_via_vlm with the import mocked to fail
        from recorder.smart_detect import _detect_via_vlm

        with patch.dict("sys.modules", {"core.vision": None}):
            # The lazy import inside _detect_via_vlm catches ImportError
            result = _detect_via_vlm(fake_screenshot)
            assert result == []

    def test_vlm_exception_returns_empty(self, fake_screenshot: np.ndarray) -> None:
        """Runtime exception from VLM returns empty list."""
        from recorder.smart_detect import _detect_via_vlm

        with patch(
            "recorder.smart_detect.first_pass_map_array",
            create=True,
            side_effect=RuntimeError("model not loaded"),
        ):
            # The function catches the import internally, so we mock at module level
            pass

        # Test via the import path inside the function
        mock_module = MagicMock()
        mock_module.first_pass_map_array.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"core.vision": mock_module}):
            result = _detect_via_vlm(fake_screenshot)
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
