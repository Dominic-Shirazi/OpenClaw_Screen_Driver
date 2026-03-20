"""Tests for locate_element_from_step() adapter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.types import ElementNotFoundError, LocateResult, Point

SAMPLE_STEP = {
    "step_index": 0,
    "node_id": "abc123def456",
    "label": "Submit",
    "element_type": "button",
    "action": "click",
    "snippet_path": "snippets/abc123def456.png",
    "embedding_path": "embeddings/abc123def456.npy",
    "anchors": {
        "visual_match": "snippets/abc123def456.png",
        "ocr_text": "Submit",
        "position_pct": {"x_pct": 0.5, "y_pct": 0.8},
        "region_hint": "bottom_center",
    },
}

ROUTINE_DIR = Path("/tmp/test_routine")


class TestLocateFromStepOcrHit:
    """OCR stage returns a match -> result.method == 'ocr'."""

    @patch("core.locate.find_text_on_screen")
    @patch("core.locate.pyautogui")
    def test_locate_from_step_ocr_hit(
        self, mock_pag: MagicMock, mock_ocr: MagicMock,
    ) -> None:
        from core.locate import locate_element_from_step

        mock_pag.size.return_value = (1920, 1080)
        mock_ocr.return_value = LocateResult(
            point=Point(960, 864),
            method="ocr",
            confidence=0.92,
        )

        # Skip stages 1 & 2 by removing snippet/embedding paths
        step = {**SAMPLE_STEP, "snippet_path": None, "embedding_path": None}
        result = locate_element_from_step(step, ROUTINE_DIR)

        assert result.method == "ocr"
        assert result.confidence == 0.92


class TestLocateFromStepPositionFallback:
    """All stages fail except position -> result.confidence == 0.3, method == 'direct'."""

    @patch("core.locate.find_text_on_screen", return_value=None)
    @patch("core.locate.pyautogui")
    def test_locate_from_step_position_fallback(
        self, mock_pag: MagicMock, mock_ocr: MagicMock,
    ) -> None:
        from core.locate import locate_element_from_step

        mock_pag.size.return_value = (1920, 1080)

        # No snippet, no embedding, skip VLM -> only position
        step = {**SAMPLE_STEP, "snippet_path": None, "embedding_path": None}
        result = locate_element_from_step(step, ROUTINE_DIR, skip_vlm=True)

        assert result.confidence == 0.3
        assert result.method == "direct"


class TestLocateFromStepSkipPosition:
    """skip_position_fallback=True + all stages fail -> ElementNotFoundError."""

    @patch("core.locate.find_text_on_screen", return_value=None)
    @patch("core.locate.pyautogui")
    def test_locate_from_step_skip_position(
        self, mock_pag: MagicMock, mock_ocr: MagicMock,
    ) -> None:
        from core.locate import locate_element_from_step

        mock_pag.size.return_value = (1920, 1080)

        step = {**SAMPLE_STEP, "snippet_path": None, "embedding_path": None}
        with pytest.raises(ElementNotFoundError):
            locate_element_from_step(
                step, ROUTINE_DIR,
                skip_vlm=True,
                skip_position_fallback=True,
            )


class TestLocateFromStepSkipVlm:
    """skip_vlm=True -> first_pass_map_array is NOT called."""

    @patch("core.locate.find_text_on_screen", return_value=None)
    @patch("core.locate.pyautogui")
    def test_locate_from_step_skip_vlm(
        self, mock_pag: MagicMock, mock_ocr: MagicMock,
    ) -> None:
        from core.locate import locate_element_from_step

        mock_pag.size.return_value = (1920, 1080)

        step = {**SAMPLE_STEP, "snippet_path": None, "embedding_path": None}

        mock_vlm = MagicMock()
        mock_vision_module = MagicMock()
        mock_vision_module.first_pass_map_array = mock_vlm

        with patch.dict("sys.modules", {"core.vision": mock_vision_module}):
            # Should NOT call VLM, should fall through to position
            result = locate_element_from_step(step, ROUTINE_DIR, skip_vlm=True)
            mock_vlm.assert_not_called()

        assert result.method == "direct"


class TestLocateFromStepSnippetResolve:
    """Snippet path is resolved against routine_dir."""

    @patch("core.locate.pyautogui")
    def test_locate_from_step_snippet_resolve(
        self, mock_pag: MagicMock,
    ) -> None:
        from core.locate import locate_element_from_step

        mock_pag.size.return_value = (1920, 1080)

        with (
            patch("cv2.imread", return_value=None) as mock_imread,
            patch("core.locate.find_text_on_screen", return_value=None),
        ):
            # With snippet_path set but cv2.imread returns None -> skips stage 1
            # We just verify the path resolution
            result = locate_element_from_step(
                SAMPLE_STEP, ROUTINE_DIR, skip_vlm=True,
            )
            expected_path = str(ROUTINE_DIR / "snippets/abc123def456.png")
            mock_imread.assert_called_once_with(expected_path)
