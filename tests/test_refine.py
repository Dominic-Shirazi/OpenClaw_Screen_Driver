"""Tests for YOLOE bbox refinement and the RefineDialog.

Covers:
- refine_bbox() with mocked find_element
- Coordinate translation (crop-relative -> screen-absolute)
- infer_bbox_at_point() with mocked results
- RefineDialog return values (mock QDialog.exec)
- RefineDialog construction doesn't crash with mock images
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.types import Point, Rect
from core.yoloe import YOLOEMatch, infer_bbox_at_point, refine_bbox


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_screen() -> np.ndarray:
    """A 1000x1600 BGR test screen (black)."""
    return np.zeros((1000, 1600, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# refine_bbox tests
# ---------------------------------------------------------------------------

class TestRefineBbox:
    """Tests for refine_bbox()."""

    @patch("core.yoloe.find_element")
    def test_returns_match_translated_to_screen(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """refine_bbox translates crop-relative coords to screen-absolute."""
        user_rect = Rect(x=400, y=300, w=100, h=80)

        # find_element will return a bbox relative to the crop
        # The crop starts at (400 - 30, 300 - 24) = (370, 276)
        crop_match = YOLOEMatch(
            bbox=Rect(x=20, y=15, w=110, h=90),
            confidence=0.85,
            center=Point(x=75, y=60),
        )
        mock_find.return_value = [crop_match]

        result = refine_bbox(fake_screen, user_rect, conf_threshold=0.25)

        assert result is not None
        assert result.confidence == 0.85
        # crop_x1 = max(0, 400 - 30) = 370
        # crop_y1 = max(0, 300 - 24) = 276
        assert result.bbox.x == 20 + 370
        assert result.bbox.y == 15 + 276
        assert result.bbox.w == 110
        assert result.bbox.h == 90

    @patch("core.yoloe.find_element")
    def test_returns_none_when_no_matches(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """refine_bbox returns None when find_element returns nothing."""
        mock_find.return_value = []
        user_rect = Rect(x=400, y=300, w=100, h=80)

        result = refine_bbox(fake_screen, user_rect)
        assert result is None

    @patch("core.yoloe.find_element")
    def test_center_is_computed_correctly(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """The returned center is the midpoint of the absolute bbox."""
        user_rect = Rect(x=500, y=400, w=200, h=100)
        crop_match = YOLOEMatch(
            bbox=Rect(x=10, y=10, w=200, h=100),
            confidence=0.90,
            center=Point(x=110, y=60),
        )
        mock_find.return_value = [crop_match]

        result = refine_bbox(fake_screen, user_rect)
        assert result is not None
        # crop_x1 = 500 - 60 = 440, crop_y1 = 400 - 30 = 370
        expected_cx = (10 + 440) + 200 // 2
        expected_cy = (10 + 370) + 100 // 2
        assert result.center.x == expected_cx
        assert result.center.y == expected_cy

    @patch("core.yoloe.find_element")
    def test_edge_clamp_near_screen_border(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """Buffer region is clamped to screen boundaries."""
        # Rect near top-left corner
        user_rect = Rect(x=5, y=5, w=20, h=20)
        crop_match = YOLOEMatch(
            bbox=Rect(x=3, y=3, w=22, h=22),
            confidence=0.70,
            center=Point(x=14, y=14),
        )
        mock_find.return_value = [crop_match]

        result = refine_bbox(fake_screen, user_rect)
        assert result is not None
        # crop_x1 = max(0, 5 - 6) = 0, crop_y1 = max(0, 5 - 6) = 0
        assert result.bbox.x == 3
        assert result.bbox.y == 3


# ---------------------------------------------------------------------------
# infer_bbox_at_point tests
# ---------------------------------------------------------------------------

class TestInferBboxAtPoint:
    """Tests for infer_bbox_at_point()."""

    @patch("core.yoloe.find_element")
    def test_returns_match_translated_to_screen(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """infer_bbox_at_point translates crop coords to screen-absolute."""
        crop_match = YOLOEMatch(
            bbox=Rect(x=50, y=60, w=80, h=40),
            confidence=0.75,
            center=Point(x=90, y=80),
        )
        mock_find.return_value = [crop_match]

        result = infer_bbox_at_point(fake_screen, x=500, y=400, radius=200)
        assert result is not None
        # region_x1 = 500 - 200 = 300, region_y1 = 400 - 200 = 200
        assert result.bbox.x == 50 + 300
        assert result.bbox.y == 60 + 200
        assert result.bbox.w == 80
        assert result.bbox.h == 40

    @patch("core.yoloe.find_element")
    def test_returns_none_when_no_matches(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """Returns None when YOLOE finds nothing."""
        mock_find.return_value = []
        result = infer_bbox_at_point(fake_screen, x=500, y=400)
        assert result is None

    @patch("core.yoloe.find_element")
    def test_clamp_near_edge(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """Search region is clamped to screen bounds near edges."""
        crop_match = YOLOEMatch(
            bbox=Rect(x=5, y=5, w=30, h=30),
            confidence=0.60,
            center=Point(x=20, y=20),
        )
        mock_find.return_value = [crop_match]

        result = infer_bbox_at_point(fake_screen, x=10, y=10, radius=200)
        assert result is not None
        # region_x1 = max(0, 10-200) = 0, region_y1 = max(0, 10-200) = 0
        assert result.bbox.x == 5
        assert result.bbox.y == 5

    @patch("core.yoloe.find_element")
    def test_uses_correct_snippet_size(
        self, mock_find: MagicMock, fake_screen: np.ndarray
    ) -> None:
        """The snippet passed to find_element is ~50x50 centered on the point."""
        mock_find.return_value = []

        infer_bbox_at_point(fake_screen, x=500, y=400, radius=200)

        assert mock_find.called
        snippet_arg = mock_find.call_args[0][0]  # first positional arg
        # snippet should be ~50x50 (may be smaller near edges)
        assert snippet_arg.shape[0] == 50
        assert snippet_arg.shape[1] == 50


# ---------------------------------------------------------------------------
# RefineDialog tests
# ---------------------------------------------------------------------------

class TestRefineDialog:
    """Tests for RefineDialog construction and return values."""

    @pytest.fixture(autouse=True)
    def _ensure_qapp(self) -> None:
        """Ensure a QApplication exists, or skip if PyQt6 is unavailable."""
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            pytest.skip("PyQt6 not installed")
            return

        if QApplication.instance() is None:
            self._app = QApplication([])

    def _make_dialog(self) -> object:
        """Build a RefineDialog with small dummy images."""
        from recorder.refine_dialog import RefineDialog

        user_crop = np.zeros((60, 80, 3), dtype=np.uint8)
        yoloe_crop = np.zeros((100, 120, 3), dtype=np.uint8)
        yoloe_rect = Rect(x=10, y=10, w=60, h=50)
        return RefineDialog(user_crop, yoloe_crop, yoloe_rect)

    def test_construction_does_not_crash(self) -> None:
        """The dialog can be instantiated without errors."""
        dlg = self._make_dialog()
        assert dlg is not None

    def test_get_result_default_is_rejected(self) -> None:
        """Default result before any interaction is ('rejected', None)."""
        dlg = self._make_dialog()
        action, rect = dlg.get_result()
        assert action == "rejected"
        assert rect is None

    def test_accept_returns_rect(self) -> None:
        """Simulating accept sets result to ('accepted', Rect)."""
        dlg = self._make_dialog()
        # Simulate accept without exec
        dlg._on_accept()
        action, rect = dlg.get_result()
        assert action == "accepted"
        assert rect is not None
        assert isinstance(rect, Rect)

    def test_reject_returns_none(self) -> None:
        """Simulating reject keeps result as ('rejected', None)."""
        dlg = self._make_dialog()
        dlg._on_reject()
        action, rect = dlg.get_result()
        assert action == "rejected"
        assert rect is None
