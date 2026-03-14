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
