"""Tests for core.detection — DetectionProvider protocol + factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGetDetector:
    """Tests for get_detector() factory."""

    def test_returns_omniparser_provider_by_default(self):
        from core.detection import get_detector
        import core.detection as det_mod

        # Reset singleton
        det_mod._detector_instance = None

        with patch("core.detection.OmniParserProvider") as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            result = get_detector({"models": {"detector": "omniparser"}})

        assert result is mock_instance
        # Reset for other tests
        det_mod._detector_instance = None

    def test_returns_singleton(self):
        from core.detection import get_detector
        import core.detection as det_mod

        det_mod._detector_instance = None

        with patch("core.detection.OmniParserProvider") as MockProvider:
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            first = get_detector({"models": {"detector": "omniparser"}})
            second = get_detector({"models": {"detector": "omniparser"}})

        assert first is second
        MockProvider.assert_called_once()
        det_mod._detector_instance = None

    def test_raises_for_unknown_detector(self):
        from core.detection import get_detector
        import core.detection as det_mod

        det_mod._detector_instance = None

        with pytest.raises(ValueError, match="Unknown detector"):
            get_detector({"models": {"detector": "nonexistent"}})

        det_mod._detector_instance = None

    def test_uses_config_when_no_arg(self):
        from core.detection import get_detector
        import core.detection as det_mod

        det_mod._detector_instance = None

        with patch("core.detection.OmniParserProvider") as MockProvider, \
             patch("core.detection.get_config",
                   return_value={"models": {"detector": "omniparser"}}):
            mock_instance = MagicMock()
            MockProvider.return_value = mock_instance

            result = get_detector()

        assert result is mock_instance
        det_mod._detector_instance = None
