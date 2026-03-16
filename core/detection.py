"""Detection provider protocol and factory.

Abstracts detection backends behind a common protocol so they can be
swapped via config. OmniParser is the first (and currently only) provider.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

import numpy as np

from core.config import get_config
from core.types import LocateResult

logger = logging.getLogger(__name__)

_detector_instance: Any = None
_detector_lock = threading.Lock()

# Lazy top-level reference so patch("core.detection.OmniParserProvider") works.
# Populated on first call to get_detector() when omniparser backend is selected.
try:
    from core.omniparser import OmniParserProvider  # type: ignore[import]
except ImportError:
    OmniParserProvider = None  # type: ignore[assignment,misc]


class DetectionProvider(Protocol):
    """Protocol for UI element detection backends."""

    def detect(self, screenshot: np.ndarray) -> list[dict[str, Any]]:
        """Detect all UI elements in screenshot.

        Returns list of candidate dicts::

            {
                "rect": {"x": int, "y": int, "w": int, "h": int},
                "type_guess": str,
                "label_guess": str,
                "confidence": float,
            }

        Args:
            screenshot: BGR numpy array of the screen.

        Returns:
            List of detected element dicts.
        """
        ...

    def detect_and_match(
        self,
        screenshot: np.ndarray,
        saved_snippet: np.ndarray,
        hint_x: int,
        hint_y: int,
        match_threshold: float = 0.7,
        search_radius: int = 400,
    ) -> LocateResult | None:
        """Detect all elements, then find best CLIP match to saved_snippet.

        Returns LocateResult if match above threshold, else None.

        Args:
            screenshot: Full-screen BGR numpy array.
            saved_snippet: Reference element image to match against.
            hint_x: Expected X center from recording time.
            hint_y: Expected Y center from recording time.
            match_threshold: Minimum CLIP similarity to accept a match.
            search_radius: Pixel radius around hint to restrict search.

        Returns:
            LocateResult if a match is found above threshold, else None.
        """
        ...


def get_detector(config: dict | None = None) -> DetectionProvider:
    """Factory that returns the configured detection provider.

    Reads ``config['models']['detector']`` to select backend.
    Returns a singleton instance (lazy-loaded, thread-safe).

    Args:
        config: Optional config dict override. If None, reads from
            :func:`core.config.get_config`.

    Returns:
        DetectionProvider instance.

    Raises:
        ValueError: If the detector name in config is not recognized.
    """
    global _detector_instance

    if _detector_instance is not None:
        return _detector_instance

    with _detector_lock:
        # Double-checked locking pattern.
        if _detector_instance is not None:
            return _detector_instance

        if config is None:
            config = get_config()

        detector_name = config.get("models", {}).get("detector", "omniparser")

        if detector_name == "omniparser":
            conf_threshold = config.get("detection", {}).get(
                "confidence_threshold", 0.3
            )
            _detector_instance = OmniParserProvider(
                confidence_threshold=conf_threshold
            )
        else:
            raise ValueError(
                f"Unknown detector: {detector_name!r}. "
                f"Supported: 'omniparser'"
            )

        logger.info("Initialized detector: %s", detector_name)
        return _detector_instance
