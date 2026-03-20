"""Semantic version bump helpers for routines.

Provides simple major.minor.patch version string manipulation
without pulling in a full semver library.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def bump_minor(version: str) -> str:
    """Increment the minor version component, resetting patch to 0.

    Args:
        version: Semantic version string (e.g. ``'1.2.3'``).

    Returns:
        Version string with minor incremented and patch reset
        (e.g. ``'1.3.0'``).
    """
    parts = version.split(".")
    parts[1] = str(int(parts[1]) + 1)
    parts[2] = "0"
    return ".".join(parts)


def bump_patch(version: str) -> str:
    """Increment the patch version component only.

    Args:
        version: Semantic version string (e.g. ``'1.2.3'``).

    Returns:
        Version string with patch incremented (e.g. ``'1.2.4'``).
    """
    parts = version.split(".")
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)
