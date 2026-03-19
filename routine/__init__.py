"""Routine file format package for OCSD.

Provides the Routine dataclass model, ordered JSON serialization,
SHA256 checksum computation, and save/load operations for ocsd-routine-v1 files.
"""

from __future__ import annotations

from routine.checksum import calculate_routine_checksum
from routine.format import VALID_CATEGORIES, Routine

__all__ = ["Routine", "VALID_CATEGORIES", "calculate_routine_checksum"]
