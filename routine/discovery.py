"""Routine discovery from ~/.ocsd/routines/ directory.

Scans for subdirectories containing routine.json files and returns
lightweight RoutineInfo objects with name, path, and schema version.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routine.migration import detect_schema_version

logger = logging.getLogger(__name__)


@dataclass
class RoutineInfo:
    """Lightweight info about a discovered routine.

    Attributes:
        name: Directory name of the routine.
        path: Absolute path to the routine directory.
        schema_version: Detected schema version (``'v0'``, ``'v1'``, or ``'unknown'``).
    """

    name: str
    path: Path
    schema_version: str


def get_routine_dir() -> Path:
    """Return the default routine storage directory.

    Returns:
        Expanded path ``~/.ocsd/routines``.
    """
    return Path.home() / ".ocsd" / "routines"


def list_routines(base_dir: Path | None = None) -> list[RoutineInfo]:
    """Enumerate routines from a directory.

    Scans subdirectories of *base_dir* for ``routine.json`` files and
    returns a sorted list of :class:`RoutineInfo` objects.

    Args:
        base_dir: Directory to scan. Defaults to :func:`get_routine_dir`.

    Returns:
        Sorted list of discovered routines. Empty list if *base_dir*
        does not exist.
    """
    if base_dir is None:
        base_dir = get_routine_dir()

    if not base_dir.exists():
        return []

    routines: list[RoutineInfo] = []

    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue

        routine_file = subdir / "routine.json"
        if not routine_file.exists():
            continue

        try:
            with open(routine_file, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            version = detect_schema_version(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read %s: %s", routine_file, exc
            )
            version = "unknown"

        routines.append(
            RoutineInfo(
                name=subdir.name,
                path=subdir,
                schema_version=version,
            )
        )

    return routines
