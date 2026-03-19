"""SHA256 checksum computation for routine integrity verification.

The checksum covers only steps and graph data (not mutable metadata like
name, description, tags, or updated_at) so that routine identity is stable
across metadata-only edits.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _json_default(obj: object) -> Any:
    """JSON serializer fallback for enums and other non-standard types.

    Args:
        obj: Object that the default JSON encoder cannot handle.

    Returns:
        The enum's .value attribute if available.

    Raises:
        TypeError: If the object cannot be serialized.
    """
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def calculate_routine_checksum(
    steps: list[dict[str, Any]],
    graph_dict: dict[str, Any],
) -> str:
    """Compute SHA256 checksum over steps and graph data.

    The checksum is deterministic: identical steps + graph always produce
    the same hash regardless of metadata differences (name, tags, etc.).

    Args:
        steps: List of step dictionaries from the routine.
        graph_dict: Graph serialization dict (from OCSDGraph.to_dict()).

    Returns:
        64-character lowercase hex SHA256 digest.
    """
    combined = {"steps": steps, "graph": graph_dict}
    data = json.dumps(
        combined,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
