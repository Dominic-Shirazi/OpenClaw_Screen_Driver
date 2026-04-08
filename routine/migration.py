"""V0-to-V1 routine migration (in-memory only).

Detects routine schema versions and upgrades ocsd-routine-v0 dicts
to ocsd-routine-v1 format without rewriting files to disk.
"""

from __future__ import annotations

import copy
import logging
import socket
from datetime import datetime, timezone
from typing import Any

from routine.checksum import calculate_routine_checksum

logger = logging.getLogger(__name__)

_SCHEMA_MAP: dict[str, str] = {
    "ocsd-routine-v0": "v0",
    "ocsd-routine-v1": "v1",
}


def detect_schema_version(data: dict[str, Any]) -> str:
    """Detect the schema version of a routine dict.

    Args:
        data: Routine dictionary with a ``$schema`` field.

    Returns:
        Version string: ``'v0'``, ``'v1'``, or ``'unknown'``.
    """
    schema = data.get("$schema", "")
    return _SCHEMA_MAP.get(schema, "unknown")


def upgrade_v0_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an ocsd-routine-v0 dict to v1 format in memory.

    Deep-copies input to avoid mutation. Adds missing v1 fields,
    converts step snippet/embedding paths to node_id-based naming,
    and adds anchors to each step.

    Args:
        data: V0 routine dictionary.

    Returns:
        New dict conforming to ocsd-routine-v1 schema.
    """
    from routine.format import _detect_platform

    d = copy.deepcopy(data)

    # Top-level v1 fields with defaults
    d["$schema"] = "ocsd-routine-v1"
    d.setdefault("version", "1.0.0")
    d.setdefault("author", socket.gethostname())
    d.setdefault("author_display", None)
    d.setdefault("category", "other")
    d.setdefault("tags", [])
    d.setdefault("programs", [])
    d.setdefault("platform", _detect_platform())
    d.setdefault("theme", None)
    if "updated_at" not in d or not d["updated_at"]:
        d["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Upgrade each step
    for step in d.get("steps", []):
        node_id = step.get("node_id", "")
        step["snippet_path"] = f"snippets/{node_id}.png"
        step["embedding_path"] = f"embeddings/{node_id}.npy"
        step.setdefault("confidence", 0.0)

        if "anchors" not in step:
            bbox_pct = step.get("bbox_pct", {})
            x_pct = bbox_pct.get("x_pct", 0.0)
            y_pct = bbox_pct.get("y_pct", 0.0)
            w_pct = bbox_pct.get("w_pct", 0.0)
            h_pct = bbox_pct.get("h_pct", 0.0)

            step["anchors"] = {
                "visual_match": f"snippets/{node_id}.png",
                "ocr_text": step.get("ocr_text"),
                "position_pct": {
                    "x_pct": x_pct + w_pct / 2,
                    "y_pct": y_pct + h_pct / 2,
                },
                "region_hint": step.get("region_hint", "center"),
            }

    # Compute checksum over upgraded content
    checksum = calculate_routine_checksum(
        d.get("steps", []), d.get("graph", {})
    )
    d["checksum"] = checksum
    d["signature"] = None

    # Reorder keys to match v1 logical order
    ordered: dict[str, Any] = {}
    key_order = [
        "$schema", "name", "version", "description",
        "author", "author_display", "created_at", "updated_at",
        "category", "tags", "programs", "platform", "theme",
        "start_from", "resolution", "steps", "graph",
        "checksum", "signature",
    ]
    for key in key_order:
        if key in d:
            ordered[key] = d[key]
    # Include any extra keys not in the standard order
    for key, val in d.items():
        if key not in ordered:
            ordered[key] = val

    logger.info(
        "Upgraded routine '%s' from v0 to v1 (in-memory)",
        data.get("name", "unknown"),
    )
    return ordered
