"""Routine dataclass model for ocsd-routine-v1 schema.

Provides the Routine dataclass with ordered JSON serialization,
save/load operations, and the build_v1_step helper for constructing
step dictionaries from raw recorder data.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from mapper.graph import OCSDGraph
from routine.checksum import _json_default, calculate_routine_checksum

logger = logging.getLogger(__name__)

VALID_CATEGORIES: list[str] = [
    "productivity",
    "web",
    "system",
    "finance",
    "creative",
    "other",
]


def _detect_platform() -> str:
    """Map sys.platform to a human-friendly platform name.

    Returns:
        Platform string: 'windows', 'macos', 'ubuntu', or raw sys.platform.
    """
    mapping = {
        "win32": "windows",
        "linux": "ubuntu",
        "darwin": "macos",
    }
    return mapping.get(sys.platform, sys.platform)


def _compute_region_hint(cx: int, cy: int, screen_w: int, screen_h: int) -> str:
    """Compute a human-readable region hint from pixel coordinates.

    Args:
        cx: Center X pixel coordinate.
        cy: Center Y pixel coordinate.
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.

    Returns:
        Region hint string like 'top_left', 'center', 'bottom_right'.
    """
    x_pct = cx / screen_w if screen_w > 0 else 0.5
    y_pct = cy / screen_h if screen_h > 0 else 0.5

    if y_pct < 0.33:
        v = "top"
    elif y_pct < 0.66:
        v = "center"
    else:
        v = "bottom"

    if x_pct < 0.33:
        h = "left"
    elif x_pct < 0.66:
        h = "center"
    else:
        h = "right"

    if v == "center" and h == "center":
        return "center"
    return f"{v}_{h}"


def _parse_direction_amount(raw: str) -> dict[str, Any]:
    """Parse a direction+amount string into a structured scroll dict.

    Accepts strings like ``"down 5"``, ``"left 10"``, ``"up 3 pages"``.

    Args:
        raw: Raw direction/amount string from tag dialog.

    Returns:
        Dict with ``direction`` (str), ``amount`` (int), and ``unit`` (str).
        Defaults: direction='down', amount=3, unit='lines'.
    """
    parts = raw.strip().split() if raw and raw.strip() else []
    direction = parts[0] if len(parts) >= 1 else "down"
    try:
        amount = int(parts[1]) if len(parts) >= 2 else 3
    except (ValueError, IndexError):
        amount = 3
    unit = parts[2] if len(parts) >= 3 else "lines"
    if unit not in ("lines", "pages", "pixels"):
        unit = "lines"
    return {"direction": direction, "amount": amount, "unit": unit}


def build_v1_step(
    step: dict[str, Any],
    index: int,
    node_id: str,
    screen_w: int,
    screen_h: int,
) -> dict[str, Any]:
    """Build a v1 step dict from raw recorder step data.

    Args:
        step: Internal step dictionary with tag_data and bbox keys.
        index: Step index in the sequence.
        node_id: Unique node identifier for this step.
        screen_w: Screen width for percentage calculations.
        screen_h: Screen height for percentage calculations.

    Returns:
        JSON-serializable step dictionary matching ocsd-routine-v1 schema.
    """
    tag_data = step.get("tag_data", {})
    bbox = step.get("bbox")
    bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (0, 0, 0, 0)

    # Compute center percentages
    cx = bbox_x + bbox_w // 2
    cy = bbox_y + bbox_h // 2
    cx_pct = cx / screen_w if screen_w > 0 else 0.5
    cy_pct = cy / screen_h if screen_h > 0 else 0.5

    region = _compute_region_hint(cx, cy, screen_w, screen_h)

    anchors = {
        "visual_match": f"snippets/{node_id}.png",
        "ocr_text": tag_data.get("ocr_text"),
        "position_pct": {"x_pct": cx_pct, "y_pct": cy_pct},
        "region_hint": region,
    }

    action = tag_data.get("action", "click")

    result = {
        "step_index": index,
        "node_id": node_id,
        "element_type": tag_data.get("element_type", "unknown"),
        "label": tag_data.get("label", ""),
        "caption": tag_data.get("caption", ""),
        "confidence": tag_data.get("confidence", 0.0),
        "action": action,
        "bbox": {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h},
        "bbox_pct": {
            "x_pct": bbox_x / screen_w if screen_w > 0 else 0.0,
            "y_pct": bbox_y / screen_h if screen_h > 0 else 0.0,
            "w_pct": bbox_w / screen_w if screen_w > 0 else 0.0,
            "h_pct": bbox_h / screen_h if screen_h > 0 else 0.0,
        },
        "region_hint": region,
        "anchors": anchors,
        "snippet_path": f"snippets/{node_id}.png",
        "embedding_path": f"embeddings/{node_id}.npy",
        "dry_run_passed": True,
    }

    # Action-specific fields
    if action == "type":
        result["text_to_type"] = tag_data.get("text_to_type", "")
        result["press_enter"] = tag_data.get("press_enter", False)
    elif action == "scroll":
        result["scroll"] = _parse_direction_amount(
            tag_data.get("direction_amount", "down 3")
        )
    elif action == "click_drag":
        drag_target = step.get("drag_target")
        if drag_target is not None:
            result["drag_target"] = drag_target
    elif action in ("read", "snip_and_search"):
        result["vlm_prompt"] = tag_data.get("vlm_prompt", "")
    elif action == "select_all_extract":
        result["vlm_prompt"] = tag_data.get("vlm_prompt", "")
    elif action == "prompt_user":
        result["question_text"] = tag_data.get("question_text", "")
    elif action == "wait":
        result["wait"] = step.get("wait_definition", {
            "condition_type": "fixed_timer",
            "timeout": 30.0,
        })
    elif action == "loop":
        result["loop"] = step.get("loop_definition")

    return result


def resolve_loop_node_ids(
    steps: list[dict[str, Any]],
    node_ids: list[str],
) -> None:
    """Resolve loop body step indices to node_ids in-place.

    Must be called after all steps have been assigned node_ids,
    before saving to routine.json.  Converts ``body_step_indices``
    (temporary integer list) into ``body_step_node_ids`` (stable
    string references) for each loop step.

    Args:
        steps: List of v1 step dicts (already built by build_v1_step).
        node_ids: Ordered list of node_ids corresponding to step indices.
    """
    for step in steps:
        if step.get("action") != "loop":
            continue
        loop_def = step.get("loop")
        if loop_def is None:
            continue
        indices = loop_def.pop("body_step_indices", [])
        loop_def["body_step_node_ids"] = [
            node_ids[i] for i in indices if i < len(node_ids)
        ]


@dataclass
class Routine:
    """Routine model following the ocsd-routine-v1 schema.

    Represents a complete recorded routine with metadata, steps,
    and a NetworkX graph for element relationships.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = field(default_factory=socket.gethostname)
    author_display: str | None = None
    category: str = "other"
    tags: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    platform: str = field(default_factory=_detect_platform)
    theme: str | None = None
    start_from: str = "desktop"
    resolution: list[int] = field(default_factory=lambda: [1920, 1080])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    steps: list[dict[str, Any]] = field(default_factory=list)
    graph: OCSDGraph = field(default_factory=OCSDGraph)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the routine to an ordered dictionary.

        Key order is fixed to match the ocsd-routine-v1 schema specification.
        Python 3.7+ dicts preserve insertion order.

        Returns:
            Ordered dict with all routine fields including computed checksum.
        """
        graph_dict = self.graph.to_dict()
        checksum = calculate_routine_checksum(self.steps, graph_dict)

        return {
            "$schema": "ocsd-routine-v1",
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "author_display": self.author_display,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "category": self.category,
            "tags": list(self.tags),
            "programs": list(self.programs),
            "platform": self.platform,
            "theme": self.theme,
            "start_from": self.start_from,
            "resolution": list(self.resolution),
            "steps": list(self.steps),
            "graph": graph_dict,
            "checksum": checksum,
            "signature": None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Routine:
        """Deserialize a routine from a dictionary.

        Validates the schema version and checksum integrity before
        constructing the Routine object.

        Args:
            data: Dictionary loaded from routine.json.

        Returns:
            A new Routine instance.

        Raises:
            ValueError: If schema version is wrong or checksum mismatches.
        """
        # Validate schema
        schema = data.get("$schema")
        if schema != "ocsd-routine-v1":
            raise ValueError(
                f"Unsupported routine schema: {schema!r}. "
                f"Expected 'ocsd-routine-v1'."
            )

        # Validate checksum
        steps = data.get("steps", [])
        graph_data = data.get("graph", {})
        stored_checksum = data.get("checksum")
        computed_checksum = calculate_routine_checksum(steps, graph_data)

        if stored_checksum != computed_checksum:
            raise ValueError(
                "Routine integrity check failed: checksum mismatch. "
                "The file may have been modified outside of OCSD."
            )

        # Build graph
        graph = OCSDGraph.from_dict(graph_data) if graph_data else OCSDGraph()

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author", socket.gethostname()),
            author_display=data.get("author_display"),
            category=data.get("category", "other"),
            tags=data.get("tags", []),
            programs=data.get("programs", []),
            platform=data.get("platform", _detect_platform()),
            theme=data.get("theme"),
            start_from=data.get("start_from", "desktop"),
            resolution=data.get("resolution", [1920, 1080]),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            steps=steps,
            graph=graph,
        )

    def save(self, directory: Path) -> None:
        """Save the routine to a directory as routine.json.

        Also creates snippets/ and embeddings/ subdirectories.

        Args:
            directory: Target directory for the routine files.
        """
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "snippets").mkdir(exist_ok=True)
        (directory / "embeddings").mkdir(exist_ok=True)

        data = self.to_dict()
        routine_path = directory / "routine.json"

        with open(routine_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)

        logger.info(
            "Saved routine '%s' v%s to %s", self.name, self.version, directory
        )

    @classmethod
    def load(cls, directory: Path) -> Routine:
        """Load a routine from a directory containing routine.json.

        Args:
            directory: Directory containing the routine.json file.

        Returns:
            A new Routine instance.

        Raises:
            FileNotFoundError: If routine.json does not exist.
            ValueError: If schema or checksum validation fails.
        """
        routine_path = directory / "routine.json"
        if not routine_path.exists():
            raise FileNotFoundError(f"Routine file not found: {routine_path}")

        with open(routine_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
