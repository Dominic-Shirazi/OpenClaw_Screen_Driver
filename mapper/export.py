"""Skill JSON serialization module for OCSD.

Provides functions to export OCSDGraph objects to JSON skill files
and import them back with integrity verification (SHA256 checksum).
Follows the ocsd-skill-v1 schema defined in the project blueprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)


def calculate_checksum(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    """Calculates SHA256 checksum of nodes and edges arrays for integrity.

    The checksum is computed over a stable JSON representation of the combined
    nodes and edges, ensuring that any modification to the graph data
    invalidates the skill file.

    Args:
        nodes: List of node data dictionaries.
        edges: List of edge data dictionaries.

    Returns:
        Hex string of the SHA256 hash.
    """
    # Create a stable representation by sorting by ID and sorting JSON keys
    stable_nodes = sorted(nodes, key=lambda x: x.get("node_id", ""))
    stable_edges = sorted(edges, key=lambda x: x.get("edge_id", ""))

    combined = {
        "nodes": stable_nodes,
        "edges": stable_edges,
    }

    # Ensure consistent separators and sorted keys for stability
    data = json.dumps(
        combined,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def export_skill(
    graph: OCSDGraph,
    name: str,
    description: str,
    author: str,
    version: str,
    target_app: str,
    target_url: Optional[str] = None,
    tags: Optional[List[str]] = None,
    os_list: Optional[List[str]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Serializes an OCSDGraph to a JSON-compatible dictionary matching the skill schema.

    Args:
        graph: The OCSDGraph to export.
        name: Human-readable name of the skill.
        description: Description of what the skill does.
        author: Author name/ID.
        version: Semver version string.
        target_app: Name of the application this skill automates.
        target_url: Optional starting URL.
        tags: Optional list of tags for categorization.
        os_list: Optional list of supported operating systems (defaults to ["windows"]).
        created_at: Optional ISO 8601 creation timestamp. If not provided,
            uses current time or oldest node's created_at.

    Returns:
        Dict following the ocsd-skill-v1 schema.
    """
    graph_dict = graph.to_dict()
    nodes = graph_dict["nodes"]
    edges = graph_dict["edges"]

    now = datetime.now(timezone.utc).isoformat()

    # Determine creation timestamp
    if not created_at:
        if nodes:
            # Use the oldest node's creation time as a heuristic for skill creation
            created_at = min(n.get("created_at", now) for n in nodes)
        else:
            created_at = now

    # Identify functional node groups
    entry_nodes = graph.get_entry_nodes()
    entry_node_id = entry_nodes[0] if entry_nodes else ""
    exit_nodes = graph.get_exit_nodes()
    read_here_nodes = graph.get_read_here_nodes()

    skill_data = {
        "$schema": "ocsd-skill-v1",
        "skill_id": graph.skill_id,
        "name": name,
        "description": description,
        "author": author,
        "version": version,
        "target_app": target_app,
        "target_url": target_url,
        "os": os_list or ["windows", "mac", "linux"],
        "created_at": created_at,
        "updated_at": now,
        "checksum": calculate_checksum(nodes, edges),
        "nodes": nodes,
        "edges": edges,
        "entry_node_id": entry_node_id,
        "exit_nodes": exit_nodes,
        "read_here_nodes": read_here_nodes,
        "tags": tags or [],
    }

    return skill_data


def save_skill_to_file(skill_data: Dict[str, Any], file_path: Path | str) -> None:
    """Saves skill dictionary to a JSON file.

    Args:
        skill_data: The skill dictionary to save.
        file_path: Path to the destination file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(skill_data, f, indent=2, ensure_ascii=False)

    logger.info("Saved skill '%s' (%s) to %s", skill_data["name"], skill_data["skill_id"][:8], path)


def load_skill_from_file(file_path: Path | str) -> Dict[str, Any]:
    """Loads a skill dictionary from a JSON file and verifies integrity.

    Args:
        file_path: Path to the skill JSON file.

    Returns:
        The loaded skill dictionary.

    Raises:
        ValueError: If checksum verification fails or schema is invalid.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        skill_data = json.load(f)

    # Basic schema check
    if skill_data.get("$schema") != "ocsd-skill-v1":
        raise ValueError(f"Unsupported skill schema version: {skill_data.get('$schema')}")

    # Verify integrity
    expected_checksum = skill_data.get("checksum")
    nodes = skill_data.get("nodes", [])
    edges = skill_data.get("edges", [])
    actual_checksum = calculate_checksum(nodes, edges)

    if expected_checksum != actual_checksum:
        logger.error("Integrity check failed for %s", path)
        raise ValueError(
            f"Skill integrity check failed for {path}. "
            "The file may have been modified outside of OCSD."
        )

    return skill_data


def import_skill(skill_data: Dict[str, Any]) -> tuple[OCSDGraph, Dict[str, Any]]:
    """Creates an OCSDGraph and metadata dict from a skill dictionary.

    Args:
        skill_data: The skill dictionary to import.

    Returns:
        A tuple of (OCSDGraph instance, metadata dictionary).
    """
    graph = OCSDGraph.from_dict(skill_data)

    # Extract metadata excluding graph components
    metadata = {
        k: v for k, v in skill_data.items()
        if k not in ("nodes", "edges", "checksum")
    }

    return graph, metadata