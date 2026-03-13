"""Intelligent change detection between OCSD graph versions.

Compares two OCSDGraph instances and produces a structured diff describing
what nodes and edges were added, removed, or moved.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)

# Threshold for position change to be considered a "move" (fraction of screen)
_MOVE_THRESHOLD = 0.05


@dataclass
class GraphDiff:
    """Structured difference between two OCSDGraph versions.

    Attributes:
        added_nodes: Node IDs present in new but not in old.
        removed_nodes: Node IDs present in old but not in new.
        moved_nodes: Node IDs present in both but with changed positions.
            Each entry maps node_id to a dict with "old" and "new" positions.
        added_edges: Edge tuples (source, target) present in new but not old.
        removed_edges: Edge tuples (source, target) present in old but not new.
    """

    added_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    moved_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    added_edges: list[tuple[str, str]] = field(default_factory=list)
    removed_edges: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Returns True if any changes were detected."""
        return bool(
            self.added_nodes
            or self.removed_nodes
            or self.moved_nodes
            or self.added_edges
            or self.removed_edges
        )


def _get_position(graph: OCSDGraph, node_id: str) -> tuple[float, float]:
    """Extracts the (x_pct, y_pct) position from a node.

    Args:
        graph: The graph containing the node.
        node_id: The node to read position from.

    Returns:
        Tuple of (x_pct, y_pct), defaulting to (0.5, 0.5) if missing.
    """
    node_data = graph.get_node(node_id)
    pos = node_data.get("relative_position", {})
    return (pos.get("x_pct", 0.5), pos.get("y_pct", 0.5))


def _position_changed(
    old_pos: tuple[float, float],
    new_pos: tuple[float, float],
) -> bool:
    """Determines if a node moved beyond the threshold.

    Args:
        old_pos: Previous (x_pct, y_pct).
        new_pos: Current (x_pct, y_pct).

    Returns:
        True if the Euclidean distance exceeds _MOVE_THRESHOLD.
    """
    dx = new_pos[0] - old_pos[0]
    dy = new_pos[1] - old_pos[1]
    return math.hypot(dx, dy) > _MOVE_THRESHOLD


def diff_graphs(old: OCSDGraph, new: OCSDGraph) -> GraphDiff:
    """Computes the structural difference between two graph versions.

    Compares node sets, edge sets, and node positions to determine
    what changed between the old and new versions.

    Args:
        old: The previous version of the graph.
        new: The current version of the graph.

    Returns:
        A GraphDiff describing all detected changes.
    """
    old_node_ids = set(old.nodes)
    new_node_ids = set(new.nodes)

    added_nodes = sorted(new_node_ids - old_node_ids)
    removed_nodes = sorted(old_node_ids - new_node_ids)

    # Check for moved nodes among those present in both
    moved_nodes: dict[str, dict[str, Any]] = {}
    common_nodes = old_node_ids & new_node_ids
    for node_id in sorted(common_nodes):
        old_pos = _get_position(old, node_id)
        new_pos = _get_position(new, node_id)
        if _position_changed(old_pos, new_pos):
            moved_nodes[node_id] = {
                "old": {"x_pct": old_pos[0], "y_pct": old_pos[1]},
                "new": {"x_pct": new_pos[0], "y_pct": new_pos[1]},
            }

    # Compare edges
    old_edges = set(old.nx_graph.edges())
    new_edges = set(new.nx_graph.edges())

    added_edges = sorted(new_edges - old_edges)
    removed_edges = sorted(old_edges - new_edges)

    result = GraphDiff(
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        moved_nodes=moved_nodes,
        added_edges=added_edges,
        removed_edges=removed_edges,
    )

    logger.info(
        "Graph diff: +%d/-%d nodes, %d moved, +%d/-%d edges",
        len(added_nodes),
        len(removed_nodes),
        len(moved_nodes),
        len(added_edges),
        len(removed_edges),
    )
    return result
