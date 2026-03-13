"""Weighted shortest-path finder for the OCSD UI automation graph.

Finds optimal paths through recorded UI flows using success-rate-weighted
edges, handles branch point resolution, fingerprint checkpoints, and
generates detailed execution plans for the replay runner.

Weight formula: weight = 1.0 - success_rate  (lower is better / more reliable)
Untried edges get an optimistic default weight of 0.5.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from core.types import PathNotFoundError
from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)


def find_path(graph: OCSDGraph, start_id: str, goal_id: str) -> list[str]:
    """Finds the best path from start_id to goal_id based on success rates.

    Uses a weighted shortest path algorithm where weight = 1.0 - success_rate.
    Untried edges (optimistic default success_rate of 0.5) get a weight of 0.5.

    Args:
        graph: The OCSDGraph to search within.
        start_id: The ID of the starting node.
        goal_id: The ID of the destination node.

    Returns:
        An ordered list of node IDs from start to goal.

    Raises:
        PathNotFoundError: If no path exists between the start and goal.
    """

    def weight_func(u: str, v: str, _: dict[str, Any]) -> float:
        """Weight function for NetworkX shortest path."""
        return 1.0 - graph.success_rate(u, v)

    try:
        path = nx.shortest_path(
            graph.nx_graph,
            source=start_id,
            target=goal_id,
            weight=weight_func,
        )
        return list(path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise PathNotFoundError(start_id, goal_id)


def find_path_through(
    graph: OCSDGraph,
    start_id: str,
    goal_id: str,
    waypoints: list[str],
) -> list[str]:
    """Finds a path that passes through required waypoints in order.

    Chains find_path calls between consecutive waypoint pairs and
    deduplicates shared nodes at join points.

    Args:
        graph: The OCSDGraph to search within.
        start_id: The ID of the starting node.
        goal_id: The ID of the destination node.
        waypoints: List of node IDs that must be visited in order.

    Returns:
        An ordered list of node IDs forming the full path.

    Raises:
        PathNotFoundError: If any segment of the path cannot be found.
    """
    full_path: list[str] = []
    current_start = start_id
    targets = waypoints + [goal_id]

    for target in targets:
        segment = find_path(graph, current_start, target)
        if not full_path:
            full_path.extend(segment)
        else:
            # Avoid duplicating the join node
            full_path.extend(segment[1:])
        current_start = target

    return full_path


def get_branch_decision(
    graph: OCSDGraph, branch_node_id: str
) -> dict[str, Any]:
    """Evaluates outgoing edges from a branch point to recommend the best path.

    Args:
        graph: The OCSDGraph to search within.
        branch_node_id: The ID of the branch_point node.

    Returns:
        A dict with:
        - "edges": list of outgoing branch edges with conditions and rates
        - "recommended": edge_id of the branch with highest success rate
        Returns an empty dict if the node is not a branch_point.
    """
    try:
        node_data = graph.get_node(branch_node_id)
        if node_data.get("element_type") != "branch_point":
            return {}
    except KeyError:
        return {}

    branch_edges = graph.get_branch_edges(branch_node_id)
    if not branch_edges:
        return {}

    recommended_edge_id = ""
    max_success_rate = -1.0

    edges_info: list[dict[str, Any]] = []
    for edge_data in branch_edges:
        target_id = edge_data["target_node_id"]
        rate = graph.success_rate(branch_node_id, target_id)

        edges_info.append({
            "edge_id": edge_data["edge_id"],
            "target_id": target_id,
            "condition": edge_data.get("branch_condition"),
            "success_rate": rate,
        })

        if rate > max_success_rate:
            max_success_rate = rate
            recommended_edge_id = edge_data["edge_id"]

    return {
        "edges": edges_info,
        "recommended": recommended_edge_id,
    }


def get_fingerprint_checkpoints(
    graph: OCSDGraph, path: list[str]
) -> list[str]:
    """Identifies fingerprint nodes along a given path.

    Args:
        graph: The OCSDGraph containing the path.
        path: List of node IDs in the path.

    Returns:
        A list of node IDs from the path that are fingerprint nodes.
    """
    checkpoints: list[str] = []
    for node_id in path:
        try:
            node_data = graph.get_node(node_id)
            if node_data.get("element_type") == "fingerprint":
                checkpoints.append(node_id)
        except KeyError:
            continue
    return checkpoints


def get_execution_plan(
    graph: OCSDGraph, start_id: str, goal_id: str
) -> dict[str, Any]:
    """Generates a detailed execution plan from start to goal.

    Args:
        graph: The OCSDGraph to search within.
        start_id: The starting node ID.
        goal_id: The goal node ID.

    Returns:
        A dict containing:
        - "path": the ordered node IDs
        - "steps": list of dicts with node_id, label, element_type, action
        - "fingerprint_checks": fingerprint node IDs along the path
        - "branch_points": branch_point node IDs along the path
        - "estimated_reliability": product of edge success_rates along path

    Raises:
        PathNotFoundError: If no path exists between start and goal.
    """
    path = find_path(graph, start_id, goal_id)
    steps: list[dict[str, Any]] = []
    reliability = 1.0
    fingerprint_checks: list[str] = []
    branch_points: list[str] = []

    for i, node_id in enumerate(path):
        node_data = graph.get_node(node_id)
        element_type = node_data.get("element_type")

        if element_type == "fingerprint":
            fingerprint_checks.append(node_id)
        elif element_type == "branch_point":
            branch_points.append(node_id)

        action: dict[str, Any] | None = None
        if i < len(path) - 1:
            next_node_id = path[i + 1]
            try:
                edge_data = graph.get_edge(node_id, next_node_id)
                action = dict(edge_data)
                reliability *= graph.success_rate(node_id, next_node_id)
            except KeyError:
                logger.warning(
                    "Edge %s -> %s missing during plan generation",
                    node_id[:8],
                    next_node_id[:8],
                )

        steps.append({
            "node_id": node_id,
            "label": node_data.get("label", ""),
            "element_type": element_type,
            "action": action,
        })

    logger.info(
        "Execution plan: %d steps, reliability=%.2f, %d fingerprints, %d branches",
        len(steps),
        reliability,
        len(fingerprint_checks),
        len(branch_points),
    )

    return {
        "path": path,
        "steps": steps,
        "fingerprint_checks": fingerprint_checks,
        "branch_points": branch_points,
        "estimated_reliability": reliability,
    }
