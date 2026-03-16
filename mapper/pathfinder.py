"""Weighted shortest-path finder for the OCSD UI automation graph.

Finds optimal paths through recorded UI flows using success-rate-weighted
edges, handles branch point resolution, fingerprint checkpoints, and
generates detailed execution plans for the replay runner.

Also provides fuzzy label search, READ_HERE node discovery, and
multi-path alternatives.

Weight formula: weight = 1.0 - success_rate  (lower is better / more reliable)
Untried edges get an optimistic default weight of 0.5.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
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


def find_by_label(graph: OCSDGraph, label: str) -> str | None:
    """Finds the best-matching node by fuzzy label comparison.

    Uses SequenceMatcher for similarity scoring. Matches against
    the human label, AI-guessed label, and OCR text.

    Args:
        graph: The OCSDGraph to search.
        label: The label text to search for.

    Returns:
        node_id of the best match, or None if no match scores above 0.5.
    """
    if not label:
        return None

    label_lower = label.strip().lower()
    best_id: str | None = None
    best_score: float = 0.0

    for node_id in graph.nodes:
        node_data = graph.get_node(node_id)

        # Score against human label
        node_label = str(node_data.get("label", "")).strip().lower()
        if node_label:
            if node_label == label_lower:
                return node_id
            score = SequenceMatcher(None, label_lower, node_label).ratio()
            if score > best_score:
                best_score = score
                best_id = node_id

        # Score against AI guess label
        ai_label = str(node_data.get("label_ai_guess", "")).strip().lower()
        if ai_label:
            if ai_label == label_lower:
                return node_id
            score = SequenceMatcher(None, label_lower, ai_label).ratio()
            if score > best_score:
                best_score = score
                best_id = node_id

        # Score against OCR text (substring match)
        ocr_text = str(node_data.get("ocr_text", "") or "").strip().lower()
        if ocr_text and label_lower in ocr_text:
            score = 0.8
            if score > best_score:
                best_score = score
                best_id = node_id

    if best_score < 0.5:
        logger.debug("No label match for '%s' (best score: %.2f)", label, best_score)
        return None

    logger.debug(
        "Label match for '%s': node %s (score: %.2f)",
        label,
        best_id[:8] if best_id else "None",
        best_score,
    )
    return best_id


def find_read_here_nodes(
    graph: OCSDGraph,
    page_hint: str | None = None,
) -> list[str]:
    """Returns all READ_HERE nodes, optionally filtered by page context.

    Args:
        graph: The OCSDGraph to search.
        page_hint: Optional page/state hint for filtering (case-insensitive).

    Returns:
        List of node_ids of READ_HERE elements.
    """
    read_nodes = graph.get_read_here_nodes()

    if page_hint is None:
        return read_nodes

    hint_lower = page_hint.strip().lower()
    filtered = []
    for nid in read_nodes:
        node_data = graph.get_node(nid)
        label = str(node_data.get("label", "")).lower()
        ocr = str(node_data.get("ocr_text", "") or "").lower()
        if hint_lower in label or hint_lower in ocr:
            filtered.append(nid)

    return filtered


def find_all_paths(
    graph: OCSDGraph,
    start_node_id: str,
    destination_node_id: str,
    max_paths: int = 5,
) -> list[list[str]]:
    """Finds multiple paths between two nodes, sorted by estimated quality.

    Args:
        graph: The OCSDGraph to search.
        start_node_id: UUID of the starting node.
        destination_node_id: UUID of the destination node.
        max_paths: Maximum number of paths to return.

    Returns:
        List of paths (each is a list of node_ids), sorted best-first.
        Empty list if no paths exist.
    """
    g = graph.nx_graph

    if start_node_id not in g.nodes or destination_node_id not in g.nodes:
        return []

    try:
        all_paths = list(
            nx.all_simple_paths(
                g,
                start_node_id,
                destination_node_id,
                cutoff=20,
            )
        )
    except nx.NodeNotFound:
        return []

    if not all_paths:
        return []

    def path_score(path: list[str]) -> float:
        if len(path) < 2:
            return 1.0
        rates = [graph.success_rate(path[i], path[i + 1]) for i in range(len(path) - 1)]
        return sum(rates) / len(rates)

    scored = sorted(all_paths, key=lambda p: (-path_score(p), len(p)))
    return scored[:max_paths]
