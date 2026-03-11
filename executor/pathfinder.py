"""BFS pathfinder over the OCSD UI map graph.

Finds paths through the directed graph from a start node to a destination.
Uses weighted BFS where edge cost = 1 - success_rate, so high-success edges
are preferred. Falls back to unweighted BFS if weighted path fails.

Also provides fuzzy label search and READ_HERE node discovery.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

import networkx as nx

from core.types import PathNotFoundError
from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)


def find_path(
    graph: OCSDGraph,
    start_node_id: str,
    destination_node_id: str,
) -> list[str]:
    """Finds the best path from start to destination node.

    Uses weighted BFS (Dijkstra) where edge cost = 1 - success_rate.
    This prefers edges that have historically succeeded more often.
    Falls back to unweighted shortest path if weighted fails.

    Args:
        graph: The OCSDGraph to search.
        start_node_id: UUID of the starting node.
        destination_node_id: UUID of the destination node.

    Returns:
        Ordered list of node_ids from start to destination (inclusive).

    Raises:
        PathNotFoundError: If no path exists between the nodes.
        KeyError: If either node_id doesn't exist in the graph.
    """
    g = graph.nx_graph

    if start_node_id not in g.nodes:
        raise KeyError(f"Start node not found: {start_node_id}")
    if destination_node_id not in g.nodes:
        raise KeyError(f"Destination node not found: {destination_node_id}")

    if start_node_id == destination_node_id:
        return [start_node_id]

    # Attempt 1: Weighted shortest path (Dijkstra)
    try:
        path = _weighted_shortest_path(graph, start_node_id, destination_node_id)
        logger.info(
            "Found weighted path: %d steps (%s -> %s)",
            len(path) - 1,
            start_node_id[:8],
            destination_node_id[:8],
        )
        return path
    except nx.NetworkXNoPath:
        logger.debug("Weighted path failed, trying unweighted")
    except nx.NodeNotFound:
        logger.debug("Node not found in weighted search, trying unweighted")

    # Attempt 2: Unweighted shortest path (BFS)
    try:
        path = nx.shortest_path(g, start_node_id, destination_node_id)
        logger.info(
            "Found unweighted path: %d steps (%s -> %s)",
            len(path) - 1,
            start_node_id[:8],
            destination_node_id[:8],
        )
        return list(path)
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        raise PathNotFoundError(start_node_id, destination_node_id) from e


def _weighted_shortest_path(
    graph: OCSDGraph,
    start_id: str,
    end_id: str,
) -> list[str]:
    """Finds shortest path using success-rate-weighted edges.

    Edge weight = 1.0 - success_rate. Edges with higher success rates
    have lower cost and are preferred. Untried edges get weight 0.5
    (optimistic default).

    Edges with success_rate < 0.5 get an additional penalty multiplier
    to strongly discourage unreliable paths when alternatives exist.

    Args:
        graph: The OCSDGraph to search.
        start_id: Start node UUID.
        end_id: End node UUID.

    Returns:
        Ordered list of node_ids.

    Raises:
        nx.NetworkXNoPath: If no path exists.
    """
    g = graph.nx_graph

    def edge_weight(u: str, v: str, data: dict[str, Any]) -> float:
        rate = graph.success_rate(u, v)
        cost = 1.0 - rate
        # Penalize edges with < 50% success rate
        if rate < 0.5 and data.get("execution_count", 0) > 0:
            cost *= 2.0
        return max(cost, 0.01)  # Ensure positive weight

    path = nx.dijkstra_path(g, start_id, end_id, weight=edge_weight)
    return list(path)


def find_by_label(graph: OCSDGraph, label: str) -> str | None:
    """Finds the best-matching node by fuzzy label comparison.

    Uses SequenceMatcher for similarity scoring. Matches against both
    the human label and the AI-guessed label.

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
            # Exact match is instant win
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

        # Score against OCR text (partial match)
        ocr_text = str(node_data.get("ocr_text", "") or "").strip().lower()
        if ocr_text and label_lower in ocr_text:
            score = 0.8  # Substring match in OCR text is a strong signal
            if score > best_score:
                best_score = score
                best_id = node_id

    # Require minimum similarity threshold
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

    READ_HERE nodes are data extraction points — elements whose values
    should be read and returned to the caller.

    Args:
        graph: The OCSDGraph to search.
        page_hint: Optional page/state hint for filtering. If provided,
                  only returns READ_HERE nodes whose label or OCR text
                  contains the hint (case-insensitive).

    Returns:
        List of node_ids of READ_HERE elements.
    """
    read_nodes = graph.get_read_here_nodes()

    if page_hint is None:
        return read_nodes

    # Filter by page hint
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

    Useful for presenting alternatives to the runner if the primary path
    fails mid-execution.

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
                cutoff=20,  # Reasonable max path length
            )
        )
    except nx.NodeNotFound:
        return []

    if not all_paths:
        return []

    # Score each path by average success rate of its edges
    def path_score(path: list[str]) -> float:
        if len(path) < 2:
            return 1.0
        rates = [graph.success_rate(path[i], path[i + 1]) for i in range(len(path) - 1)]
        return sum(rates) / len(rates)

    # Sort by score descending (best paths first), break ties by length (shorter first)
    scored = sorted(all_paths, key=lambda p: (-path_score(p), len(p)))

    return scored[:max_paths]
