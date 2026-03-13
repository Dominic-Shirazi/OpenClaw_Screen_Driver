"""NetworkX directed graph wrapper for the OCSD UI map.

Nodes represent UI elements. Edges represent actions/transitions between them.
Provides full CRUD, execution tracking, serialization, and query helpers.

All node/edge schemas follow the blueprint spec exactly:
- Nodes: node_id, skill_id, element_type, layer, label, embeddings, position, stats
- Edges: edge_id, source/target, action_type, payload, branching, execution stats
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import networkx as nx

from core.types import Rect
from recorder.element_types import ElementType

logger = logging.getLogger(__name__)

# Valid layer types (will become a proper enum in mapper/layers.py in Stage 2)
_VALID_LAYERS = {"os_ui", "app_persistent", "page_specific"}


def _now_iso() -> str:
    """Returns current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _gen_id() -> str:
    """Generates a new UUID4 string."""
    return str(uuid.uuid4())


def _region_hint(x_pct: float, y_pct: float) -> str:
    """Computes a human-readable region hint from percentage coordinates.

    Args:
        x_pct: Horizontal position 0.0 (left) to 1.0 (right).
        y_pct: Vertical position 0.0 (top) to 1.0 (bottom).

    Returns:
        Region hint string like "top_left", "center", "bottom_right".
    """
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


class OCSDGraph:
    """Directed graph representing the OCSD UI map.

    Wraps networkx.DiGraph with schema enforcement, execution tracking,
    and serialization.
    """

    def __init__(self, skill_id: str = "") -> None:
        """Initializes an empty OCSD graph.

        Args:
            skill_id: The skill ID this graph belongs to.
        """
        self._graph = nx.DiGraph()
        self.skill_id = skill_id

    # -------------------------------------------------------------------
    # Node CRUD
    # -------------------------------------------------------------------

    def add_node(
        self,
        element_type: str,
        label: str,
        *,
        node_id: str | None = None,
        layer: str = "page_specific",
        label_ai_guess: str = "",
        ocr_text: str | None = None,
        embedding_id: str = "",
        snippet_path: str = "",
        x_pct: float = 0.5,
        y_pct: float = 0.5,
        w_pct: float = 0.0,
        h_pct: float = 0.0,
        resolution: tuple[int, int] = (1920, 1080),
        confidence_threshold: float = 0.75,
        context_links: list[dict[str, str]] | None = None,
    ) -> str:
        """Adds a new element node to the graph.

        Args:
            element_type: ElementType value string (e.g., "button", "textbox").
            label: Human-provided label for this element.
            node_id: Optional explicit UUID. Generated if not provided.
            layer: Layer type: "os_ui", "app_persistent", or "page_specific".
            label_ai_guess: VLM's label guess.
            ocr_text: Visible text extracted by OCR/VLM.
            embedding_id: FAISS index key for this element's visual embedding.
            snippet_path: Path to the reference image crop.
            x_pct: Horizontal center position as fraction of screen width (0.0-1.0).
            y_pct: Vertical center position as fraction of screen height (0.0-1.0).
            w_pct: Bounding box width as fraction of screen width (0.0-1.0). 0 = point click.
            h_pct: Bounding box height as fraction of screen height (0.0-1.0). 0 = point click.
            resolution: Screen resolution when this node was recorded [w, h].
            confidence_threshold: Minimum match confidence for locating this element.
            context_links: Navigational context — list of dicts describing where
                this element leads or what page/state it belongs to. Each dict
                has keys like "url", "page_title", "app_state", "parent_region".
                Used for building the site/app GPS map.

        Returns:
            The node_id of the newly added node.

        Raises:
            ValueError: If element_type or layer is invalid.
        """
        # Validate element_type
        try:
            ElementType(element_type)
        except ValueError:
            raise ValueError(
                f"Invalid element_type '{element_type}'. "
                f"Valid values: {[e.value for e in ElementType]}"
            )

        # Validate layer
        if layer not in _VALID_LAYERS:
            raise ValueError(
                f"Invalid layer '{layer}'. Valid values: {sorted(_VALID_LAYERS)}"
            )

        nid = node_id or _gen_id()
        now = _now_iso()

        self._graph.add_node(
            nid,
            node_id=nid,
            skill_id=self.skill_id,
            element_type=element_type,
            layer=layer,
            label=label,
            label_ai_guess=label_ai_guess,
            ocr_text=ocr_text,
            embedding_id=embedding_id,
            snippet_path=snippet_path,
            relative_position={
                "x_pct": x_pct,
                "y_pct": y_pct,
                "w_pct": w_pct,
                "h_pct": h_pct,
                "region_hint": _region_hint(x_pct, y_pct),
            },
            last_seen_resolution=list(resolution),
            confidence_threshold=confidence_threshold,
            context_links=context_links or [],
            verified_count=0,
            fail_count=0,
            created_at=now,
            updated_at=now,
        )

        logger.debug("Added node %s: %s (%s)", nid[:8], label, element_type)
        return nid

    def get_node(self, node_id: str) -> dict[str, Any]:
        """Returns the full data dict for a node.

        Args:
            node_id: The UUID of the node.

        Returns:
            Dict with all node attributes.

        Raises:
            KeyError: If node_id does not exist.
        """
        if node_id not in self._graph.nodes:
            raise KeyError(f"Node not found: {node_id}")
        return dict(self._graph.nodes[node_id])

    def update_node(self, node_id: str, **attrs: Any) -> None:
        """Updates attributes on an existing node.

        Args:
            node_id: The UUID of the node.
            **attrs: Key-value pairs to update.

        Raises:
            KeyError: If node_id does not exist.
        """
        if node_id not in self._graph.nodes:
            raise KeyError(f"Node not found: {node_id}")

        # Validate element_type if being updated
        if "element_type" in attrs:
            try:
                ElementType(attrs["element_type"])
            except ValueError:
                raise ValueError(f"Invalid element_type: {attrs['element_type']}")

        # Validate layer if being updated
        if "layer" in attrs:
            if attrs["layer"] not in _VALID_LAYERS:
                raise ValueError(f"Invalid layer: {attrs['layer']}")

        attrs["updated_at"] = _now_iso()
        self._graph.nodes[node_id].update(attrs)

    def remove_node(self, node_id: str) -> None:
        """Removes a node and all its connected edges.

        Args:
            node_id: The UUID of the node.

        Raises:
            KeyError: If node_id does not exist.
        """
        if node_id not in self._graph.nodes:
            raise KeyError(f"Node not found: {node_id}")
        self._graph.remove_node(node_id)
        logger.debug("Removed node %s", node_id[:8])

    @property
    def node_count(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def nodes(self) -> list[str]:
        """Returns a list of all node IDs."""
        return list(self._graph.nodes)

    # -------------------------------------------------------------------
    # Edge CRUD
    # -------------------------------------------------------------------

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        action_type: str,
        *,
        edge_id: str | None = None,
        action_payload: str | None = None,
        is_branch: bool = False,
        branch_condition: str | None = None,
    ) -> str:
        """Adds a directed edge (action/transition) between two nodes.

        Args:
            source_node_id: Node where the action is performed.
            target_node_id: Node representing the result state.
            action_type: ElementType value describing what action was taken.
            edge_id: Optional explicit UUID. Generated if not provided.
            action_payload: Text to type, key to press, etc.
            is_branch: Whether this edge represents a conditional path.
            branch_condition: Human-described condition for this branch.

        Returns:
            The edge_id of the newly added edge.

        Raises:
            KeyError: If source or target node doesn't exist.
        """
        if source_node_id not in self._graph.nodes:
            raise KeyError(f"Source node not found: {source_node_id}")
        if target_node_id not in self._graph.nodes:
            raise KeyError(f"Target node not found: {target_node_id}")

        eid = edge_id or _gen_id()

        self._graph.add_edge(
            source_node_id,
            target_node_id,
            edge_id=eid,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            action_type=action_type,
            action_payload=action_payload,
            is_branch=is_branch,
            branch_condition=branch_condition,
            execution_count=0,
            success_count=0,
        )

        logger.debug(
            "Added edge %s: %s -> %s (%s)",
            eid[:8],
            source_node_id[:8],
            target_node_id[:8],
            action_type,
        )
        return eid

    def get_edge(self, source_id: str, target_id: str) -> dict[str, Any]:
        """Returns the full data dict for an edge.

        Args:
            source_id: Source node UUID.
            target_id: Target node UUID.

        Returns:
            Dict with all edge attributes.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self._graph.has_edge(source_id, target_id):
            raise KeyError(f"Edge not found: {source_id} -> {target_id}")
        return dict(self._graph.edges[source_id, target_id])

    def get_edge_by_id(self, edge_id: str) -> dict[str, Any]:
        """Looks up an edge by its edge_id.

        Args:
            edge_id: The UUID of the edge.

        Returns:
            Dict with all edge attributes.

        Raises:
            KeyError: If no edge with that ID exists.
        """
        for u, v, data in self._graph.edges(data=True):
            if data.get("edge_id") == edge_id:
                return dict(data)
        raise KeyError(f"Edge not found by ID: {edge_id}")

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Removes an edge between two nodes.

        Args:
            source_id: Source node UUID.
            target_id: Target node UUID.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self._graph.has_edge(source_id, target_id):
            raise KeyError(f"Edge not found: {source_id} -> {target_id}")
        self._graph.remove_edge(source_id, target_id)
        logger.debug("Removed edge %s -> %s", source_id[:8], target_id[:8])

    @property
    def edge_count(self) -> int:
        """Returns the number of edges in the graph."""
        return self._graph.number_of_edges()

    # -------------------------------------------------------------------
    # Execution tracking
    # -------------------------------------------------------------------

    def record_execution(
        self, source_id: str, target_id: str, success: bool
    ) -> None:
        """Records the result of executing an edge (action).

        Updates the edge's execution_count and success_count.
        Also updates the target node's verified_count or fail_count.

        Args:
            source_id: Source node UUID.
            target_id: Target node UUID.
            success: Whether the action succeeded.
        """
        # Update edge stats
        if self._graph.has_edge(source_id, target_id):
            edge_data = self._graph.edges[source_id, target_id]
            edge_data["execution_count"] = edge_data.get("execution_count", 0) + 1
            if success:
                edge_data["success_count"] = edge_data.get("success_count", 0) + 1

        # Update target node stats
        if target_id in self._graph.nodes:
            node_data = self._graph.nodes[target_id]
            if success:
                node_data["verified_count"] = node_data.get("verified_count", 0) + 1
            else:
                node_data["fail_count"] = node_data.get("fail_count", 0) + 1
            node_data["updated_at"] = _now_iso()

    def success_rate(self, source_id: str, target_id: str) -> float:
        """Returns the success rate for an edge.

        Args:
            source_id: Source node UUID.
            target_id: Target node UUID.

        Returns:
            Float between 0.0 and 1.0. Returns 0.5 if never executed
            (optimistic default to allow first attempt).
        """
        if not self._graph.has_edge(source_id, target_id):
            return 0.0

        data = self._graph.edges[source_id, target_id]
        total = data.get("execution_count", 0)
        if total == 0:
            return 0.5  # Optimistic default for untried edges
        return data.get("success_count", 0) / total

    # -------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------

    def get_nodes_by_type(self, element_type: str) -> list[str]:
        """Returns all node IDs matching the given element type.

        Args:
            element_type: ElementType value string.

        Returns:
            List of matching node IDs.
        """
        return [
            nid
            for nid, data in self._graph.nodes(data=True)
            if data.get("element_type") == element_type
        ]

    def get_nodes_by_layer(self, layer: str) -> list[str]:
        """Returns all node IDs in the given layer.

        Args:
            layer: Layer type string.

        Returns:
            List of matching node IDs.
        """
        return [
            nid
            for nid, data in self._graph.nodes(data=True)
            if data.get("layer") == layer
        ]

    def get_successors(self, node_id: str) -> list[str]:
        """Returns all direct successor node IDs.

        Args:
            node_id: The UUID of the node.

        Returns:
            List of successor node IDs.
        """
        if node_id not in self._graph.nodes:
            return []
        return list(self._graph.successors(node_id))

    def get_predecessors(self, node_id: str) -> list[str]:
        """Returns all direct predecessor node IDs.

        Args:
            node_id: The UUID of the node.

        Returns:
            List of predecessor node IDs.
        """
        if node_id not in self._graph.nodes:
            return []
        return list(self._graph.predecessors(node_id))

    def get_entry_nodes(self) -> list[str]:
        """Returns nodes with no predecessors (potential start points)."""
        return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

    def get_exit_nodes(self) -> list[str]:
        """Returns nodes with no successors (potential end points)."""
        return [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]

    def get_read_here_nodes(self) -> list[str]:
        """Returns all READ_HERE nodes (data extraction points)."""
        return self.get_nodes_by_type("read_here")

    def get_fingerprint_nodes(self) -> list[str]:
        """Returns all FINGERPRINT nodes (app identity checkpoints)."""
        return self.get_nodes_by_type("fingerprint")

    def get_destination_nodes(self) -> list[str]:
        """Returns all DESTINATION nodes (success state markers)."""
        return self.get_nodes_by_type("destination")

    def get_branch_points(self) -> list[str]:
        """Returns all BRANCH_POINT nodes (conditional forks)."""
        return self.get_nodes_by_type("branch_point")

    def get_branch_edges(self, node_id: str) -> list[dict[str, Any]]:
        """Returns all outgoing edges from a branch point with their conditions.

        Args:
            node_id: The UUID of a branch_point node.

        Returns:
            List of edge data dicts, each with branch_condition and target info.
        """
        if node_id not in self._graph.nodes:
            return []
        result = []
        for _, target, data in self._graph.out_edges(node_id, data=True):
            if data.get("is_branch", False):
                edge_info = dict(data)
                edge_info["target_label"] = self._graph.nodes[target].get("label", "")
                result.append(edge_info)
        return result

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Exposes the underlying NetworkX DiGraph for pathfinding etc."""
        return self._graph

    # -------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializes the entire graph to a JSON-compatible dict.

        Returns:
            Dict with "skill_id", "nodes" (list), and "edges" (list).
        """
        nodes_list = []
        for nid, data in self._graph.nodes(data=True):
            node_dict = dict(data)
            nodes_list.append(node_dict)

        edges_list = []
        for u, v, data in self._graph.edges(data=True):
            edge_dict = dict(data)
            edges_list.append(edge_dict)

        return {
            "skill_id": self.skill_id,
            "nodes": nodes_list,
            "edges": edges_list,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCSDGraph:
        """Deserializes a graph from a dict (e.g., loaded from JSON skill file).

        Args:
            data: Dict with "skill_id", "nodes", and "edges".

        Returns:
            A new OCSDGraph populated with the provided data.
        """
        graph = cls(skill_id=data.get("skill_id", ""))

        # Add nodes
        for node_data in data.get("nodes", []):
            nid = node_data.get("node_id")
            if nid is None:
                logger.warning("Skipping node without node_id")
                continue
            # Add node directly to NetworkX graph to preserve all attributes
            graph._graph.add_node(nid, **node_data)

        # Add edges
        for edge_data in data.get("edges", []):
            src = edge_data.get("source_node_id")
            tgt = edge_data.get("target_node_id")
            if src is None or tgt is None:
                logger.warning("Skipping edge without source/target")
                continue
            if src not in graph._graph.nodes or tgt not in graph._graph.nodes:
                logger.warning(
                    "Skipping edge %s -> %s: node not in graph",
                    src[:8] if src else "?",
                    tgt[:8] if tgt else "?",
                )
                continue
            graph._graph.add_edge(src, tgt, **edge_data)

        logger.info(
            "Loaded graph: %d nodes, %d edges",
            graph.node_count,
            graph.edge_count,
        )
        return graph

    def __repr__(self) -> str:
        return (
            f"OCSDGraph(skill_id={self.skill_id!r}, "
            f"nodes={self.node_count}, edges={self.edge_count})"
        )
