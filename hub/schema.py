"""JSON schema validator for OCSD skill files.

Validates that skill data dicts contain all required fields with
correct types before they are loaded or shared.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Top-level required fields and their expected types
_REQUIRED_TOP_LEVEL: dict[str, type | tuple[type, ...]] = {
    "$schema": str,
    "skill_id": str,
    "name": str,
    "nodes": list,
    "edges": list,
}

# Required fields for each node
_REQUIRED_NODE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "node_id": str,
    "element_type": str,
    "label": str,
}

# Required fields for each edge
_REQUIRED_EDGE_FIELDS: dict[str, type | tuple[type, ...]] = {
    "edge_id": str,
    "source_node_id": str,
    "target_node_id": str,
    "action_type": str,
}


def _check_required_fields(
    data: dict[str, Any],
    required: dict[str, type | tuple[type, ...]],
    context: str,
) -> list[str]:
    """Checks that a dict contains all required fields with correct types.

    Args:
        data: The dict to validate.
        required: Mapping of field name to expected type(s).
        context: Human-readable description for error messages
            (e.g., "top-level", "node[0]").

    Returns:
        List of validation error strings (empty if valid).
    """
    errors: list[str] = []
    for field_name, expected_type in required.items():
        if field_name not in data:
            errors.append(f"{context}: missing required field '{field_name}'")
        elif not isinstance(data[field_name], expected_type):
            actual = type(data[field_name]).__name__
            if isinstance(expected_type, tuple):
                expected_name = "/".join(t.__name__ for t in expected_type)
            else:
                expected_name = expected_type.__name__
            errors.append(
                f"{context}: field '{field_name}' should be {expected_name}, "
                f"got {actual}"
            )
    return errors


def validate_skill(data: dict) -> list[str]:
    """Validates an OCSD skill data dict against the required schema.

    Checks for:
    - Required top-level fields: $schema, skill_id, name, nodes, edges
    - Required node fields: node_id, element_type, label
    - Required edge fields: edge_id, source_node_id, target_node_id, action_type

    Args:
        data: The skill data dict to validate.

    Returns:
        List of validation error strings. An empty list means the skill
        is valid.
    """
    errors: list[str] = []

    # Validate top-level fields
    errors.extend(_check_required_fields(data, _REQUIRED_TOP_LEVEL, "top-level"))

    # Validate nodes (only if nodes field exists and is a list)
    nodes = data.get("nodes")
    if isinstance(nodes, list):
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(f"nodes[{i}]: expected a dict, got {type(node).__name__}")
                continue
            errors.extend(
                _check_required_fields(node, _REQUIRED_NODE_FIELDS, f"nodes[{i}]")
            )

    # Validate edges (only if edges field exists and is a list)
    edges = data.get("edges")
    if isinstance(edges, list):
        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"edges[{i}]: expected a dict, got {type(edge).__name__}")
                continue
            errors.extend(
                _check_required_fields(edge, _REQUIRED_EDGE_FIELDS, f"edges[{i}]")
            )

    if errors:
        logger.warning("Skill validation found %d errors", len(errors))
        for err in errors:
            logger.debug("  %s", err)
    else:
        logger.debug("Skill validation passed")

    return errors
