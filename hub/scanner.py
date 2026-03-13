"""Automated malicious pattern detection for recorded skills.

Scans skill data for suspicious patterns that might indicate malicious
intent, such as password exfiltration, excessive automation, or
unauthorized system access.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Thresholds
_MAX_MOUSE_STEPS = 100
_MAX_RISK_SCORE = 1.0

# Patterns that suggest password / secret entry
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"api[_\-]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"private[_\-]?key", re.IGNORECASE),
]

# Patterns for URLs and emails being typed
_URL_PATTERN = re.compile(
    r"https?://[^\s]+", re.IGNORECASE
)
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# System paths that skills should not touch
_SYSTEM_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"C:\\Windows\\System32", re.IGNORECASE),
    re.compile(r"/etc/", re.IGNORECASE),
    re.compile(r"/usr/bin/", re.IGNORECASE),
    re.compile(r"/System/", re.IGNORECASE),
    re.compile(r"\\AppData\\", re.IGNORECASE),
    re.compile(r"~/.ssh", re.IGNORECASE),
    re.compile(r"\.env\b", re.IGNORECASE),
]


@dataclass
class ScanResult:
    """Result of scanning a skill for malicious patterns.

    Attributes:
        is_safe: Whether the skill passed all safety checks.
        warnings: Human-readable descriptions of detected issues.
        risk_score: Aggregate risk score from 0.0 (safe) to 1.0 (dangerous).
    """

    is_safe: bool = True
    warnings: list[str] = field(default_factory=list)
    risk_score: float = 0.0


def _count_mouse_steps(nodes: list[dict]) -> int:
    """Counts nodes that represent mouse movement actions.

    Args:
        nodes: List of node dicts from the skill data.

    Returns:
        Number of mouse-movement-like nodes.
    """
    mouse_types = {"mouse_move", "drag", "scroll", "click"}
    return sum(
        1 for n in nodes
        if n.get("element_type", "").lower() in mouse_types
    )


def _collect_typed_text(edges: list[dict]) -> list[str]:
    """Extracts all text payloads from type/keystroke edges.

    Args:
        edges: List of edge dicts from the skill data.

    Returns:
        List of non-empty action_payload strings.
    """
    type_actions = {"type_text", "keystroke", "type"}
    payloads: list[str] = []
    for edge in edges:
        if edge.get("action_type", "").lower() in type_actions:
            payload = edge.get("action_payload", "")
            if payload:
                payloads.append(payload)
    return payloads


def _collect_all_text(skill_data: dict) -> list[str]:
    """Gathers all text content from nodes and edges for scanning.

    Args:
        skill_data: The full skill dict.

    Returns:
        List of all text strings found in labels, payloads, paths, etc.
    """
    texts: list[str] = []
    for node in skill_data.get("nodes", []):
        for key in ("label", "ocr_text", "snippet_path", "action_payload"):
            val = node.get(key, "")
            if val:
                texts.append(str(val))
    for edge in skill_data.get("edges", []):
        for key in ("action_payload", "branch_condition"):
            val = edge.get(key, "")
            if val:
                texts.append(str(val))
    return texts


def scan_skill(skill_data: dict) -> ScanResult:
    """Scans a skill for potentially malicious patterns.

    Checks for:
    - Excessive mouse movements (>100 steps)
    - Typing of URLs or email addresses
    - Typing of password/key/secret patterns
    - References to sensitive system paths

    Args:
        skill_data: The full skill dict with "nodes" and "edges" keys.

    Returns:
        A ScanResult with safety verdict, warnings, and risk score.
    """
    result = ScanResult()
    nodes = skill_data.get("nodes", [])
    edges = skill_data.get("edges", [])
    risk_increments: list[float] = []

    # Check 1: Excessive mouse movements
    mouse_steps = _count_mouse_steps(nodes)
    if mouse_steps > _MAX_MOUSE_STEPS:
        warning = (
            f"Excessive mouse movements detected: {mouse_steps} steps "
            f"(threshold: {_MAX_MOUSE_STEPS})"
        )
        result.warnings.append(warning)
        risk_increments.append(0.2)
        logger.warning(warning)

    # Check 2: Typed text analysis
    typed_texts = _collect_typed_text(edges)
    for text in typed_texts:
        # URL typing
        if _URL_PATTERN.search(text):
            warning = f"Skill types a URL: {text[:80]!r}"
            result.warnings.append(warning)
            risk_increments.append(0.15)
            logger.warning(warning)

        # Email typing
        if _EMAIL_PATTERN.search(text):
            warning = f"Skill types an email address: {text[:80]!r}"
            result.warnings.append(warning)
            risk_increments.append(0.15)
            logger.warning(warning)

        # Secret/password patterns
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                warning = (
                    f"Skill types text matching secret pattern "
                    f"'{pattern.pattern}': {text[:80]!r}"
                )
                result.warnings.append(warning)
                risk_increments.append(0.3)
                logger.warning(warning)
                break  # One match per text is enough

    # Check 3: System path references
    all_texts = _collect_all_text(skill_data)
    for text in all_texts:
        for pattern in _SYSTEM_PATH_PATTERNS:
            if pattern.search(text):
                warning = (
                    f"Reference to sensitive system path "
                    f"(pattern '{pattern.pattern}'): {text[:80]!r}"
                )
                result.warnings.append(warning)
                risk_increments.append(0.25)
                logger.warning(warning)
                break  # One match per text is enough

    # Compute final risk score
    if risk_increments:
        result.risk_score = min(sum(risk_increments), _MAX_RISK_SCORE)

    # Determine safety verdict
    result.is_safe = result.risk_score < 0.5

    logger.info(
        "Scan complete: safe=%s, risk=%.2f, warnings=%d",
        result.is_safe,
        result.risk_score,
        len(result.warnings),
    )
    return result
