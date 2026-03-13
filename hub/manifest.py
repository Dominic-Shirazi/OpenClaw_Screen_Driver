"""Skill versioning and metadata management.

Provides a dataclass for skill manifests and functions to load/save
them as JSON files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillManifest:
    """Metadata and versioning information for an OCSD skill.

    Attributes:
        skill_id: Unique identifier for the skill.
        name: Human-readable skill name.
        version: Semantic version string (e.g., "1.0.0").
        author: Name or handle of the skill creator.
        description: Brief description of what the skill does.
        target_app: The application this skill automates (e.g., "Chrome").
        os_list: Operating systems this skill supports.
        tags: Searchable tags for discovery.
        checksum: SHA-256 hex digest of the skill file for integrity.
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 last-updated timestamp.
        download_url: URL to fetch the skill package from the hub.
    """

    skill_id: str
    name: str
    version: str = "0.1.0"
    author: str = ""
    description: str = ""
    target_app: str = ""
    os_list: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    checksum: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    download_url: str = ""


def load_manifest(path: Path) -> SkillManifest:
    """Loads a SkillManifest from a JSON file.

    Args:
        path: Path to the JSON manifest file.

    Returns:
        A populated SkillManifest instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        TypeError: If the JSON structure doesn't match SkillManifest fields.
    """
    logger.debug("Loading manifest from %s", path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = SkillManifest(**data)
    logger.info("Loaded manifest: %s v%s", manifest.name, manifest.version)
    return manifest


def save_manifest(manifest: SkillManifest, path: Path) -> None:
    """Saves a SkillManifest to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        manifest: The manifest to serialize.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved manifest to %s", path)
