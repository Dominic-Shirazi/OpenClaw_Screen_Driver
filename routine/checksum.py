"""SHA256 checksum computation for routine integrity verification.

The checksum covers only steps and graph data (not mutable metadata like
name, description, tags, or updated_at) so that routine identity is stable
across metadata-only edits.
"""

from __future__ import annotations

# Stub — will be implemented in GREEN phase
