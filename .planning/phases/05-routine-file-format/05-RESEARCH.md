# Phase 5: Routine File Format - Research

**Researched:** 2026-03-18
**Domain:** JSON schema design, file serialization, NetworkX graph round-trip, cross-platform metadata detection
**Confidence:** HIGH

## Summary

Phase 5 evolves the existing `ocsd-routine-v0` format (written by Phase 4's `RecordSession._save_routine()`) into a fully specified `ocsd-routine-v1` schema. The codebase already has strong foundations: `mapper/export.py` provides a proven SHA256 checksum pattern, `mapper/graph.py` provides `OCSDGraph.to_dict()`/`from_dict()` for graph serialization, and `recorder/record_session.py` shows the current v0 step format that must be evolved.

The primary work is: (1) define the v1 JSON schema with all new metadata fields, (2) build a `Routine` model class for load/save/validate, (3) rename snippet/embedding files from `step_NN` to `{node_id}`, (4) implement checksum integrity, (5) add auto-detection of platform/programs/theme metadata, and (6) ensure perfect graph round-trip fidelity.

**Primary recommendation:** Use a dataclass-based `Routine` model in a new `routine/format.py` module. Reuse the checksum pattern from `mapper/export.py` verbatim. Keep the v0-to-v1 migration path as a pure in-memory transform (detect `$schema` on load, upgrade transparently, do not rewrite to disk).

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Separate formats**: `ocsd-routine-v1` for recorded routines, `ocsd-skill-v1` stays for the OpenClaw skill layer. No converter/bridge needed
- **Integrity**: `$schema` version tag + SHA256 checksum over steps + graph data + empty `signature` field placeholder for future Hub trust scoring
- **V0->V1 migration**: Build the auto-upgrade path (detect schema version on load, transparently upgrade in memory) but leave it OFF by default. Don't rewrite v0 files to disk until a verification method is in place
- **Both co-exist, cross-validated**: routine.json contains a flat `steps` array AND the full NetworkX graph dict. Conflict resolution: prefer steps if disagreement
- **Full OCSDGraph schema**: Graph section includes everything from `OCSDGraph.to_dict()` -- node IDs, layers, execution stats, context_links
- **1:1 step-node mapping**: Each step references a graph node by `node_id`. Step order = execution order
- **Linear only in V1**: Steps are a flat ordered list. No branching/looping in format structure
- **Exact OCSDGraph.to_dict()**: Graph section is raw output of `OCSDGraph.to_dict()`. Load back with `OCSDGraph.from_dict()`. Zero translation layer
- **Recording resolution stored**: Top-level `resolution` field ([w, h])
- **Four anchor strategies**: `visual_match`, `ocr_text`, `position_pct`, `region_hint`. All four always populated if data exists
- **Snippets at original resolution**: No normalization. Replay engine handles scaling
- **Semver versioning**: `version` field uses semantic versioning, starts at "1.0.0"
- **Author = machine identity**: Auto-populated with hostname. `author_display` for optional user display name
- **Structured categories + free tags**: `category` from fixed list + free-form `tags`
- **Programs, platform, theme**: Auto-detected during recording. User can edit later
- **Pretty-printed**: 2-space indentation, `ensure_ascii=False`
- **Logical field ordering**: `$schema` -> identity -> author/timestamps -> metadata -> start_from, resolution -> steps -> graph -> checksum, signature
- **Directory scan**: Enumerate `~/.ocsd/routines/` subdirectories containing `routine.json`. No index file
- **Node ID based naming**: `snippets/{node_id}.png`, `embeddings/{node_id}.npy`
- **Raw .npy vectors**: One file per element's CLIP embedding. No FAISS index per routine
- **Graceful degradation**: Missing snippets/embeddings = skip that anchor strategy, never hard-fail

### Claude's Discretion
- Exact SHA256 checksum implementation details (what fields to include/exclude)
- Internal Routine model class design (dataclass, Pydantic, plain dict)
- How to implement theme auto-detection from screenshot luminance
- How to detect foreground app name cross-platform (Windows vs Ubuntu)
- Migration function internals for v0->v1 upgrade
- Whether to use OrderedDict or custom JSON encoder for field ordering

### Deferred Ideas (OUT OF SCOPE)
- Routine-to-skill bridge/converter
- FAISS index per routine
- Per-routine VLM fallback config
- SQLite routine catalog
- Embedding deduplication across routines
- "Screen CLI" mode

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FMT-01 | Routine stored as human-readable JSON with steps, bbox, region_hint, snippet/embedding paths | v1 schema definition with pretty-printing, logical field ordering, node_id-based asset paths |
| FMT-02 | Each step includes VLM metadata (element_type, label, caption, confidence) | Step schema evolution from v0 `_step_to_json()` adding anchor strategies and VLM fields |
| FMT-03 | Snippets stored as +30% padded PNGs in snippets/ directory | Existing `_save_snippets_and_embeddings()` adapted for node_id naming |
| FMT-04 | CLIP embeddings stored as .npy files in embeddings/ directory | Same adaptation -- `embeddings/{node_id}.npy` instead of `step_NN.npy` |
| FMT-05 | NetworkX graph serializes to/from routine.json (graph round-trips) | `OCSDGraph.to_dict()`/`from_dict()` used directly, verified by round-trip test |
| FMT-06 | Storage layout: `~/.ocsd/routines/{name}/routine.json` + snippets/ + embeddings/ | Already created by Phase 4; this phase formalizes and validates |
| FMT-07 | Routine file includes metadata: name, version, created, author, description, tags | Extended metadata: category, programs, platform, theme, author_display, signature placeholder |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| json (stdlib) | 3.x | JSON serialization/deserialization | Already used by project, no external dependency |
| hashlib (stdlib) | 3.x | SHA256 checksum computation | Already proven in `mapper/export.py` |
| pathlib (stdlib) | 3.x | Cross-platform file paths | CLAUDE.md requirement |
| dataclasses (stdlib) | 3.x | Routine model class | Lightweight, no external dep, type-hintable |
| socket (stdlib) | 3.x | `gethostname()` for author field | Cross-platform hostname detection |
| numpy | existing | `.npy` file I/O for embeddings | Already a project dependency |
| networkx | existing | Graph serialization via OCSDGraph | Already a project dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cv2 (opencv) | existing | Snippet PNG I/O, luminance calculation for theme detection | Already imported in capture/record_session |
| pyautogui | existing | Screen resolution detection | Already used in `_save_routine()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| dataclass | Pydantic | Pydantic adds validation DSL but is an extra dependency; dataclass + manual validation is sufficient and lighter |
| OrderedDict | Custom JSONEncoder with key ordering | Custom encoder is cleaner -- a subclass of `json.JSONEncoder` that sorts keys in the desired logical order |
| json.dumps sort_keys | Custom key ordering function | `sort_keys=True` alphabetizes; we want logical ordering. Use a helper that builds an OrderedDict in desired order before serialization |

## Architecture Patterns

### Recommended Project Structure
```
routine/
    __init__.py
    format.py          # Routine dataclass, save/load/validate, v1 schema
    migration.py       # v0->v1 upgrade logic (in-memory only)
    discovery.py       # Enumerate routines from ~/.ocsd/routines/
    checksum.py        # SHA256 computation (reuse pattern from mapper/export.py)
```

### Pattern 1: Routine Dataclass Model
**What:** A `@dataclass` representing a complete routine with typed fields, plus `to_dict()`, `from_dict()`, `save()`, `load()` methods.
**When to use:** All routine I/O.
**Example:**
```python
# Source: project convention (dataclass pattern)
from __future__ import annotations

import hashlib
import json
import logging
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mapper.graph import OCSDGraph

logger = logging.getLogger(__name__)

VALID_CATEGORIES = [
    "productivity", "web", "system", "finance", "creative", "other",
]

@dataclass
class Routine:
    """In-memory representation of an ocsd-routine-v1 file."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = field(default_factory=socket.gethostname)
    author_display: str | None = None
    category: str = "other"
    tags: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    platform: str = field(default_factory=lambda: _detect_platform())
    theme: str | None = None
    start_from: str = "desktop"
    resolution: list[int] = field(default_factory=lambda: [1920, 1080])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    steps: list[dict[str, Any]] = field(default_factory=list)
    graph: OCSDGraph = field(default_factory=OCSDGraph)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to ordered dict for JSON output."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Routine:
        """Deserialize from loaded JSON dict."""
        ...

    def save(self, directory: Path) -> None:
        """Write routine.json to directory."""
        ...

    @classmethod
    def load(cls, directory: Path) -> Routine:
        """Load routine from directory containing routine.json."""
        ...
```

### Pattern 2: Ordered JSON Serialization
**What:** Build a plain dict with keys in the exact logical order specified in CONTEXT.md, then `json.dump()` it.
**When to use:** Every save operation.
**Example:**
```python
def _ordered_routine_dict(routine: Routine) -> dict[str, Any]:
    """Build dict with keys in schema-specified order."""
    graph_dict = routine.graph.to_dict()
    steps = routine.steps
    checksum = calculate_routine_checksum(steps, graph_dict)

    return {
        "$schema": "ocsd-routine-v1",
        # Identity
        "name": routine.name,
        "version": routine.version,
        "description": routine.description,
        # Author / timestamps
        "author": routine.author,
        "author_display": routine.author_display,
        "created_at": routine.created_at,
        "updated_at": routine.updated_at,
        # Metadata
        "category": routine.category,
        "tags": routine.tags,
        "programs": routine.programs,
        "platform": routine.platform,
        "theme": routine.theme,
        # Recording context
        "start_from": routine.start_from,
        "resolution": routine.resolution,
        # Data
        "steps": steps,
        "graph": graph_dict,
        # Integrity
        "checksum": checksum,
        "signature": None,
    }
```

Python 3.7+ dicts maintain insertion order, so this pattern preserves the logical field ordering without needing OrderedDict.

### Pattern 3: Checksum Computation
**What:** SHA256 over stable JSON of steps + graph data, reusing the pattern from `mapper/export.py`.
**When to use:** Save and load (integrity verification).
**Example:**
```python
def calculate_routine_checksum(
    steps: list[dict[str, Any]],
    graph_dict: dict[str, Any],
) -> str:
    """SHA256 over steps + graph for integrity verification."""
    combined = {
        "steps": steps,
        "graph": graph_dict,
    }
    data = json.dumps(
        combined,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
```

### Pattern 4: v1 Step Schema
**What:** Evolved step format with node_id references and anchor strategies.
**Example:**
```python
def _build_v1_step(
    step: dict,
    index: int,
    node_id: str,
    screen_w: int,
    screen_h: int,
) -> dict[str, Any]:
    """Build a v1 step dict from internal step data."""
    tag_data = step.get("tag_data", {})
    bbox = step.get("bbox")
    bx, by, bw, bh = bbox if bbox else (0, 0, 0, 0)

    x_pct = bx / screen_w if screen_w > 0 else 0.0
    y_pct = by / screen_h if screen_h > 0 else 0.0
    w_pct = bw / screen_w if screen_w > 0 else 0.0
    h_pct = bh / screen_h if screen_h > 0 else 0.0
    cx_pct = (bx + bw / 2) / screen_w if screen_w > 0 else 0.5
    cy_pct = (by + bh / 2) / screen_h if screen_h > 0 else 0.5

    region = _compute_region_hint(
        bx + bw // 2, by + bh // 2, screen_w, screen_h,
    )

    return {
        "step_index": index,
        "node_id": node_id,
        "element_type": tag_data.get("element_type", "unknown"),
        "label": tag_data.get("label", ""),
        "caption": tag_data.get("caption", ""),
        "confidence": tag_data.get("confidence", 0.0),
        "action": tag_data.get("action", "click"),
        "bbox": {"x": bx, "y": by, "w": bw, "h": bh},
        "bbox_pct": {
            "x_pct": x_pct,
            "y_pct": y_pct,
            "w_pct": w_pct,
            "h_pct": h_pct,
        },
        "region_hint": region,
        "anchors": {
            "visual_match": f"snippets/{node_id}.png",
            "ocr_text": tag_data.get("ocr_text"),
            "position_pct": {
                "x_pct": cx_pct,
                "y_pct": cy_pct,
            },
            "region_hint": region,
        },
        "snippet_path": f"snippets/{node_id}.png",
        "embedding_path": f"embeddings/{node_id}.npy",
        "dry_run_passed": True,
    }
```

### Anti-Patterns to Avoid
- **Storing graph AND steps with different data:** Steps and graph nodes must have 1:1 mapping. Node IDs in steps must match graph node IDs exactly.
- **Relative paths outside the routine directory:** All asset paths (snippet_path, embedding_path) must be relative to the routine directory, never absolute.
- **Writing v0 files to disk during migration:** The CONTEXT.md explicitly says do NOT rewrite v0 files to disk -- only upgrade in memory.
- **Using step index for file naming:** v0 used `step_00.png`; v1 uses `{node_id}.png`. This is critical for future element sharing.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Graph serialization | Custom graph-to-JSON | `OCSDGraph.to_dict()`/`from_dict()` | Already tested, handles nodes+edges+metadata |
| Checksum computation | New hash approach | Adapt `mapper/export.calculate_checksum()` pattern | Proven stable, sort_keys+separators prevents drift |
| UUID generation | Custom ID scheme | `uuid.uuid4()` (already used in OCSDGraph) | Standard, collision-resistant |
| Screenshot luminance | Complex color analysis | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` + `np.mean()` | Simple mean grayscale value, threshold at ~128 for dark/light |
| Cross-platform hostname | Per-OS detection code | `socket.gethostname()` | Works on Windows, macOS, Linux identically |

## Common Pitfalls

### Pitfall 1: Graph Round-Trip Lossy Serialization
**What goes wrong:** Graph data gets subtly different after save -> load -> save cycle. Floating point drift, key reordering, or missing attributes.
**Why it happens:** NetworkX stores arbitrary Python objects in node/edge data. JSON serialization may lose type information (tuples become lists, etc.).
**How to avoid:** Use the round-trip test pattern from `test_export.py`. Compare `to_dict()` output before and after a save/load cycle using `json.dumps(sort_keys=True)` comparison. The existing `OCSDGraph.from_dict()` already handles this correctly by passing `**node_data` directly.
**Warning signs:** Test failures on round-trip assertion, especially with resolution tuples becoming lists.

### Pitfall 2: Step-Graph Cross-Validation Mismatch
**What goes wrong:** Steps reference node_ids that don't exist in the graph, or graph has nodes not referenced by any step.
**Why it happens:** Steps and graph are built separately and can drift during save.
**How to avoid:** Validate at save time: every step's `node_id` must exist in the graph, and every graph node must be referenced by exactly one step (for v1 linear routines). Warn but don't fail on load -- prefer steps.
**Warning signs:** KeyError when loading graph nodes referenced by steps.

### Pitfall 3: Checksum Including Mutable Fields
**What goes wrong:** Checksum includes `updated_at` or `version` fields, causing verification failure after legitimate metadata edits.
**Why it happens:** Overly broad checksum scope.
**How to avoid:** Checksum should cover only `steps` + `graph` data (the behavioral content). Metadata fields like name, description, tags, version, timestamps are excluded. This matches the integrity goal: detecting if the automation behavior was tampered with.
**Warning signs:** Checksum mismatch after changing routine name or bumping version.

### Pitfall 4: Platform-Specific Path Separators in JSON
**What goes wrong:** Snippet paths stored as `snippets\node_id.png` on Windows, breaking on Linux/macOS.
**Why it happens:** Using `str(Path(...))` on Windows produces backslashes.
**How to avoid:** Always use forward slashes in JSON asset paths. Build paths as strings with `/` separator, or use `.as_posix()` on Path objects.
**Warning signs:** File not found errors on cross-platform routine sharing.

### Pitfall 5: Theme Auto-Detection Returning Wrong Results
**What goes wrong:** Screenshot taken with overlay visible skews luminance calculation.
**Why it happens:** Theme detection runs at wrong time in the pipeline.
**How to avoid:** Detect theme from the clean screenshot (the one captured after overlay hides). Use the full screenshot's mean luminance: `< 128 = "dark"`, `>= 128 = "light"`.
**Warning signs:** All routines tagged as "dark" regardless of actual app theme.

## Code Examples

### Existing v0 Format (from record_session.py line 1016)
```python
# Source: recorder/record_session.py:_save_routine()
routine_data = {
    "$schema": "ocsd-routine-v0",
    "name": self._routine_name,
    "description": f"Recorded routine: {self._routine_name}",
    "start_from": self._start_from,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "resolution": [screen_w, screen_h],
    "steps": [...],
    "graph": graph.to_dict(),
}
```

### v0 Step Format (from record_session.py line 63)
```python
# Source: recorder/record_session.py:_step_to_json()
{
    "step_index": index,
    "node_id": step.get("node_id", str(uuid4())),
    "element_type": tag_data.get("element_type", "unknown"),
    "label": tag_data.get("label", ""),
    "caption": tag_data.get("caption", ""),
    "action": tag_data.get("action", "click"),
    "bbox": {"x": bbox_x, "y": bbox_y, "w": bbox_w, "h": bbox_h},
    "bbox_pct": {"x_pct": ..., "y_pct": ..., "w_pct": ..., "h_pct": ...},
    "region_hint": "...",
    "snippet_path": "snippets/step_00.png",       # <-- v0: index-based
    "embedding_path": "embeddings/step_00.npy",   # <-- v0: index-based
    "confidence": 0.0,
    "dry_run_passed": True,
}
```

### Existing Checksum Pattern (from mapper/export.py)
```python
# Source: mapper/export.py:calculate_checksum()
def calculate_checksum(nodes, edges):
    stable_nodes = sorted(nodes, key=lambda x: x.get("node_id", ""))
    stable_edges = sorted(edges, key=lambda x: x.get("edge_id", ""))
    combined = {"nodes": stable_nodes, "edges": stable_edges}
    data = json.dumps(combined, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
```

### Foreground App Detection (Cross-Platform)
```python
# Source: research recommendation
def detect_foreground_app() -> str:
    """Detect the foreground application name."""
    if sys.platform == "win32":
        # Already available: core/capture.py:get_window_title()
        from core.capture import get_window_title
        title = get_window_title()
        # Extract app name from title (e.g., "Document - Word" -> "Word")
        return title.rsplit(" - ", 1)[-1] if " - " in title else title
    elif sys.platform == "linux":
        try:
            import subprocess
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                title = result.stdout.strip()
                return title.rsplit(" - ", 1)[-1] if " - " in title else title
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return ""
```

### Theme Detection from Screenshot
```python
# Source: research recommendation
def detect_theme(screenshot: np.ndarray) -> str | None:
    """Detect dark/light theme from screenshot luminance."""
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    if mean_lum < 128:
        return "dark"
    return "light"
```

### Graph Round-Trip Verification Pattern
```python
# Source: adapted from tests/test_export.py
def verify_graph_roundtrip(graph: OCSDGraph) -> bool:
    """Verify that to_dict -> from_dict produces identical graph."""
    d1 = graph.to_dict()
    restored = OCSDGraph.from_dict(d1)
    d2 = restored.to_dict()
    # Compare via stable JSON serialization
    j1 = json.dumps(d1, sort_keys=True, ensure_ascii=False)
    j2 = json.dumps(d2, sort_keys=True, ensure_ascii=False)
    return j1 == j2
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v0 index-based naming (`step_00.png`) | v1 node_id-based naming (`{uuid}.png`) | Phase 5 | Enables future element sharing across routines |
| v0 minimal metadata | v1 full metadata (category, tags, programs, platform, theme) | Phase 5 | Supports routine discovery and filtering |
| No integrity verification | SHA256 checksum on steps+graph | Phase 5 | Detects tampering for Hub trust scoring |
| Steps only | Steps + graph co-existing | Phase 4 introduced, Phase 5 formalizes | Graph enables V2 graph-primary migration |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | none -- pytest runs from project root |
| Quick run command | `python -m pytest tests/test_routine_format.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FMT-01 | routine.json is human-readable with steps, bbox, region_hint, paths | unit | `python -m pytest tests/test_routine_format.py::test_save_produces_readable_json -x` | Wave 0 |
| FMT-02 | Each step has element_type, label, caption, confidence | unit | `python -m pytest tests/test_routine_format.py::test_step_vlm_metadata -x` | Wave 0 |
| FMT-03 | Snippets as padded PNGs in snippets/ | unit | `python -m pytest tests/test_routine_format.py::test_snippet_naming -x` | Wave 0 |
| FMT-04 | Embeddings as .npy in embeddings/ | unit | `python -m pytest tests/test_routine_format.py::test_embedding_naming -x` | Wave 0 |
| FMT-05 | Graph round-trips through routine.json | unit | `python -m pytest tests/test_routine_format.py::test_graph_roundtrip -x` | Wave 0 |
| FMT-06 | Storage layout correct | unit | `python -m pytest tests/test_routine_format.py::test_storage_layout -x` | Wave 0 |
| FMT-07 | Metadata fields present | unit | `python -m pytest tests/test_routine_format.py::test_metadata_fields -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_routine_format.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_routine_format.py` -- covers FMT-01 through FMT-07
- [ ] No new conftest fixtures needed -- existing conftest pre-mocks heavy deps; routine tests use stdlib only

## Open Questions

1. **Action type field values for v1**
   - What we know: v0 uses `tag_data.get("action", "click")` which comes from the tag dialog dropdown
   - What's unclear: Phase 6 defines 12 action types (ACT-01 through ACT-12). Should v1 schema anticipate these values?
   - Recommendation: Use the action string from tag dialog as-is. Phase 6 will define the enum. The format just stores whatever string is provided.

2. **v0 files on disk from Phase 4 testing**
   - What we know: Phase 4 created v0 routines during development/testing
   - What's unclear: Are there real v0 routines in `~/.ocsd/routines/` that need migration?
   - Recommendation: Build migration code but leave it off by default (per CONTEXT.md). Test with synthetic v0 data.

## Sources

### Primary (HIGH confidence)
- `mapper/export.py` -- SHA256 checksum pattern, save/load with integrity
- `mapper/graph.py` -- OCSDGraph.to_dict()/from_dict() serialization
- `recorder/record_session.py` -- Current v0 format, step schema, save flow
- `core/embeddings.py` -- CLIP embedding generation and .npy storage
- `core/capture.py` -- Snippet save, window title detection, screenshot
- `tests/test_export.py` -- Round-trip test pattern for graph serialization
- `05-CONTEXT.md` -- All locked decisions and discretion areas

### Secondary (MEDIUM confidence)
- Python 3.7+ dict ordering guarantee -- dicts maintain insertion order (PEP 468, well-established)
- `socket.gethostname()` cross-platform behavior -- verified on Windows, standard POSIX behavior on Linux/macOS

### Tertiary (LOW confidence)
- `xdotool` availability on Ubuntu -- commonly installed but not guaranteed; needs graceful fallback

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all stdlib + already-installed deps
- Architecture: HIGH - clear evolution from existing v0 code with well-defined schema
- Pitfalls: HIGH - based on actual code analysis of existing serialization patterns

**Research date:** 2026-03-18
**Valid until:** 2026-04-18 (stable domain, no external API dependencies)
