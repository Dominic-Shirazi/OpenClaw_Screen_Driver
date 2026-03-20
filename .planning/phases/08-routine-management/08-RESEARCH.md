# Phase 8: Routine Management - Research

**Researched:** 2026-03-20
**Domain:** Routine CRUD operations with overlay-integrated update flow
**Confidence:** HIGH

## Summary

Phase 8 adds four management operations to existing routines: update (guided replay with per-step editing), fork (deep copy with optional truncation), delete (hard `shutil.rmtree`), and inspect (Rich terminal table). The update flow is the most complex -- it reuses Phase 7's replay infrastructure (locate cascade, overlay adapter) but pauses at each step for user editing controls instead of auto-continuing. The fork, delete, and inspect operations are straightforward file-system and data-model operations.

The codebase already has all the building blocks: `Routine.load()`/`.save()` for JSON I/O with checksum, `locate_element_from_step()` for per-step element location, `ReplayOverlayAdapter` for thread-safe overlay updates, `ToolbarMode` enum for context-sensitive toolbar buttons, and `mini_dialogs.py` for pre-filled parameter editing. The primary new code is an `UpdateSession` orchestrator (analogous to `RecordSession`), a new `ToolbarMode.UPDATE` with OK/Edit/Fork/Delete buttons, version bumping helpers, and terminal-facing entry points for fork/delete/inspect.

**Primary recommendation:** Build `UpdateSession` as a new class in `routine/update_session.py` that wraps the same locate-and-display loop as `run_routine()` but pauses per step, exposes toolbar controls, and delegates to `RecordSession`-style capture pipeline for Edit Step. Keep fork/delete/inspect as standalone functions in `routine/management.py`.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Update flow = guided replay model**: Overlay walks through each step; purple shimmer while AI locates, highlight when found, shimmer turns red (user in control)
- **Per-step controls via toolbar**: OK/Continue, Edit Step (full re-record), Fork Here, Delete Step
- **No inserts in V1**: Users cannot insert new steps. Fork-and-re-record is the workaround
- **Non-element steps**: Show mini-dialog pre-filled with current values for modification
- **Element not found**: Show "Could not find [label]" with Skip/Edit/Delete/Abort options
- **Save behavior**: End of walkthrough offers Save & Replace or Save as New
- **Version bump**: Auto-bump minor for step changes (1.0.0 -> 1.1.0), patch for metadata-only
- **Asset alignment**: Unchanged steps keep original snippets/embeddings; edited steps get new ones; deleted steps' assets removed
- **Fork = full copy**: Copy all snippet PNGs and .npy files. Independent routines, no shared state
- **Fork from update**: "Fork Here" at step N creates routine with steps 1..N, drops rest, continues recording
- **Fork naming**: "Save As" dialog pre-filled with original name; error if name exists after editing
- **Standalone fork**: Copy entire routine, then open in update flow
- **Delete = hard delete with confirmation**: Terminal confirmation, `shutil.rmtree` the directory
- **No running check**: Delete allowed even during execution; runner fails gracefully
- **Inspect = Rich table**: Step#, action type, element label, bbox, snippet exists columns
- **Inspect header**: Name, version, step count, programs, platform, created date
- **--json flag**: Output full routine.json for machine consumption

### Claude's Discretion
- How to structure the UpdateSession class (reuse RecordSession patterns or separate)
- How to coordinate overlay state transitions between replay-mode and recording-mode during Edit Step
- Terminal confirmation prompt implementation (Rich.prompt or input())
- Exact Rich table column widths and formatting
- How to handle graph consistency when steps are deleted or re-ordered
- How to implement "Save As" name dialog (Rich prompt or custom widget)

### Deferred Ideas (OUT OF SCOPE)
- Insert step during update -- V2, fork-and-re-record is V1 workaround
- Program/extension dependency model -- V2
- Soft delete with trash/recovery -- V2
- Fork auto-skip-to-divergence -- V2
- Routine diff/comparison view -- V2

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MGMT-01 | Update routine -- step-through existing with controls | UpdateSession class reusing runner locate cascade + overlay adapter pattern; new ToolbarMode.UPDATE with OK/Edit/Fork/Delete buttons |
| MGMT-02 | Fork routine -- copy under new name | `shutil.copytree` for directory copy + Routine.load/save for name change; "Fork Here" truncation in UpdateSession |
| MGMT-03 | Delete routine | `shutil.rmtree` with Rich.Confirm confirmation prompt; use `discovery.get_routine_dir()` for path resolution |
| MGMT-04 | Inspect routine -- print step summary | Rich Table with routine metadata header; `--json` flag for raw routine.json output |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyQt6 | >=6.5 | Overlay for update flow | Already project standard; update flow reuses overlay |
| Rich | >=13.0 | Terminal tables, confirmations | Already in `tui` optional dep; used by record_flow.py and tui.py |
| NetworkX | >=3.0 | Graph consistency after step changes | Already project standard for OCSDGraph |
| shutil (stdlib) | N/A | `copytree` for fork, `rmtree` for delete | Standard library, already used in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| packaging | (stdlib in 3.11+) | Semver version parsing and bumping | Simple string split is sufficient; no extra dep needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual semver parse | `packaging.version` | Overkill for simple X.Y.Z bump; manual split works fine |
| Rich Confirm | `input()` | Rich already a project dep; consistent with existing record_flow.py pattern |

**Installation:**
```bash
# No new dependencies -- everything already in pyproject.toml
# Rich is in [tui] optional group, already installed for recording flow
```

## Architecture Patterns

### Recommended Project Structure
```
routine/
    management.py          # fork_routine(), delete_routine(), inspect_routine()
    update_session.py      # UpdateSession class (guided replay with editing)
    version.py             # bump_minor(), bump_patch() helpers

recorder/overlay/
    toolbar_panel.py       # Add ToolbarMode.UPDATE entry
    state.py               # No changes needed (REPLAYING + RECORDING already exist)
    controller.py          # Add update-mode convenience methods
```

### Pattern 1: UpdateSession as Modified Replay
**What:** UpdateSession wraps the locate-per-step loop from `run_routine()` but instead of auto-continuing, it pauses at each step and waits for user input via toolbar buttons.
**When to use:** Whenever the update flow is active.
**Example:**
```python
# Source: Derived from runner.py run_routine() + record_session.py patterns
class UpdateSession:
    def __init__(self, controller, routine: Routine, routine_dir: Path):
        self._controller = controller
        self._routine = routine
        self._routine_dir = routine_dir
        self._current_step = 0
        self._modified_steps: list[dict] = list(routine.steps)
        self._deleted_indices: set[int] = set()
        self._bridge = PipelineBridge()
        # Wire toolbar button_clicked -> _on_toolbar_action

    def _advance_to_step(self, index: int) -> None:
        """Locate element for step[index], show overlay, wait for user."""
        step = self._modified_steps[index]
        # Purple shimmer while locating
        self._controller.set_replay_mode(True)
        # Locate in background thread
        threading.Thread(target=self._locate_step, args=(step,)).start()

    def _on_toolbar_action(self, action: str) -> None:
        """Handle OK/Edit/Fork/Delete from toolbar."""
        match action:
            case "ok": self._keep_and_advance()
            case "edit": self._enter_edit_mode()
            case "fork_here": self._fork_at_current()
            case "delete_step": self._delete_current_step()
```

### Pattern 2: State Transitions During Edit Step
**What:** When user clicks "Edit Step", overlay transitions from REPLAYING (purple) to RECORDING (red). The full capture pipeline runs (screenshot -> detect -> VLM -> tag dialog). On tag confirm, transition back to REPLAYING and advance.
**When to use:** Edit Step action in update flow.
**Example:**
```python
# Transition flow:
# REPLAYING (purple, showing step highlight)
#   -> User clicks "Edit Step"
#   -> RECORDING (red, click-through off, capture pipeline)
#   -> Tag dialog appears with current values
#   -> User confirms -> new snippet/embedding saved
#   -> REPLAYING (purple, advance to next step)
```

### Pattern 3: Fork as Directory Copy + Truncation
**What:** Fork creates a full directory copy with `shutil.copytree`, then loads the copy, modifies name/metadata, optionally truncates steps, and saves.
**When to use:** Standalone fork or "Fork Here" from update flow.
**Example:**
```python
def fork_routine(
    source_dir: Path,
    new_name: str,
    truncate_at: int | None = None,
) -> Path:
    target_dir = get_routine_dir() / new_name
    shutil.copytree(source_dir, target_dir)
    routine = Routine.load(target_dir)
    routine.name = new_name
    routine.version = "1.0.0"  # Reset version for fork
    if truncate_at is not None:
        routine.steps = routine.steps[:truncate_at]
        # Clean up orphaned snippets/embeddings
        _remove_orphaned_assets(target_dir, routine.steps)
    routine.save(target_dir)
    return target_dir
```

### Pattern 4: Version Bumping
**What:** Simple string-split semver bump. Minor bump for structural changes (steps added/removed/edited), patch for metadata-only.
**When to use:** On save after update flow completes.
**Example:**
```python
def bump_minor(version: str) -> str:
    parts = version.split(".")
    major, minor = int(parts[0]), int(parts[1])
    return f"{major}.{minor + 1}.0"

def bump_patch(version: str) -> str:
    parts = version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    return f"{major}.{minor}.{patch + 1}"
```

### Anti-Patterns to Avoid
- **Modifying routine.steps in-place during walkthrough:** Build a new list of modified steps; only commit changes on save. This avoids partial-save corruption if user aborts mid-update.
- **Sharing snippet files between original and fork:** Always deep copy. Shared references cause cascade failures when one routine is deleted.
- **Removing assets during walkthrough:** Only clean up orphaned snippets/embeddings at save time, not during the step-by-step walk.
- **Re-using the same RecordSession for Edit Step:** Create a lightweight "single-step capture" flow rather than spinning up a full RecordSession. RecordSession manages multi-step recording state that conflicts with update flow.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Directory copy | Manual file-by-file copy | `shutil.copytree()` | Handles symlinks, permissions, nested dirs correctly |
| Directory delete | Manual file iteration + os.remove | `shutil.rmtree()` | Already used in project (run_log.py); handles non-empty dirs |
| Rich terminal table | Manual string formatting | `rich.table.Table` | Already used in tui.py; handles column widths, colors, alignment |
| Confirmation prompts | Custom input loop | `rich.prompt.Confirm` | Already used in record_flow.py; consistent UX |
| Thread-safe overlay updates | Manual QTimer.singleShot | `PipelineBridge` pyqtSignal | Established pattern from Phase 4; thread-safe AutoConnection |

**Key insight:** Every building block already exists in the codebase. The update flow is a composition of existing patterns (locate cascade + overlay adapter + toolbar mode + capture pipeline), not new infrastructure.

## Common Pitfalls

### Pitfall 1: Graph Inconsistency After Step Deletion
**What goes wrong:** Deleting a step from `routine.steps` but leaving its node and edges in the NetworkX graph causes graph serialization to reference non-existent steps.
**Why it happens:** The graph stores `node_id` references that must match steps.
**How to avoid:** When deleting steps, also remove the corresponding node from `OCSDGraph` and clean up edges. Rebuild step indices after deletion.
**Warning signs:** Checksum validation fails after load; graph.to_dict() contains orphan node_ids.

### Pitfall 2: Loop Body References After Step Deletion
**What goes wrong:** Loop steps reference `body_step_node_ids`. If a body step is deleted, the loop references a non-existent node_id.
**Why it happens:** Loop definitions store hard references to other steps' node_ids.
**How to avoid:** When deleting a step, check all loop steps for references to the deleted node_id. Remove the reference from body_step_node_ids. If all body steps are removed, delete the loop step too.
**Warning signs:** Runner crashes with KeyError when trying to find loop body steps.

### Pitfall 3: Stale Assets After Edit Step
**What goes wrong:** When a step is re-recorded via Edit, the old snippet PNG and embedding .npy remain on disk if the node_id changes.
**Why it happens:** Edit Step may generate a new node_id, leaving orphaned files.
**How to avoid:** Keep the same node_id when editing a step; just overwrite the snippet and embedding files. Only generate new node_ids for truly new steps.
**Warning signs:** Disk space grows; snippets/ directory contains unreferenced files.

### Pitfall 4: Overlay State Leak During Edit-Cancel
**What goes wrong:** User starts Edit Step (RECORDING state) then cancels. Overlay stays in RECORDING mode instead of returning to REPLAYING.
**Why it happens:** Missing state cleanup on cancel/dismiss path.
**How to avoid:** Every Edit Step entry must have a matching exit that restores REPLAYING state, regardless of outcome (confirm, dismiss, or abort).
**Warning signs:** Red shimmer persists when it should be purple; toolbar shows wrong buttons.

### Pitfall 5: Concurrent Modification During Fork
**What goes wrong:** Fork copies the directory while the runner is actively writing to it (e.g., a run is in progress), causing partial copies.
**Why it happens:** No locking mechanism on routine directories.
**How to avoid:** Fork copies from a loaded `Routine` object, not live filesystem reads. Load first, then copy. Accept that runs/ subdir may be partial (acceptable per CONTEXT.md: "No running check").
**Warning signs:** Forked routine has truncated JSON or missing snippets.

## Code Examples

### Update Flow Toolbar Mode
```python
# Source: Extending recorder/overlay/toolbar_panel.py pattern
# Add to ToolbarMode enum:
class ToolbarMode(Enum):
    RECORDING = auto()
    TAG_OPEN = auto()
    DRY_RUN = auto()
    VALIDATING = auto()
    BBOX_EDITING = auto()
    UPDATE = auto()        # [OK] [Edit Step] [Fork Here] [Delete Step]
    UPDATE_NOT_FOUND = auto()  # [Skip] [Edit] [Delete] [Abort]

# Add to _MODE_BUTTONS dict:
ToolbarMode.UPDATE: [
    ("OK", "ok"),
    ("Edit Step", "edit"),
    ("Fork Here", "fork_here"),
    ("Delete Step", "delete_step"),
],
ToolbarMode.UPDATE_NOT_FOUND: [
    ("Skip", "skip"),
    ("Edit", "edit"),
    ("Delete", "delete_step"),
    ("Abort Update", "abort"),
],
```

### Delete Routine
```python
# Source: Derived from routine/run_log.py shutil.rmtree pattern
import shutil
from rich.prompt import Confirm

def delete_routine(routine_dir: Path) -> bool:
    routine = Routine.load(routine_dir)
    if not Confirm.ask(
        f'Delete routine "{routine.name}" and all its files? '
        "This cannot be undone."
    ):
        return False
    shutil.rmtree(routine_dir)
    logger.info("Deleted routine '%s' at %s", routine.name, routine_dir)
    return True
```

### Inspect Routine
```python
# Source: Derived from recorder/tui.py Rich Table pattern
from rich.console import Console
from rich.table import Table

def inspect_routine(routine_dir: Path, as_json: bool = False) -> None:
    routine = Routine.load(routine_dir)
    if as_json:
        import json
        print(json.dumps(routine.to_dict(), indent=2))
        return

    console = Console()
    # Header
    console.print(f"[bold]{routine.name}[/bold] v{routine.version}")
    console.print(f"Steps: {len(routine.steps)} | Platform: {routine.platform}")
    console.print(f"Programs: {', '.join(routine.programs) or 'none'}")
    console.print(f"Created: {routine.created_at}\n")

    # Step table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", width=4)
    table.add_column("Action", width=14)
    table.add_column("Label", width=30)
    table.add_column("Bbox", width=20)
    table.add_column("Snippet", width=4, justify="center")

    for step in routine.steps:
        bbox = step.get("bbox", {})
        bbox_str = f"({bbox.get('x',0)},{bbox.get('y',0)}) {bbox.get('w',0)}x{bbox.get('h',0)}"
        snippet_exists = (routine_dir / step.get("snippet_path", "")).exists()
        table.add_row(
            str(step.get("step_index", "?")),
            step.get("action", "?"),
            step.get("label", "")[:30],
            bbox_str,
            "[green]X[/green]" if snippet_exists else "[red]-[/red]",
        )
    console.print(table)
```

### Version Bump on Save
```python
# Source: Derived from CONTEXT.md version bump decision
def save_updated_routine(
    routine: Routine,
    routine_dir: Path,
    structural_change: bool = True,
) -> None:
    if structural_change:
        routine.version = bump_minor(routine.version)
    else:
        routine.version = bump_patch(routine.version)
    routine.updated_at = datetime.now(timezone.utc).isoformat()
    routine.save(routine_dir)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No management ops | Phase 8 adds update/fork/delete/inspect | Phase 8 | Users can modify routines without re-recording from scratch |
| Routine version static | Auto-bumped on update save | Phase 8 | Enables routine versioning and audit trail |

**Deprecated/outdated:**
- None; this is net-new functionality building on established Phase 4-7 patterns.

## Open Questions

1. **Graph rebuild after step deletion**
   - What we know: OCSDGraph stores nodes with node_ids matching steps. Deleting steps requires removing nodes and edges.
   - What's unclear: Whether to rebuild the entire graph from scratch or surgically remove nodes. OCSDGraph has `remove_node()` but edge cleanup may leave dangling references.
   - Recommendation: Rebuild graph from modified step list after all edits. Simpler and guarantees consistency. The graph is small (typically <20 nodes).

2. **"Save As" dialog during update flow**
   - What we know: User needs to enter a new name at end of update walkthrough if choosing "Save as New".
   - What's unclear: Whether to use a Rich terminal prompt (blocking the overlay) or an overlay mini-dialog widget.
   - Recommendation: Use a mini-dialog overlay widget (similar to PromptDialog). The overlay is already visible; switching to terminal breaks flow. Pre-fill with original name.

3. **How Edit Step re-captures a single step**
   - What we know: Full RecordSession handles multi-step recording with complex state machine. Edit Step needs a subset: capture -> detect -> VLM -> tag -> dry-run.
   - What's unclear: Whether to extract a "single step capture" from RecordSession or reuse it wholesale.
   - Recommendation: Extract the capture pipeline into a reusable helper or run a "single-step RecordSession" that auto-completes after one step. The UpdateSession can set a flag to limit RecordSession to one capture cycle.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/test_routine_management.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MGMT-01 | Update session keeps/edits/deletes steps | unit | `python -m pytest tests/test_update_session.py -x` | -- Wave 0 |
| MGMT-01 | Update toolbar mode has correct buttons | unit | `python -m pytest tests/test_toolbar_panel.py::test_update_mode -x` | -- Wave 0 |
| MGMT-01 | Version auto-bumps on save | unit | `python -m pytest tests/test_routine_management.py::test_version_bump -x` | -- Wave 0 |
| MGMT-02 | Fork creates independent copy | unit | `python -m pytest tests/test_routine_management.py::test_fork_routine -x` | -- Wave 0 |
| MGMT-02 | Fork Here truncates at step N | unit | `python -m pytest tests/test_routine_management.py::test_fork_truncate -x` | -- Wave 0 |
| MGMT-03 | Delete removes directory | unit | `python -m pytest tests/test_routine_management.py::test_delete_routine -x` | -- Wave 0 |
| MGMT-03 | Delete requires confirmation | unit | `python -m pytest tests/test_routine_management.py::test_delete_confirmation -x` | -- Wave 0 |
| MGMT-04 | Inspect outputs Rich table | unit | `python -m pytest tests/test_routine_management.py::test_inspect_routine -x` | -- Wave 0 |
| MGMT-04 | Inspect --json outputs valid JSON | unit | `python -m pytest tests/test_routine_management.py::test_inspect_json -x` | -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_routine_management.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_routine_management.py` -- covers MGMT-02, MGMT-03, MGMT-04 (fork, delete, inspect)
- [ ] `tests/test_update_session.py` -- covers MGMT-01 (update flow state machine, step manipulation)
- [ ] Version bump helpers in `routine/version.py` need corresponding test functions

## Sources

### Primary (HIGH confidence)
- `routine/format.py` -- Routine dataclass, load/save, build_v1_step, checksum recalculation
- `routine/discovery.py` -- list_routines(), get_routine_dir() for path resolution
- `routine/runner.py` -- run_routine() step loop pattern, RunEvent enum, failure cascade
- `routine/replay_overlay.py` -- ReplayOverlayAdapter thread-safe signal pattern
- `routine/checksum.py` -- calculate_routine_checksum() for integrity after modifications
- `recorder/overlay/toolbar_panel.py` -- ToolbarMode enum, _MODE_BUTTONS dict, button creation pattern
- `recorder/overlay/state.py` -- OverlayState enum with REPLAYING state
- `recorder/overlay/controller.py` -- set_replay_mode(), replay API methods
- `recorder/record_session.py` -- RecordSession orchestrator pattern, PipelineBridge usage
- `recorder/record_flow.py` -- Entry point pattern (TUI prompt -> overlay -> session -> event loop)
- `recorder/overlay/mini_dialogs.py` -- WaitDialog/PromptDialog/LoopDialog for pre-filled editing

### Secondary (MEDIUM confidence)
- `mapper/graph.py` -- OCSDGraph CRUD methods for node/edge management during step deletion

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new deps
- Architecture: HIGH -- patterns directly derived from existing record_session.py and runner.py
- Pitfalls: HIGH -- identified from actual code structure (graph refs, loop body refs, asset lifecycle)

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable, internal codebase patterns)
