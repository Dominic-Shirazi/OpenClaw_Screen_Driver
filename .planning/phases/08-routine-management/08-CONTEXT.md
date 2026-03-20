# Phase 8: Routine Management - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Update, fork, delete, and inspect saved routines without re-recording from scratch. The update flow is a guided replay where the user walks through each step with editing controls. Fork creates an independent copy. Delete removes the routine directory. Inspect prints a human-readable step summary to the terminal. Does NOT add new action types (Phase 6), replay engine features (Phase 7), TUI/CLI entry points (Phase 9), or API endpoints (Phase 10).

</domain>

<decisions>
## Implementation Decisions

### Update flow (MGMT-01)
- **Guided replay model**: Update flow launches the overlay and walks through each step like a replay-with-editing. Purple shimmer while AI locates the element, highlight when found, shimmer turns red (user in control)
- **Per-step controls via toolbar**: OK/Continue (executes the action, moves to next), Edit Step (full re-record: click new element, AI bbox, VLM, tag dialog), Fork Here (save-as-new from this point), Delete Step (remove this step)
- **No inserts in V1**: Users cannot insert new steps between existing ones. If they need to add steps, they fork from the desired point and record new steps. Insert deferred to future version
- **Non-element steps (wait, loop, prompt_user)**: Show the mini-dialog pre-filled with current values. User can modify parameters and confirm, or keep as-is
- **Element not found**: Display "Could not find [element label]" with options: Skip (keep step as-is), Edit (re-capture via full pipeline), Delete (remove step), Abort (cancel update)
- **Save behavior**: At end of walkthrough, user chooses Save & Replace (overwrites original) or Save as New (creates new routine with new name)
- **Version bump**: Auto-bump minor version on save (1.0.0 → 1.1.0 for step changes, patch for metadata-only edits). Automatic, user never thinks about it
- **Asset alignment**: Steps that are kept unchanged retain their original snippets/embeddings. Steps that are re-captured via Edit get new snippets/embeddings. Deleted steps' assets are removed. Checksum recalculated on save

### Fork behavior (MGMT-02)
- **Full copy**: Copy all snippet PNGs and embedding .npy files to the new routine directory. Independent routines, no shared state
- **Fork from update**: "Fork Here" at step N creates a new routine with steps 1..N from the original. Steps N+1 onward are dropped. User continues recording NEW steps from that point
- **Fork naming**: "Save As" dialog pre-filled with original routine name. User edits to new name. If name still exists after editing, show error and let them re-enter
- **Standalone fork**: Fork from CLI/API (not during update) copies the entire routine under a new name, then opens it in the update flow for modification

### Delete behavior (MGMT-03)
- **Hard delete with confirmation**: Terminal confirmation prompt: "Delete routine [name] and all its files? This cannot be undone." Then `shutil.rmtree` the routine directory
- **No running check**: Delete is allowed even if the routine is currently being executed. The runner will fail gracefully when it can't find its files
- **Deletes everything**: routine.json, snippets/, embeddings/, runs/ — entire routine directory removed

### Inspect output (MGMT-04)
- **Rich table**: Rich-formatted terminal table with columns: step#, action type, element label, bbox, snippet exists (checkmark). Colorized
- **Header + steps**: Brief header section (name, version, step count, programs, platform, created date) above the step table
- **--json flag**: With `--json`, output the full routine.json content for machine consumption. Consistent with CLI-09 requirement (all commands support --json)

### Claude's Discretion
- How to structure the UpdateSession class (reuse RecordSession patterns or separate)
- How to coordinate overlay state transitions between replay-mode and recording-mode during Edit Step
- Terminal confirmation prompt implementation (Rich.prompt or input())
- Exact Rich table column widths and formatting
- How to handle graph consistency when steps are deleted or re-ordered
- How to implement "Save As" name dialog (Rich prompt or custom widget)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Management requirements
- `.planning/REQUIREMENTS.md` — MGMT-01 through MGMT-04 define all routine management requirements
- `.planning/ROADMAP.md` — Phase 8 success criteria (4 criteria that must be TRUE)

### Routine format and discovery
- `routine/format.py` — `Routine` dataclass with `load()`/`save()`, `build_v1_step()`, v1 schema with steps, graph, checksum. The update flow modifies and re-saves routines using this
- `routine/discovery.py` — `list_routines()`, `get_routine_dir()` for finding routines on disk
- `routine/checksum.py` — `calculate_routine_checksum()` for integrity verification after modifications

### Replay infrastructure (update flow reuses this)
- `routine/runner.py` — `run_routine()`, `RunEvent`, `preflight_check()`. The update walkthrough is a modified replay that pauses per step for user input
- `routine/replay_overlay.py` — `ReplayOverlayAdapter` for thread-safe overlay updates. Update flow needs similar adapter
- `core/locate.py` — `locate_element_from_step()` for finding elements during update walkthrough

### Recording infrastructure (edit step reuses this)
- `recorder/record_session.py` — `RecordSession` orchestrator. Edit Step should reuse the capture-detect-VLM-tag pipeline
- `recorder/overlay/toolbar_panel.py` — `ToolbarMode` enum, toolbar button management. Update flow needs new toolbar mode
- `recorder/overlay/tag_dialog_panel.py` — Tag dialog for editing step fields
- `recorder/overlay/mini_dialogs.py` — Wait/loop/prompt mini-dialogs for non-element step editing

### Overlay state machine
- `recorder/overlay/state.py` — `OverlayState` enum (READY, RECORDING, PAUSED, REPLAYING). Update flow transitions between REPLAYING (AI locating) and RECORDING (user editing)
- `recorder/overlay/controller.py` — `OverlayController` API including replay mode methods from Phase 7

### Prior phase context
- `.planning/phases/04-record-flow/04-CONTEXT.md` — Recording pipeline, dry-run flow, save format
- `.planning/phases/05-routine-file-format/05-CONTEXT.md` — v1 schema, versioning, asset storage
- `.planning/phases/06-action-types/06-CONTEXT.md` — All 12 action types, mini-dialogs
- `.planning/phases/07-run-flow/07-CONTEXT.md` — Replay overlay UX, purple shimmer, failure cascade

### Project constraints
- `.planning/PROJECT.md` — Human-like execution non-negotiable, routine auditability
- `CLAUDE.md` — Type hints, logging, thread safety, venv isolation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `routine/runner.py:run_routine()`: Step-sequential replay engine. Update walkthrough is essentially run_routine() that pauses per step for user input instead of auto-continuing
- `routine/replay_overlay.py:ReplayOverlayAdapter`: Thread-safe adapter from runner events to overlay. Update flow needs a similar adapter (or extends this one) with additional events for user controls
- `core/locate.py:locate_element_from_step()`: Locates elements from v1 step dicts. Used during update walkthrough to find each step's target
- `recorder/record_session.py:RecordSession`: Full capture-detect-VLM-tag pipeline. "Edit Step" action should reuse this for re-recording a step
- `recorder/overlay/toolbar_panel.py`: Toolbar with configurable modes (RECORDING, TAG_OPEN, DRY_RUN). Needs new UPDATE mode with OK/Edit/Fork/Delete buttons
- `recorder/overlay/mini_dialogs.py`: Wait/loop/prompt mini-dialogs. Reused for non-element step editing during update
- `routine/discovery.py:list_routines()`: Enumerate routines for selection
- `routine/format.py:Routine.save()`: Save routine with checksum recalculation

### Established Patterns
- PipelineBridge signals for thread-safe background→main delivery
- OverlayState enum for shimmer color transitions (REPLAYING=purple, RECORDING=red)
- ToolbarMode enum for context-sensitive toolbar buttons
- AnimationClock.register() for timed animations
- Routine.load() / Routine.save() for JSON I/O with checksum

### Integration Points
- `recorder/overlay/toolbar_panel.py` — Add UPDATE toolbar mode with OK/Edit/Fork/Delete buttons
- `recorder/overlay/state.py` — Transitions between REPLAYING and RECORDING during update
- `routine/format.py:Routine` — Methods for step modification (delete, replace, reorder)
- `routine/discovery.py` — Used by delete and fork to find/verify routines

</code_context>

<specifics>
## Specific Ideas

- Update flow is a "guided replay with editing controls" — reuses Phase 7 replay infrastructure with per-step pause points
- Purple shimmer = AI locating, red shimmer = user in control. Same color language from Phase 7 but with different toolbar buttons
- "Fork Here" creates a clean break: steps before the fork point are copied, everything after is fresh recording. User gets a "Save As" dialog pre-filled with original name
- "Save As" dialog shows original name for reference — user edits it to something similar (e.g., "Chrome_Google_IFL" → "Chrome_Google_Search")
- Non-element steps (wait, loop, prompt_user) show their mini-dialogs pre-filled so users can tweak parameters without re-creating from scratch
- Version auto-bumps so users never think about it — minor for structural changes, patch for metadata edits

</specifics>

<deferred>
## Deferred Ideas

- Insert step during update — V2, fork-and-re-record is the V1 workaround
- Program/extension dependency model — V2, routines as "extensions" of program routines (e.g., Chrome → site-specific routines)
- Soft delete with trash/recovery — V2, hard delete is sufficient for V1
- Fork auto-skip-to-divergence — V2, requires composition intelligence to detect shared prefix
- Routine diff/comparison view — V2, compare two routine versions side-by-side

</deferred>

---

*Phase: 08-routine-management*
*Context gathered: 2026-03-20*
