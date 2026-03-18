# Phase 4: Record Flow - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

End-to-end recording session wiring: F2→click→hide→screenshot→AI bbox→scan animation→VLM→tag dialog→dry-run→validate→loop→save. Connects all Phase 1-3 overlay infrastructure (shimmer, scan animation, tag dialog, toolbar) into a working recording flow. Replaces the legacy `record_controller.py` flow with the new overlay-integrated pipeline. Does not define the final routine file format (Phase 5) or implement all action types (Phase 6), but saves enough data to round-trip through Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Pre-recording setup (REC-01, REC-02)
- **Naming**: Rich TUI prompt in the terminal before overlay launches. User types routine name, then overlay starts. No Qt UI needed for naming
- **Window minimize**: Auto-minimize all windows (Win+D equivalent) to establish a clean desktop baseline before recording begins
- **Minimize failure**: If some windows can't be minimized, warn the user and offer "Continue" or "Cancel" options — don't silently proceed or block forever
- **Start state**: After naming, ask user: "Does this routine start from the desktop or from an already-open app?" — user chooses per routine. This determines whether the first recorded step is opening an app or an in-app action

### Click-to-tag pipeline (REC-04, REC-05, REC-06, REC-07)
- **Bbox detection strategy**: Cascade — click-local crop first (~240px centered on click), run OmniParser on that region. If no element found, fall back to full-screen detection. Click location is sent to OmniParser as context. Crop coordinates are adjusted so detection results map back to screen space correctly
- **Drag-highlight capture**: AI proposes a tighter bbox from the user's drag region. User can reject and keep their original drag rect. Shown as morph animation that the user can undo
- **Bbox editing**: Both click AND drag captures allow the user to edit/adjust the bbox before accepting the AI's proposal. The bbox overlay is editable (drag corners) in both cases
- **VLM timing**: Sequential — after bbox is accepted, scan animation plays, VLM runs, tag dialog opens with results. One element at a time. Card glow pulses as loading indicator (per Phase 3)
- **VLM failure handling**: Retry VLM once with shorter timeout. If still fails, open tag dialog with partial data from detection pipeline (type_guess, florence_caption). User completes the rest manually

### Dry run behavior (REC-08, REC-09, REC-10)
- **Countdown UX**: Cursor-following countdown spinner — mouse changes to a spinning "thinking" logo with a countdown that follows the cursor. Mouse stays free to move during countdown
- **Dry-run is mandatory**: Every step must be dry-run tested before it's saved. No skip option
- **Overlay during execution**: Overlay switches to click-through mode during dry-run execution (stays visible, shimmer still running, but fully transparent to input). Does NOT fully hide
- **Validation**: User confirms result — toolbar shows Yes / No (edit step) / Retry after action executes. User is always the validator during recording
- **Edit on failure**: "No (edit step)" gives full redo — user can either edit the tag dialog fields OR go back and re-click/re-drag to capture a different element entirely
- **Retry behavior**: Full 3-2-1 countdown plays again before retry. Consistent experience
- **Loop-back**: After user confirms "Yes", brief green success flash (~500ms — donut cloud or bbox flashes green), then straight back to recording mode (red shimmer, crosshair) for next element
- **No mid-recording undo**: Once a step is confirmed via dry-run, it's locked. Edit/delete of past steps happens in the Update flow (Phase 8)

### Save and abort flow (REC-11)
- **Save format**: Full routine.json + snippet PNGs + CLIP embeddings. Phase 5 can refine the JSON schema but all data is captured now
- **Save location**: `~/.ocsd/routines/{name}/` — routine.json + snippets/ + embeddings/ inside. Exactly as spec'd in requirements
- **Save error handling**: Show error on overlay, let user fix the issue, retry. Don't lose session data in memory
- **ESC abort**: Confirmation prompt — "Are you sure? All recorded steps from this session will be lost." (industry-standard discard confirmation). If confirmed, discard everything — no partial routines on disk. If no steps captured yet, close silently

### Claude's Discretion
- Exact cursor-following countdown spinner implementation (Qt custom cursor, small overlay widget, etc.)
- How to structure the interim routine.json before Phase 5 finalizes the format
- Detection pipeline threading and async orchestration
- How to implement the "start from desktop vs already-open app" choice in the routine metadata
- Green success flash implementation (which overlay element flashes)
- Error retry UX details (overlay panel vs toast vs toolbar message)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Record flow requirements
- `.planning/REQUIREMENTS.md` — REC-01 through REC-11 define all record flow requirements
- `.planning/ROADMAP.md` — Phase 4 success criteria (6 criteria that must be TRUE)

### Phase 1-3 foundation (build on top of this)
- `recorder/overlay/controller.py` — OverlayController public API: show/close, state transitions, HUD panel API, scan/donut API
- `recorder/overlay/view.py` — OverlayView: scene management, hide_for_capture/show_after_capture, drag-to-draw, HUD panels
- `recorder/overlay/state.py` — OverlayState enum (READY/RECORDING/PAUSED), transition table, STATE_COLORS
- `recorder/overlay/scan_layer.py` — Scan animation: start_scan(), receive_fitted_bbox()
- `recorder/overlay/tag_dialog_panel.py` — Tag dialog: show_dialog(), dismiss(), get_form_data(), typewriter fill
- `recorder/overlay/toolbar_panel.py` — Toolbar: ToolbarMode (RECORDING/TAG_OPEN/DRY_RUN), set_mode()
- `recorder/overlay/donut_cloud_layer.py` — Donut cloud: accept() for green transition
- `recorder/overlay/shimmer_layer.py` — Shimmer: set_state(), set_avoidance_rects()

### Legacy record controller (reference — being replaced)
- `recorder/record_controller.py` — Legacy flow with smart_detect, _try_refine_bbox, _auto_snip, _save_recording, _save_snippets_and_embeddings. Contains reusable logic for bbox refinement, VLM labeling, snippet saving, and graph export

### Core pipeline modules
- `core/capture.py` — screenshot_full(), screenshot_region(), save_snippet()
- `core/detection.py` — get_detector() for OmniParser element detection
- `core/vision.py` — analyze_crop_array() for VLM element analysis
- `core/embeddings.py` — generate_embedding(), save_to_index() for CLIP/FAISS
- `core/executor.py` — Human-like mouse/keyboard execution for dry-run
- `mapper/graph.py` — OCSDGraph for building step sequences
- `mapper/export.py` — export_skill(), save_skill_to_file() for JSON output

### Prior phase context
- `.planning/phases/01-overlay-foundation/01-CONTEXT.md` — Layer-based composition, DPI correctness, clean capture cycle
- `.planning/phases/02-overlay-animations/02-CONTEXT.md` — Animation patterns, scan sequence, color language (red=processing, green=ready)
- `.planning/phases/03-overlay-hud-panels/03-CONTEXT.md` — Tag dialog appearance, typewriter fill, toolbar modes, VLM glow-as-loading

### Project constraints
- `.planning/PROJECT.md` — Overlay screenshot cleanliness non-negotiable, VLM with manual fallback, human-like execution
- `CLAUDE.md` — Code quality rules, type hints, logging, thread safety, venv isolation

### Codebase patterns
- `.planning/codebase/CONVENTIONS.md` — Naming, imports, error handling, logging patterns
- `.planning/codebase/ARCHITECTURE.md` — Layer architecture, event-driven patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `record_controller._try_refine_bbox()`: IoU-based bbox refinement against OmniParser detections — can be adapted for the cascade detection approach
- `record_controller._auto_snip()`: Click-local detection with radius crop — close to the new click-local strategy, needs coord adjustment
- `record_controller._save_snippets_and_embeddings()`: 30% padded crop + CLIP embedding pipeline — reuse directly
- `record_controller._save_recording()`: NetworkX graph building + export — adapt for new routine.json format
- `OverlayController` API: Already has show_tag_dialog(), start_scan(), finish_scan(), show_donut_cloud(), set_toolbar_mode() — the pipeline wires these together
- `OverlayView.hide_for_capture()` / `show_after_capture()`: Clean capture cycle with clock stop/start
- `ToolbarMode` enum: RECORDING, TAG_OPEN, DRY_RUN already defined in Phase 3

### Established Patterns
- Each layer = independent QGraphicsObject subclass (Phase 1/2/3 pattern)
- AnimationClock.register(callback) for all timed animations
- Controller lazy-imports View to avoid circular deps
- _SignalBridge QObject for thread-safe background→main thread communication
- State-driven visual updates via apply_state() hook

### Integration Points
- `OverlayController._handle_toggle()` — F2 state transition, where recording starts
- `OverlayController._handle_close()` — Ctrl+Q/ESC handler, where save/abort fires
- `OverlayView.mouseReleaseEvent()` — Drag-to-draw completion, fires on_selection callback
- `recorder/hotkeys.py` — Global hotkey listener (F2, Ctrl+Q)
- `recorder/element_types.py` — ElementType enum for tag dialog dropdowns

</code_context>

<specifics>
## Specific Ideas

- Cursor-following countdown spinner for dry-run — the countdown rides along with the mouse, not stuck in one place. Should feel like the system is "thinking" before acting
- User is ALWAYS the validator during recording dry-run — no automated pass/fail. Yes / No (edit step) / Retry
- Brief green flash after confirmed step gives satisfying visual closure before next element
- Click location sent to OmniParser as context — the detector "knows where we're looking"
- Overlay stays in click-through mode during dry-run execution (shimmer still visible, not fully hidden)
- "Start from desktop vs already-open app" per-routine choice affects how the routine is later replayed (Phase 7)
- Confirmation prompt on ESC abort uses industry-standard discard language

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-record-flow*
*Context gathered: 2026-03-18*
