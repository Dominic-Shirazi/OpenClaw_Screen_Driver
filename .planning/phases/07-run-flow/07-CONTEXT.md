# Phase 7: Run Flow - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Execute a saved routine reliably with human-like motion, visible step progress, and graceful failure handling. Builds a new `routine/runner.py` that reads v1 `routine.json` steps sequentially, locates each target element via the 5-stage cascade, executes actions with human-like timing, validates results, and handles failures with a multi-stage recovery cascade. Completes the condition engine stubs (`element_appears`, `text_matches`) left by Phase 6. Does NOT build the TUI/CLI interfaces (Phase 9), routine management (Phase 8), or API endpoints (Phase 10).

</domain>

<decisions>
## Implementation Decisions

### Replay progress UX (RUN-09)
- **Shimmer color**: Purple = "automation active". New `REPLAYING` state added to OverlayState enum and ShimmerLayer. Distinct from green (ready) and red (recording)
- **Status badge**: Small frosted-glass pill at top-center of screen. Shows step label only: "Step 3/8: Click Submit". No locate method details in badge (those go to structured logging)
- **Target highlight**: Quick purple glow around located element bbox (~300ms) before mouse moves to it. Shows the user/debugger what the system found. Hides before screenshot
- **Camera flash**: 15px border "camera flash" after each screenshot, before returning to purple shimmer
- **Step result**: Badge text updates with OK/RETRY status. No border color flash on step completion
- **Completion**: Immediately clear all overlay UI on routine completion. No delay, no flash. Purple disappearing = replay done. Primary consumers are AI agents, not human spectators
- **Overlay optional**: `replay.show_overlay` config in ocsd.yaml (default `true`). API/CLI can run without overlay for headless/unattended runs. Not a per-run flag — a global config setting
- **Clean capture cycle**: All overlay elements (shimmer, badge, bbox highlight) hide before screenshots, same as recording. Reuse existing `hide_for_capture()`/`show_after_capture()` cycle

### Failure cascade (RUN-07, RUN-08)
- **Stage 1 — Retry at position**: Re-run full 5-stage cascade at recorded position, 2 attempts
- **Stage 2 — Region scan**: Compute the 25% screen region the element should be in (from recorded position + element size). Run full 5-stage cascade over that region
- **Stage 3 — Full-screen scan**: Single full-screen 5-stage cascade
- **Stage 4 — LiteLLM AI fallback**: Send full routine context (all steps, current position, screenshots) to LiteLLM with smart prompting: "Here's the routine, we're on step N, this is the screen — where should we click?" Basic implementation using existing LiteLLM infrastructure, not cloud-specific
- **Stage 5 — Abort with annotated log**: Generate detailed failure log with all cascade attempt results, confidence scores, and annotated screenshots (bboxes drawn around detected elements). Return simple summary to caller: "Failed at step N: Could not find [element]. VLM suggests [page description]"
- **Never blind-click**: If all stages fail, abort. Never click on uncertain targets

### Run logs
- **Location**: `~/.ocsd/routines/{name}/runs/{run_id}/` — each run gets a timestamped directory with log file + annotated screenshots
- **Self-cleaning**: Auto-prune on new run completion. Keep last 5 successful runs, last 10 failed runs. Older runs deleted automatically
- **Structured logging**: Python logging at INFO level for all step execution. "Step 3/8: Click 'Submit' at (450,320) via clip [0.92]"

### Run lifecycle (RUN-01)
- **Pre-flight validation**: Full validation before execution — check asset files exist (snippet PNGs, embedding .npy), verify routine checksum, test VLM connectivity
- **VLM unavailable**: Return status to caller (prompt caller). Let the agent/user decide whether to proceed without VLM or wait. Don't auto-proceed, don't auto-abort
- **Desktop setup**: For `start_from='desktop'`, auto-minimize all windows (Win+D). For `start_from='app'`, skip minimize. Build as a pluggable pre-run step that V2 GPS can override (e.g., "chrome is already open, skip to step 3")
- **Resolution mismatch**: Silent proceed. No check, no warning. Let the cascade and percentage-based coordinates handle it. If elements can't be found, the failure cascade kicks in naturally

### Runner architecture
- **New module**: `routine/runner.py` — iterates v1 steps sequentially. Reuses `core/locate.py` and `core/executor.py` but has its own orchestration. Existing `mapper/runner.py` stays for legacy graph-based replay and V2 GPS
- **Loop replay**: Inline re-execution. When the runner hits a loop step, it re-executes the body steps (referenced by `body_step_node_ids`) in order, checking the exit condition after each iteration via `core/conditions.py` ConditionChecker
- **Human-like execution**: All mouse moves via Bezier curves with overshoot-and-correct. Typing uses per-letter delays with typo simulation. Configurable `human_delay` multiplier (0=instant, 1.0=natural, 5.0=slow). Already implemented in `core/executor.py`

### Condition engine completion
- **element_appears**: Region-first search, then full screen. Adaptive polling — first 3 polls use stages 1-3 only (OmniParser, CLIP, OCR) for speed. After 3 polls, include VLM (stage 4) on subsequent polls. Position fallback (stage 5) never used for condition checking
- **text_matches**: Fuzzy OCR matching via `core/ocr.py find_text_on_screen()` with Levenshtein distance tolerance. Handles OCR imperfections (e.g., "Subm1t" matches "Submit"). Region-scoped using recorded position hint
- **Poll interval**: 2 seconds default, configurable in ocsd.yaml

### Claude's Discretion
- Exact Levenshtein distance threshold for fuzzy text matching
- How to annotate screenshots with bboxes in failure logs (cv2, PIL, etc.)
- LiteLLM smart prompt design for the AI fallback stage
- Status badge styling details (font, opacity, padding)
- Run directory naming convention (timestamp format, run_id generation)
- How to detect which 25% region an element belongs to from recorded position
- Pre-flight validation ordering and error aggregation
- How to structure the pluggable pre-run step for V2 GPS extensibility

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Run flow requirements
- `.planning/REQUIREMENTS.md` — RUN-01 through RUN-09 define all run flow requirements
- `.planning/ROADMAP.md` — Phase 7 success criteria (5 criteria that must be TRUE)

### Existing locate cascade
- `core/locate.py` — 5-stage cascade: OmniParser detect+match, CLIP embedding, OCR text match, VLM full scan, position fallback. This is the foundation for replay element location

### Existing executor (human-like execution)
- `core/executor.py` — `click()`, `right_click()`, `double_click()`, `drag()`, `type_text()`, `scroll()`, `press_enter()`, `hotkey()`, `prompt_user_blocking()`, `select_all_extract()`. Bezier curves, doughnut offset, typo simulation, thread-safe via `_lock`. Configurable `human_delay` multiplier

### Existing replay infrastructure (reference — being replaced)
- `mapper/runner.py` — `run_skill()`, `run_path()`, `execute_node()`, `RunnerEventType`. Graph-based replay engine. Phase 7 builds a new step-sequential runner, but this shows established patterns for event callbacks, retry loops, and validation

### Routine format
- `routine/format.py` — `Routine` dataclass with `load()`/`save()`, `build_v1_step()`, v1 schema. Step anchors (visual_match, ocr_text, position_pct, region_hint) define which locate strategies to try
- `routine/checksum.py` — SHA256 integrity verification

### Condition engine (stubs to complete)
- `core/conditions.py` — `ConditionChecker` with `poll_until()`. `_check_element_appears()` and `_check_text_matches()` are stubs returning False. Phase 7 must implement these using the locate cascade

### Post-action validation
- `mapper/validator.py` — `validate_action()` (pixel diff + VLM), `quick_check()`, `validate_element_located()`, `validate_destination()`. Used for post-action verification (RUN-06)

### Overlay infrastructure (for replay UX)
- `recorder/overlay/shimmer_layer.py` — ShimmerLayer with `set_state()`. Needs new purple REPLAYING color
- `recorder/overlay/state.py` — OverlayState enum. Needs REPLAYING state
- `recorder/overlay/view.py` — `hide_for_capture()`/`show_after_capture()` for clean capture cycle
- `recorder/overlay/controller.py` — OverlayController API for showing/hiding overlay elements

### Core pipeline
- `core/capture.py` — `screenshot_full()`, `screenshot_region()`, `load_snippet()`
- `core/detection.py` — `get_detector()` for OmniParser
- `core/vision.py` — `analyze_crop_array()` for VLM analysis
- `core/embeddings.py` — `generate_embedding()`, `get_embedding_by_id()` for CLIP/FAISS
- `core/ocr.py` — `find_text_on_screen()` for OCR text search
- `core/config.py` — YAML config with `get_config()`

### Prior phase context
- `.planning/phases/04-record-flow/04-CONTEXT.md` — start_from field, dry-run flow, save format
- `.planning/phases/05-routine-file-format/05-CONTEXT.md` — v1 schema, anchor strategies, graceful degradation
- `.planning/phases/06-action-types/06-CONTEXT.md` — All 12 action types, condition engine, loop body format, prompt_user mechanism

### Project constraints
- `.planning/PROJECT.md` — Human-like execution non-negotiable, never blind-click when uncertain, overlay screenshot cleanliness
- `CLAUDE.md` — Type hints, logging, thread safety, venv isolation, cross-platform awareness

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/locate.py:locate_element()`: Full 5-stage cascade already implemented. Called with graph + node_id. Phase 7 runner needs to adapt this to work with v1 step dicts (anchors, snippet paths) instead of graph nodes
- `core/executor.py`: All human-like execution functions ready. Click variants, type, scroll, drag, prompt_user_blocking, select_all_extract. Thread-safe, configurable human_delay
- `core/conditions.py:ConditionChecker`: Poll loop infrastructure built. Just needs element_appears and text_matches implementations
- `mapper/validator.py:validate_action()`: Pixel diff + VLM post-action validation. Reuse for RUN-06
- `mapper/runner.py:execute_node()`: Pattern for locate → screenshot → execute → validate → record stats. Adapt for step-sequential runner
- `routine/format.py:Routine.load()`: Load routine from disk with checksum validation. Pre-flight uses this
- `recorder/overlay/view.py:hide_for_capture()/show_after_capture()`: Clean capture cycle ready for replay
- `recorder/overlay/shimmer_layer.py`: Shimmer infrastructure ready — just needs purple color constant

### Established Patterns
- PipelineBridge signals for thread-safe background→main delivery
- AnimationClock.register() for timed animations
- ConditionChecker.poll_until() blocks on calling thread (background thread expected)
- OCSDGraph node data includes relative_position, label, element_type, ocr_text
- Step anchors define which cascade stages to try (visual_match, ocr_text, position_pct, region_hint)

### Integration Points
- `core/conditions.py:_check_element_appears()` — Replace stub with locate cascade call
- `core/conditions.py:_check_text_matches()` — Replace stub with fuzzy OCR search
- `recorder/overlay/state.py` — Add REPLAYING state with purple color
- `recorder/overlay/shimmer_layer.py` — Support purple shimmer for REPLAYING state
- `core/config.py` — Add `replay` config section (show_overlay, auto_minimize, etc.)

</code_context>

<specifics>
## Specific Ideas

- Purple = "automation active" — distinct from green (ready) and red (recording). This color language should eventually propagate back to recording dry-run execution (shimmer turns purple during countdown execution, returns to red for validation prompt)
- Overlay is primarily a "hands off my keyboard" indicator for when an AI agent triggers a routine. No human is watching. So: minimal UX, instant cleanup, no delays
- The 25% overlapping region scan for failure recovery ensures elements near edges aren't missed. The system already knows roughly where the element should be from recorded position — compute one region, not nine
- Run logs with annotated screenshots enable post-mortem debugging. Bboxes drawn on screenshots show what the system detected vs what it was looking for
- LiteLLM AI fallback uses existing infrastructure — not a new cloud integration, just a smart prompt sent to whatever VLM proxy is configured. The prompt includes full routine context so the model can reason about the workflow state
- Self-cleaning run directory prevents disk bloat from repeated automated runs. 5 success + 10 failure is enough history for debugging without unbounded growth
- Fuzzy OCR matching handles real-world OCR imperfections. A Levenshtein threshold lets "Subm1t" match "Submit" without false positives
- Adaptive condition polling starts cheap (stages 1-3) and escalates to VLM after 3 polls. Keeps poll cycles under 2s for the first 6 seconds, then accepts slower polls for reliability

</specifics>

<deferred>
## Deferred Ideas

- Purple shimmer during recording dry-run execution (Phase 4 retroactive) — shimmer turns purple during countdown execution, returns to red for "did that work?" validation prompt. Consistent with "purple = automation active" color language
- Cloud AI computer-use model multi-provider fallback chain (Claude, Gemini, Molmo) — V2, extensibility hooks built in V1 via LiteLLM basic implementation
- V2 GPS skip-ahead — runner can start from any step when GPS knows app state (e.g., chrome already open, skip desktop setup). Pluggable pre-run step supports this
- Rich TUI progress display during replay — Phase 9
- API run status with per-step output — Phase 10

</deferred>

---

*Phase: 07-run-flow*
*Context gathered: 2026-03-19*
