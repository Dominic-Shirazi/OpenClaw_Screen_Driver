# Phase 6: Action Types - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement all 12 action types so they are capturable during recording and executable during replay. Connects the tag dialog's existing conditional fields (Phase 3), the recording pipeline (Phase 4), the routine format (Phase 5), and the existing executor functions (`core/executor.py`) into a complete action type system. Adds toolbar quick-add buttons for non-element actions (wait, loop, prompt_user, look-here). Builds a shared condition-checking engine used by both wait and loop actions.

Does NOT build the full replay engine (Phase 7), routine management (Phase 8), or AI-driven decision branching (V2).

</domain>

<decisions>
## Implementation Decisions

### Capture trigger model
- **Click-based actions** (click, double_click, right_click, click_drag, type, scroll): User clicks/drags the element normally, then changes the action type in the tag dialog dropdown. The click captures the target; the dropdown says what to DO with it
- **Non-element actions** (wait, loop, prompt_user, look-here): Dedicated toolbar quick-add buttons. Four buttons added to the toolbar in RECORDING mode:
  - **Look Here** — user drags a region after pressing. Always a region drag (never click). Captures area for read/snip_and_search. VLM analyzes the region
  - **Add Wait** — opens mini-dialog to set wait condition and timeout
  - **Add Loop** — opens loop definition dialog (step range + condition)
  - **Add Prompt** — opens mini-dialog to set question text for prompt_user step

### Click action variants (ACT-01, ACT-02, ACT-03)
- click, double_click, right_click are distinct action types in the tag dialog dropdown
- All three use donut distribution targeting during replay
- Right-click and double-click are captured via the tag dialog dropdown change (user clicks element normally, then changes action to right_click or double_click)

### Click-drag (ACT-04)
- User clicks source element, tag dialog shows click_drag action type, conditional "drag target hint" field appears
- Second element (drag target) captured by clicking it — two bboxes stored per step
- During replay: click source, drag to target with Bezier curve

### Type action (ACT-05)
- User clicks the text input element, changes action to "type" in tag dialog
- Text to type is entered in the conditional "text_to_type" field
- Optional "press Enter" checkbox for form submission
- During replay: human-like typing with per-letter delays, variance, and typo simulation (already in core/executor.py)

### Data-returning actions (ACT-06, ACT-07, ACT-08)
- **Look Here button** triggers these — user always drags a region
- Tag dialog action dropdown distinguishes read vs snip_and_search vs select_all_extract
- **During recording dry-run**: No extra UI for viewing extracted data. The existing dry-run flow (countdown → execute → Yes/No/Retry) is sufficient. Read/snip actions are observation steps — the screen doesn't change, which is self-evident confirmation
- **During replay**: Step output available in API run status response (each step has an `output` field). Disk writes to temp folder only when debug/verbose flag is set. Default off for disk writes
- **select_all_extract (ACT-08)**: Ctrl+A → Ctrl+C → read clipboard. If clipboard empty/unchanged, fall back to VLM screenshot analysis. Clipboard-preferred, VLM-fallback

### Scroll (ACT-09)
- User clicks the scrollable area, changes action to "scroll" in tag dialog
- Conditional "direction_amount" field for direction (up/down/left/right) and amount (lines, pages, pixels)
- During replay: uses existing `core/executor.py:scroll()` function

### Wait action (ACT-10)
- **Timeout always required** — default 30s. Safety net, never wait forever
- **Four condition types**: fixed timer, element appears, screen change, custom VLM check
- **Shared condition-checking engine** with loop — same "poll and check" module. Wait is essentially a single-step loop with no body actions
- **Poll interval**: Configurable, default 2 seconds. User can adjust in ocsd.yaml config
- **Element appears**: Same capture flow as loop exit — click/describe the target element
- **Screen change**: Pixel diff threshold against screenshot taken at wait start
- **Custom VLM check**: Natural-language condition (e.g., "loading spinner is gone"). Slowest but most flexible

### Loop action (ACT-11)
- **Definition flow**: User presses "Add Loop" → mini-dialog shows numbered list of recorded steps → user selects start and end step range → picks exit condition
- **Exit condition capture for "element appears"**: System starts replaying the loop body steps. User presses Enter each iteration (manual advance). When the exit condition element appears on screen, user clicks/selects it. AI describes it, user refines the "why" (stop condition description)
- **Four exit conditions**: N iterations, element appears, text matches, prompt_user
  - Build extensible — N iterations is minimum viable, others can land incrementally
  - All conditions except N iterations have a **max iterations** safety limit (mandatory)
- **Prompt-user exit condition**: Can include screenshot (whole screen or "look" region) alongside the question text
- **Nesting allowed**: A loop body can contain other loop steps
- **No mid-session edits**: Loop body is fixed once defined. Editing is a Phase 8 concern
- **No loop-level dry-run**: Individual steps were already dry-run tested when recorded. Loop definition itself skips dry-run
- **Max iterations exceeded**: Prompt user with context — "Loop reached max iterations (N tries) looking for '{stop condition}'. Keep looking or abort?" with optional screenshot. Future extensibility: AI model fallback chain before prompting (V2)
- **Replay visibility**: Individual iterations shown as separate step progress (e.g., "Loop iteration 3/10: Step 5 (scroll down)")

### Prompt user (ACT-12)
- Pauses routine during replay, sends question text + optional screenshot via API `/respond` endpoint
- Routine does NOT resume until a response arrives
- No AI decision branching in V1 — prompt_user collects a text response, that's it. The routine continues with the same next step regardless of the answer

### Shared condition-checking engine
- Wait and loop use the **same underlying module** for condition polling
- Internally: screenshot → run OmniParser/CLIP/OCR/VLM depending on condition type → check match → sleep poll_interval → repeat until match or timeout/max-iterations
- Configurable poll interval (default 2s) from ocsd.yaml
- Reuses the existing 5-stage locate cascade where applicable (element appears condition)

### Claude's Discretion
- How to structure the shared condition-checking engine internally (class vs functions, async vs threaded)
- Mini-dialog UI implementation for wait/loop/prompt setup (reuse tag dialog patterns or separate widget)
- How to implement "Look Here" toolbar button state transition (button press → region-drag mode → tag dialog)
- Exact loop step-range selector UI (list with checkboxes, range slider, etc.)
- How to store two bboxes for click_drag steps in the routine format
- Adaptive polling optimizations within the 2s default interval
- How to present the loop "play iterations until you see it" UX for element-appears condition

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Action type requirements
- `.planning/REQUIREMENTS.md` — ACT-01 through ACT-12 define all action type requirements
- `.planning/ROADMAP.md` — Phase 6 success criteria (5 criteria that must be TRUE)

### Existing executor (already implements 6 action types)
- `core/executor.py` — `click()`, `right_click()`, `double_click()`, `drag()`, `type_text()`, `scroll()` with human-like timing, Bezier curves, typo simulation. Thread-safe via `_lock`

### Tag dialog conditional fields (already wired)
- `recorder/overlay/tag_dialog_panel.py` — `_CONDITIONAL_FIELDS` dict (line 127): type→text_to_type/press_enter, scroll→direction_amount, wait→condition_timeout, click_drag→drag_target_hint, read/snip_and_search→vlm_prompt, prompt_user→question_text

### Recording pipeline
- `recorder/record_session.py` — RecordSession orchestrator with phase state machine, capture-detect-VLM-tag pipeline
- `recorder/overlay/toolbar_panel.py` — ToolbarMode enum, toolbar button management. New quick-add buttons go here
- `recorder/overlay/record_phase.py` — RecordPhase enum with all sub-states

### Routine format
- `routine/format.py` — `build_v1_step()` for step construction, v1 schema
- `routine/checksum.py` — SHA256 integrity verification

### Core pipeline
- `core/capture.py` — `screenshot_full()`, `screenshot_region()`, `save_snippet()`
- `core/detection.py` — `get_detector()` for OmniParser element detection
- `core/vision.py` — `analyze_crop_array()` for VLM analysis
- `core/embeddings.py` — `generate_embedding()` for CLIP/FAISS

### Existing replay (reference for execution patterns)
- `mapper/runner.py` — `run_skill()`, `execute_node()` — existing replay engine using graph traversal. Phase 7 will build the new routine-based replay

### Prior phase context
- `.planning/phases/04-record-flow/04-CONTEXT.md` — Dry-run flow, save/abort, VLM failure handling
- `.planning/phases/05-routine-file-format/05-CONTEXT.md` — v1 schema, step format, anchor strategies

### Project constraints
- `.planning/PROJECT.md` — Human-like execution non-negotiable, VLM with manual fallback, overlay screenshot cleanliness
- `CLAUDE.md` — Type hints, logging, thread safety, venv isolation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/executor.py`: Already implements click, right_click, double_click, drag, type_text, scroll with human-like timing. These are the replay-side execution functions — Phase 6 wires them to the recording capture + format pipeline
- `tag_dialog_panel.py:_CONDITIONAL_FIELDS`: All conditional fields already built. No new tag dialog fields needed — just wiring the data flow from these fields into the step format
- `recorder/record_session.py`: RecordSession orchestrator already handles the click→detect→VLM→tag→dry-run pipeline. Non-click actions extend this with new entry points (toolbar buttons)
- `recorder/overlay/toolbar_panel.py`: ToolbarMode and button management. New quick-add buttons are additive

### Established Patterns
- Each action type's recording is a state machine transition in RecordSession (RecordPhase enum)
- PipelineBridge signals for thread-safe background→main delivery (detection, VLM, execution all happen on background threads)
- Tag dialog get_form_data() returns dict with action_type and conditional fields — downstream code reads this dict
- AnimationClock.register() for any timed UI (countdown, polling indicator)

### Integration Points
- `recorder/overlay/toolbar_panel.py` — Add 4 new quick-add buttons (Look Here, Add Wait, Add Loop, Add Prompt)
- `recorder/record_session.py` — New handlers for non-click action initiation (toolbar button callbacks)
- `routine/format.py:build_v1_step()` — Extend step format for action-type-specific fields (loop range, wait condition, drag target, etc.)
- `core/executor.py` — May need new functions for select_all_extract (Ctrl+A + Ctrl+C + clipboard read)

</code_context>

<specifics>
## Specific Ideas

- Wait and loop share the same condition-checking engine — "wait is a bodyless loop." Build one module, two entry points
- Loop "element appears" capture: replay the loop body with manual Enter advance until the element shows up, then click it. AI describes, user refines the "why." Very interactive, very intuitive
- Max iterations is always required as a safety net — never loop or wait forever. Even element-appears and text-match conditions have a max
- When loop hits max iterations: prompt user with full context about what it was looking for. Future: AI model fallback chain (cloud computer-use models, Molmo) before prompting — build the extensibility hooks now
- "Look Here" is always a region drag, never a point click. Consistent gesture for observation steps
- select_all_extract: clipboard-first (Ctrl+A → Ctrl+C → read clipboard), VLM-fallback if clipboard empty
- Data-returning step outputs: available in API, written to temp folder only with debug flag. Default off for disk
- Individual loop iterations visible as separate step progress in API/TUI — debuggable

</specifics>

<deferred>
## Deferred Ideas

- AI-driven decision branching at prompt_user steps — V2, requires context-aware action selection beyond simple text responses
- Cloud AI computer-use model fallback for loop/wait condition checking — V2, extensibility hooks built in V1
- Local model offload (Molmo) for condition checking — V2, same extensibility path
- Loop body editing during recording session — Phase 8 Update flow
- "Take next action" based on routine context — V2, needs full app mapping / GPS

</deferred>

---

*Phase: 06-action-types*
*Context gathered: 2026-03-19*
