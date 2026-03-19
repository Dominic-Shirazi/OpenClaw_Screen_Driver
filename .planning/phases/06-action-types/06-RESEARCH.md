# Phase 6: Action Types - Research

**Researched:** 2026-03-19
**Domain:** Recording pipeline action type capture + routine format extension + condition engine
**Confidence:** HIGH

## Summary

Phase 6 wires all 12 action types into the recording pipeline so they are capturable during recording and executable during dry-run. The existing codebase already has strong foundations: `core/executor.py` implements click, right_click, double_click, drag, type_text, scroll, and hotkey with human-like timing. The tag dialog's `_CONDITIONAL_FIELDS` dict already maps action types to their UI fields. The `RecordSession` state machine handles the click-through-tag-dry-run pipeline. The v1 routine format (`routine/format.py`) has `build_v1_step()` for step construction.

The primary new work is: (1) extending `build_v1_step()` to include action-type-specific fields in the step dict, (2) adding 4 toolbar quick-add buttons for non-element actions, (3) building a shared condition-checking engine for wait and loop, (4) implementing the "Look Here" region-drag flow for read/snip_and_search/select_all_extract, (5) adding a `select_all_extract` executor function (Ctrl+A, Ctrl+C, clipboard read with VLM fallback), and (6) wiring the loop definition dialog with step-range selection and exit condition capture.

**Primary recommendation:** Build in layers -- first extend the step format and simple action type wiring (click variants, type, scroll), then the toolbar buttons and "Look Here" flow, then the condition engine (shared by wait and loop), and finally the loop definition UX which is the most complex piece.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Click-based actions (click, double_click, right_click, click_drag, type, scroll): User clicks/drags the element normally, then changes the action type in the tag dialog dropdown
- Non-element actions (wait, loop, prompt_user, look-here): Dedicated toolbar quick-add buttons
- "Look Here" is always a region drag, never a point click
- Wait timeout always required (default 30s)
- Wait and loop use the same shared condition-checking engine
- Loop definition: user selects start/end step range from numbered list, picks exit condition
- Loop exit condition "element appears" capture: system replays body, user presses Enter each iteration, clicks exit element when it appears
- Four loop exit conditions: N iterations, element appears, text matches, prompt_user (N iterations is minimum viable)
- All conditions except N iterations have mandatory max iterations safety limit
- Max iterations exceeded: prompt user with context
- Nesting allowed (loop body can contain other loops)
- No mid-session loop edits (Phase 8)
- No loop-level dry-run (individual steps already tested)
- prompt_user pauses routine, sends question + optional screenshot via API /respond, waits for response
- No AI decision branching in V1
- Data-returning step outputs: available in API, temp folder only with debug flag
- select_all_extract: clipboard-preferred (Ctrl+A + Ctrl+C), VLM-fallback
- click_drag: two bboxes per step (source + drag target)

### Claude's Discretion
- How to structure the shared condition-checking engine internally (class vs functions, async vs threaded)
- Mini-dialog UI implementation for wait/loop/prompt setup (reuse tag dialog patterns or separate widget)
- How to implement "Look Here" toolbar button state transition (button press -> region-drag mode -> tag dialog)
- Exact loop step-range selector UI (list with checkboxes, range slider, etc.)
- How to store two bboxes for click_drag steps in the routine format
- Adaptive polling optimizations within the 2s default interval
- How to present the loop "play iterations until you see it" UX for element-appears condition

### Deferred Ideas (OUT OF SCOPE)
- AI-driven decision branching at prompt_user steps (V2)
- Cloud AI computer-use model fallback for loop/wait condition checking (V2)
- Local model offload (Molmo) for condition checking (V2)
- Loop body editing during recording session (Phase 8)
- "Take next action" based on routine context (V2)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ACT-01 | `click` -- left click with donut distribution targeting | Already implemented in `core/executor.py:click()`. Phase 6 wires tag dialog action type into step format |
| ACT-02 | `double_click` -- double left click | Already implemented in `core/executor.py:double_click()`. Wire dropdown selection to step format |
| ACT-03 | `right_click` -- right click (context menu) | Already implemented in `core/executor.py:right_click()`. Wire dropdown selection to step format |
| ACT-04 | `click_drag` -- click source, drag to target | `core/executor.py:drag()` exists. New: two-bbox capture flow, `drag_target` field in step format |
| ACT-05 | `type` -- keyboard input with human-like timing | `core/executor.py:type_text()` exists. Wire `text_to_type` and `press_enter` from tag dialog conditional fields |
| ACT-06 | `read` -- OCR/VLM extract text from region | "Look Here" toolbar button + region drag + VLM analysis. New step format field: `vlm_prompt` |
| ACT-07 | `snip_and_search` -- crop region, VLM analysis, structured result | Same "Look Here" flow as ACT-06, different action type in dropdown |
| ACT-08 | `select_all_extract` -- Ctrl+A, Ctrl+C, clipboard read, VLM fallback | New executor function. "Look Here" or element click + action type change |
| ACT-09 | `scroll` -- scroll within element | `core/executor.py:scroll()` exists. Wire `direction_amount` from tag dialog |
| ACT-10 | `wait` -- wait for condition | New toolbar button, condition mini-dialog, shared condition engine |
| ACT-11 | `loop` -- repeat steps until condition | New toolbar button, loop definition dialog, shared condition engine, step range |
| ACT-12 | `prompt_user` -- pause routine, API /respond | New toolbar button, question text mini-dialog, API stub for /respond |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyQt6 | >=6.5 | Overlay UI, toolbar buttons, mini-dialogs | Project decision, all HUD built on this |
| pyautogui | >=0.9.54 | Mouse/keyboard automation, clipboard via hotkey | Already used in executor.py |
| mss | >=9.0 | Screenshot capture | Already used in capture.py |
| opencv-python | >=4.8 | Image comparison for pixel-diff condition | Already used throughout |
| numpy | >=1.24 | Array operations for screenshot comparison | Already used throughout |
| pyperclip | latest | Cross-platform clipboard read for select_all_extract | Preferred over platform-specific ctypes |

### Supporting (already in project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| networkx | >=3.0 | Graph structure in routine format | Loop edges (back-edges) in graph |
| pyyaml | >=6.0 | Config (poll interval, timeouts) | Condition engine reads ocsd.yaml |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pyperclip for clipboard | `pyautogui` hotkey + tkinter clipboard | pyperclip is simpler, cross-platform, dedicated purpose |
| Threaded condition polling | asyncio polling | Project uses threading throughout (executor, detection, VLM). Stay consistent |

**Installation:**
```bash
pip install pyperclip
```

Note: `pyperclip` should be added to `pyproject.toml` dependencies. It has no heavy deps and works cross-platform (uses pbcopy/xclip/win32 under the hood).

## Architecture Patterns

### Recommended Project Structure
```
core/
    executor.py          # Add select_all_extract(), press_enter()
    conditions.py        # NEW: Shared condition-checking engine
recorder/
    record_session.py    # Extend with non-click action handlers
    overlay/
        toolbar_panel.py # Add 4 quick-add buttons in RECORDING mode
        mini_dialogs.py  # NEW: Wait/Loop/Prompt mini-dialog widgets
        record_phase.py  # Add new phases: REGION_DRAG, LOOP_DEFINE
routine/
    format.py            # Extend build_v1_step() for action-type fields
```

### Pattern 1: Action Type Dispatch in Dry-Run
**What:** The `_execute_dry_run` method in RecordSession currently always calls `exec_click()`. Extend it with action-type dispatch.
**When to use:** Every action type needs its own dry-run behavior.
**Example:**
```python
# In RecordSession._execute_dry_run()
action = self._current_step.get("tag_data", {}).get("action", "click")
match action:
    case "click":
        exec_click(center_x, center_y)
    case "double_click":
        exec_double_click(center_x, center_y)
    case "right_click":
        exec_right_click(center_x, center_y)
    case "type":
        exec_click(center_x, center_y)
        text = self._current_step["tag_data"].get("text_to_type", "")
        exec_type_text(text)
        if self._current_step["tag_data"].get("press_enter"):
            pyautogui.press("enter")
    case "scroll":
        direction, amount = parse_direction_amount(
            self._current_step["tag_data"].get("direction_amount", "")
        )
        exec_scroll(center_x, center_y, direction, amount)
    case "click_drag":
        # Source already at center_x, center_y
        # Target from drag_target bbox
        exec_drag(center_x, center_y, target_x, target_y)
    case "read" | "snip_and_search":
        # Observation-only: no screen changes
        pass
    case "select_all_extract":
        exec_select_all_extract()  # Ctrl+A, Ctrl+C, read clipboard
    case "wait" | "loop" | "prompt_user":
        pass  # No dry-run for these
```

### Pattern 2: Shared Condition Engine (class-based, threaded)
**What:** A `ConditionChecker` class that polls a condition on a background thread.
**When to use:** Both wait and loop actions need condition checking.
**Recommendation:** Use a class with a `check()` method per condition type and a `poll_until()` driver.
**Example:**
```python
# core/conditions.py
class ConditionChecker:
    """Polls a condition until met or timeout/max-iterations reached."""

    def __init__(
        self,
        condition_type: str,  # "fixed_timer", "element_appears", "screen_change", "vlm_check", "text_matches", "n_iterations"
        params: dict,
        poll_interval: float = 2.0,
        timeout: float = 30.0,
        max_iterations: int | None = None,
    ) -> None: ...

    def poll_until(self, callback: Callable[[ConditionResult], None]) -> None:
        """Run polling on current thread. Calls callback with result."""
        ...

    def _check_element_appears(self) -> bool:
        """Screenshot + locate cascade for target element."""
        ...

    def _check_screen_change(self, baseline: np.ndarray) -> bool:
        """Pixel-diff threshold against baseline screenshot."""
        ...

    def _check_vlm(self, prompt: str) -> bool:
        """VLM natural-language condition check."""
        ...

    def _check_text_matches(self, target_text: str) -> bool:
        """OCR scan for target text in region."""
        ...
```

### Pattern 3: Toolbar Quick-Add Flow
**What:** New toolbar buttons in RECORDING mode that create non-element steps without the click-detect-VLM pipeline.
**When to use:** wait, loop, prompt_user, and look-here actions.
**Example:**
```python
# Extend _MODE_BUTTONS in toolbar_panel.py
ToolbarMode.RECORDING: [
    ("Pause", "pause"),
    ("Undo Last", "undo_last"),
    ("Look Here", "look_here"),      # NEW
    ("Add Wait", "add_wait"),         # NEW
    ("Add Loop", "add_loop"),         # NEW
    ("Add Prompt", "add_prompt"),     # NEW
],
```

### Pattern 4: Two-Bbox Click-Drag Step Format
**What:** Store both source and target bboxes for click_drag steps.
**Recommendation:** Add a `drag_target` sub-object alongside the existing `bbox` field.
```json
{
    "action": "click_drag",
    "bbox": {"x": 100, "y": 200, "w": 50, "h": 30},
    "drag_target": {
        "bbox": {"x": 400, "y": 300, "w": 60, "h": 40},
        "anchors": { "visual_match": "snippets/{node_id}_target.png", ... }
    }
}
```

### Pattern 5: Loop Step Format
**What:** Loop steps reference a range of existing step indices and an exit condition.
```json
{
    "action": "loop",
    "loop": {
        "body_steps": [2, 3, 4],
        "exit_condition": {
            "type": "element_appears",
            "element_description": "Submit button",
            "element_anchors": { ... },
            "max_iterations": 10
        }
    }
}
```

### Anti-Patterns to Avoid
- **Building separate wait and loop condition checkers:** They must share the same engine. Wait is literally a bodyless loop with one exit condition.
- **Infinite loops without safety:** Every condition-based wait/loop MUST have a timeout or max-iterations cap. No exceptions.
- **Synchronous condition checking on the Qt main thread:** Condition polling involves screenshots and VLM calls -- always run on a background thread, deliver results via PipelineBridge.
- **Storing loop body as step copies:** Loop should reference step indices, not duplicate step data. This avoids data inconsistency.
- **Platform-specific clipboard code without fallback:** Use pyperclip or guard all clipboard code with platform checks.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Clipboard access | ctypes calls to Win32 API | `pyperclip.paste()` | Cross-platform, handles encoding, well-tested |
| Image diff for screen change | Custom pixel comparison | `cv2.absdiff()` + `cv2.countNonZero()` | OpenCV's C implementation is orders of magnitude faster |
| Human-like mouse/keyboard | New automation primitives | Existing `core/executor.py` functions | Already battle-tested with Bezier, donut, typo sim |
| Condition polling loop | Custom threading code | `threading.Thread` + `time.sleep` loop | Simple, consistent with project patterns |
| Cross-platform Enter key | Platform-specific keycode | `pyautogui.press("enter")` | Already used throughout |

**Key insight:** 6 of the 12 executor functions already exist. The recording-side wiring and format extension are the real work, not new automation primitives.

## Common Pitfalls

### Pitfall 1: Clipboard Race Condition in select_all_extract
**What goes wrong:** Ctrl+A -> Ctrl+C fires too fast; clipboard hasn't updated when you read it.
**Why it happens:** OS clipboard update is asynchronous. Win32 and X11 have different latencies.
**How to avoid:** Add a human-delay-scaled sleep (200-500ms) between Ctrl+C and clipboard read. Retry once if clipboard is empty. Compare clipboard content before/after to detect unchanged.
**Warning signs:** Tests pass locally, fail in CI or on slower machines.

### Pitfall 2: Condition Engine Blocking the UI
**What goes wrong:** Polling screenshots + VLM checks freeze the overlay.
**Why it happens:** Running condition checks on the main Qt thread.
**How to avoid:** Always run ConditionChecker.poll_until() on a background thread. Emit results via PipelineBridge signals. Use the established project pattern.
**Warning signs:** Overlay becomes unresponsive during wait/loop execution.

### Pitfall 3: Loop Step Index Invalidation
**What goes wrong:** Loop body references step indices [2,3,4], but user deletes step 1 during editing (Phase 8).
**Why it happens:** Index-based references break on insertion/deletion.
**How to avoid:** Use `node_id` references instead of raw step indices. node_ids are UUIDs and stable across edits.
**Warning signs:** Not a Phase 6 problem (no editing), but design for it now.

### Pitfall 4: Tag Dialog Dropdown Not Updating Conditional Fields
**What goes wrong:** User changes action type in dropdown but conditional fields don't appear.
**Why it happens:** The `_CONDITIONAL_FIELDS` dict is already wired, but the recording pipeline may not correctly pass the action type through to dry-run dispatch.
**How to avoid:** Ensure `tag_data["action"]` is read from `get_form_data()` (not just `element_type`), and that dry-run dispatch keys off this field.
**Warning signs:** The action field in saved routine.json is always "click" regardless of dropdown selection.

### Pitfall 5: "Look Here" Region Drag Conflicting with Element Selection
**What goes wrong:** User presses "Look Here" button, then clicks/drags, but the existing selection handler treats it as a normal element capture.
**Why it happens:** `on_selection()` in RecordSession only knows about AWAITING_CLICK phase.
**How to avoid:** Add a new RecordPhase (e.g., `AWAITING_REGION_DRAG`) that "Look Here" enters. The selection handler checks this phase and routes to the observation-step pipeline instead of the click-detect pipeline.
**Warning signs:** Clicking "Look Here" then dragging starts the full detection/VLM pipeline instead of the read/snip flow.

### Pitfall 6: Nested Loop Serialization
**What goes wrong:** A loop body containing another loop creates circular or ambiguous step references.
**Why it happens:** If loop body is stored as step index ranges, an inner loop's body could overlap or reference the outer loop.
**How to avoid:** Each loop step stores `body_step_node_ids` (not index ranges). The replay engine processes loops recursively using node_ids. Inner loops are just regular steps within the body.
**Warning signs:** Routine JSON has overlapping loop body ranges.

## Code Examples

### Extending build_v1_step for Action-Type Fields
```python
# In routine/format.py:build_v1_step()
# After the base step dict construction, add action-type-specific fields:

action = tag_data.get("action", "click")
v1_step["action"] = action

if action == "type":
    v1_step["text_to_type"] = tag_data.get("text_to_type", "")
    v1_step["press_enter"] = tag_data.get("press_enter", False)
elif action == "scroll":
    v1_step["scroll"] = _parse_direction_amount(
        tag_data.get("direction_amount", "down 3")
    )
elif action == "click_drag":
    # drag_target populated by second capture flow
    v1_step["drag_target"] = step.get("drag_target")
elif action in ("read", "snip_and_search"):
    v1_step["vlm_prompt"] = tag_data.get("vlm_prompt", "")
elif action == "select_all_extract":
    v1_step["vlm_prompt"] = tag_data.get("vlm_prompt", "")
elif action == "wait":
    v1_step["wait"] = _parse_wait_condition(
        tag_data.get("condition_timeout", "30")
    )
elif action == "loop":
    v1_step["loop"] = step.get("loop_definition")
elif action == "prompt_user":
    v1_step["question_text"] = tag_data.get("question_text", "")
```

### select_all_extract Executor Function
```python
# In core/executor.py
def select_all_extract(dry_run: bool = False) -> str:
    """Select all text and extract via clipboard, with VLM fallback.

    Returns:
        Extracted text content.
    """
    if dry_run:
        return ""

    import pyperclip

    # Save current clipboard
    old_clipboard = pyperclip.paste()

    # Select all + copy
    hotkey("ctrl", "a")
    _hsleep(0.15)
    hotkey("ctrl", "c")
    _hsleep(0.3)  # Wait for clipboard update

    # Read clipboard
    new_clipboard = pyperclip.paste()

    if new_clipboard and new_clipboard != old_clipboard:
        return new_clipboard

    # Fallback: VLM screenshot analysis
    logger.info("Clipboard empty/unchanged, falling back to VLM")
    from core.capture import screenshot_full
    from core.vision import analyze_crop_array
    screenshot = screenshot_full()
    result = analyze_crop_array(screenshot, "Extract all visible text")
    return result.get("text", "") if result else ""
```

### Pixel Diff for Screen Change Condition
```python
# In core/conditions.py
def check_screen_changed(
    baseline: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.05,
) -> bool:
    """Compare two screenshots using pixel difference.

    Args:
        baseline: Reference screenshot (BGR).
        current: Current screenshot (BGR).
        threshold: Fraction of pixels that must differ (0.0-1.0).

    Returns:
        True if screen has changed beyond threshold.
    """
    import cv2

    gray_base = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_base, gray_curr)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return (changed_pixels / total_pixels) > threshold
```

### New RecordPhase States
```python
# Add to record_phase.py
class RecordPhase(Enum):
    # ... existing phases ...

    AWAITING_REGION_DRAG = auto()
    """After 'Look Here' button, waiting for user to drag a region."""

    LOOP_DEFINING = auto()
    """Loop definition dialog open, user selecting step range and condition."""

    WAIT_CONFIGURING = auto()
    """Wait condition mini-dialog open."""

    PROMPT_CONFIGURING = auto()
    """Prompt question mini-dialog open."""
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single bbox per step | Two bboxes for click_drag | Phase 6 | Format extension, no breaking change |
| Click-only dry-run | Action-type-dispatched dry-run | Phase 6 | `_execute_dry_run` becomes a match/case dispatcher |
| Only click-initiated steps | Toolbar quick-add for non-element actions | Phase 6 | New RecordPhase states, new toolbar buttons |

**Deprecated/outdated:**
- None. This phase extends existing patterns, doesn't replace them.

## Open Questions

1. **Mini-dialog widget reuse vs new**
   - What we know: Tag dialog is a QGraphicsObject with card glow, frosted glass, typewriter. It's complex.
   - What's unclear: Whether wait/loop/prompt mini-dialogs should reuse TagDialogPanel (complex but consistent) or be simpler standalone QGraphicsObject widgets.
   - Recommendation: Build lightweight mini-dialog widgets that share the frosted glass + card glow visual style from `hud_common.py` but are simpler than TagDialogPanel. The wait/prompt dialogs only need 1-3 fields. TagDialogPanel is overkill.

2. **Loop "element appears" capture UX flow details**
   - What we know: User clicks "Add Loop" -> selects step range -> picks exit condition. For "element appears", system replays loop body with manual Enter advance.
   - What's unclear: Exact state transitions during the interactive loop capture. How does the user signal "I see the exit element now"?
   - Recommendation: Add a RecordPhase.LOOP_ITERATION_ADVANCE state. After each loop body replay, show toolbar with "Next Iteration" and "Exit Element Found" buttons. User clicks "Exit Element Found" then clicks/drags the element on screen. Standard capture pipeline runs for that element.

3. **click_drag second capture trigger**
   - What we know: User clicks source element, tag dialog shows click_drag with "drag target hint" field. Then somehow captures the second element.
   - What's unclear: Does the second capture happen after tag dialog confirm? Or does the user need to confirm, then click the target?
   - Recommendation: After confirming the tag dialog for a click_drag action, transition to a new AWAITING_DRAG_TARGET phase. The next click/drag captures the target element. Run the capture-detect-VLM pipeline again for just the target, then merge both into one step.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in pyproject.toml) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/test_record_session.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ACT-01 | click action in step format | unit | `python -m pytest tests/test_action_types.py::test_click_step_format -x` | -- Wave 0 |
| ACT-02 | double_click in step format | unit | `python -m pytest tests/test_action_types.py::test_double_click_step_format -x` | -- Wave 0 |
| ACT-03 | right_click in step format | unit | `python -m pytest tests/test_action_types.py::test_right_click_step_format -x` | -- Wave 0 |
| ACT-04 | click_drag two-bbox format | unit | `python -m pytest tests/test_action_types.py::test_click_drag_two_bbox -x` | -- Wave 0 |
| ACT-05 | type with text_to_type and press_enter | unit | `python -m pytest tests/test_action_types.py::test_type_step_format -x` | -- Wave 0 |
| ACT-06 | read step captures vlm_prompt | unit | `python -m pytest tests/test_action_types.py::test_read_step_format -x` | -- Wave 0 |
| ACT-07 | snip_and_search step format | unit | `python -m pytest tests/test_action_types.py::test_snip_search_step_format -x` | -- Wave 0 |
| ACT-08 | select_all_extract with clipboard + VLM fallback | unit | `python -m pytest tests/test_action_types.py::test_select_all_extract -x` | -- Wave 0 |
| ACT-09 | scroll step with direction_amount | unit | `python -m pytest tests/test_action_types.py::test_scroll_step_format -x` | -- Wave 0 |
| ACT-10 | wait step with condition_timeout | unit | `python -m pytest tests/test_condition_engine.py::test_wait_condition -x` | -- Wave 0 |
| ACT-11 | loop step with body_steps and exit condition | unit | `python -m pytest tests/test_condition_engine.py::test_loop_definition -x` | -- Wave 0 |
| ACT-12 | prompt_user step with question_text | unit | `python -m pytest tests/test_action_types.py::test_prompt_user_step_format -x` | -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_action_types.py tests/test_condition_engine.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_action_types.py` -- covers ACT-01 through ACT-09, ACT-12 (step format and dry-run dispatch)
- [ ] `tests/test_condition_engine.py` -- covers ACT-10, ACT-11 (wait/loop condition checking)
- [ ] `pyperclip` added to `pyproject.toml` dependencies

## Sources

### Primary (HIGH confidence)
- `core/executor.py` -- read directly, verified all 6 existing action functions
- `recorder/overlay/tag_dialog_panel.py` -- read directly, verified `_CONDITIONAL_FIELDS` dict at line 127
- `recorder/record_session.py` -- read directly, verified state machine and dry-run flow
- `routine/format.py` -- read directly, verified `build_v1_step()` and Routine model
- `recorder/overlay/toolbar_panel.py` -- read directly, verified ToolbarMode and button definitions
- `recorder/overlay/pipeline_bridge.py` -- read directly, verified 6 pyqtSignals for thread-safe delivery
- `recorder/overlay/record_phase.py` -- read directly, verified 10 RecordPhase states

### Secondary (MEDIUM confidence)
- pyperclip cross-platform clipboard support -- well-known library, standard recommendation for Python clipboard access
- cv2.absdiff for pixel comparison -- standard OpenCV pattern for image differencing

### Tertiary (LOW confidence)
- None. All findings verified against codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project except pyperclip (well-known)
- Architecture: HIGH -- patterns extend established codebase conventions (PipelineBridge, RecordPhase state machine, ToolbarMode)
- Pitfalls: HIGH -- identified from reading actual code patterns and known cross-platform issues

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (stable -- internal codebase patterns, no fast-moving external deps)
