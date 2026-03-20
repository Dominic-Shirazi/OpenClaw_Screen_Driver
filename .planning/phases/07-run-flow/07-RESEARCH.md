# Phase 7: Run Flow - Research

**Researched:** 2026-03-19
**Domain:** Routine replay engine, element location, failure recovery, overlay replay UX
**Confidence:** HIGH

## Summary

Phase 7 builds the step-sequential routine runner (`routine/runner.py`) that reads a v1 `routine.json`, locates each target element via the existing 5-stage cascade, executes actions with the existing human-like executor, validates results, and handles failures with a multi-stage recovery cascade. It also completes the condition engine stubs (`element_appears`, `text_matches`) in `core/conditions.py`, adds replay overlay UX (purple shimmer, status badge, target highlight), and implements run logging with self-cleaning directories.

The codebase is mature for this phase. All building blocks exist: `core/locate.py` (5-stage cascade), `core/executor.py` (Bezier mouse, typo typing), `core/conditions.py` (poll loop infrastructure), `mapper/validator.py` (post-action validation), and `routine/format.py` (Routine.load/save). The legacy `mapper/runner.py` provides established patterns for event callbacks, retry loops, and execution flow. The new runner adapts these patterns for step-sequential v1 routines instead of graph-based replay.

**Primary recommendation:** Build `routine/runner.py` as a new module that iterates v1 steps sequentially, adapting `core/locate.py` to accept step anchor dicts instead of graph nodes. Reuse `core/executor.py` and `mapper/validator.py` directly. Complete condition engine stubs using the locate cascade for `element_appears` and fuzzy OCR for `text_matches`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Replay progress UX (RUN-09)**: Purple = "automation active" shimmer. Small frosted-glass pill badge at top-center showing step label only. Quick purple glow around located element bbox (~300ms). Camera flash after screenshots. Badge text updates with OK/RETRY. Immediately clear all overlay on completion. `replay.show_overlay` config in ocsd.yaml (default `true`). Reuse existing `hide_for_capture()`/`show_after_capture()` cycle.
- **Failure cascade (RUN-07, RUN-08)**: 5-stage recovery: retry at position (2x) -> 25% region scan -> full-screen scan -> LiteLLM AI fallback -> abort with annotated log. Never blind-click.
- **Run logs**: `~/.ocsd/routines/{name}/runs/{run_id}/` with auto-prune (keep last 5 successful, 10 failed). Structured logging at INFO level.
- **Run lifecycle (RUN-01)**: Pre-flight validation (assets exist, checksum, VLM connectivity). VLM unavailable returns status to caller. `start_from='desktop'` auto-minimizes (Win+D). `start_from='app'` skips. Pluggable pre-run step for V2 GPS. Silent proceed on resolution mismatch.
- **Runner architecture**: New `routine/runner.py`, step-sequential. Existing `mapper/runner.py` stays for legacy. Loop replay inline re-execution via `body_step_node_ids`. Human-like execution via existing `core/executor.py`. Configurable `human_delay` multiplier.
- **Condition engine completion**: `element_appears` uses region-first then full screen, adaptive polling (stages 1-3 first 3 polls, then include VLM). `text_matches` uses fuzzy OCR with Levenshtein distance. 2-second default poll interval.

### Claude's Discretion
- Exact Levenshtein distance threshold for fuzzy text matching
- How to annotate screenshots with bboxes in failure logs (cv2, PIL, etc.)
- LiteLLM smart prompt design for the AI fallback stage
- Status badge styling details (font, opacity, padding)
- Run directory naming convention (timestamp format, run_id generation)
- How to detect which 25% region an element belongs to from recorded position
- Pre-flight validation ordering and error aggregation
- How to structure the pluggable pre-run step for V2 GPS extensibility

### Deferred Ideas (OUT OF SCOPE)
- Purple shimmer during recording dry-run execution (Phase 4 retroactive)
- Cloud AI computer-use model multi-provider fallback chain (V2)
- V2 GPS skip-ahead
- Rich TUI progress display during replay (Phase 9)
- API run status with per-step output (Phase 10)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RUN-01 | User selects routine from TUI or triggers via CLI/API | Pre-flight validation + `Routine.load()` + `routine/runner.py` entry point. Phase 7 builds the runner engine; TUI/CLI wiring is Phase 9 |
| RUN-02 | Per-step element location using 5-stage cascade | Adapt `core/locate.py:locate_element()` to accept v1 step anchor dicts instead of graph node_ids |
| RUN-03 | Human-like mouse execution via Bezier curves with overshoot-and-correct | Already implemented in `core/executor.py:_bezier_move()` with doughnut offset. Fully reusable |
| RUN-04 | Human-like typing with per-letter delays, variance, occasional typo + correction | Already implemented in `core/executor.py:type_text()`. Fully reusable |
| RUN-05 | Configurable execution speed via human_delay multiplier | Already implemented in `core/executor.py:_exec_cfg()` and `_hsleep()`. Config at `execution.human_delay` |
| RUN-06 | Post-action validation via pixel-diff + optional VLM confirmation | Reuse `mapper/validator.py:validate_action()` directly |
| RUN-07 | Failure cascade: auto-retry -> widen search -> prompt -> log | New 5-stage failure cascade in `routine/runner.py` with LiteLLM AI fallback |
| RUN-08 | Never silently fail or blind-click when unsure | Cascade stage 5 aborts with annotated log. No fallback to position-only during failure recovery |
| RUN-09 | Step-by-step replay status visible to user | Purple REPLAYING shimmer, status badge, target highlight, camera flash overlay elements |
</phase_requirements>

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyQt6 | >=6.5 | Overlay UI for replay status badge, shimmer, target highlight | Project UI framework |
| opencv-python | >=4.8 | Screenshot annotation for failure logs (draw bboxes) | Already used for capture/detection |
| pyautogui | * | Mouse/keyboard automation, screen size | Already used by executor |
| numpy | * | Embedding comparisons, image arrays | Already used throughout |
| pytesseract | * | OCR for text_matches condition | Already used in `core/ocr.py` |

### New Dependency
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rapidfuzz | >=3.0 | Levenshtein distance for fuzzy OCR text matching | `text_matches` condition - compare OCR output against target text with tolerance for OCR errors |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rapidfuzz | python-Levenshtein | rapidfuzz is 40% faster, MIT license vs GPLv2, more string metrics |
| rapidfuzz | manual Levenshtein | Hand-rolling is slower and more error-prone for edge cases |
| cv2 for bbox annotation | Pillow | cv2 is already imported throughout, no new dependency |

**Installation:**
```bash
pip install rapidfuzz>=3.0
```

## Architecture Patterns

### New Module Structure
```
routine/
    runner.py          # NEW: Step-sequential replay engine
    format.py          # Existing: Routine dataclass + load/save
    checksum.py        # Existing: SHA256 integrity
    discovery.py       # Existing: List routines from disk

core/
    locate.py          # MODIFIED: Add locate_element_from_step() for v1 steps
    conditions.py      # MODIFIED: Complete element_appears + text_matches stubs
    config.py          # MODIFIED: Add replay config section defaults

recorder/overlay/
    state.py           # MODIFIED: Add REPLAYING state
    shimmer_layer.py   # MODIFIED: Add purple color for REPLAYING
    status_badge.py    # NEW: Frosted-glass pill showing step progress
    target_highlight.py # NEW: Quick purple glow around located element
    controller.py      # MODIFIED: Add replay-mode API methods
    view.py            # MODIFIED: Create/manage new replay layers
```

### Pattern 1: Step-Sequential Runner with Event Callbacks
**What:** `routine/runner.py` iterates steps, emitting events at each stage for overlay/logging consumers
**When to use:** All routine replay execution
**Example:**
```python
# Source: Adapted from mapper/runner.py patterns
from enum import Enum, auto
from typing import Any, Callable, Protocol

class RunEvent(Enum):
    RUN_START = auto()
    STEP_START = auto()
    ELEMENT_LOCATED = auto()
    ACTION_EXECUTED = auto()
    VALIDATION_PASSED = auto()
    STEP_FAILED = auto()
    STEP_COMPLETE = auto()
    RUN_COMPLETE = auto()
    RUN_FAILED = auto()

class RunCallback(Protocol):
    def __call__(self, event: RunEvent, data: dict[str, Any]) -> None: ...

def run_routine(
    routine: Routine,
    *,
    callback: RunCallback | None = None,
    show_overlay: bool = True,
) -> RunResult:
    """Execute a routine step-by-step."""
    ...
```

### Pattern 2: Locate Adapter for V1 Steps
**What:** Adapt `locate_element()` to work with v1 step anchor dicts instead of graph node_ids
**When to use:** Every step in routine replay
**Example:**
```python
# New function in core/locate.py
def locate_element_from_step(
    step: dict[str, Any],
    routine_dir: Path,
) -> LocateResult:
    """Locate element using v1 step anchors.

    Extracts snippet_path, embedding_path, ocr_text, position_pct,
    and region_hint from the step dict and runs the 5-stage cascade.
    """
    anchors = step.get("anchors", {})
    # Stage 1: OmniParser with snippet from routine_dir / snippet_path
    # Stage 2: CLIP with embedding from routine_dir / embedding_path
    # Stage 3: OCR with anchors["ocr_text"]
    # Stage 4: VLM with step["label"]
    # Stage 5: Position from anchors["position_pct"]
    ...
```

### Pattern 3: Failure Recovery Cascade
**What:** Multi-stage recovery when initial locate fails
**When to use:** When `locate_element_from_step()` raises `ElementNotFoundError`
**Example:**
```python
def _failure_cascade(
    step: dict[str, Any],
    routine: Routine,
    routine_dir: Path,
    run_log_dir: Path,
) -> LocateResult | None:
    """5-stage failure recovery cascade.

    Stage 1: Retry full cascade at recorded position (2 attempts)
    Stage 2: 25% region scan around recorded position
    Stage 3: Full-screen 5-stage cascade
    Stage 4: LiteLLM AI fallback with full context
    Stage 5: Abort with annotated failure log
    """
    ...
```

### Pattern 4: Adaptive Condition Polling
**What:** `element_appears` starts cheap (stages 1-3) and escalates to VLM after 3 polls
**When to use:** Wait steps with `element_appears` condition, loop exit conditions
**Example:**
```python
def _check_element_appears(self) -> bool:
    """Full implementation using locate cascade."""
    iteration = self._iteration_count  # Track internally
    # First 3 polls: stages 1-3 only (fast)
    # After 3 polls: include VLM stage 4
    use_vlm = iteration >= 3
    try:
        result = locate_element_from_step(
            self.params.get("step", {}),
            routine_dir=self.params.get("routine_dir", Path(".")),
            skip_vlm=not use_vlm,
            skip_position_fallback=True,  # Never use position for conditions
        )
        return result is not None
    except ElementNotFoundError:
        return False
```

### Pattern 5: Run Log Directory with Self-Cleaning
**What:** Each run creates `~/.ocsd/routines/{name}/runs/{run_id}/` with log + screenshots
**When to use:** Every routine execution
**Example:**
```python
import uuid
from datetime import datetime

def _create_run_dir(routine_dir: Path) -> Path:
    """Create timestamped run directory."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = routine_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _prune_old_runs(routine_dir: Path) -> None:
    """Keep last 5 successful + 10 failed runs, delete older."""
    runs_dir = routine_dir / "runs"
    if not runs_dir.exists():
        return
    # Sort by mtime, read result.json from each, partition by success/fail
    # Delete excess beyond 5 success / 10 failure
    ...
```

### Anti-Patterns to Avoid
- **Graph dependency in runner:** Phase 7 runner MUST NOT require an OCSDGraph. It works with v1 step dicts directly. The graph exists in `routine.json` but the runner iterates `steps[]` sequentially.
- **Shared mutable state:** Runner executes on a background thread. Never mutate overlay state directly -- emit events and let the Qt thread handle visual updates via PipelineBridge signals.
- **Blocking the Qt thread:** All locate/execute/validate calls happen on a background thread. Only overlay updates (badge text, shimmer state, highlight) happen on the main thread via signal delivery.
- **Full cascade on every condition poll:** The first 3 polls of `element_appears` should skip VLM (stage 4) for speed. Only escalate after 3 cheap attempts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Fuzzy text matching | Custom Levenshtein | `rapidfuzz.distance.Levenshtein` | Edge cases with unicode, empty strings, performance |
| Bezier mouse movement | New mouse path logic | `core/executor.py:_bezier_move()` | Already implemented with doughnut offset, thread-safe |
| Post-action validation | New pixel diff logic | `mapper/validator.py:validate_action()` | Already has pixel diff + VLM flow |
| Routine loading | Custom JSON parser | `routine/format.py:Routine.load()` | Handles checksum validation, schema detection |
| Thread-safe Qt signals | Custom queue/callback | `PipelineBridge` pattern with pyqtSignal | AutoConnection handles thread delivery |
| Screenshot bbox annotation | Custom drawing | `cv2.rectangle()` + `cv2.putText()` | cv2 already imported, trivial API |
| Condition polling loop | Custom while loop | `ConditionChecker.poll_until()` | Already handles timeout, cancellation, iteration counting |

**Key insight:** Almost every building block exists. Phase 7's value is orchestration -- connecting existing pieces into a coherent replay flow, not building new primitives.

## Common Pitfalls

### Pitfall 1: Graph-Coupled Locate
**What goes wrong:** `locate_element()` currently takes `graph: OCSDGraph, node_id: str` -- it cannot be called with a v1 step dict
**Why it happens:** The existing function was built for graph-based replay
**How to avoid:** Create `locate_element_from_step()` that accepts step anchors dict and routine_dir. Internally builds the same data structures the cascade expects. Do NOT refactor the existing function (it's still used by `mapper/runner.py`)
**Warning signs:** ImportError or TypeError when trying to call locate_element without a graph

### Pitfall 2: Overlay Thread Safety
**What goes wrong:** Runner thread updates overlay directly, causing Qt crashes
**Why it happens:** Qt widgets are not thread-safe, must be modified from main thread only
**How to avoid:** Runner emits events via callback. A Qt-side adapter receives events and updates overlay via PipelineBridge signals or QMetaObject.invokeMethod
**Warning signs:** "QObject: Cannot create children for a parent in a different thread"

### Pitfall 3: Screenshot Contamination
**What goes wrong:** Overlay elements appear in screenshots used for locate/validate
**Why it happens:** Forgetting to call `hide_for_capture()` before screenshots
**How to avoid:** All screenshots during replay must use the hide/capture/show cycle. The runner's locate function must coordinate with overlay visibility
**Warning signs:** Locate cascade finds the status badge or purple glow instead of target element

### Pitfall 4: Condition Polling Blocks Forever
**What goes wrong:** `element_appears` or `text_matches` never returns True, runner hangs
**Why it happens:** No timeout set, or timeout too high
**How to avoid:** Always pass a reasonable timeout to ConditionChecker (from step's wait/loop definition). The existing poll_until already handles timeouts correctly
**Warning signs:** Runner appears frozen on a wait/loop step

### Pitfall 5: Run Directory Disk Bloat
**What goes wrong:** Annotated screenshots from every failed attempt accumulate
**Why it happens:** No cleanup after runs, especially in automated/CI scenarios
**How to avoid:** Self-cleaning prune function runs after each run completion. Also limit annotated screenshot count per run (only save key screenshots: pre-failure, each cascade stage result)
**Warning signs:** `~/.ocsd/routines/*/runs/` grows to GB-scale

### Pitfall 6: Locate Adapter Missing Snippet/Embedding Files
**What goes wrong:** Stage 1/2 of cascade fail because snippet PNGs or .npy files aren't at expected paths
**Why it happens:** v1 step stores relative paths (`snippets/abc123.png`) but caller doesn't resolve against routine directory
**How to avoid:** `locate_element_from_step()` takes `routine_dir: Path` and resolves all relative paths. Pre-flight validation checks all referenced files exist before run starts

## Code Examples

### Locate Element from V1 Step (New Function)
```python
# Source: Adapted from core/locate.py:locate_element()
def locate_element_from_step(
    step: dict[str, Any],
    routine_dir: Path,
    *,
    skip_vlm: bool = False,
    skip_position_fallback: bool = False,
) -> LocateResult:
    """Locate element using v1 step anchors instead of graph node.

    Args:
        step: V1 step dict with anchors, snippet_path, embedding_path, label.
        routine_dir: Base directory for resolving relative paths.
        skip_vlm: Skip VLM stage (stage 4) for faster polling.
        skip_position_fallback: Skip position stage (stage 5) for conditions.
    """
    anchors = step.get("anchors", {})
    pos_pct = anchors.get("position_pct", {})
    sw, sh = pyautogui.size()
    hint_x = int(pos_pct["x_pct"] * sw) if "x_pct" in pos_pct else None
    hint_y = int(pos_pct["y_pct"] * sh) if "y_pct" in pos_pct else None

    # Stage 1: OmniParser detect+match
    snippet_rel = step.get("snippet_path", "")
    if snippet_rel:
        snippet_path = routine_dir / snippet_rel
        # ... load and match (same logic as locate_element stage 1)

    # Stage 2: CLIP embedding
    emb_rel = step.get("embedding_path", "")
    if emb_rel:
        emb_path = routine_dir / emb_rel
        # ... load .npy and compare (same logic as stage 2)

    # Stage 3: OCR text match
    ocr_text = anchors.get("ocr_text")
    if ocr_text:
        # ... find_text_on_screen with hint (same as stage 3)

    # Stage 4: VLM (skip if skip_vlm=True)
    if not skip_vlm:
        label = step.get("label", "")
        # ... VLM scan (same as stage 4)

    # Stage 5: Position fallback (skip if skip_position_fallback=True)
    if not skip_position_fallback and hint_x is not None:
        # ... return position with low confidence

    raise ElementNotFoundError(
        step.get("node_id", "unknown"),
        f"All locate stages failed for step {step.get('step_index', '?')}"
    )
```

### Fuzzy Text Matching for text_matches Condition
```python
# Source: rapidfuzz documentation
from rapidfuzz.distance import Levenshtein

def _check_text_matches(self) -> bool:
    """Fuzzy OCR text matching with Levenshtein tolerance."""
    target = self.params.get("target_text", "")
    if not target:
        return False

    # OCR the relevant region
    hint = self.params.get("position_hint", {})
    result = find_text_on_screen(
        target,
        hint_x=hint.get("x"),
        hint_y=hint.get("y"),
        search_radius=400,
    )
    if result is not None:
        return True  # Exact substring match found

    # Fuzzy fallback: OCR full region and check Levenshtein distance
    img = self._take_screenshot()
    if img is None:
        return False
    blocks = ocr_with_boxes(img)

    threshold = self.params.get("fuzzy_threshold", 3)  # max edit distance
    for block in blocks:
        dist = Levenshtein.distance(target.lower(), block["text"].lower())
        if dist <= threshold:
            return True
    return False
```

### 25% Region Computation for Failure Recovery
```python
# Source: Research recommendation
def _compute_search_region(
    pos_pct: dict[str, float],
    screen_w: int,
    screen_h: int,
) -> tuple[int, int, int, int]:
    """Compute 25% overlapping region around recorded position.

    Returns:
        (rx, ry, rw, rh) in absolute pixels.
    """
    cx = int(pos_pct.get("x_pct", 0.5) * screen_w)
    cy = int(pos_pct.get("y_pct", 0.5) * screen_h)

    # 25% of screen with some overlap
    rw = int(screen_w * 0.35)  # Slightly larger than 25% for overlap
    rh = int(screen_h * 0.35)

    # Center region on recorded position, clamp to screen
    rx = max(0, min(cx - rw // 2, screen_w - rw))
    ry = max(0, min(cy - rh // 2, screen_h - rh))

    return rx, ry, rw, rh
```

### Screenshot Annotation for Failure Logs
```python
# Source: OpenCV drawing API
import cv2

def annotate_screenshot(
    img: np.ndarray,
    bboxes: list[dict],
    label: str = "",
) -> np.ndarray:
    """Draw bboxes and labels on a screenshot for failure logs.

    Args:
        img: BGR screenshot array.
        bboxes: List of {x, y, w, h, label, confidence, color} dicts.
        label: Overall annotation label.
    """
    annotated = img.copy()
    for bbox in bboxes:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        color = bbox.get("color", (0, 255, 0))  # Green default
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        text = f"{bbox.get('label', '')} {bbox.get('confidence', 0):.2f}"
        cv2.putText(annotated, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated
```

### LiteLLM AI Fallback Prompt Design
```python
# Source: Research recommendation
def _build_ai_fallback_prompt(
    routine: Routine,
    step_index: int,
    step: dict,
    screenshot_b64: str,
) -> str:
    """Build smart prompt for LiteLLM AI fallback (Stage 4 of failure cascade).

    Includes full routine context so the model can reason about workflow state.
    """
    steps_summary = "\n".join(
        f"  Step {s['step_index']}: {s['action']} on '{s['label']}'"
        + (" [DONE]" if s['step_index'] < step_index else
           " [CURRENT]" if s['step_index'] == step_index else "")
        for s in routine.steps
    )

    return f"""You are helping an automation system find a UI element on screen.

Routine: "{routine.name}" - {routine.description}
Steps:
{steps_summary}

Current step {step_index}: {step['action']} on "{step['label']}"
Element type: {step['element_type']}
OCR text hint: {step.get('anchors', {}).get('ocr_text', 'none')}

The screenshot shows the current screen state. The system could not find
the target element using visual matching, CLIP embeddings, OCR, or VLM detection.

Where on the screen is the element "{step['label']}" that the user needs to
{step['action']}? Respond with JSON: {{"x": pixel_x, "y": pixel_y, "confidence": 0.0-1.0, "reasoning": "..."}}
If the element is not visible, respond: {{"x": -1, "y": -1, "confidence": 0.0, "reasoning": "element not visible because..."}}"""
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Graph-based replay (`mapper/runner.py`) | Step-sequential replay (`routine/runner.py`) | Phase 7 | Simpler model for v1 routines, graph-based stays for V2 GPS |
| `locate_element(graph, node_id)` | `locate_element_from_step(step, routine_dir)` | Phase 7 | Decouples locate from graph dependency |
| Condition stubs (always False) | Full condition implementations | Phase 7 | Enables wait/loop actions to actually work |

**Deprecated/outdated:**
- `mapper/runner.py`: Not deprecated, but Phase 7's `routine/runner.py` is the primary replay engine for v1 routines going forward

## Open Questions

1. **Levenshtein threshold value**
   - What we know: OCR errors typically produce 1-2 character substitutions (e.g., "1" for "l", "0" for "O")
   - Recommendation: Use `max(2, len(target) // 5)` -- at least 2 edits, plus 1 per 5 chars of target length. This handles short labels (3-5 chars, threshold=2) and longer text (10 chars, threshold=2). Cap at 5 to avoid false positives.

2. **Overlay integration without importing recorder in routine/**
   - What we know: `routine/runner.py` should not directly depend on `recorder/overlay/` (circular dependency risk)
   - Recommendation: Runner emits events via callback function. A separate `routine/replay_overlay.py` adapter subscribes to events and drives overlay updates via PipelineBridge or OverlayController. This keeps the runner headless-capable.

3. **VLM connectivity pre-flight test**
   - What we know: LiteLLM proxy URL is in config. Need a lightweight test.
   - Recommendation: Send a minimal text-only prompt to the configured model with a 5-second timeout. If it fails, report to caller but don't auto-abort.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.0 with pytest-mock >=3.0 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `python -m pytest tests/test_routine_runner.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RUN-01 | Pre-flight validates assets, checksum, VLM | unit | `python -m pytest tests/test_routine_runner.py::test_preflight -x` | Wave 0 |
| RUN-02 | Locate cascade with v1 step anchors | unit | `python -m pytest tests/test_routine_runner.py::test_locate_from_step -x` | Wave 0 |
| RUN-03 | Bezier mouse movement (existing) | unit | `python -m pytest tests/test_step_replay.py -x` | Exists |
| RUN-04 | Type with typo simulation (existing) | unit | `python -m pytest tests/test_step_replay.py -x` | Exists |
| RUN-05 | human_delay multiplier (existing) | unit | `python -m pytest tests/test_step_replay.py -x` | Exists |
| RUN-06 | Post-action validation via validator | unit | `python -m pytest tests/test_routine_runner.py::test_post_action_validation -x` | Wave 0 |
| RUN-07 | 5-stage failure cascade | unit | `python -m pytest tests/test_routine_runner.py::test_failure_cascade -x` | Wave 0 |
| RUN-08 | Never blind-click on uncertainty | unit | `python -m pytest tests/test_routine_runner.py::test_never_blind_click -x` | Wave 0 |
| RUN-09 | Purple shimmer + status badge overlay | unit | `python -m pytest tests/test_replay_overlay.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_routine_runner.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_routine_runner.py` -- covers RUN-01, RUN-02, RUN-06, RUN-07, RUN-08
- [ ] `tests/test_replay_overlay.py` -- covers RUN-09 (purple shimmer, badge, highlight)
- [ ] `tests/test_condition_completion.py` -- covers element_appears and text_matches implementations
- [ ] `rapidfuzz>=3.0` added to pyproject.toml dependencies

## Sources

### Primary (HIGH confidence)
- `core/locate.py` -- Current 5-stage cascade implementation (read directly)
- `core/executor.py` -- Human-like execution API (read directly)
- `core/conditions.py` -- ConditionChecker with stubs (read directly)
- `routine/format.py` -- Routine dataclass, v1 schema (read directly)
- `mapper/runner.py` -- Legacy runner patterns (read directly)
- `mapper/validator.py` -- Post-action validation (read directly)
- `recorder/overlay/state.py` -- OverlayState enum (read directly)
- `recorder/overlay/shimmer_layer.py` -- Shimmer rendering (read directly)
- `recorder/overlay/pipeline_bridge.py` -- Thread-safe signals (read directly)
- `recorder/overlay/controller.py` -- Overlay public API (read directly)

### Secondary (MEDIUM confidence)
- [RapidFuzz GitHub](https://github.com/rapidfuzz) -- Fuzzy string matching library
- [2025 Fuzzy Matching Benchmarks](https://similarity-api.com/blog/speed-benchmarks) -- Performance comparison showing rapidfuzz 40% faster than python-Levenshtein

### Tertiary (LOW confidence)
- LiteLLM smart prompt design -- based on general VLM prompting patterns, not verified against specific OCSD LiteLLM proxy configuration

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project except rapidfuzz (well-established)
- Architecture: HIGH -- patterns derived directly from existing codebase (mapper/runner.py, locate.py)
- Pitfalls: HIGH -- identified from reading actual code dependencies and threading patterns
- Overlay UX: MEDIUM -- new overlay elements (badge, highlight) follow established patterns but are new widgets
- LiteLLM fallback: MEDIUM -- prompt design is speculative, needs validation against actual VLM behavior

**Research date:** 2026-03-19
**Valid until:** 2026-04-19 (stable domain, all internal codebase)
