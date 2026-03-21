# Phase 11: Integration Wiring - Research

**Researched:** 2026-03-21
**Domain:** Integration glue -- wiring existing overlay and scanner code into runtime execution paths
**Confidence:** HIGH

## Summary

This phase closes two integration gaps identified in the v1.0 milestone audit. Both features (ReplayOverlayAdapter and hub scanner) are already implemented and tested in isolation -- they just need wiring into the actual execution paths.

For RUN-09, the `ReplayOverlayAdapter` class at `routine/replay_overlay.py` is complete and correct. It bridges `RunEvent` callbacks from the runner's background thread to `OverlayController` methods on the Qt main thread via `pyqtSignal`. The CLI and TUI run paths currently call `run_routine()` without any `callback=` argument, and without creating a `QApplication` or `OverlayController`. The existing recording flow at `recorder/record_flow.py:cmd_record()` provides the exact pattern for QApplication lifecycle management that should be replicated.

For SEC-01, `hub/scanner.py::scan_skill()` exists and works, but expects a legacy `skill_data` dict with `nodes`/`edges` keys. The v1 routine format uses `steps` (a flat array of step dicts). A thin `scan_routine()` adapter function is needed to convert between formats, plus a call site in the runner preflight or a wrapper around `run_routine()`.

**Primary recommendation:** Wire overlay into CLI/TUI using the same `QApplication.instance() or QApplication(sys.argv)` singleton pattern from `record_flow.py`. Add `scan_routine()` as a format adapter in `hub/scanner.py` and call it from `run_routine()` preflight.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- CLI `run_command()` at `cli/app.py:155` must instantiate `QApplication` + `OverlayController` + `ReplayOverlayAdapter` and pass adapter as `callback=` to `run_routine()`
- TUI run dispatch at `cli/tui.py:370` must do the same overlay wiring
- API run path (`api/server.py`) stays headless -- no overlay needed there (acceptable per audit)
- Purple shimmer, StatusBadge (current step), TargetHighlight, and CameraFlash must all activate during replay
- Add `scan_routine()` wrapper in `hub/scanner.py` that adapts v1 routine format (steps array) to scanner input format
- Call scanner before execution in `run_routine()` preflight or equivalent entry point
- Block execution if scanner returns threats
- Scanner must work for all entry points: CLI, TUI, and API

### Claude's Discretion
- How to manage QApplication lifecycle in CLI/TUI (singleton, per-run, etc.)
- Whether scanner call goes in `run_routine()` itself or a wrapper
- Error UX when scanner blocks a routine

### Deferred Ideas (OUT OF SCOPE)
None -- this phase closes the final v1.0 gaps.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| RUN-09 | Step-by-step replay status visible to user (which step is executing) | ReplayOverlayAdapter is complete at `routine/replay_overlay.py`. Needs wiring into CLI `run_command()` and TUI `run_tui()` run dispatch. Pattern established by `recorder/record_flow.py::cmd_record()` for QApplication lifecycle. |
| SEC-01 | Hub scanner flags suspicious URLs, prompt injection, data exfiltration, keylogger patterns | `hub/scanner.py::scan_skill()` exists with full pattern detection. Needs `scan_routine()` adapter for v1 format + call site in runner preflight. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyQt6 | (existing) | QApplication, OverlayController, overlay view | Already the project's UI framework per CLAUDE.md |
| routine.runner | (existing) | `run_routine()` with `callback=` parameter | Already supports RunCallback protocol |
| routine.replay_overlay | (existing) | `ReplayOverlayAdapter` -- thread-safe bridge | Already implemented, just needs instantiation |
| hub.scanner | (existing) | `scan_skill()` pattern detection | Already implemented, needs format adapter |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| threading | stdlib | Background thread for `run_routine()` in TUI/CLI with overlay | Runner must run on background thread while Qt event loop runs on main thread |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| QApplication singleton | Fresh QApplication per run | Singleton (`QApplication.instance() or QApplication(sys.argv)`) is already the project pattern in `record_flow.py` |
| Scanner in `run_routine()` preflight | Scanner as separate middleware | Preflight is cleaner -- scanner runs before any step execution, consistent with existing preflight_check pattern |

## Architecture Patterns

### Recommended Project Structure
No new files needed. Changes to existing files only:
```
cli/app.py              # Wire overlay into run_command()
cli/tui.py              # Wire overlay into TUI run dispatch
hub/scanner.py           # Add scan_routine() adapter
routine/runner.py        # Add scanner call to preflight
```

### Pattern 1: QApplication + Overlay + Background Runner (for CLI/TUI)
**What:** Create QApplication, OverlayController, and ReplayOverlayAdapter on main thread. Run `run_routine()` on a background thread with `callback=adapter`. Call `app.exec()` to pump the Qt event loop until the run completes.
**When to use:** CLI `run_command()` and TUI `run` dispatch.
**Example:**
```python
# Source: recorder/record_flow.py (existing pattern)
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)

controller = OverlayController()
adapter = ReplayOverlayAdapter(controller)

controller.show()

# Run routine on background thread
def _run_thread():
    try:
        result = run_routine(routine_dir=path, callback=adapter)
    finally:
        app.quit()  # Exit Qt event loop when done

thread = threading.Thread(target=_run_thread, daemon=True)
thread.start()

app.exec()  # Blocks until run completes
```

### Pattern 2: Scanner Format Adapter
**What:** Convert v1 routine format (steps array) to the legacy scanner input format (nodes/edges dict).
**When to use:** `scan_routine()` wrapper function.
**Example:**
```python
# Source: analysis of hub/scanner.py scan_skill() expectations
def scan_routine(routine_data: dict) -> ScanResult:
    """Adapt v1 routine format to scanner input and scan.

    Args:
        routine_data: Dict with 'steps' key (v1 format).

    Returns:
        ScanResult from the scanner.
    """
    steps = routine_data.get("steps", [])
    # Map steps to nodes (scanner checks element_type, label, ocr_text, etc.)
    nodes = []
    edges = []
    for step in steps:
        nodes.append(step)  # Steps already have element_type, label, etc.
        # Steps with action_type/action map to edges for typed text checks
        if step.get("action") in ("type", "type_text"):
            edges.append({
                "action_type": step.get("action", ""),
                "action_payload": step.get("text_to_type", ""),
            })
    return scan_skill({"nodes": nodes, "edges": edges})
```

### Pattern 3: Scanner in Runner Preflight
**What:** Call `scan_routine()` in the `run_routine()` function after loading the routine but before execution.
**When to use:** All execution paths (CLI, TUI, API) automatically get scanning.
**Example:**
```python
# In run_routine(), after loading routine but before step loop:
from hub.scanner import scan_routine

scan_result = scan_routine(routine.to_dict())
if not scan_result.is_safe:
    # Block execution, emit PREFLIGHT_FAILED
    _emit(callback, RunEvent.PREFLIGHT_FAILED, {
        "errors": scan_result.warnings
    })
    # Return failure RunResult
```

### Anti-Patterns to Avoid
- **Creating QApplication in a thread:** QApplication MUST be on the main thread. Always create it before spawning the runner thread.
- **Blocking the Qt event loop:** Never call `run_routine()` directly on the main thread when overlay is active -- it blocks the event loop and freezes the overlay.
- **Multiple QApplication instances:** Use the `QApplication.instance() or QApplication(sys.argv)` pattern consistently. Never create a second QApplication.
- **Scanner per entry point:** Don't add scanner calls in CLI, TUI, and API separately. Put it in `run_routine()` so all paths get it automatically.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread-safe Qt updates | Custom thread synchronization | `ReplayOverlayAdapter` with `pyqtSignal` | Already built and tested, handles thread boundary correctly |
| QApplication lifecycle | New lifecycle manager | `QApplication.instance() or QApplication(sys.argv)` + `setQuitOnLastWindowClosed(False)` | Proven pattern from record_flow.py |
| Scanner format conversion | Complex mapping layer | Thin adapter that maps steps to nodes/edges | Scanner already handles all pattern detection |

**Key insight:** Everything needed already exists. This phase is purely wiring -- connecting existing pieces. No new features or libraries.

## Common Pitfalls

### Pitfall 1: QApplication Already Exists
**What goes wrong:** If TUI or CLI code path already created a QApplication instance elsewhere, creating another one crashes.
**Why it happens:** Qt enforces a single QApplication per process.
**How to avoid:** Always use `QApplication.instance() or QApplication(sys.argv)` -- the `or` clause only creates if none exists.
**Warning signs:** "QApplication was already created" error at runtime.

### Pitfall 2: Runner Thread vs Qt Main Thread
**What goes wrong:** Overlay freezes or crashes because `run_routine()` blocks the Qt event loop.
**Why it happens:** Qt event loop must run on the main thread. If `run_routine()` (which does time.sleep, network calls, etc.) runs on the main thread, no Qt events process.
**How to avoid:** Always run `run_routine()` on a daemon thread. The `ReplayOverlayAdapter` uses `pyqtSignal` to safely cross the thread boundary back to the main thread.
**Warning signs:** Frozen overlay, no purple shimmer animation, UI not updating.

### Pitfall 3: app.quit() Timing
**What goes wrong:** Qt event loop doesn't exit after run completes, or exits before run finishes.
**Why it happens:** `app.exec()` blocks indefinitely unless `app.quit()` is called.
**How to avoid:** Call `app.quit()` in the runner thread's finally block, after `run_routine()` returns (success or failure). Use `setQuitOnLastWindowClosed(False)` so window close doesn't quit prematurely.
**Warning signs:** Process hangs after run completes, or overlay disappears mid-run.

### Pitfall 4: Scanner Blocking User Routines
**What goes wrong:** Legitimate routines get blocked because they type URLs or reference system paths.
**Why it happens:** Scanner risk threshold (0.5) can be exceeded by multiple low-risk warnings.
**How to avoid:** Log warnings clearly so users understand what triggered the block. Consider adding a `--force` / `--skip-scan` flag or config option for power users. For now, the requirement says "block execution if scanner returns threats" -- but the UX should be clear about why.
**Warning signs:** Users confused about why their routine won't run.

### Pitfall 5: TUI Event Loop Interaction
**What goes wrong:** TUI Rich Live display conflicts with Qt event loop.
**Why it happens:** Rich Live uses terminal control sequences; Qt event loop takes over the process.
**How to avoid:** Follow the established sequential gate pattern: TUI exits completely, then Qt runs, then TUI resumes. The TUI `run` handler already does `minimize_terminal()` before the run -- the overlay wiring fits inside this existing try/finally block.
**Warning signs:** Garbled terminal output, Rich and Qt fighting for stdin/stdout.

## Code Examples

### CLI run_command() Overlay Wiring
```python
# In cli/app.py run_command(), replace the direct run_routine() call:
import sys
import threading

from PyQt6.QtWidgets import QApplication

from recorder.overlay.controller import OverlayController
from routine.replay_overlay import ReplayOverlayAdapter
from routine.runner import run_routine

app = QApplication.instance() or QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)

controller = OverlayController()
adapter = ReplayOverlayAdapter(controller)
controller.show()

run_result_holder = [None]
run_exc_holder = [None]

def _run():
    try:
        run_result_holder[0] = run_routine(
            routine_dir=run_path, callback=adapter
        )
    except Exception as exc:
        run_exc_holder[0] = exc
    finally:
        app.quit()

thread = threading.Thread(target=_run, daemon=True)
thread.start()
app.exec()

if run_exc_holder[0] is not None:
    raise run_exc_holder[0]
result = run_result_holder[0]
```

### scan_routine() Adapter
```python
# In hub/scanner.py:
def scan_routine(routine_data: dict) -> ScanResult:
    """Scan a v1 routine for malicious patterns.

    Adapts the v1 routine format (steps array) to the legacy
    scanner input format (nodes/edges dict) and delegates to
    scan_skill().

    Args:
        routine_data: Routine dict with 'steps' key.

    Returns:
        ScanResult with safety verdict and warnings.
    """
    steps = routine_data.get("steps", [])
    nodes = list(steps)  # Steps have element_type, label, etc.
    edges = []
    for step in steps:
        action = step.get("action", "")
        if action in ("type", "type_text", "keystroke"):
            edges.append({
                "action_type": action,
                "action_payload": step.get("text_to_type", ""),
            })
    return scan_skill({"nodes": nodes, "edges": edges})
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct `run_routine()` call (headless) | `run_routine(callback=adapter)` with overlay | Phase 11 | Users see step-by-step progress during replay |
| No runtime security scanning | `scan_routine()` in preflight | Phase 11 | Every routine scanned before execution |

## Open Questions

1. **Scanner `--force` flag**
   - What we know: Scanner may block legitimate routines that type URLs
   - What's unclear: Whether to add a `--skip-scan` CLI flag in this phase
   - Recommendation: Add a `--no-scan` flag to `run_command()` and a config option. Low effort, high user-friendliness. Claude's discretion per CONTEXT.md.

2. **TUI run already uses minimize_terminal**
   - What we know: TUI run handler calls `minimize_terminal()` before run
   - What's unclear: Whether overlay is visible when terminal is minimized
   - Recommendation: The overlay is a separate transparent fullscreen window -- it should be visible regardless of terminal state. No issue expected, but worth verifying.

3. **API scanner integration path**
   - What we know: SEC-01 requires scanner for "all entry points: CLI, TUI, and API"
   - What's unclear: Whether scanner goes in `run_routine()` (covers all) or needs separate wiring
   - Recommendation: Put scanner in `run_routine()` preflight. This automatically covers CLI, TUI, and API paths with zero additional wiring. If scanner is too slow, it can be made async in the API path later.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `python -m pytest tests/ -x --timeout=30` |
| Full suite command | `python -m pytest tests/ --timeout=60` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RUN-09 | CLI run wires overlay adapter as callback | unit (mock) | `python -m pytest tests/test_cli.py -x -k "run"` | Partial (test_cli.py exists but no overlay test) |
| RUN-09 | TUI run wires overlay adapter as callback | unit (mock) | `python -m pytest tests/test_cli.py -x -k "tui"` | Partial |
| RUN-09 | ReplayOverlayAdapter receives events during run | unit | `python -m pytest tests/test_replay_overlay.py -x` | Yes |
| SEC-01 | scan_routine() adapts v1 format correctly | unit | `python -m pytest tests/test_scanner.py -x` | Wave 0 (needs test_scanner.py) |
| SEC-01 | run_routine() blocks unsafe routines | unit | `python -m pytest tests/test_routine_runner.py -x -k "scan"` | Wave 0 |
| SEC-01 | Scanner works for CLI, TUI, API paths | integration | `python -m pytest tests/test_routine_runner.py -x -k "preflight"` | Partial |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_replay_overlay.py tests/test_routine_runner.py tests/test_cli.py -x --timeout=30`
- **Per wave merge:** `python -m pytest tests/ --timeout=60`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_scanner.py` -- test scan_routine() format adapter with v1 routine data
- [ ] Add scan-preflight test cases to `tests/test_routine_runner.py` -- verify unsafe routines are blocked
- [ ] Add overlay-wiring test cases to `tests/test_cli.py` -- verify callback is passed to run_routine

## Sources

### Primary (HIGH confidence)
- `routine/replay_overlay.py` -- Complete ReplayOverlayAdapter implementation
- `hub/scanner.py` -- Complete scan_skill() implementation
- `routine/runner.py` -- run_routine() with callback= protocol
- `cli/app.py` -- Current CLI run_command() without overlay
- `cli/tui.py` -- Current TUI run dispatch without overlay
- `recorder/record_flow.py` -- Established QApplication lifecycle pattern
- `recorder/overlay/controller.py` -- OverlayController with full replay API
- `recorder/overlay/state.py` -- OverlayState.REPLAYING state definition
- `.planning/v1.0-MILESTONE-AUDIT.md` -- Gap analysis with exact locations

### Secondary (MEDIUM confidence)
- None needed -- all integration points are in the codebase

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies
- Architecture: HIGH -- existing patterns (record_flow.py) provide exact template
- Pitfalls: HIGH -- well-understood Qt threading model, documented in project decisions

**Research date:** 2026-03-21
**Valid until:** Indefinite -- pure integration of existing stable code
