# Phase 11: Integration Wiring - Context

**Gathered:** 2026-03-21
**Status:** Ready for planning
**Source:** v1.0 Milestone Audit (`.planning/v1.0-MILESTONE-AUDIT.md`)

<domain>
## Phase Boundary

Wire two existing-but-disconnected features into runtime execution paths:
1. ReplayOverlayAdapter → CLI/TUI run commands
2. Security scanner → all routine execution paths (with v1 format support)

This phase writes NO new features — only integration glue.

</domain>

<decisions>
## Implementation Decisions

### RUN-09: Replay Overlay Wiring
- CLI `run_command()` at `cli/app.py:155` must instantiate `QApplication` + `OverlayController` + `ReplayOverlayAdapter` and pass adapter as `callback=` to `run_routine()`
- TUI run dispatch at `cli/tui.py:370` must do the same overlay wiring
- API run path (`api/server.py`) stays headless — no overlay needed there (acceptable per audit)
- Purple shimmer, StatusBadge (current step), TargetHighlight, and CameraFlash must all activate during replay

### SEC-01: Scanner Integration
- Add `scan_routine()` wrapper in `hub/scanner.py` that adapts v1 routine format (steps array) → scanner input format
- Call scanner before execution in `run_routine()` preflight or equivalent entry point
- Block execution if scanner returns threats
- Scanner must work for all entry points: CLI, TUI, and API

### Claude's Discretion
- How to manage QApplication lifecycle in CLI/TUI (singleton, per-run, etc.)
- Whether scanner call goes in `run_routine()` itself or a wrapper
- Error UX when scanner blocks a routine

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Audit & Gaps
- `.planning/v1.0-MILESTONE-AUDIT.md` — Exact gap descriptions, locations, and fix suggestions

### Overlay System
- `recorder/overlay/replay_overlay_adapter.py` — The adapter class that needs wiring
- `recorder/overlay/overlay_controller.py` — Controller that manages overlay lifecycle
- `recorder/overlay_items.py` — Visual items (shimmer, badge, highlight, flash)

### Run Paths
- `cli/app.py` — CLI run command (line ~155)
- `cli/tui.py` — TUI run dispatch (line ~370)
- `recorder/runner.py` — `run_routine()` function
- `api/server.py` — API run path (headless, no changes needed)

### Scanner
- `hub/scanner.py` — `scan_skill()` function that needs v1 format wrapper

</canonical_refs>

<specifics>
## Specific Ideas

From audit fix suggestions:
- CLI and TUI run paths need `QApplication` + `OverlayController` + `ReplayOverlayAdapter`, passed as `callback=` to `run_routine()`
- Add `scan_routine()` wrapper that adapts v1 routine format → scanner input, call from `run_routine()` preflight

</specifics>

<deferred>
## Deferred Ideas

None — this phase closes the final v1.0 gaps.

</deferred>

---

*Phase: 11-integration-wiring*
*Context gathered: 2026-03-21 from v1.0 Milestone Audit*
