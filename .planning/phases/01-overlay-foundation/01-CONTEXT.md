# Phase 1: Overlay Foundation - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a reliable, DPI-correct, compositor-aware transparent fullscreen overlay shell that other phases build on. Delivers: click-through passthrough, F2 state machine, clean-capture cycle (overlay fully hidden before screenshots), and cross-platform foundation (Windows verified, Linux stubs). No animations (Phase 2), no HUD panels (Phase 3), no record flow wiring (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Rebuild strategy
- Delete existing overlay code (overlay.py, overlay_view.py, overlay_items.py) and start fresh
- Existing code is broken and monolithic — not worth retrofitting
- New overlay uses **layer-based composition**: each visual concern is its own QGraphicsItem subclass added/removed from the scene independently
- The view/shell is a thin container that manages the scene and delegates to layers

### File layout
- One Python file per layer — maximum isolation, easy to find, easy to test individually
- Phase 1 layers: `border_layer.py`, `click_catcher_layer.py`, `mode_indicator_layer.py`, `bbox_layer.py`
- Plus `overlay_shell.py` (thin QGraphicsView container) and `overlay_controller.py` (mode/state management)

### Phase 1 layers
- **BorderLayer** — colored border around screen edges (green=idle, red=recording). Static color, no animation yet
- **ClickCatcherLayer** — invisible full-screen rect for mouse hit-testing in record mode
- **ModeIndicatorLayer** — text label showing current mode and hotkeys
- **BboxOverlayLayer** — renders detected element bounding boxes with resize handles. Needed for capture cycle testing

### Clean-capture cycle
- Shell coordinates hide-all: single `hide_for_capture()` call hides the entire window at once
- Layers don't need to know about capture — shell handles it
- Capture delay is config-driven with safe default: `capture_delay_ms` in config.yaml, default 100ms
- Verification: manual visual check test (triggers capture cycle, saves screenshot for human review)

### F2 state machine
- Two states: idle (green) and recording (red). "Ready" and "paused" are the same green idle state — no visual distinction
- F2 is the primary toggle hotkey, Ctrl+R is a fallback (laptop Fn key accessibility)
- Ctrl+Q saves and finishes, ESC aborts
- Transitions are instant color swap in Phase 1 — Phase 2 adds shimmer animation

### Cross-platform scope
- Windows: fully built and verified in Phase 1
- Linux/Ubuntu: platform guards and X11/XCB stubs that launch without crashing. No verification on Linux yet
- Wayland auto-detection (force XCB backend) included in stubs

### Claude's Discretion
- DPI strategy (physical pixels vs Qt logical with conversion layer) — pick what's most reliable for mss coordinate matching
- Exact file location within recorder/ package
- OverlayController API surface design
- Layer base class / protocol design

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Overlay requirements
- `.planning/REQUIREMENTS.md` — OVLY-01 through OVLY-05 define the 5 overlay foundation requirements
- `.planning/ROADMAP.md` — Phase 1 success criteria (5 criteria that must be TRUE)

### Existing code (reference only — being deleted and rebuilt)
- `recorder/overlay_view.py` — Current PyQt6 overlay window (Win32 flags, DPI setup, click/drag patterns)
- `recorder/overlay.py` — Current OverlayController (mode switching, click-through logic)
- `recorder/overlay_items.py` — Current bbox/handle rendering

### Project constraints
- `.planning/PROJECT.md` — Tech stack constraints, overlay screenshot cleanliness is non-negotiable
- `.planning/STATE.md` — Blockers: 80ms DWM flush concern, QObject+QGraphicsItem MRO ordering

### Codebase patterns
- `.planning/codebase/CONVENTIONS.md` — Naming, imports, error handling, logging patterns
- `.planning/codebase/STRUCTURE.md` — Where new code goes, file naming conventions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `core/config.py` (get_config()) — YAML config loader, use for `capture_delay_ms` setting
- `core/capture.py` — mss screenshot capture, will be called after overlay hide
- `core/types.py` — Point, Rect dataclasses for coordinate types
- `recorder/hotkeys.py` — Global hotkey listener (Win32/pynput), reuse for F2/Ctrl+R/Ctrl+Q

### Established Patterns
- Win32 layered window flags (WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW) — proven pattern from existing overlay
- WA_TransparentForMouseEvents for non-Windows click-through — proven fallback
- QTimer.singleShot(0, ...) for post-init setup — established pattern
- `from __future__ import annotations` + type hints on all signatures — mandatory

### Integration Points
- `recorder/record_controller.py` — Will create and manage the overlay (Phase 4 wiring, but controller API must be stable)
- `main.py` — DPI awareness setup (ctypes.windll.shcore.SetProcessDpiAwareness) already done here
- `config.yaml` — New `overlay.capture_delay_ms` config key needed

</code_context>

<specifics>
## Specific Ideas

- "The previous one was so broken and the wrong design. We need it to be more modular so if something breaks it's not everything that breaks or the entire layout/view"
- Layer-based composition was chosen specifically for fault isolation — one broken layer shouldn't crash the overlay
- Each layer in its own file for maximum isolation and independent testability

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-overlay-foundation*
*Context gathered: 2026-03-16*
