# Project Research Summary

**Project:** OCSD — OpenClaw Screen Driver
**Domain:** Screen automation tool with cinematic PyQt6 overlay, Rich TUI, and MCP-compatible FastAPI API
**Researched:** 2026-03-16
**Confidence:** HIGH (all research files HIGH or MEDIUM-HIGH with primary source verification)

## Executive Summary

OCSD is a local-first screen automation tool that records UI interactions once and replays them using a 5-stage AI element-location cascade. The milestone under research adds three new capability layers on top of the existing core (CLIP, FAISS, OmniParser, LiteLLM, NetworkX, mss): a cinematic PyQt6 overlay with animated recording feedback, a Rich TUI pre-launch launcher, and a FastAPI server exposed as MCP tools for agent consumption. The recommended approach is a strict layered build order — overlay window foundation first, then animation items, then the full record flow wired to the overlay, then TUI and API independently in parallel. The entire system depends on a single architectural guarantee: the overlay must be fully hidden and the compositor must have flushed before any screenshot is taken; this contract must be established at the base class level before any other work proceeds.

The key technical risks are all well-understood and have concrete mitigations. The most dangerous is the hide-before-screenshot timing gap on Windows DWM: `QWidget.hide()` does not block until the compositor repaints, so a mandatory 30-80ms sleep after `processEvents()` is required before every `mss` capture. The second structural risk is the event loop cohabitation problem: Rich `Live` must never run concurrently with `QApplication.exec()` (sequential pre-launch gate solves this), and uvicorn must be started with `install_signal_handlers = False` to avoid signal ownership conflicts with Qt. DPI correctness — multiplying all logical Qt coordinates by `devicePixelRatio()` before saving to routine.json — is a correctness issue that is expensive to fix after routines have been recorded, so it must be addressed in Phase 1.

The recommended stack is almost entirely existing project dependencies. New additions are minimal and purpose-matched: `typer[all]` for the CLI, `fastapi-mcp` for zero-config MCP exposure, `questionary` for arrow-key-navigable TUI prompts, and `PyQt6-Frameless-Window` for the cross-platform frameless transparent overlay base. No new heavy dependencies are introduced. The cinematic overlay uses only Qt's built-in animation framework (QPropertyAnimation, QTimer, QPainter) — no external animation libraries. This milestone is well-scoped for the existing stack.

## Key Findings

### Recommended Stack

The overlay, TUI, and API layers each have a clear library choice with no viable alternatives for the OCSD constraints. PyQt6's built-in animation framework (QPropertyAnimation + QEasingCurve + QTimer-driven paintEvent) is the only Python path to Bézier-driven shimmer, scan lines, and frosted-glass HUDs in a transparent overlay window. Rich + questionary is the correct TUI choice — Textual's full async event loop would conflict with Qt; Rich's `Live`+`Layout`+`Panel` is sufficient for a launcher menu. `fastapi-mcp` (tadata-org) converts existing FastAPI routes to MCP tools in three lines of code, making it the only library purpose-built for OCSD's situation (existing FastAPI app, not building MCP-first).

**Core technologies:**
- **PyQt6 6.10.x**: Cinematic overlay and animation engine — only viable path for transparent composited overlay in Python; already in project
- **PyQt6-Frameless-Window 0.8.0**: Frameless transparent overlay base class — handles `WA_TranslucentBackground` + DWM acrylic blur boilerplate; saves ~100 lines of ctypes
- **Rich 14.3.3**: Pre-launch TUI launcher and loading screen — already in project; `Live` + `Layout` + `Panel` + `Spinner` covers all V1 TUI needs
- **questionary 2.1.1**: Arrow-key-navigable prompts inside Rich TUI — fills the gap Rich's built-in `Prompt` leaves (no arrow-key navigation)
- **typer[all] 0.24.1**: CLI entry point (`ocsd record/run/list/inspect/update/fork`) — wraps click, integrates with Rich for colored help, type-hint-driven subcommands
- **fastapi-mcp 0.4.0**: Zero-config MCP exposure of existing FastAPI endpoints — three lines of code; auto-derives MCP tool names from OpenAPI operation IDs

### Expected Features

**Must have (table stakes):**
- F2 toggle: ready → recording → ready state machine with visible shimmer border (red/green)
- Click capture flow: overlay hide → screenshot → AI bbox → scan animation → tag dialog (VLM fill + manual fallback)
- Routine save to `~/.ocsd/routines/{name}/routine.json` with snippets and embeddings
- Replay with 5-stage location cascade and failure cascade (retry → widen → prompt/log)
- Step-by-step replay with visible progress
- Routine browser: `ocsd list` (Rich table) and `ocsd run <name>`
- Rich TUI main menu (record / run / list / update / fork) + loading screen with bundled demos
- FastAPI endpoints: `/routines`, `/routines/{id}/run`, `/status`, `/abort`
- MCP tool mapping via fastapi-mcp (tool names follow `ocsd/routines/run` pattern per SEP-986)
- CLI entry point: `ocsd record/run/list/inspect/update/fork` with `--json` flag
- ESC abort and Ctrl+Q save shortcuts

**Should have (competitive — add after V1 core is validated):**
- Cinematic scan animation: perimeter trace + sweep line on element detection
- Frosted-glass HUD tag dialog with typewriter VLM auto-fill
- Donut cloud probability visualizer (click target confidence rings)
- Per-step dry run: Enter → countdown → execute → validate
- Routine update flow (step-through and edit existing routines)

**Defer (V2+):**
- Full app page mapping and graph exploration
- Remote routine hub and cloud sync
- Routine composition/chaining
- Voice input (faster-whisper stub only in V1)
- Raycast-style GUI launcher
- macOS support

### Architecture Approach

The system has five distinct layers that must be built in strict dependency order. The Qt application layer (overlay window + animation items) is the critical path — everything else (record flow, TUI, API) attaches to it. The single most important architectural constraint is the hide-flush-capture-show cycle: the overlay must disappear completely before mss touches the framebuffer. Rich TUI is a pre-launch blocking gate — it returns a `(command, kwargs)` tuple and exits before `QApplication()` is created; the two event loops never run concurrently. FastAPI/uvicorn runs as a daemon thread with `install_signal_handlers = False`, sharing state with Qt only via `threading.Lock`-protected objects and Qt cross-thread signals.

**Major components:**
1. **CinematicOverlayView + OverlayController** — transparent fullscreen Qt window, state machine (PASSTHROUGH/RECORD/SCANNING/IDLE), hide/capture/show cycle
2. **Overlay animation items** — BorderGlowItem (QPropertyAnimation shimmer), ScanAnimItems (QTimer 60fps perimeter trace), DonutProbItem (probability rings), HUDTagPanel (frosted glass dialog + typewriter)
3. **RecordController** — orchestrates F2→hide→screenshot→AI bbox→tag→save flow; wires all overlay components together
4. **ExecuteController / mapper layer** — loads routine.json, runs 5-stage location cascade, fires progress events; already substantially built
5. **Rich TUI (tui.py)** — pre-launch blocking menu; no Qt dependency; can be built in parallel with API
6. **FastAPI server (api/server.py)** — REST/MCP endpoints in daemon thread; no imports from recorder/ in V1

### Critical Pitfalls

1. **Screenshot contains overlay artifacts** — `hide()` does not block until DWM repaints. Use: `hide()` → `processEvents()` → `time.sleep(0.03)` on Windows → capture. Never use `QTimer.singleShot(0, ...)` alone. Minimum 80ms empirically safe; test on slowest target machine.

2. **DPI coordinate mismatch on HiDPI displays** — Qt reports logical coordinates; mss captures physical pixels. All bbox coordinates saved to routine.json must be multiplied by `QApplication.primaryScreen().devicePixelRatio()`. This is catastrophically expensive to fix post-recording. Fix it in Phase 1 before any recording logic is wired.

3. **Wayland click-through silently broken** — `WindowTransparentForInput` has no Wayland equivalent. Force X11 backend: `os.environ["QT_QPA_PLATFORM"] = "xcb"` before `QApplication()` when `XDG_SESSION_TYPE == "wayland"`.

4. **Animation `self.update()` inside `paintEvent` creates infinite repaint loop** — CPU hits 100%. Drive all animation ticks from a single `QTimer` at 16ms. Never call `self.update()` inside `paintEvent`. Use `QPropertyAnimation` for property interpolation — it drives updates correctly.

5. **FastAPI/uvicorn signal handler conflict** — `uvicorn.run()` in a thread raises `ValueError: signal only works in main thread`. Use `uvicorn.Server(config)` with `server.install_signal_handlers = lambda: None`. Alternatively spawn as a daemon `multiprocessing.Process`.

## Implications for Roadmap

Based on the combined research, the architecture's build-order implications (ARCHITECTURE.md §Build Order Implications) and the pitfall-to-phase mapping (PITFALLS.md §Pitfall-to-Phase Mapping) converge strongly on a 3-phase structure. The overlay foundation is the critical path and must come first — everything else depends on it. TUI and API are independent of each other and can be parallelized in Phase 3, but both depend on the record/replay flow being stable.

### Phase 1: Cinematic Overlay Foundation

**Rationale:** The overlay window and its hide/capture/show contract are the foundation for all other work. DPI correctness and compositor timing must be validated before any capture logic is wired. PITFALLS.md maps 5 of 7 critical pitfalls to this phase. Building animations bottom-up (border glow first → scan items → donut → HUD) validates each Qt pattern before building on it.

**Delivers:** Transparent fullscreen overlay window with correct DPI handling, click-through passthrough mode, F2 state machine (ready/recording), shimmer border glow, scan animation, donut probability visualizer, HUD tag dialog with VLM typewriter fill, and the full record flow (click → hide → capture → AI bbox → tag → save to routine.json).

**Addresses from FEATURES.md:** F2 toggle with border glow, click capture flow, VLM auto-fill with manual fallback, routine save format, cinematic scan animation, frosted-glass HUD, donut cloud visualizer.

**Avoids from PITFALLS.md:** Screenshot overlay contamination (hide+sleep+capture pattern), DPI mismatch (devicePixelRatio multiplication from day one), Wayland click-through (xcb fallback at startup), animation infinite repaint loop (QTimer-driven tick pattern established first), multi-monitor wrong screen (explicit screen selection).

**Stack:** PyQt6 6.10.x, PyQt6-Frameless-Window 0.8.0, QPropertyAnimation, QTimer, QPainter.

### Phase 2: Record/Replay Polish and Routine Management

**Rationale:** With the overlay foundation proven, wire the record flow fully to the execution layer, validate the 5-stage location cascade in the dry-run loop, and surface failure cascade feedback visibly. This phase makes the "show it once, it never forgets" promise demonstrably true before adding external interfaces.

**Delivers:** Full record-replay loop with dry-run per step, step-visible replay progress, failure cascade user notification, routine browser (`ocsd list`), update and fork flows. The routine.json format is finalized here based on what the record flow actually produces.

**Addresses from FEATURES.md:** Replay with location cascade, step-visible progress, failure notification, routine browser, dry-run per step, routine update/fork flow, `prompt_user` action type.

**Avoids from PITFALLS.md:** Full-resolution 4K screenshot memory exhaustion (log rotation), scene.clear() flash on candidate refresh (in-place item updates), QApplication.processEvents() spin-wait (QThread workers for blocking operations).

**Stack:** Existing mapper/ layer (runner, orchestrator, validator, pathfinder), NetworkX, core.locate cascade.

### Phase 3: TUI, CLI, and API

**Rationale:** TUI and API have no dependency on each other. Both depend on the record/replay flow (Phase 2) being stable so the CLI commands and API endpoints have real behavior to invoke. This phase can be split into two parallel tracks.

**Delivers:** `ocsd` CLI entry point with all subcommands and `--json` output flag, Rich TUI launcher (main menu + loading screen with bundled demos), FastAPI REST endpoints (`/routines`, `/routines/{id}/run`, `/status`, `/abort`), MCP tool mapping via fastapi-mcp with SEP-986-compliant operation IDs.

**Addresses from FEATURES.md:** CLI entry point, Rich TUI main menu + loading screen, FastAPI endpoints, MCP tool API, `prompt_user` /respond endpoint.

**Avoids from PITFALLS.md:** Rich + PyQt6 event loop conflict (Rich is pre-launch only, sequential gate), FastAPI signal handler conflict (`install_signal_handlers = False`), Rich TUI styling degraded in threads (never start Rich in a background thread), FastAPI endpoint security (bind to 127.0.0.1 only).

**Stack:** typer[all] 0.24.1, questionary 2.1.1, Rich 14.3.3, fastapi-mcp 0.4.0, uvicorn daemon thread.

### Phase Ordering Rationale

- **Overlay first** because all 5 critical overlay pitfalls (DPI, compositor timing, Wayland, animation CPU, multi-monitor) compound each other. Fixing DPI after recording routines exist is catastrophic. Establishing the hide-capture-show contract first means every downstream component inherits correct behavior.
- **Record/replay second** because the TUI and API both invoke it. Building CLI commands before the underlying behavior is stable is premature.
- **TUI and API last** because they are the thinnest layers (Rich is a launcher, fastapi-mcp is three lines of code) and have the most well-documented patterns. They are also independent of each other within Phase 3.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (Overlay Foundation):** The `QObject` + `QGraphicsItem` multiple-inheritance pattern for `pyqtProperty` on graphics items is sparsely documented in PyQt6 specifically. MRO ordering (`QObject` must come first) is a known foot-gun. Verify the BorderGlowItem pattern with a minimal prototype before building all animation items on top of it.
- **Phase 1 (HUD Tag Panel):** True frosted glass (DWM acrylic) vs. fake semi-transparent background: the platform-specific DWM path via `DwmSetWindowAttribute` is obscure and may not be worth V1 complexity. Research confirms fake semi-transparent is the recommended V1 approach, but DWM real-blur should be validated early on Windows to know whether the V1 fallback is visually acceptable.

Phases with well-documented patterns (skip research-phase):
- **Phase 3 (TUI + CLI):** Rich + questionary + typer patterns are thoroughly documented with multiple primary sources. No research needed before implementing.
- **Phase 3 (API + MCP):** fastapi-mcp zero-config integration is a three-line addition to existing FastAPI app. Pattern verified via GitHub source and PyPI. No research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All versions PyPI-verified. Compatibility matrix confirmed against Python 3.12.10. Alternatives explicitly evaluated and dismissed with rationale. |
| Features | MEDIUM-HIGH | Table stakes derived from competitor analysis (Power Automate, UiPath, Pulover). Differentiators grounded in OCSD's existing capabilities. Anti-features backed by domain reasoning. Overlay UX confidence MEDIUM (OBS forum heuristics, no direct PyQt overlay UX studies). |
| Architecture | HIGH | Based on existing codebase inspection + Qt/FastAPI official docs. Hide-capture-show pattern confirmed via Qt forum. Uvicorn daemon thread pattern confirmed via uvicorn issue #650 and production community reports. |
| Pitfalls | HIGH | 6 of 7 critical pitfalls verified against official documentation, Qt forum threads, or GitHub issues. DWM compositor timing confirmed empirically (Qt forum). Rich threading confirmed via Rich GitHub issues #1530, #2665. |

**Overall confidence:** HIGH

### Gaps to Address

- **Hide delay exact value:** Research confirms 80ms as empirically safe on Windows, but the correct value may vary across GPU drivers and DWM configurations. Build in a configurable `capture_delay_ms` setting (YAML config) rather than hardcoding, so it can be tuned per deployment.
- **Frosted glass visual quality threshold:** The fake semi-transparent HUD approach is the V1 recommendation, but whether it meets the "cinematic" quality bar is a judgment call that should be validated in Phase 1 before the HUD is fully built out.
- **`prompt_user` action conflict with unattended replay:** Routines containing `prompt_user` steps block until an agent responds on the API. The `requires_human: true` flag in routine.json is specified but the exact behavior (timeout? skip? fail?) under unattended conditions is not yet defined.
- **MCP tool description quality:** `fastapi-mcp` auto-derives MCP tool descriptions from OpenAPI docstrings. LLM performance with MCP tools is significantly better with well-crafted descriptions. Budget time for curating FastAPI route docstrings beyond the minimum.

## Sources

### Primary (HIGH confidence)
- PyPI package registry — PyQt6 6.10.2, fastapi-mcp 0.4.0, PyQt6-Frameless-Window 0.8.0, questionary 2.1.1, typer 0.24.1, rich 14.3.3 (all version-verified)
- Qt official docs: `doc.qt.io/qt-6` — QPropertyAnimation, QGraphicsItem, QEasingCurve, QGraphicsBlurEffect, WA_TranslucentBackground, High DPI, Wayland
- FastAPI GitHub issue #650 + uvicorn Discussion #1103 — daemon thread signal handler workaround confirmed
- Rich GitHub issues #1530, #2665 — threading and isatty limitations confirmed
- MCP Specification 2025-11-25 + SEP-986 — tool naming conventions
- OCSD existing codebase: `recorder/overlay.py`, `overlay_view.py`, `overlay_items.py`, `api/server.py`

### Secondary (MEDIUM confidence)
- Qt Forum threads: frosted glass DWM (110293), hide/show timing (136295), click-through Wayland (154266), opacity Wayland (158586), multi-monitor geometry (139569)
- GitHub: tadata-org/fastapi_mcp — zero-config MCP exposure pattern
- GitHub: zhiyiYo/PyQt-Frameless-Window — Win10/Win11 acrylic blur support
- pythonguis.com: QPropertyAnimation tutorial, paintEvent/update() loop warning
- Power Automate Desktop (Microsoft Learn), UiPath docs — competitor feature baseline
- Rich docs: Live Display — Layout + Panel + Live patterns

### Tertiary (MEDIUM-LOW confidence)
- OBS forum: full-screen recording indicator patterns — used for overlay UX heuristics (not a PyQt source)
- Medium: Seamless desktop widgets with PyQt6 (`@hudbeard`) — WA_TranslucentBackground patterns
- 8 TUI Patterns (Medium) — TUI design pattern validation

---
*Research completed: 2026-03-16*
*Ready for roadmap: yes*
