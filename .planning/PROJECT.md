# OCSD (OpenClaw Screen Driver)

## What This Is

The recording studio for the OpenClaw ecosystem. OCSD watches you do something once, then does it for you forever — with human-like mouse movement, typing, and error correction. No code. No API. No MCP server. Record a routine in 30 seconds, share it with anyone. "Muscle Memory — Show it once, it never forgets."

AI helps you **record**. Replay is pure automation — no tokens, no API keys, no data leaving your machine.

## Core Value

A non-technical person can record a multi-step screen routine in under 2 minutes and replay it reliably on any resolution — the "air fryer" for macros.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. Inferred from existing codebase. -->

- ✓ Screenshot capture with multi-monitor support — `core/capture.py` (mss)
- ✓ 5-stage element location cascade (pixel match → OmniParser+CLIP → OCR → VLM → position fallback) — `core/locate.py`
- ✓ CLIP embeddings + FAISS vector search for element matching — `core/embeddings.py`
- ✓ OmniParser (YOLOv8) + Florence-2 detection pipeline — `core/detection.py`, `core/omniparser.py`
- ✓ Tesseract OCR with cross-platform binary discovery — `core/ocr.py`
- ✓ VLM analysis via LiteLLM (OpenAI-compatible) — `core/vision.py`
- ✓ Human-like mouse/keyboard execution (Bézier curves, typo simulation) — `core/executor.py`
- ✓ NetworkX graph model for action sequences — `mapper/graph.py`
- ✓ JSON graph serialization/export — `mapper/export.py`
- ✓ Replay engine with validation and recovery — `mapper/runner.py`, `mapper/validator.py`
- ✓ YAML config with env-var overrides and deep merge — `core/config.py`
- ✓ Hub security scanner (malicious pattern detection) — `hub/scanner.py`
- ✓ Basic PyQt6 overlay window with hotkey support — `recorder/overlay_view.py`
- ✓ DPI awareness on Windows — `main.py`

### Active

<!-- V1 scope. Building toward these. -->

- [ ] Cinematic overlay: shimmer border glow (green=ready, red=recording), smooth state transitions
- [ ] Cinematic overlay: element scan animation (perimeter trace, scan line sweep, bbox animation)
- [ ] Cinematic overlay: donut cloud probability visualizer for click targeting
- [ ] Cinematic overlay: frosted-glass HUD tag dialog with typewriter fill
- [ ] Cinematic overlay: floating draggable toolbar (context-sensitive actions)
- [ ] Cinematic overlay: all elements hide before screenshots (clean capture)
- [ ] Record flow: F2 toggle (ready↔recording), Ctrl+Q save, ESC abort
- [ ] Record flow: click capture → hide overlay → screenshot → AI bbox fitting
- [ ] Record flow: drag-highlight capture → AI tightens bbox
- [ ] Record flow: smart crop (+30% padding) → VLM analysis → tag dialog
- [ ] Record flow: VLM auto-fill with manual fallback (graceful degradation if VLM fails)
- [ ] Record flow: dry run per step (Enter → countdown → execute → validate)
- [ ] Record flow: save routine file (JSON + snippets/ + embeddings/)
- [ ] Run flow: routine selection → per-step element location via cascade
- [ ] Run flow: human-like execution with configurable delay/typo settings
- [ ] Run flow: failure cascade (auto-retry → widen search → prompt user/agent → log)
- [ ] Routine file format: human-readable JSON with steps, bbox, region_hint, snippet/embedding paths, VLM metadata
- [ ] Routine storage layout: `~/.ocsd/routines/{name}/routine.json` + `snippets/` + `embeddings/`
- [ ] NetworkX graph evolves to serialize as routine.json format
- [ ] Action types: click, double_click, right_click, click_drag, type, read, snip_and_search, select_all_extract, scroll, wait, loop, prompt_user
- [ ] `prompt_user` action: pause routine, send screenshot + question to human or agent, wait for response
- [ ] Update routine flow: step-through existing routine, edit/delete/insert steps
- [ ] Fork routine flow: copy routine under new name, modify as needed
- [ ] Rich TUI: routine browser, model loading status, record/run/update/fork menu
- [ ] Rich TUI: loading screen with community routines scrolling (bundled demos in V1)
- [ ] CLI: `ocsd record/run/list/inspect/update/fork/hub` commands with `--json` flag
- [ ] FastAPI endpoints: `/routines`, `/routines/{id}/run`, status, respond, screenshot, abort
- [ ] MCP-compatible tool names mapped 1:1 from REST endpoints

### Out of Scope

<!-- Explicit V2+ boundaries. -->

- Full app page mapping (scan everything) — V2, needs detection pipeline maturity
- UI GPS / automated graph traversal — V3, needs full app maps from V2
- Automated trust scoring — V2, needs community scale
- Routine composition/chaining — V2, one routine = one sequence in V1
- Fork auto-skip-to-divergence — V2, requires composition intelligence
- GUI launcher (Raycast-style) — V2, terminal + overlay is V1
- Voice input (faster-whisper) — V2, stubbed only
- Remote Routine Hub — V2, V1 hub is local-only with bundled demos
- Android/mobile — future, not V1
- macOS support — future, Windows + Ubuntu only for V1

## Context

**Brownfield project.** ~70% of the core detection + location + execution pipeline is built and working. The new work is primarily:
1. **Cinematic overlay** — the biggest chunk: animations, scan effects, HUD panels (built fresh, existing overlay code is reference only)
2. **Record flow** — end-to-end F2→click→bbox→tag→dry-run→save wiring
3. **Routine file format** — evolve NetworkX graph serialization to the new routine.json spec
4. **TUI + CLI** — Rich terminal launcher, `ocsd` command interface
5. **API endpoints** — new routine-centric REST/MCP endpoints

**VLM setup:** LiteLLM proxy running locally with fallback chains (vision/quick/planning/coding model groups). Sometimes flaky — recording must gracefully degrade to manual entry when VLM fails.

**Platform:** Windows-first development, Ubuntu second. No macOS for V1.

**Target users:** AI enthusiasts/agent builders (OpenClaw ecosystem) AND power users automating repetitive tasks. The routine executor can be a human or an AI agent — it's agnostic.

## Constraints

- **Tech stack**: PyQt6 overlay, Rich TUI, FastAPI API, NetworkX graphs — decided, not negotiable
- **Platform**: Windows + Ubuntu. macOS deferred.
- **VLM dependency**: LiteLLM proxy must be running for AI-assisted recording; manual fallback required when it's down
- **No cloud dependency at runtime**: Replay is 100% local. AI only needed at record time.
- **Overlay screenshot cleanliness**: ALL overlay elements MUST hide before any screenshot capture. Non-negotiable.
- **Human-like execution**: Mouse moves via Bézier curves with overshoot. Typing has variance and typo simulation. This is core identity, not optional.
- **Routine auditability**: JSON must be human-readable. Every step inspectable. Trust through transparency.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Overlay built fresh (reference only from existing code) | Existing overlay lacks cinematic animations; faster to build correctly than retrofit | — Pending |
| NetworkX stays, routine.json is its serialization | Graph model is proven; new format is a view, not a replacement | — Pending |
| VLM with manual fallback | LiteLLM sometimes flaky; recording must work even if VLM is down | — Pending |
| Terminal + overlay (no GUI launcher) | Smallest surface area; agent/API users skip TUI entirely | — Pending |
| Windows + Ubuntu only | Developer's available platforms; macOS deferred | — Pending |
| Local-only hub in V1 | Network layer not worth building until community exists | — Pending |

---
*Last updated: 2026-03-16 after initialization*
