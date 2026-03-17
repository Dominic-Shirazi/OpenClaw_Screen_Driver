# Roadmap: OCSD V1

## Overview

OCSD V1 builds the recording studio for the OpenClaw ecosystem on top of an existing core detection and execution pipeline (~70% built). The journey moves from the overlay window foundation up through cinematic animations and HUD panels, then wires the full record flow to produce portable routine files, adds all supported action types, completes the replay engine, enables routine management, and finally delivers the terminal interfaces (TUI, CLI) and the API/MCP layer that lets agents consume everything. Each phase delivers one complete, independently verifiable capability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Overlay Foundation** - Transparent fullscreen overlay with DPI correctness, click-through passthrough, F2 state machine, and guaranteed clean-capture cycle
- [ ] **Phase 2: Overlay Animations** - 60fps border shimmer glow, element scan animation, bbox morph, and donut cloud probability visualizer
- [ ] **Phase 3: Overlay HUD Panels** - Frosted-glass tag dialog with typewriter VLM fill and floating draggable toolbar
- [ ] **Phase 4: Record Flow** - End-to-end F2->click->bbox->VLM->tag->dry-run->save wiring
- [ ] **Phase 5: Routine File Format** - Human-readable routine.json with NetworkX serialization, snippet storage, and embedding paths
- [ ] **Phase 6: Action Types** - All 12 action types implemented and capturable during recording
- [ ] **Phase 7: Run Flow** - 5-stage cascade replay, human-like execution, failure cascade, and step-visible progress
- [ ] **Phase 8: Routine Management** - Update, fork, delete, and inspect operations on saved routines
- [ ] **Phase 9: TUI and CLI** - Rich pre-launch launcher with loading screen and full `ocsd` CLI entry point
- [ ] **Phase 10: API and MCP** - FastAPI REST endpoints, MCP tool mapping via fastapi-mcp, and security scanning

## Phase Details

### Phase 1: Overlay Foundation
**Goal**: The overlay window is a reliable, DPI-correct, compositor-aware fullscreen shell that other components can build on
**Depends on**: Nothing (first phase)
**Requirements**: OVLY-01, OVLY-02, OVLY-03, OVLY-04, OVLY-05
**Success Criteria** (what must be TRUE):
  1. Overlay renders as a transparent fullscreen window; clicks pass through to the desktop when in passthrough mode
  2. All overlay visual elements disappear completely before any mss screenshot, and the compositor has flushed (no overlay artifacts appear in captured images)
  3. All overlay-saved coordinates match mss physical pixel positions on both 1x and 2x DPI displays
  4. F2 toggles the overlay between ready (green shimmer) and recording (red shimmer) states with a smooth visual transition
  5. Overlay launches successfully on Windows and on Ubuntu under X11/XCB (Wayland session forces xcb backend automatically)
**Plans**: 3 plans
Plans:
- [ ] 01-01-PLAN.md -- Core infrastructure: state machine, DPI utilities, platform helpers, capture guard
- [ ] 01-02-PLAN.md -- Overlay view shell, 4 layer items, and controller
- [ ] 01-03-PLAN.md -- Integration tests and visual verification

### Phase 2: Overlay Animations
**Goal**: Cinematic animation items run at 60fps without CPU spike and give users clear visual feedback during element capture
**Depends on**: Phase 1
**Requirements**: ANIM-01, ANIM-02, ANIM-03, ANIM-04, ANIM-05, ANIM-06
**Success Criteria** (what must be TRUE):
  1. Border shimmer glow pulses along screen edges at 60fps — green when ready, red when recording — with no visible frame drops
  2. After a click is captured, the overlay shows a glowing perimeter trace around the detected element followed by a scan line sweep
  3. The rough selection bounding box visibly morphs to the AI-fitted tight bbox with a smooth eased animation
  4. The donut cloud renders concentric probability rings centered on the click target
  5. No animation drives `self.update()` from within `paintEvent` — CPU stays under 10% during idle animation
**Plans**: 3 plans
Plans:
- [ ] 02-01-PLAN.md -- Animation clock infrastructure and border shimmer glow layer
- [ ] 02-02-PLAN.md -- Element scan animation and bbox morph capability
- [ ] 02-03-PLAN.md -- Donut cloud visualizer and full animation integration into view/controller

### Phase 3: Overlay HUD Panels
**Goal**: Users can interact with a frosted-glass tag dialog and a floating toolbar during recording without leaving the overlay
**Depends on**: Phase 2
**Requirements**: HUD-01, HUD-02, HUD-03, HUD-04, HUD-05, HUD-06
**Success Criteria** (what must be TRUE):
  1. Tag dialog slides up from the captured element as a dark frosted-glass panel with thin fonts and glow accents
  2. All VLM-filled fields in the tag dialog animate simultaneously with a typewriter effect on first capture
  3. Editing an existing routine step shows pre-filled tag dialog fields with no typewriter animation
  4. Tag dialog action dropdown shows conditional fields that change based on the selected action type
  5. Floating toolbar is draggable, context-sensitive (different buttons for recording vs dry-run vs tag-open states), and hides fully before any screenshot
**Plans**: 3 plans
Plans:
- [ ] 03-01-PLAN.md -- HUD shared constants, card glow helper, and typewriter engine
- [ ] 03-02-PLAN.md -- Tag dialog panel with frosted glass, typewriter fill, and conditional fields
- [ ] 03-03-PLAN.md -- Floating toolbar panel and full HUD integration into view/controller

### Phase 4: Record Flow
**Goal**: A user can press F2, click through a multi-step workflow, dry-run each step, and save a complete routine file
**Depends on**: Phase 3
**Requirements**: REC-01, REC-02, REC-03, REC-04, REC-05, REC-06, REC-07, REC-08, REC-09, REC-10, REC-11
**Success Criteria** (what must be TRUE):
  1. User names the routine before the overlay enters recording mode, and all windows minimize to desktop baseline
  2. Pressing F2 starts recording; each click triggers hide-screenshot-AI-bbox-scan-tag-dialog without any overlay artifact in the captured screenshot
  3. Drag-highlight capture shows the AI tightening the bbox from the rough selection to the actual element boundary
  4. VLM auto-fills the tag dialog fields; if VLM times out or fails, user can manually enter type, label, and caption without the recording session aborting
  5. Enter on a captured step triggers 3-2-1 countdown, executes the action, and validates the result — mouse is never locked during countdown
  6. Ctrl+Q saves the completed routine to `~/.ocsd/routines/{name}/` and ESC cleanly aborts without saving
**Plans**: TBD

### Phase 5: Routine File Format
**Goal**: Routine files are portable, human-readable, and carry all data needed for reliable cross-resolution replay
**Depends on**: Phase 4
**Requirements**: FMT-01, FMT-02, FMT-03, FMT-04, FMT-05, FMT-06, FMT-07
**Success Criteria** (what must be TRUE):
  1. A saved routine.json is readable by a human without tooling — steps, bboxes, region hints, and VLM metadata are plainly structured
  2. Each step in routine.json includes element_type, label, caption, and confidence from VLM analysis
  3. Snippet PNGs (30% padded) exist in `snippets/` and embedding .npy files exist in `embeddings/` alongside routine.json
  4. The NetworkX graph round-trips through routine.json with no data loss — load->serialize->load produces identical graphs
  5. Routine files include top-level metadata: name, version, created, author, description, and tags
**Plans**: TBD

### Phase 6: Action Types
**Goal**: All 12 action types are capturable during recording and executable during replay
**Depends on**: Phase 5
**Requirements**: ACT-01, ACT-02, ACT-03, ACT-04, ACT-05, ACT-06, ACT-07, ACT-08, ACT-09, ACT-10, ACT-11, ACT-12
**Success Criteria** (what must be TRUE):
  1. Click, double-click, right-click, and click-drag are all capturable as distinct action types with correct bbox data
  2. Type action records human-like timing metadata and replays with per-letter delays, variance, and typo simulation
  3. Read, snip-and-search, and select-all-extract actions are capturable as distinct steps and return structured text results during replay
  4. Scroll, wait, loop, and prompt_user actions are capturable with their parameters (direction/amount, condition, iteration count, question text)
  5. A prompt_user step pauses the routine and surfaces its question via the API `/respond` endpoint — the routine does not resume until a response arrives
**Plans**: TBD

### Phase 7: Run Flow
**Goal**: A saved routine executes reliably with human-like motion, visible step progress, and graceful failure handling
**Depends on**: Phase 6
**Requirements**: RUN-01, RUN-02, RUN-03, RUN-04, RUN-05, RUN-06, RUN-07, RUN-08, RUN-09
**Success Criteria** (what must be TRUE):
  1. User selects a routine from the TUI browser or CLI and execution begins with no additional setup
  2. Each step locates its target element using the 5-stage cascade (pixel -> CLIP -> OCR -> VLM -> position) — visible in step status
  3. Mouse moves to targets via Bezier curves with overshoot-and-correct; typing uses per-letter delays with occasional typo and self-correction
  4. Execution speed is configurable via human_delay multiplier — setting 0 runs instantly, 1.0 is natural, 5.0 is slow
  5. When a step fails: auto-retries 2x, widens search to full screen, then surfaces the failure to the user or agent via API — never blind-clicks when uncertain
**Plans**: TBD

### Phase 8: Routine Management
**Goal**: Users can update, fork, delete, and inspect saved routines without re-recording from scratch
**Depends on**: Phase 7
**Requirements**: MGMT-01, MGMT-02, MGMT-03, MGMT-04
**Success Criteria** (what must be TRUE):
  1. Update flow steps through an existing routine with Enter=keep, E=edit, D=delete, I=insert controls that save the modified routine
  2. Fork creates a copy of the routine under a new name and opens it in the update flow for modification
  3. Delete removes the routine directory and its routine.json, snippets, and embeddings
  4. Inspect prints a human-readable step summary — action type, element label, and bbox for each step — without launching the overlay
**Plans**: TBD

### Phase 9: TUI and CLI
**Goal**: Users can drive all of OCSD from a terminal — a Rich loading screen and menu for interactive use, and `ocsd` subcommands for direct invocation
**Depends on**: Phase 8
**Requirements**: TUI-01, TUI-02, TUI-03, TUI-04, TUI-05, TUI-06, CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, CLI-07, CLI-08, CLI-09
**Success Criteria** (what must be TRUE):
  1. `ocsd` with no arguments opens a Rich TUI loading screen (with bundled demo routines scrolling) followed by a menu: Record / Run / Update / Fork — all navigable with arrow keys
  2. TUI shows model loading status and greys out V2+ features with "Feature inbound" labels
  3. `ocsd record "Name"`, `ocsd run "Name"`, `ocsd list`, `ocsd inspect`, `ocsd update`, `ocsd fork`, and `ocsd hub search` all work from the command line
  4. All commands output clean human-formatted tables by default and switch to machine-readable JSON with `--json`
  5. TUI exits completely before `QApplication()` is created — the Rich event loop and Qt event loop never run concurrently
**Plans**: TBD

### Phase 10: API and MCP
**Goal**: Agents and external tools can discover, run, and control routines via a local REST API with MCP-compatible tool names
**Depends on**: Phase 9
**Requirements**: API-01, API-02, API-03, API-04, API-05, API-06, API-07, API-08, API-09, SEC-01, SEC-02, SEC-03
**Success Criteria** (what must be TRUE):
  1. `GET /routines`, `GET /routines/{id}`, `POST /routines/{id}/run`, `GET /runs/{run_id}/status`, `POST /runs/{run_id}/respond`, `GET /runs/{run_id}/screenshot`, and `POST /runs/{run_id}/abort` all return correct responses
  2. All FastAPI endpoints are exposed as MCP tools via fastapi-mcp with SEP-986-compliant tool names
  3. FastAPI server starts as a daemon thread with `install_signal_handlers = False` — Qt signal ownership is never contested
  4. Hub scanner flags suspicious URLs, prompt injection patterns, data exfiltration patterns, and keylogger signatures before any routine is executed
  5. Server only binds to 127.0.0.1 — no remote access in V1; routine replay is 100% local with no cloud calls
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Overlay Foundation | 3/3 | Complete | 2026-03-16 |
| 2. Overlay Animations | 0/3 | Not started | - |
| 3. Overlay HUD Panels | 0/3 | Not started | - |
| 4. Record Flow | 0/TBD | Not started | - |
| 5. Routine File Format | 0/TBD | Not started | - |
| 6. Action Types | 0/TBD | Not started | - |
| 7. Run Flow | 0/TBD | Not started | - |
| 8. Routine Management | 0/TBD | Not started | - |
| 9. TUI and CLI | 0/TBD | Not started | - |
| 10. API and MCP | 0/TBD | Not started | - |
