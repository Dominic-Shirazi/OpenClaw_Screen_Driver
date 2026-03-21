# Requirements: OCSD V1

**Defined:** 2026-03-16
**Core Value:** A non-technical person can record a multi-step screen routine in under 2 minutes and replay it reliably on any resolution.

## v1 Requirements

### Overlay Foundation

- [x] **OVLY-01**: Overlay renders as transparent fullscreen window with click-through passthrough when not interacting
- [x] **OVLY-02**: Overlay hides ALL visual elements before any screenshot capture (80ms+ DWM flush)
- [x] **OVLY-03**: Overlay handles DPI scaling correctly — all coordinates use physical pixels matching mss output
- [x] **OVLY-04**: Overlay state machine: ready (green) ↔ recording (red) ↔ paused (green) with smooth transitions
- [x] **OVLY-05**: Overlay works on Windows (primary) and Ubuntu (X11/XCB fallback for Wayland)

### Overlay Animations

- [x] **ANIM-01**: Border shimmer glow — faded moving pulse along screen edges, green=ready, red=recording
- [x] **ANIM-02**: Element scan animation — glowing perimeter trace around selection, scan line sweep top→bottom then left→right
- [x] **ANIM-03**: Bbox animation — smooth morph from rough selection to AI-fitted bounding box
- [x] **ANIM-04**: Donut cloud — probability density visualizer showing where simulated clicks will land
- [x] **ANIM-05**: All animations run at 60fps via QTimer (16ms interval), never trigger repaint loops
- [x] **ANIM-06**: All overlay elements use fluid transitions — slides, fades, scales, eases — nothing "appears"

### Overlay HUD

- [x] **HUD-01**: Tag dialog — dark frosted-glass panel slides up from element with thin fonts and glow accents
- [x] **HUD-02**: Tag dialog typewriter fill — all VLM fields animate simultaneously on first capture
- [x] **HUD-03**: Tag dialog pre-fills without animation when editing existing routine steps
- [x] **HUD-04**: Tag dialog includes action dropdown with conditional fields per action type
- [x] **HUD-05**: Floating toolbar — persistent, draggable, context-sensitive (recording vs dry run vs tag dialog)
- [x] **HUD-06**: Floating toolbar hides during screenshot captures (same as all overlay elements)

### Record Flow

- [x] **REC-01**: User names routine before recording starts
- [x] **REC-02**: System auto-minimizes all windows to desktop as reproducible baseline
- [x] **REC-03**: F2 toggles recording on/off (ready↔recording), Ctrl+Q saves and finishes, ESC aborts
- [x] **REC-04**: Click capture — overlay hides → clean screenshot → AI bbox fitting → overlay returns with scan animation
- [x] **REC-05**: Drag-highlight capture — user drags region → AI tightens bbox to actual element inside highlight
- [x] **REC-06**: Smart crop — final bbox + 30% pixel padding → VLM analysis → tag dialog
- [x] **REC-07**: VLM auto-fill with manual fallback — if VLM fails or times out, user can manually enter type/label/caption
- [x] **REC-08**: Dry run per step — Enter → 3-2-1 countdown → execute action → validate result
- [x] **REC-09**: Dry run does NOT block/lock the mouse — Enter + countdown is sufficient safety
- [x] **REC-10**: After dry run, loop back for next element (still in recording mode) until Ctrl+Q
- [x] **REC-11**: Save routine file on Ctrl+Q — routine.json + snippets/ + embeddings/

### Run Flow

- [x] **RUN-01**: User selects routine from TUI or triggers via CLI/API
- [x] **RUN-02**: Per-step element location using 5-stage cascade (pixel → CLIP → OCR → VLM → position)
- [x] **RUN-03**: Human-like mouse execution via Bézier curves with overshoot-and-correct
- [x] **RUN-04**: Human-like typing with per-letter delays, variance, occasional typo + correction
- [x] **RUN-05**: Configurable execution speed — human_delay multiplier (0=instant, 1.0=normal, 5.0=slow)
- [x] **RUN-06**: Post-action validation via pixel-diff + optional VLM confirmation
- [x] **RUN-07**: Failure cascade — auto-retry (2x) → widen search to full screen → prompt user/agent → log
- [x] **RUN-08**: Never silently fail or blind-click when unsure — always surface uncertainty
- [ ] **RUN-09**: Step-by-step replay status visible to user (which step is executing)

### Action Types

- [x] **ACT-01**: `click` — left click with donut distribution targeting
- [x] **ACT-02**: `double_click` — double left click
- [x] **ACT-03**: `right_click` — right click (context menu)
- [x] **ACT-04**: `click_drag` — click source, drag to target (second bbox captured during recording)
- [x] **ACT-05**: `type` — keyboard input with human-like timing and typo simulation
- [x] **ACT-06**: `read` — OCR/VLM extract text from element, return to caller
- [x] **ACT-07**: `snip_and_search` — crop region → VLM analysis → return structured result
- [x] **ACT-08**: `select_all_extract` — Ctrl+A → feed to local AI → summarize/extract
- [x] **ACT-09**: `scroll` — scroll within element or page
- [x] **ACT-10**: `wait` — wait for screen change / element appear / timer / custom condition
- [x] **ACT-11**: `loop` — repeat previous N steps until condition met (element appears / text matches / N iterations / prompt_user)
- [x] **ACT-12**: `prompt_user` — pause routine, send screenshot + question via API, wait for human or agent response

### Routine Format

- [x] **FMT-01**: Routine stored as human-readable JSON (routine.json) with steps, bbox, region_hint, snippet/embedding paths
- [x] **FMT-02**: Each step includes VLM metadata (element_type, label, caption, confidence)
- [x] **FMT-03**: Snippets stored as +30% padded PNGs in snippets/ directory
- [x] **FMT-04**: CLIP embeddings stored as .npy files in embeddings/ directory
- [x] **FMT-05**: NetworkX graph serializes to/from routine.json format (graph stays under the hood)
- [x] **FMT-06**: Storage layout: `~/.ocsd/routines/{name}/routine.json` + `snippets/` + `embeddings/`
- [x] **FMT-07**: Routine file includes metadata: name, version, created, author, description, tags

### Routine Management

- [x] **MGMT-01**: Update routine — step-through existing, Enter to keep / E to edit / D to delete / I to insert
- [x] **MGMT-02**: Fork routine — copy under new name, modify as needed (manual step-through in V1)
- [x] **MGMT-03**: Delete routine
- [x] **MGMT-04**: Inspect routine — print human-readable step summary

### TUI

- [x] **TUI-01**: Rich TUI as home base — routine browser with searchable list
- [x] **TUI-02**: TUI shows model loading status
- [x] **TUI-03**: TUI loading screen with bundled demo routines scrolling (marketing during load)
- [x] **TUI-04**: TUI menu: Record Routine | Run Routine | Update Routine | Fork Routine
- [x] **TUI-05**: Greyed-out V2+ features with "Feature inbound" labels
- [x] **TUI-06**: TUI runs before Qt event loop (sequential gate, not concurrent)

### CLI

- [x] **CLI-01**: `ocsd` launches TUI (no arguments)
- [x] **CLI-02**: `ocsd record "Name"` — record new routine (skip TUI, straight to overlay)
- [x] **CLI-03**: `ocsd run "Name"` or `ocsd run path/to/routine.json` — run routine
- [x] **CLI-04**: `ocsd list` (human table) and `ocsd list --json` (machine-readable)
- [x] **CLI-05**: `ocsd inspect "Name"` — print routine steps
- [x] **CLI-06**: `ocsd update "Name"` — enter update flow
- [x] **CLI-07**: `ocsd fork "Name" "NewName"` — fork routine
- [x] **CLI-08**: `ocsd hub search "query"` — search local routine hub
- [x] **CLI-09**: All commands support `--json` flag for structured output

### API

- [x] **API-01**: `GET /routines` — list all routines (filterable by tags, name)
- [x] **API-02**: `GET /routines/{id}` — get routine metadata + steps
- [x] **API-03**: `POST /routines/{id}/run` — start execution, return run_id
- [x] **API-04**: `GET /runs/{run_id}/status` — execution status (running/waiting/paused/completed/failed)
- [x] **API-05**: `POST /runs/{run_id}/respond` — respond to prompt_user step
- [x] **API-06**: `GET /runs/{run_id}/screenshot` — current screen state
- [x] **API-07**: `POST /runs/{run_id}/abort` — cancel running routine
- [x] **API-08**: MCP-compatible tool names mapped 1:1 from REST endpoints (via fastapi-mcp)
- [x] **API-09**: FastAPI server runs as daemon process/thread with `install_signal_handlers=False`

### Security

- [ ] **SEC-01**: Hub scanner flags suspicious URLs, prompt injection, data exfiltration, keylogger patterns
- [x] **SEC-02**: Routine files are auditable — every step human-readable in JSON
- [x] **SEC-03**: No cloud dependency at runtime — replay is 100% local

## v2 Requirements

### Hub & Community

- **HUB-01**: Remote Routine Hub with network sync
- **HUB-02**: Automated trust scoring based on community signals
- **HUB-03**: Verified creator badges

### Advanced Flows

- **ADV-01**: Routine composition/chaining (run A then B)
- **ADV-02**: Fork auto-skip-to-divergence (detect shared prefix, skip to first different step)
- **ADV-03**: Full app page mapping (scan entire UI)
- **ADV-04**: UI GPS / automated graph traversal

### Platform & Input

- **PLT-01**: macOS support
- **PLT-02**: Android support
- **PLT-03**: Voice input during recording (faster-whisper)
- **PLT-04**: GUI launcher (Raycast-style)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time mouse move recording | Produces fragile replays; Bézier generation from click targets is more resilient |
| Absolute coordinate fallback as primary | Breaks on different resolutions; position is stage 5 cascade only |
| Cloud sync of routines | Network layer not worth building until community exists |
| OAuth / API authentication | V1 is local-only; auth adds complexity without users |
| Progress indicator during screenshot hide | Would appear in the screenshot — defeats the purpose |
| Real-time chat/video in routines | Out of domain — OCSD is screen automation, not communication |

## Traceability

<!-- Updated during roadmap creation — 2026-03-16 -->

| Requirement | Phase | Status |
|-------------|-------|--------|
| OVLY-01 | Phase 1 | Complete |
| OVLY-02 | Phase 1 | Complete |
| OVLY-03 | Phase 1 | Complete |
| OVLY-04 | Phase 1 | Complete |
| OVLY-05 | Phase 1 | Complete |
| ANIM-01 | Phase 2 | Complete |
| ANIM-02 | Phase 2 | Complete |
| ANIM-03 | Phase 2 | Complete |
| ANIM-04 | Phase 2 | Complete |
| ANIM-05 | Phase 2 | Complete |
| ANIM-06 | Phase 2 | Complete |
| HUD-01 | Phase 3 | Complete |
| HUD-02 | Phase 3 | Complete |
| HUD-03 | Phase 3 | Complete |
| HUD-04 | Phase 3 | Complete |
| HUD-05 | Phase 3 | Complete |
| HUD-06 | Phase 3 | Complete |
| REC-01 | Phase 4 | Complete |
| REC-02 | Phase 4 | Complete |
| REC-03 | Phase 4 | Complete |
| REC-04 | Phase 4 | Complete |
| REC-05 | Phase 4 | Complete |
| REC-06 | Phase 4 | Complete |
| REC-07 | Phase 4 | Complete |
| REC-08 | Phase 4 | Complete |
| REC-09 | Phase 4 | Complete |
| REC-10 | Phase 4 | Complete |
| REC-11 | Phase 4 | Complete |
| FMT-01 | Phase 5 | Complete |
| FMT-02 | Phase 5 | Complete |
| FMT-03 | Phase 5 | Complete |
| FMT-04 | Phase 5 | Complete |
| FMT-05 | Phase 5 | Complete |
| FMT-06 | Phase 5 | Complete |
| FMT-07 | Phase 5 | Complete |
| ACT-01 | Phase 6 | Complete |
| ACT-02 | Phase 6 | Complete |
| ACT-03 | Phase 6 | Complete |
| ACT-04 | Phase 6 | Complete |
| ACT-05 | Phase 6 | Complete |
| ACT-06 | Phase 6 | Complete |
| ACT-07 | Phase 6 | Complete |
| ACT-08 | Phase 6 | Complete |
| ACT-09 | Phase 6 | Complete |
| ACT-10 | Phase 6 | Complete |
| ACT-11 | Phase 6 | Complete |
| ACT-12 | Phase 6 | Complete |
| RUN-01 | Phase 7 | Complete |
| RUN-02 | Phase 7 | Complete |
| RUN-03 | Phase 7 | Complete |
| RUN-04 | Phase 7 | Complete |
| RUN-05 | Phase 7 | Complete |
| RUN-06 | Phase 7 | Complete |
| RUN-07 | Phase 7 | Complete |
| RUN-08 | Phase 7 | Complete |
| RUN-09 | Phase 11 | Pending |
| MGMT-01 | Phase 8 | Complete |
| MGMT-02 | Phase 8 | Complete |
| MGMT-03 | Phase 8 | Complete |
| MGMT-04 | Phase 8 | Complete |
| TUI-01 | Phase 9 | Complete |
| TUI-02 | Phase 9 | Complete |
| TUI-03 | Phase 9 | Complete |
| TUI-04 | Phase 9 | Complete |
| TUI-05 | Phase 9 | Complete |
| TUI-06 | Phase 9 | Complete |
| CLI-01 | Phase 9 | Complete |
| CLI-02 | Phase 9 | Complete |
| CLI-03 | Phase 9 | Complete |
| CLI-04 | Phase 9 | Complete |
| CLI-05 | Phase 9 | Complete |
| CLI-06 | Phase 9 | Complete |
| CLI-07 | Phase 9 | Complete |
| CLI-08 | Phase 9 | Complete |
| CLI-09 | Phase 9 | Complete |
| API-01 | Phase 10 | Complete |
| API-02 | Phase 10 | Complete |
| API-03 | Phase 10 | Complete |
| API-04 | Phase 10 | Complete |
| API-05 | Phase 10 | Complete |
| API-06 | Phase 10 | Complete |
| API-07 | Phase 10 | Complete |
| API-08 | Phase 10 | Complete |
| API-09 | Phase 10 | Complete |
| SEC-01 | Phase 11 | Pending |
| SEC-02 | Phase 5 | Complete |
| SEC-03 | Phase 10 | Complete |

**Coverage:**
- v1 requirements: 87 total
- Mapped to phases: 87
- Unmapped: 0

---
*Requirements defined: 2026-03-16*
*Last updated: 2026-03-16 — traceability populated by roadmapper*
