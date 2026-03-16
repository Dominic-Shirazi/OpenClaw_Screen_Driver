# Feature Research

**Domain:** Screen automation recording tool with cinematic overlay, Rich TUI, and MCP-compatible API
**Researched:** 2026-03-16
**Confidence:** MEDIUM-HIGH (primary sources: official docs, community best practices, competitor analysis)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Record/stop hotkey (F2 toggle) | Every recorder has a global hotkey; users won't hunt for a button | LOW | Already partially wired; needs reliable state machine |
| Visual recording state indicator | Users need clear "am I recording?" confirmation — red = recording, green = ready is universal convention | LOW | Shimmer border glow addresses this; must be unmissable |
| Abort/cancel without saving | ESC-to-abort is muscle memory for every power user; no abort = scary tool | LOW | Maps to ESC key handler |
| Click capture with bbox confirmation | Users expect to see what got captured before committing; Power Automate, UiPath both do preview | MEDIUM | Hide overlay → screenshot → AI bbox → show confirmation |
| Saved routine that replays | The whole point: record once, replay forever | HIGH | Already exists in runner.py; needs routine.json format |
| Step-by-step replay with status | Users need to see which step is running; pure silent replay feels broken | MEDIUM | Progress bar or HUD during run flow |
| Routine list / browser | Users accumulate routines; need to see and select them | LOW | Rich TUI table or CLI `ocsd list` |
| Failure notification with reason | When replay fails, users need to know why (element not found, timeout, etc.) | MEDIUM | Already in failure cascade; needs user-visible output |
| CLI entry point | Power users and agents expect `ocsd <command>` not a GUI | LOW | `pyproject.toml` script entry point |
| Routine portability (file format) | Users want to share routines; human-readable JSON is expected | MEDIUM | routine.json spec — fully planned in PROJECT.md |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Cinematic overlay animations (shimmer border, scan line, bbox trace) | Transforms a utilitarian tool into something people *show off*; makes AI activity legible and satisfying rather than invisible | HIGH | QPropertyAnimation + custom paintEvent; WA_TranslucentBackground required; must hide before screenshots |
| Frosted-glass HUD tag dialog with typewriter fill | VLM auto-fill feels magical vs. a plain dialog box; differentiates AI-assisted recording from dumb record-and-play | HIGH | QPainter semi-transparent background + animation loop for typewriter effect |
| Donut cloud probability visualizer | Makes confidence visible — users see *why* the AI picked a click target; builds trust in the system | HIGH | Custom QPainter draw; normalized probability → ring radius mapping |
| 5-stage location cascade (pixel → CLIP → OCR → VLM → position) | Other tools fail silently on UI changes; the cascade degrades gracefully and logs exactly which stage found the element | HIGH | Already built; the key is surfacing cascade stage in replay UI |
| Human-like mouse execution (Bézier curves + overshoot) | Bypasses bot detection; feels "real" to anti-cheat and monitoring systems; competitors use linear moves | MEDIUM | Already built in executor.py; needs to be surfaced as a visible setting |
| VLM auto-fill with manual fallback | AI accelerates tagging but doesn't block recording when VLM is down; graceful degradation wins trust | MEDIUM | LiteLLM proxy call with timeout + UI fallback to manual text entry |
| `prompt_user` action type | Enables human-in-the-loop workflows; no other macro tool pauses mid-routine to ask an agent a question | HIGH | Suspends replay, sends screenshot + question, waits for response via API or TUI |
| MCP tool API (FastAPI + MCP-compatible names) | AI agents can use OCSD routines as first-class tools without custom integration; directly targets the OpenClaw agent ecosystem | MEDIUM | `fastapi-mcp` library auto-converts FastAPI routes to MCP tools; tool names must follow SEP-986 (a-z, _, /) convention |
| Rich TUI loading screen with bundled demos | Onboarding experience: shows community routines scrolling while models load; first impression matters for developer tools | MEDIUM | Rich `Live` display with `Panel` + `Table`; bundled demo routines in package |
| Drag-highlight capture with AI bbox tightening | More precise than single-click capture; users can manually indicate a region and let AI refine it | MEDIUM | PyAutoGUI drag capture + VLM bbox refinement call |
| Dry run per step (Enter → countdown → execute → validate) | Lets users verify each step before committing; Power Automate has no per-step dry run; reduces "I just broke something" anxiety | MEDIUM | Step-by-step mode in runner.py with Enter prompt and visual countdown |
| Routine audit trail (human-readable JSON) | Every step, bbox, VLM metadata, and embedding path is inspectable in a text editor; builds trust through transparency | LOW | Already mandated in PROJECT.md; complexity is in schema design |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full app page mapping (scan everything) | "Map the whole UI so I don't have to record step by step" | Requires detection pipeline maturity that doesn't exist yet; produces noisy graphs that confuse replay; V2 scope | Record individual routines; compose in V2 when composition is built |
| GUI launcher (Raycast-style window) | Looks polished; "normal" apps have GUIs | Doubles surface area to maintain; agent/API callers skip it entirely; TUI + overlay covers all real use cases for V1 | Rich TUI + overlay toolbar covers the V1 need |
| Voice input during recording | Hands-free tagging sounds useful | faster-whisper adds heavy dependency; real-time transcription introduces latency in the capture loop; adds GPU memory pressure during recording | Stub interface only in V1; activate in V2 when pipeline is stable |
| Remote routine hub (cloud sync) | "Share routines with my team" | Network layer not worth building until community exists; security model for shared automation code is non-trivial | Local-only hub in V1 with bundled demo routines; remote hub in V2 |
| Real-time mouse move recording | "Record exactly what I did" | Produces thousands of coordinate events; replays are fragile on different resolutions; linear moves bypass bot detection | Record click targets and let executor generate human-like moves via Bézier curves |
| Routine composition/chaining | "Run routine A then B" | Requires composition intelligence to handle state handoff between routines; V2 scope | One routine = one sequence in V1; chain via API caller in V2 |
| Auto-trust scoring of community routines | "Show me which routines are safe" | Needs community scale to generate meaningful signal; false confidence is worse than no score | Manual audit + scanner.py for malicious pattern detection; automated scoring in V2 |
| Absolute coordinate fallback as primary | "Just click at X,Y if element not found" | Breaks on different monitor sizes, DPI, and window positions | Position fallback is stage 5 of cascade only — last resort, never primary |
| Progress bar for screenshot hide/show | "Show me when the overlay is hidden" | The whole point of hiding is a clean capture; a progress bar would appear in the screenshot | Simply hide all elements synchronously before capture; no indicator needed |

---

## Feature Dependencies

```
[Cinematic Overlay: state machine (ready/recording/locating)]
    └──requires──> [F2 hotkey handler]
    └──requires──> [Overlay hides before screenshot]
                       └──requires──> [QWidget show/hide synchronization]

[Record flow: click capture → bbox → tag dialog]
    └──requires──> [Cinematic Overlay: scan animation]
    └──requires──> [VLM auto-fill]
                       └──enhances──> [Frosted-glass HUD tag dialog]
    └──requires──> [Manual tag fallback]  ← graceful degrade if VLM fails

[Dry run per step]
    └──requires──> [Routine file format (routine.json)]
    └──requires──> [5-stage location cascade]
    └──enhances──> [Donut cloud probability visualizer]

[FastAPI REST endpoints]
    └──requires──> [Routine file format]
    └──requires──> [Replay engine (runner.py)]
    └──enhances──> [MCP tool API]
                       └──requires──> [FastAPI endpoints with operation IDs]

[Rich TUI launcher]
    └──requires──> [CLI entry point (ocsd command)]
    └──requires──> [Routine list/browser]
    └──enhances──> [Rich TUI loading screen with bundled demos]

[`prompt_user` action type]
    └──requires──> [FastAPI /respond endpoint]
    └──requires──> [Replay engine pause/resume]
    └──conflicts──> [Unattended background replay]  ← prompt_user blocks until answered

[Routine update flow]
    └──requires──> [Routine file format]
    └──requires──> [Step-by-step dry run]

[Fork routine flow]
    └──requires──> [Routine storage layout (~/.ocsd/routines/)]
    └──requires──> [Routine file format]
```

### Dependency Notes

- **Cinematic overlay requires overlay hides before screenshot:** ALL overlay elements must be fully hidden before mss captures; any visible widget appears in the screenshot and corrupts bbox fitting. This is non-negotiable and must be wired at the base overlay class level, not per-feature.
- **MCP tool API requires FastAPI endpoints with operation IDs:** `fastapi-mcp` and `fastapi_mcp` both use OpenAPI operation IDs to name MCP tools. Every FastAPI route must have an explicit `operation_id` following SEP-986 conventions (`a-z`, `_`, `/`).
- **`prompt_user` conflicts with unattended replay:** A routine with `prompt_user` steps cannot run fully automated without an agent listener on the API. This should be surfaced clearly in routine metadata (e.g., `requires_human: true` flag in routine.json).
- **Dry run enhances donut cloud visualizer:** The donut cloud showing click probability only makes sense when the user is watching step-by-step; it is inappropriate during fast batch replay.
- **VLM auto-fill enhances tag dialog:** Tag dialog must work without VLM (manual input is the fallback path). VLM fills the fields and the user confirms or edits — never blocks on VLM response.

---

## MVP Definition

### Launch With (V1)

Minimum viable product — what's needed to validate "show it once, it never forgets."

- [ ] F2 toggle: ready → recording → ready state machine with shimmer border glow (green/red)
- [ ] Click capture flow: overlay hide → screenshot → AI bbox → scan animation → tag dialog (VLM fill + manual fallback)
- [ ] Routine save to `~/.ocsd/routines/{name}/routine.json` with snippets and embeddings
- [ ] Replay with 5-stage cascade, failure cascade (retry → widen → prompt/log), step-visible progress
- [ ] Routine browser: `ocsd list` (Rich table) and `ocsd run <name>`
- [ ] Rich TUI: main menu (record / run / list / update / fork) + loading screen
- [ ] FastAPI endpoints: `/routines`, `/routines/{id}/run`, `/status`, `/abort`
- [ ] MCP tool names mapped from FastAPI operation IDs (fastapi-mcp library)
- [ ] CLI: `ocsd record/run/list/inspect/update/fork` with `--json` output flag
- [ ] ESC abort and Ctrl+Q save shortcuts

### Add After Validation (V1.x)

Features to add once core record-run loop is validated.

- [ ] Donut cloud probability visualizer — adds insight but not needed for basic trust
- [ ] `prompt_user` action type — needs the REST /respond endpoint to be battle-tested first
- [ ] Drag-highlight capture — single-click capture validates the core loop first
- [ ] Dry run per step — useful once users have routines to debug
- [ ] Routine update flow (step-through and edit) — users need routines first before they need to edit them

### Future Consideration (V2+)

Features explicitly deferred per PROJECT.md scope.

- [ ] Full app page mapping — needs detection pipeline maturity
- [ ] Remote routine hub — needs community scale
- [ ] Routine composition/chaining — needs composition intelligence
- [ ] Voice input (faster-whisper) — heavy dependency, stub only in V1
- [ ] GUI launcher (Raycast-style) — terminal + overlay covers V1
- [ ] Auto-trust scoring — needs community data
- [ ] macOS support — Windows + Ubuntu only for V1

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| F2 hotkey state machine + shimmer border | HIGH | LOW | P1 |
| Overlay hides before screenshot | HIGH | LOW | P1 (blocker for all capture) |
| Click capture → bbox → tag dialog | HIGH | HIGH | P1 |
| VLM auto-fill with manual fallback | HIGH | MEDIUM | P1 |
| Routine save (routine.json format) | HIGH | MEDIUM | P1 |
| Replay with location cascade | HIGH | LOW (already built) | P1 |
| Rich TUI main menu | MEDIUM | LOW | P1 |
| CLI entry point + commands | HIGH | LOW | P1 |
| FastAPI endpoints | HIGH | MEDIUM | P1 |
| MCP tool mapping | HIGH | LOW (fastapi-mcp) | P1 |
| Cinematic scan animation (perimeter trace) | MEDIUM | HIGH | P2 |
| Frosted-glass HUD tag dialog | MEDIUM | HIGH | P2 |
| Rich TUI loading screen with demos | MEDIUM | LOW | P2 |
| Step-visible replay progress | MEDIUM | LOW | P2 |
| Failure cascade user notification | HIGH | LOW | P2 |
| Dry run per step | MEDIUM | MEDIUM | P2 |
| Donut cloud probability visualizer | LOW | HIGH | P3 |
| `prompt_user` action type | MEDIUM | HIGH | P3 |
| Drag-highlight capture | LOW | MEDIUM | P3 |
| Routine update/fork flow | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

---

## Competitor Feature Analysis

| Feature | Power Automate Desktop | Pulover's Macro Creator | UiPath | OCSD Approach |
|---------|----------------------|------------------------|--------|---------------|
| Recording state indicator | Toolbar button state change | Hotkey toggles tray icon | Floating toolbar | Full-screen shimmer border glow (unmissable) |
| Element capture | Click + UI Automation inspector | Click coord or image | Click + selector builder | Click + AI bbox fitting (5-stage cascade) |
| Element labeling | Auto-labelled by UI tree | None (position only) | Auto from UI tree | VLM-generated label with manual override |
| Replay failure handling | Stop on error | Stop on error | Error handler flow | Cascade: retry → widen → prompt/log |
| Human-like mouse moves | No (linear) | No (linear) | No (linear) | Bézier curves + overshoot (built in) |
| Dry run / step preview | Full replay only | Full replay only | Step highlight in designer | Per-step dry run with countdown |
| AI-assisted recording | Copilot (cloud) | None | AI suggestions (cloud) | Local VLM via LiteLLM (offline-capable) |
| API / agent integration | Power Platform REST (cloud) | None | Orchestrator REST (enterprise) | FastAPI + MCP tools (local, agent-native) |
| Routine format | Proprietary PAD format | AHK script | Proprietary XAML | Human-readable JSON (auditable) |
| TUI / terminal interface | None | None | None | Rich TUI launcher + `ocsd` CLI |

---

## Domain-Specific Notes

### Overlay Design: Why Cinematic Matters

Most automation tools make their recording state invisible or relegate it to a tray icon. Users consistently miss whether recording is active. The shimmer border approach (full-screen colored glow) makes state unmissable without covering screen content. This is a UX gap all major competitors have. Confidence: MEDIUM (derived from OBS forum discussions and UX heuristics; no direct competitor analysis of PyQt overlay design).

### MCP Tool Naming: Follow SEP-986

The MCP spec is actively evolving (latest spec 2025-11-25). Tool names must use `a-z`, `A-Z`, `0-9`, `_`, `-`, `.`, `/`. LiteLLM namespaces tools by MCP server name. Recommended pattern: `ocsd/routines/run`, `ocsd/routines/list`, etc. Using `fastapi-mcp` (`tadata-org/fastapi_mcp`) or FastMCP (`gofastmcp.com`) for auto-conversion is acceptable for bootstrapping, but curated tool descriptions matter more than auto-conversion quality — LLMs perform significantly better with well-described tools. Confidence: HIGH (official MCP spec + SEP-986 issue + FastMCP docs).

### Rich TUI: Separation of Concerns

The proven pattern (2025) is: Rich for static/semi-static display, Textual for fully interactive apps. Since OCSD's TUI is primarily a launcher menu (not a live editor), Rich + `questionary` for prompts is the correct choice — lower dependency weight than Textual, sufficient for the use case. Use a centralized `theme.py` to avoid hardcoded color strings throughout TUI code. Confidence: HIGH (multiple official sources + PyPI analytics).

### Frosted Glass: Platform Limitation

True frosted glass (blur-behind) requires compositor support. On Windows, `WA_TranslucentBackground` + `DWM` compositing gives real translucency. On Ubuntu (X11), the result depends on the compositor (Compton/Picom). The fallback — semi-transparent dark fill with QPainter — looks identical to 90% of users. Build for the fallback first; add DWM blur as an enhancement for Windows. Confidence: MEDIUM (Qt forum discussions + Qt wiki; no first-party Qt frosted glass guide).

---

## Sources

- [Pulover's Macro Creator](https://www.macrocreator.com/) — competitor feature baseline
- [Power Automate Desktop — Recording Flows (Microsoft Learn)](https://learn.microsoft.com/en-us/power-automate/desktop-flows/recording-flow) — enterprise macro recorder UI patterns
- [PyQt6 QPropertyAnimation Tutorial (pythonguis.com)](https://www.pythonguis.com/tutorials/pyqt6-animated-widgets/) — animation implementation
- [QGraphicsOpacityEffect — Qt for Python](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QGraphicsOpacityEffect.html) — overlay transparency
- [QWidget Semi-transparent Background Color — Qt Wiki](https://wiki.qt.io/QWidget_Semi-transparent_Background_Color) — frosted glass approach
- [Rich Library — GitHub](https://github.com/Textualize/rich) — TUI implementation
- [8 TUI Patterns (Medium)](https://medium.com/@Nexumo_/8-tui-patterns-to-turn-python-scripts-into-apps-ce6f964d3b6f) — TUI design patterns
- [FastAPI-MCP — GitHub (tadata-org)](https://github.com/tadata-org/fastapi_mcp) — REST to MCP mapping
- [FastMCP FastAPI Integration](https://gofastmcp.com/integrations/fastapi) — MCP auto-conversion patterns
- [MCP Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — MCP tool naming rules
- [SEP-986: Tool Name Format (MCP GitHub)](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/986) — tool naming conventions
- [Best Practices: Mapping REST to MCP Tools (Zuplo)](https://zuplo.com/learning-center/mapping-rest-apis-to-mcp-tools) — MCP tool design
- [Macro Recorder Limitations (macrorecorder.org)](https://macrorecorder.org/2024/10/26/what-are-the-limitations-of-a-macro-recorder/) — anti-feature analysis
- [OBS Full-Screen Recording Indicator (AHK)](https://obsproject.com/forum/threads/full-screen-recording-indicator-with-autohotkey-ahk-script.183719/) — overlay state indicator precedent

---
*Feature research for: Screen automation recording tool with cinematic overlay, Rich TUI, and MCP API*
*Researched: 2026-03-16*
