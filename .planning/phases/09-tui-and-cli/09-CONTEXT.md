# Phase 9: TUI and CLI - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the terminal interfaces for OCSD: a Rich TUI loading screen with feature ticker and interactive menu for first-time/interactive use, plus a full `ocsd` CLI with subcommands for direct invocation. The TUI is a sequential gate that initializes models before any action. CLI subcommands always show TUI first (for terminal launches). API-triggered runs (Phase 10) skip TUI. Does NOT build API endpoints (Phase 10), new action types (Phase 6), or replay engine features (Phase 7).

</domain>

<decisions>
## Implementation Decisions

### TUI loading screen (TUI-01, TUI-02, TUI-03)
- **Feature ticker**: Loading screen shows Rich spinner + progress bars for model loading. Below that, a scrolling feature ticker loaded from a data file (YAML or markdown). Mix of shipped features and "coming soon" items. Easy to update as product evolves
- **Data file for ticker**: Features loaded from a `features.yml` or `features.md` file bundled with the package. Each entry has text + status (shipped/coming_soon). No hardcoded strings
- **Model status**: Progress bars for each model loading (OmniParser, CLIP, VLM connectivity check). Shows which models are ready
- **No demo routines built**: User will create real demo routines when making their first few routines. TUI infrastructure supports loading them from `~/.ocsd/routines/` when they exist

### TUI menu (TUI-04, TUI-05)
- **Arrow-key navigable list**: Vertical list with highlight: Record Routine / Run Routine / Update Routine / Fork Routine / List Routines. Arrow keys + Enter to select
- **V2+ features greyed out**: Items like "Hub Browse", "Voice Record" appear greyed out with "Feature inbound" label. User sees what's coming but can't select
- **Action menu first**: Menu shows actions (Record/Run/Update/Fork). When user picks Run or Update, THEN show a searchable routine list to select from. Two-step: action → target
- **Searchable routine list**: Type to filter routines in real-time. Arrow keys to navigate filtered results. Enter to select. Good for users with many routines

### CLI framework (CLI-01 through CLI-09)
- **Typer**: Use Typer (built on Click) for CLI framework. Type hints for arguments, auto-generated help. Add `typer` to pyproject.toml dependencies
- **Always show TUI first**: Even CLI subcommands like `ocsd run "Name"` show the TUI loading screen first. The loading screen serves as the initialization gate (model loading, VLM connectivity). Only API-triggered runs (Phase 10) skip TUI
- **Both name and path**: `ocsd run "My Routine"` looks up by name in `~/.ocsd/routines/`. `ocsd run path/to/routine.json` runs from any location. Auto-detect by checking if argument is a file path
- **--speed flag**: `ocsd run "Name" --speed 0` for instant, `--speed 1.0` for normal, `--speed 5.0` for slow. Overrides config for this run
- **--json flag**: All commands support `--json` for machine-readable output (CLI-09)
- **--param flag**: `ocsd run "Name" --param search_term="news"` passes variable values without interactive prompting. For agent use
- **Rich error panels**: Red-bordered Rich panel with error title, message, and fix suggestion. e.g., "Routine not found. Run `ocsd list` to see available routines."
- **Hub search stub**: `ocsd hub search` prints "Hub coming soon in V2. Browse routines at [GitHub URL]." No actual search

### Pre-run variable prompts
- **Pre-run collection**: Before replay starts, scan all steps for variable inputs (`input_spec.type == "variable"`). Prompt user for ALL values upfront with hints from the routine creator. Then execute with no interruptions
- **Hints from routine**: Variable prompts include hint text from the routine (e.g., "What would you like to Google?" for a search_term variable). Hints are stored in the step's `input_spec` and editable by the routine creator
- **--param override**: Agent/scripts can pass `--param key=value` to skip interactive prompts entirely. Multiple `--param` flags allowed
- **Inline variable resolution deferred**: Using output from previous steps as input to later steps is V2

### TUI-to-Qt handoff (TUI-06)
- **Terminal minimizes**: When overlay launches, terminal window auto-minimizes. Clean desktop during operation
- **Restore with summary**: After overlay closes, terminal un-minimizes, prints a brief summary (routine saved, replay complete, etc.), then returns to TUI menu or exits depending on launch mode
- **TUI for terminal only**: TUI shows when launched from terminal. API-triggered runs (Phase 10) skip TUI — the API server is already running, models are loaded
- **Sequential gate**: TUI Rich event loop completes fully and is destroyed before QApplication() is created. Two event loops never coexist

### Claude's Discretion
- Exact Typer app structure (single file vs subcommand modules)
- How to implement terminal minimize/restore cross-platform (Windows vs Linux)
- Feature ticker animation speed and visual style
- How to detect whether an argument is a routine name or file path
- Rich table column formatting for `ocsd list` and `ocsd inspect`
- How to implement searchable routine list (Rich Live vs textual vs custom)
- Model loading progress bar implementation (real progress vs estimated)
- How to structure the features.yml/md data file

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### TUI and CLI requirements
- `.planning/REQUIREMENTS.md` — TUI-01 through TUI-06 and CLI-01 through CLI-09 define all requirements
- `.planning/ROADMAP.md` — Phase 9 success criteria (5 criteria that must be TRUE)

### Existing entry point (being replaced)
- `main.py` — Current argparse-based entry point. Phase 9 replaces this with Typer-based `ocsd` CLI

### Routine operations (Phase 8 — all implemented)
- `routine/management.py` — `fork_routine()`, `delete_routine()`, `inspect_routine()` for CLI wiring
- `routine/update_session.py` — `UpdateSession` for update/fork flows
- `routine/discovery.py` — `list_routines()`, `get_routine_dir()` for routine enumeration
- `routine/format.py` — `Routine.load()` for loading routines before replay/update

### Replay engine (Phase 7)
- `routine/runner.py` — `run_routine()`, `preflight_check()`, `RunEvent`, `RunResult` for replay
- `routine/replay_overlay.py` — `ReplayOverlayAdapter` for overlay during replay

### Recording infrastructure
- `recorder/record_session.py` — `RecordSession` for recording new routines
- `recorder/overlay/controller.py` — `OverlayController.show()` for launching overlay

### Core pipeline
- `core/config.py` — `get_config()`, `load_config()` for YAML config
- `core/executor.py` — `_resolve_input_text()` for variable input resolution

### Prior phase context
- `.planning/phases/07-run-flow/07-CONTEXT.md` — Overlay optional via config, pre-flight validation, VLM prompt caller
- `.planning/phases/08-routine-management/08-CONTEXT.md` — Update flow, fork, delete, inspect decisions

### Project constraints
- `.planning/PROJECT.md` — "Terminal + overlay (no GUI launcher)", Rich TUI sequential gate
- `CLAUDE.md` — Type hints, logging, venv isolation, cross-platform awareness

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `routine/management.py`: `fork_routine()`, `delete_routine()`, `inspect_routine()` — CLI commands wire directly to these
- `routine/runner.py:run_routine()`: CLI `run` command calls this with collected params
- `routine/update_session.py:UpdateSession`: CLI `update` command instantiates this
- `routine/discovery.py:list_routines()`: CLI `list` command uses this
- `routine/format.py:Routine.load()`: Load routine for inspect/run/update
- `core/executor.py:_resolve_input_text()`: Variable resolution from `execution_params` dict — pre-run prompts feed into this
- `recorder/record_session.py:RecordSession`: CLI `record` command wraps this

### Established Patterns
- `Routine.load(path)` / `Routine.save(path)` for JSON I/O
- `list_routines()` returns `RoutineInfo` objects with name, path, schema_version
- `run_routine(routine, callback, params)` for replay execution
- `OverlayController.show()` for launching overlay (lazy-imports View)
- PipelineBridge signals for thread-safe communication
- `get_config()` for YAML config access

### Integration Points
- `main.py` — Replace with Typer app entry point (`ocsd/cli.py` or similar)
- `pyproject.toml` — Add `typer` dependency, add `[project.scripts]` entry point for `ocsd`
- `core/config.py` — Already has `load_config()` called at startup
- `recorder/overlay/controller.py:OverlayController.show()` — Called after TUI exits for record/run/update

</code_context>

<specifics>
## Specific Ideas

- Feature ticker loaded from a data file — easy to update as product evolves. Mix of shipped features ("Record once, replay forever") and coming soon ("Voice-recorded routines", "Routine Hub marketplace")
- TUI always shows first even for CLI subcommands — serves as the model initialization gate. This is important because most future use is agent-driven CLI use that requires warmed models
- Terminal minimizes during overlay operation, restores with summary after. Clean desktop experience
- Pre-run variable prompts with creator-defined hints — "What would you like to Google?" rather than just "Enter search_term:". Makes routines self-documenting
- `--param` flag for agents to pass all variables without interactive prompting. Essential for automation
- Rich error panels with fix suggestions — helps users recover from common mistakes

</specifics>

<deferred>
## Deferred Ideas

- Inline variable resolution (using output from step N as input to step N+1) — V2, requires runtime data flow between steps
- Remote Routine Hub with network sync — V2, V1 hub is local stub
- Voice input during recording (faster-whisper) — V2, stubbed only
- GUI launcher (Raycast-style) — V2, terminal + overlay is V1
- Bundled demo routines — user will create when making first routines, not pre-built

</deferred>

---

*Phase: 09-tui-and-cli*
*Context gathered: 2026-03-20*
