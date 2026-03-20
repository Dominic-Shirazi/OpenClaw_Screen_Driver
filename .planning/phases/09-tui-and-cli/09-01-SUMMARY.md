---
phase: 09-tui-and-cli
plan: 01
subsystem: cli
tags: [typer, rich, cli, entry-point]

requires:
  - phase: 08-routine-management
    provides: fork_routine, delete_routine, inspect_routine, UpdateSession
  - phase: 05-routine-format
    provides: Routine.load, discovery.list_routines
  - phase: 07-run-flow
    provides: run_routine, RunResult
provides:
  - "ocsd CLI entry point with all subcommands via Typer"
  - "Shared output helpers (json/table dual output, error panels, path resolution)"
  - "Feature ticker data for TUI loading screen"
  - "Cross-platform terminal minimize/restore utility"
affects: [09-tui-and-cli, 10-api-and-mcp]

tech-stack:
  added: [typer, rich]
  patterns: [lazy-import for heavy modules, --json flag on all applicable commands, --speed wiring via config override]

key-files:
  created:
    - cli/__init__.py
    - cli/app.py
    - cli/output.py
    - cli/features.yml
    - cli/_minimize.py
    - tests/test_cli.py
  modified:
    - pyproject.toml

key-decisions:
  - "Typer with lazy imports for all heavy backends (Qt, runner, recorder) to keep CLI startup fast"
  - "--speed flag wires to config human_delay override with try/finally restore pattern"
  - "Read-only commands (list, inspect, fork, delete) skip loading screen since no model warm-up needed"

patterns-established:
  - "Lazy-import pattern: heavy modules imported inside command functions, not at module level"
  - "Dual output: every data command supports --json for machine-readable output"
  - "resolve_routine_path: unified name-or-path resolution used by all commands"

requirements-completed: [CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, CLI-07, CLI-08, CLI-09]

duration: 8min
completed: 2026-03-20
---

# Phase 09 Plan 01: CLI Package Summary

**Typer CLI with 7 subcommands, hub stub, --speed/--json/--param flags, and shared output helpers replacing legacy argparse entry point**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-20T22:30:04Z
- **Completed:** 2026-03-20T22:38:05Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Complete `ocsd` CLI with record, run, list, inspect, update, fork, delete subcommands and hub V2 stub
- --speed flag correctly wires to config human_delay with restore-in-finally pattern
- 19 passing tests covering all commands, output helpers, and edge cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Create CLI package with Typer app, output helpers, features data, and minimize utility** - `8afa499` (feat)
2. **Task 2: CLI command tests using Typer CliRunner** - `2c9df83` (test)

## Files Created/Modified
- `cli/__init__.py` - Package init
- `cli/app.py` - Typer app with all subcommands and hub sub-app
- `cli/output.py` - show_error, resolve_routine_path, output_routines, parse_params
- `cli/features.yml` - 12 feature entries (8 shipped, 4 coming_soon) for TUI loading screen
- `cli/_minimize.py` - Cross-platform terminal minimize/restore via Win32 ctypes
- `pyproject.toml` - typer+rich as core deps, entry point -> cli.app:main, cli* in packages
- `tests/test_cli.py` - 19 tests covering all CLI commands with mocked backends

## Decisions Made
- Used lazy imports throughout cli/app.py to keep CLI startup fast (no Qt, no heavy AI imports at import time)
- --speed wiring uses config dict mutation with try/finally restore rather than a separate config parameter
- Read-only commands skip show_loading_screen since they don't need model warm-up

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CLI package ready, TUI module (cli.tui) expected by plan 09-02
- All subcommands wire to existing backends via lazy imports

---
*Phase: 09-tui-and-cli*
*Completed: 2026-03-20*
