---
phase: 09-tui-and-cli
plan: 02
subsystem: ui
tags: [rich, tui, arrow-menu, routine-browser, cross-platform-keys]

requires:
  - phase: 05-routine-schema
    provides: "RoutineInfo dataclass and list_routines discovery"
provides:
  - "Rich TUI with loading screen, arrow-key menu, and searchable routine browser"
  - "Cross-platform keypress reader (cli/_keys.py)"
  - "Feature ticker from features.yml"
affects: [09-tui-and-cli, 10-api-and-mcp]

tech-stack:
  added: [rich-live, rich-progress, pyyaml-features]
  patterns: [lazy-import-for-heavy-modules, cross-platform-keypress-reader]

key-files:
  created:
    - cli/__init__.py
    - cli/_keys.py
    - cli/tui.py
    - cli/features.yml
  modified:
    - tests/test_tui.py

key-decisions:
  - "Lazy imports for all action handlers to avoid loading Qt/heavy modules at menu time"
  - "MENU_ITEMS as module-level constant list of (label, command, enabled) tuples for testability"
  - "read_key uses msvcrt on Windows, tty/termios on Unix with always-restore terminal settings"

patterns-established:
  - "Arrow-key menu: Rich Live with read_key loop, skip-disabled navigation, wrap-around"
  - "Type-to-filter browser: filter_text accumulation with live table re-render"

requirements-completed: [TUI-01, TUI-02, TUI-03, TUI-04, TUI-05]

duration: 6min
completed: 2026-03-20
---

# Phase 9 Plan 02: Rich TUI Summary

**Rich TUI with model loading progress, feature ticker from YAML, arrow-key navigable menu with greyed-out V2+ items, and searchable type-to-filter routine browser**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-20T22:30:13Z
- **Completed:** 2026-03-20T22:36:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Loading screen with OmniParser/CLIP/VLM progress bars and feature ticker from features.yml
- Arrow-key navigable menu with Record/Run/Update/Fork/List/Inspect items and V2+ greyed out as "Feature inbound"
- Searchable routine browser with type-to-filter, arrow navigation, and enter-to-select
- Cross-platform keypress reader supporting Windows (msvcrt) and Unix (tty/termios)
- 14 passing unit tests covering features, menu, browser, keys, and disabled item behavior

## Task Commits

Each task was committed atomically:

1. **Task 1: Create cross-platform key reader and TUI module** - `cf7c23a` (feat)
2. **Task 2: TUI unit tests** - `91a02cf` (test)

## Files Created/Modified
- `cli/__init__.py` - Package init for CLI module
- `cli/_keys.py` - Cross-platform single-keypress reader with normalized key names
- `cli/tui.py` - Rich TUI: loading screen, arrow-key menu, routine browser, main entry point
- `cli/features.yml` - Feature list for ticker display (shipped + coming_soon)
- `tests/test_tui.py` - 14 unit tests for TUI functionality

## Decisions Made
- Lazy imports for all action handlers (record_flow, run_routine, etc.) to avoid loading Qt/heavy modules at menu time
- MENU_ITEMS as module-level constant for easy testing and inspection
- read_key uses msvcrt on Windows, tty/termios on Unix with always-restore pattern
- Disabled menu items cannot be selected via Enter; navigation skips them with wrap-around

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created cli/__init__.py package file**
- **Found during:** Task 1
- **Issue:** cli/ directory did not exist; needed __init__.py for Python package
- **Fix:** Created empty cli/__init__.py
- **Files modified:** cli/__init__.py
- **Committed in:** cf7c23a

**2. [Rule 3 - Blocking] Created cli/features.yml**
- **Found during:** Task 1
- **Issue:** features.yml did not exist yet (Plan 01 may not have run)
- **Fix:** Created features.yml with shipped and coming_soon features
- **Files modified:** cli/features.yml
- **Committed in:** cf7c23a

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary for module to function. No scope creep.

## Issues Encountered
- Test patches needed to target `routine.discovery.list_routines` instead of `cli.tui.list_routines` because list_routines is lazy-imported inside the function, not at module level
- Windows msvcrt patch needed `patch.object(msvcrt, "getwch")` instead of `patch("cli._keys.msvcrt")` for same lazy-import reason

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TUI module ready for integration with CLI entry point
- All menu commands dispatch to lazy-imported handlers
- Feature ticker extensible via features.yml

---
*Phase: 09-tui-and-cli*
*Completed: 2026-03-20*
