---
phase: 08-routine-management
plan: 02
subsystem: routine-management
tags: [update-session, toolbar-mode, guided-replay, graph-rebuild, version-bump]

requires:
  - phase: 08-routine-management-01
    provides: fork_routine, bump_minor, bump_patch, delete_routine, inspect_routine
provides:
  - UpdateSession guided-replay orchestrator with keep/delete/edit/fork controls
  - UPDATE and UPDATE_NOT_FOUND toolbar modes for overlay HUD
affects: [09-tui-and-cli, 10-api-and-mcp]

tech-stack:
  added: []
  patterns: [guided-replay-state-machine, graph-rebuild-on-mutation]

key-files:
  created:
    - routine/update_session.py
    - tests/test_update_session.py
  modified:
    - recorder/overlay/toolbar_panel.py

key-decisions:
  - "Graph rebuilt from scratch on save (not mutated in place) for consistency with fork_routine pattern"
  - "fork_routine base_dir defaults to routine_dir.parent for UpdateSession fork flow"

patterns-established:
  - "UpdateSession copies steps to working list; mutations tracked via index sets; applied atomically on save"

requirements-completed: [MGMT-01]

duration: 3min
completed: 2026-03-20
---

# Phase 8 Plan 02: UpdateSession Orchestrator Summary

**Guided-replay UpdateSession with step-by-step keep/delete/edit/fork controls, graph rebuild, loop cleanup, and UPDATE toolbar modes**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-20T04:48:43Z
- **Completed:** 2026-03-20T04:51:28Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added UPDATE and UPDATE_NOT_FOUND toolbar modes with 4 buttons each
- Implemented UpdateSession orchestrator with full step walkthrough lifecycle
- Graph rebuilt from surviving steps after deletions with proper edge connectivity
- Loop body references to deleted node_ids automatically cleaned
- Orphaned snippet/embedding files removed on save
- Version auto-bumps minor for structural changes, patch otherwise
- 10 passing tests covering all UpdateSession behaviors

## Task Commits

Each task was committed atomically:

1. **Task 1: Add UPDATE toolbar modes** - `500e470` (feat)
2. **Task 2 RED: Failing tests** - `4c18733` (test)
3. **Task 2 GREEN: UpdateSession implementation** - `332943b` (feat)

## Files Created/Modified
- `routine/update_session.py` - UpdateSession class with keep/delete/edit/fork/save workflow
- `tests/test_update_session.py` - 10 test cases covering navigation, deletion, graph cleanup, loop refs, versioning, fork
- `recorder/overlay/toolbar_panel.py` - Added UPDATE and UPDATE_NOT_FOUND enum values and button definitions

## Decisions Made
- Graph rebuilt from scratch on save (not mutated in place) for consistency with fork_routine pattern from Plan 01
- fork_routine base_dir defaults to routine_dir.parent so UpdateSession fork creates sibling directories

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 8 (Routine Management) fully complete
- All MGMT requirements implemented: fork, delete, inspect (Plan 01) and update flow (Plan 02)
- Ready for Phase 9: TUI and CLI

---
*Phase: 08-routine-management*
*Completed: 2026-03-20*
