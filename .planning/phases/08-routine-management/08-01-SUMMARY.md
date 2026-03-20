---
phase: 08-routine-management
plan: 01
subsystem: routine
tags: [shutil, rich, semver, fork, crud]

requires:
  - phase: 05-routine-format
    provides: Routine dataclass with save/load, OCSDGraph, checksum
provides:
  - fork_routine with optional truncation and orphan cleanup
  - delete_routine with Rich confirmation
  - inspect_routine with Rich table and JSON output
  - bump_minor and bump_patch version helpers
affects: [08-routine-management, 09-cli-tui]

tech-stack:
  added: []
  patterns: [shutil.copytree for fork, shutil.rmtree for delete, Rich Table for inspect]

key-files:
  created:
    - routine/version.py
    - routine/management.py
    - tests/test_routine_management.py
  modified: []

key-decisions:
  - "Graph rebuilt from scratch on truncation (new OCSDGraph with kept node_ids only)"
  - "Orphan cleanup scans snippets/ and embeddings/ by node_id filename convention"

patterns-established:
  - "Management functions accept base_dir parameter for testability (defaults to get_routine_dir)"
  - "skip_confirm parameter on destructive operations for programmatic use"

requirements-completed: [MGMT-02, MGMT-03, MGMT-04]

duration: 2min
completed: 2026-03-20
---

# Phase 8 Plan 1: Routine Management Operations Summary

**Fork/delete/inspect CRUD operations with Rich prompts, optional truncation, and 12 passing tests**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T04:45:07Z
- **Completed:** 2026-03-20T04:47:09Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- Version helpers (bump_minor, bump_patch) for semantic version manipulation
- fork_routine creates independent copies via shutil.copytree with optional truncation that removes orphaned assets
- delete_routine uses Rich Confirm prompt with skip_confirm for programmatic use
- inspect_routine outputs Rich table (step#, action, label, bbox, snippet) or JSON
- 12 comprehensive tests covering all behaviors

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `93e1c54` (test)
2. **Task 1 GREEN: Implementation** - `5020abf` (feat)

## Files Created/Modified
- `routine/version.py` - bump_minor and bump_patch semantic version helpers
- `routine/management.py` - fork_routine, delete_routine, inspect_routine functions
- `tests/test_routine_management.py` - 12 test functions with helper _make_test_routine

## Decisions Made
- Graph rebuilt from scratch on truncation rather than trying to prune existing graph (simpler, avoids edge cleanup)
- Orphan cleanup uses node_id filename convention (snippets/{node_id}.png, embeddings/{node_id}.npy)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Management functions ready for UpdateSession (Plan 02) "Fork Here" and "Save & Replace" flows
- Version helpers available for fork version reset

---
*Phase: 08-routine-management*
*Completed: 2026-03-20*
