---
phase: 05-routine-file-format
plan: 02
subsystem: format
tags: [migration, discovery, node-id, v0-to-v1, dataclass]

requires:
  - phase: 05-routine-file-format
    provides: "Routine dataclass, build_v1_step, checksum from Plan 01"
  - phase: 04-record-flow
    provides: "RecordSession with v0 save logic"
provides:
  - "v0-to-v1 in-memory migration with node_id path conversion"
  - "Routine discovery from ~/.ocsd/routines/ directories"
  - "RecordSession saves ocsd-routine-v1 format with Routine model"
  - "Node_id-based snippet and embedding file naming"
affects: [06-action-types, 07-run-flow, 08-routine-management]

tech-stack:
  added: []
  patterns: [in-memory-migration, directory-discovery, auto-detect-metadata]

key-files:
  created:
    - routine/migration.py
    - routine/discovery.py
    - tests/test_routine_migration.py
  modified:
    - recorder/record_session.py

key-decisions:
  - "Migration is in-memory only -- no disk rewrite of v0 files until verification method exists"
  - "Auto-detect theme from screenshot luminance (dark < 128, light >= 128)"
  - "Auto-detect foreground program via core.capture.get_window_title on Windows only"

patterns-established:
  - "Migration pattern: deep-copy input, upgrade fields, reorder keys to match v1 spec"
  - "Discovery pattern: scan subdirectories for routine.json, return lightweight RoutineInfo dataclass"

requirements-completed: [FMT-03, FMT-04, FMT-06]

duration: 4min
completed: 2026-03-19
---

# Phase 5 Plan 02: Migration, Discovery, and RecordSession v1 Integration Summary

**V0-to-v1 in-memory migration with node_id-based paths, routine directory discovery, and RecordSession producing ocsd-routine-v1 via Routine model**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-19T00:30:19Z
- **Completed:** 2026-03-19T00:34:19Z
- **Tasks:** 2 (Task 1 TDD, Task 2 auto)
- **Files modified:** 4

## Accomplishments
- V0-to-v1 migration converts all fields including node_id-based snippet/embedding paths and anchors
- Routine discovery scans ~/.ocsd/routines/ subdirectories for routine.json files with schema detection
- RecordSession._save_routine now creates Routine model and calls routine.save() for v1 output
- Snippet PNGs and embedding NPYs use {node_id} naming instead of step_NN

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for migration and discovery** - `7e93af0` (test)
2. **Task 1 GREEN: Implement migration.py and discovery.py** - `6c76578` (feat)
3. **Task 2: Update RecordSession to produce v1 format** - `ba22f4d` (feat)

## Files Created/Modified
- `routine/migration.py` - detect_schema_version and upgrade_v0_to_v1 with key reordering
- `routine/discovery.py` - RoutineInfo dataclass, get_routine_dir, list_routines scanner
- `tests/test_routine_migration.py` - 17 tests covering migration, detection, and discovery
- `recorder/record_session.py` - _step_to_json delegates to build_v1_step; _save_routine uses Routine model; node_id naming for assets

## Decisions Made
- Migration is in-memory only (no disk rewrite of v0 files) per CONTEXT.md decision
- Theme auto-detection uses simple luminance threshold (mean gray < 128 = dark)
- Foreground program detection is Windows-only via core.capture.get_window_title, fails gracefully on other platforms

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Routine format pipeline is complete: record -> v1 save -> discovery -> migration
- Ready for Phase 6 (action types) and Phase 7 (run flow) to consume v1 routines
- Phase 8 (routine management) can use discovery module for listing/organizing routines

---
*Phase: 05-routine-file-format*
*Completed: 2026-03-19*
