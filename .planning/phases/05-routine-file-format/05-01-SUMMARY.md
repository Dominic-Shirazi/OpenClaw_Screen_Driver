---
phase: 05-routine-file-format
plan: 01
subsystem: format
tags: [json, sha256, dataclass, networkx, serialization]

requires:
  - phase: 04-record-flow
    provides: "v0 routine format and RecordSession save logic"
provides:
  - "Routine dataclass with to_dict/from_dict/save/load"
  - "calculate_routine_checksum for integrity verification"
  - "build_v1_step helper for v0-to-v1 step conversion"
  - "VALID_CATEGORIES constant for routine categorization"
affects: [05-routine-file-format, 06-action-types, 07-run-flow]

tech-stack:
  added: []
  patterns: [dataclass-model, ordered-json-serialization, checksum-integrity]

key-files:
  created:
    - routine/__init__.py
    - routine/format.py
    - routine/checksum.py
    - tests/test_routine_format.py
  modified: []

key-decisions:
  - "Checksum covers steps+graph only (not mutable metadata) for stable routine identity"
  - "Platform detection maps win32/linux/darwin to windows/ubuntu/macos"
  - "_json_default fallback for enum serialization reused from mapper/export.py pattern"

patterns-established:
  - "Routine model: dataclass with to_dict/from_dict for JSON round-trip"
  - "Checksum stability: SHA256 over content-only fields, metadata changes don't invalidate"
  - "Key order: $schema first, signature last, for human-readable JSON"

requirements-completed: [FMT-01, FMT-02, FMT-05, FMT-07]

duration: 3min
completed: 2026-03-19
---

# Phase 5 Plan 01: Routine File Format Summary

**Routine dataclass with ordered ocsd-routine-v1 JSON, SHA256 checksum over steps+graph, and NetworkX graph round-trip**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-19T00:26:03Z
- **Completed:** 2026-03-19T00:28:33Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Routine dataclass serializes to human-readable JSON with exact key ordering per v1 spec
- SHA256 checksum is stable across metadata-only changes (name, tags, description, updated_at)
- NetworkX graph round-trips through to_dict/from_dict with zero data loss
- Save/load cycle creates routine.json with snippets/ and embeddings/ directories
- Load validates both schema version and checksum integrity

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for routine format** - `44f3803` (test)
2. **Task 1 GREEN: Implement routine dataclass and checksum** - `5c9a8b6` (feat)

## Files Created/Modified
- `routine/__init__.py` - Package init with public exports (Routine, VALID_CATEGORIES, calculate_routine_checksum)
- `routine/format.py` - Routine dataclass with to_dict/from_dict/save/load and build_v1_step helper
- `routine/checksum.py` - SHA256 checksum computation over steps+graph data
- `tests/test_routine_format.py` - 17 tests covering schema, key order, checksum, round-trip, save/load

## Decisions Made
- Checksum covers steps+graph only (not mutable metadata) so routine identity is stable across edits to name/tags/description
- Platform detection maps sys.platform values: win32->windows, linux->ubuntu, darwin->macos
- Reused _json_default enum fallback pattern from mapper/export.py for consistency

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- routine/ package is importable and tested, ready for Plan 02 (v0-to-v1 migration, RecordSession integration)
- build_v1_step helper ready for RecordSession to adopt in place of _step_to_json

---
*Phase: 05-routine-file-format*
*Completed: 2026-03-19*
