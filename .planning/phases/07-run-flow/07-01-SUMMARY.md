---
phase: 07-run-flow
plan: 01
subsystem: core
tags: [locate, conditions, rapidfuzz, levenshtein, ocr, vlm, cascade]

requires:
  - phase: 06-action-types
    provides: condition engine stubs (element_appears, text_matches), v1 step format
provides:
  - locate_element_from_step() adapter for v1 routine step dicts
  - Complete element_appears condition with adaptive VLM escalation
  - Complete text_matches condition with fuzzy OCR via rapidfuzz
  - rapidfuzz project dependency
affects: [07-run-flow, runner, replay]

tech-stack:
  added: [rapidfuzz]
  patterns: [adaptive-vlm-escalation, fuzzy-levenshtein-matching]

key-files:
  created:
    - tests/test_locate_adapter.py
  modified:
    - core/locate.py
    - core/conditions.py
    - pyproject.toml
    - tests/test_condition_engine.py

key-decisions:
  - "Adaptive VLM: first 3 polls skip VLM for fast checking, then escalate to VLM on poll 4+"
  - "Levenshtein threshold: min(5, max(2, len/5)) balances typo tolerance and false positives"
  - "skip_position_fallback=True always for conditions -- never blind-click for condition checks"

patterns-established:
  - "Adaptive escalation: cheap stages first, expensive stages after N failures"
  - "Step-dict locate: extract anchors from v1 step dict instead of graph node"

requirements-completed: [RUN-02]

duration: 4min
completed: 2026-03-20
---

# Phase 7 Plan 01: Locate Adapter and Condition Engine Completion Summary

**locate_element_from_step() 5-stage cascade for v1 step dicts with adaptive VLM escalation and fuzzy OCR text matching via rapidfuzz**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-20T02:41:01Z
- **Completed:** 2026-03-20T02:45:10Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added locate_element_from_step() that mirrors the existing 5-stage cascade but works with v1 step dicts instead of graph nodes
- Replaced element_appears stub with adaptive VLM escalation (skips VLM for first 3 polls, enables on 4th)
- Replaced text_matches stub with exact OCR + fuzzy Levenshtein fallback using rapidfuzz
- Full test coverage with 12 new tests (5 locate adapter + 7 condition completions)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add locate_element_from_step() and complete condition stubs** - `d5c3bf9` (feat)
2. **Task 2: Tests for locate adapter and condition completions** - `4511edc` (test)

## Files Created/Modified
- `core/locate.py` - Added locate_element_from_step() with 5-stage cascade for v1 step dicts
- `core/conditions.py` - Replaced element_appears and text_matches stubs with full implementations
- `pyproject.toml` - Added rapidfuzz>=3.0 dependency
- `tests/test_locate_adapter.py` - 5 tests for locate adapter (OCR hit, position fallback, skip position, skip VLM, snippet resolve)
- `tests/test_condition_engine.py` - 7 new tests for condition completions (element_appears success/failure/adaptive VLM, text_matches exact/fuzzy/no match/empty)

## Decisions Made
- Adaptive VLM escalation: first 3 polls skip VLM (stages 1-3 only) for fast polling, then include VLM on poll 4+ for higher reliability
- Levenshtein distance threshold formula: min(5, max(2, len(target)//5)) -- at least 2 edits tolerance, scales with text length, capped at 5
- Position fallback always skipped for conditions (skip_position_fallback=True) -- blind-clicking is never acceptable for condition verification

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- locate_element_from_step() is ready for the runner (Plan 03) to use for step replay
- Condition engine is complete for wait/loop actions during replay
- Next plan (07-02) can build the REPLAYING overlay state with purple shimmer

## Self-Check: PASSED

All 5 files verified on disk. Both commits (d5c3bf9, 4511edc) verified in git log.

---
*Phase: 07-run-flow*
*Completed: 2026-03-20*
