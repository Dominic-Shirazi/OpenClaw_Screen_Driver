---
phase: 06-action-types
plan: 04
subsystem: ui
tags: [pyqt6, mini-dialog, wait-condition, prompt-user, threading, overlay]

requires:
  - phase: 06-action-types-02
    provides: "ConditionChecker engine with fixed_timer, screen_change, element_appears, vlm_check"
  - phase: 06-action-types-03
    provides: "Toolbar quick-add buttons (add_wait, add_prompt), RecordPhase WAIT_CONFIGURING/PROMPT_CONFIGURING"
provides:
  - "WaitDialog mini-dialog for wait condition configuration (4 condition types + timeout)"
  - "PromptDialog mini-dialog for prompt_user question text input"
  - "Controller show/hide methods for wait and prompt dialogs"
  - "prompt_user_blocking executor with respond_to_prompt API unblock mechanism"
  - "RecordSession handlers wiring toolbar buttons to mini-dialogs to step creation"
affects: [07-replay-engine, 08-routine-management, 10-api]

tech-stack:
  added: []
  patterns:
    - "Mini-dialog QGraphicsObject with proxy widgets for lightweight config forms"
    - "threading.Event for blocking executor function with API-triggered resume"

key-files:
  created:
    - recorder/overlay/mini_dialogs.py
  modified:
    - recorder/record_session.py
    - recorder/overlay/controller.py
    - recorder/overlay/view.py
    - core/executor.py
    - tests/test_condition_engine.py

key-decisions:
  - "Generic show_mini_dialog/hide_mini_dialog on view instead of per-dialog methods"
  - "prompt_user_blocking uses module-level threading.Event for simplicity"

patterns-established:
  - "Mini-dialog pattern: frosted glass QGraphicsObject with confirm/dismiss signals, positioned center-screen"
  - "Blocking executor pattern: threading.Event cleared before wait, set by external trigger"

requirements-completed: [ACT-10, ACT-12]

duration: 4min
completed: 2026-03-19
---

# Phase 6 Plan 4: Wait & Prompt Dialog Flows Summary

**WaitDialog mini-dialog with 4 condition types + timeout, PromptDialog for question text, and prompt_user_blocking executor with threading.Event resume mechanism**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-19T21:59:58Z
- **Completed:** 2026-03-19T22:04:09Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- WaitDialog with condition type dropdown (fixed_timer, element_appears, screen_change, vlm_check), timeout field, and conditional param field
- PromptDialog with question text input for prompt_user recording steps
- RecordSession handlers wire toolbar buttons to mini-dialogs, creating steps that skip dry-run
- prompt_user_blocking blocks executor thread until respond_to_prompt is called (API /respond stub)
- Controller and View have show/hide methods for both dialogs

## Task Commits

Each task was committed atomically:

1. **Task 1: WaitDialog and PromptDialog mini-dialog widgets** - `2e2f129` (feat)
2. **Task 2: Wire wait/prompt handlers, controller methods, prompt_user executor** - `9094883` (feat)

## Files Created/Modified
- `recorder/overlay/mini_dialogs.py` - WaitDialog and PromptDialog QGraphicsObject classes with frosted glass style
- `recorder/record_session.py` - _handle_add_wait, _handle_add_prompt with dialog creation and step recording
- `recorder/overlay/controller.py` - show/hide_wait_dialog, show/hide_prompt_dialog public API
- `recorder/overlay/view.py` - Generic show_mini_dialog/hide_mini_dialog for dialog lifecycle
- `core/executor.py` - prompt_user_blocking and respond_to_prompt with threading.Event
- `tests/test_condition_engine.py` - Dialog signal tests, wait step format, prompt blocking tests

## Decisions Made
- Used generic show_mini_dialog/hide_mini_dialog on OverlayView instead of per-dialog methods (cleaner, extensible)
- prompt_user_blocking uses module-level threading.Event (simplest correct mechanism, easily testable)
- Wait and prompt steps skip dry-run entirely (wait has no visible action, prompt is a flow control step)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added View mini-dialog support**
- **Found during:** Task 2 (controller dialog methods)
- **Issue:** Plan specified controller methods but OverlayView had no mini-dialog infrastructure
- **Fix:** Added _mini_dialog instance var and generic show_mini_dialog/hide_mini_dialog methods to view.py
- **Files modified:** recorder/overlay/view.py
- **Verification:** Controller methods delegate correctly through view
- **Committed in:** 9094883 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary view-level plumbing to support controller API. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Wait and prompt_user recording flows complete
- Plan 05 (loop definition) can proceed -- loop uses same ConditionChecker engine
- Phase 7 replay engine can consume wait steps with wait_definition and call ConditionChecker
- Phase 10 API can implement /respond endpoint calling respond_to_prompt

---
*Phase: 06-action-types*
*Completed: 2026-03-19*
