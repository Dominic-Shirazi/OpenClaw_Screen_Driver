---
phase: 7
slug: run-flow
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_routine_runner.py -x -q` |
| **Full suite command** | `python -m pytest tests/test_routine_runner.py tests/test_condition_engine.py tests/test_run_logs.py -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_routine_runner.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/test_routine_runner.py tests/test_condition_engine.py tests/test_run_logs.py -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | RUN-01 | unit | `pytest tests/test_routine_runner.py -k test_load_and_preflight` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | RUN-02 | unit | `pytest tests/test_routine_runner.py -k test_locate_from_step` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | RUN-03, RUN-04 | unit | `pytest tests/test_routine_runner.py -k test_action_dispatch` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 1 | RUN-05 | unit | `pytest tests/test_routine_runner.py -k test_human_delay` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 2 | RUN-06 | unit | `pytest tests/test_routine_runner.py -k test_post_action_validation` | ❌ W0 | ⬜ pending |
| 07-03-02 | 03 | 2 | RUN-07, RUN-08 | unit | `pytest tests/test_routine_runner.py -k test_failure_cascade` | ❌ W0 | ⬜ pending |
| 07-04-01 | 04 | 2 | RUN-09 | unit | `pytest tests/test_routine_runner.py -k test_overlay_replay` | ❌ W0 | ⬜ pending |
| 07-05-01 | 05 | 3 | RUN-02 | unit | `pytest tests/test_condition_engine.py -k test_element_appears` | ❌ W0 | ⬜ pending |
| 07-05-02 | 05 | 3 | RUN-02 | unit | `pytest tests/test_condition_engine.py -k test_text_matches` | ❌ W0 | ⬜ pending |
| 07-06-01 | 06 | 3 | RUN-07 | unit | `pytest tests/test_run_logs.py -k test_run_log_creation` | ❌ W0 | ⬜ pending |
| 07-06-02 | 06 | 3 | RUN-07 | unit | `pytest tests/test_run_logs.py -k test_self_cleaning` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_routine_runner.py` — stubs for RUN-01 through RUN-09
- [ ] `tests/test_condition_engine.py` — extend existing with element_appears and text_matches tests
- [ ] `tests/test_run_logs.py` — stubs for run log creation and self-cleaning
- [ ] `rapidfuzz` — add to pyproject.toml dependencies

*Existing test infrastructure (pytest, conftest) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Purple shimmer overlay during replay | RUN-09 | Requires visual Qt overlay on display | Launch replay with `replay.show_overlay: true`, verify purple border |
| Status badge shows step info | RUN-09 | Visual UI element | Verify top-center pill shows "Step N/M: Label" during replay |
| Target bbox highlight before click | RUN-09 | Visual animation timing | Verify ~300ms purple glow around target before mouse moves |
| Camera flash after screenshot | RUN-09 | Visual overlay timing | Verify 15px border flash after each screenshot capture |
| Mouse Bezier movement is visible | RUN-03 | Visual path inspection | Watch mouse path during replay, verify curved non-linear movement |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
