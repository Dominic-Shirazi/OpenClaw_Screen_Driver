---
phase: 6
slug: action-types
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml `[tool.pytest]` |
| **Quick run command** | `python -m pytest tests/ -x -q --tb=short` |
| **Full suite command** | `python -m pytest tests/ -v --tb=long` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q --tb=short`
- **After every plan wave:** Run `python -m pytest tests/ -v --tb=long`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | ACT-01,02,03 | unit | `pytest tests/test_action_types.py -k "click"` | Created by Plan 01 | pending |
| 06-01-02 | 01 | 1 | ACT-05 | unit | `pytest tests/test_action_types.py -k "type_step"` | Created by Plan 01 | pending |
| 06-01-03 | 01 | 1 | ACT-09 | unit | `pytest tests/test_action_types.py -k "scroll"` | Created by Plan 01 | pending |
| 06-02-01 | 02 | 1 | ACT-08 | unit | `pytest tests/test_condition_engine.py -k "select_all"` | Created by Plan 02 | pending |
| 06-02-02 | 02 | 1 | ACT-10 (engine) | unit | `pytest tests/test_condition_engine.py -k "timer or screen_change"` | Created by Plan 02 | pending |
| 06-03-01 | 03 | 2 | ACT-04 | unit | `pytest tests/test_action_types.py -k "click_drag"` | Created by Plan 03 | pending |
| 06-03-02 | 03 | 2 | ACT-06,07 | unit | `pytest tests/test_action_types.py -k "read"` | Created by Plan 03 | pending |
| 06-04-01 | 04 | 2 | ACT-10 (dialog) | unit | `pytest tests/test_condition_engine.py -k "dialog or wait_step"` | Created by Plan 04 | pending |
| 06-04-02 | 04 | 2 | ACT-12 | unit | `pytest tests/test_condition_engine.py -k "prompt_user"` | Created by Plan 04 | pending |
| 06-05-01 | 05 | 3 | ACT-11 | unit | `pytest tests/test_action_types.py -k "loop"` | Created by Plan 05 | pending |
| 06-05-02 | 05 | 3 | ACT-11 (resolve) | unit | `pytest tests/test_action_types.py -k "resolve_loop"` | Created by Plan 05 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_action_types.py` — created by Plan 01 Task 1 (step format + dry-run dispatch tests)
- [ ] `tests/test_condition_engine.py` — created by Plan 02 Task 1 (condition engine + select_all_extract tests)
- [ ] `tests/conftest.py` — shared fixtures (mock executor, mock pipeline bridge)
- [ ] pytest already installed in venv

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| click_drag two-capture UX flow | ACT-04 | Requires GUI interaction with overlay | Record a drag action, verify two bboxes captured in step JSON |
| prompt_user pauses routine | ACT-12 | Requires running API server + client | Start replay, hit prompt_user step, POST to /respond, verify resume |
| Human-like typing with variance | ACT-05 | Timing behavior is visual/perceptual | Replay type action, observe per-letter delays vary naturally |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
