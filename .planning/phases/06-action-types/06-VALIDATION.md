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
| 06-01-01 | 01 | 1 | ACT-01 | unit | `pytest tests/test_action_click.py` | ❌ W0 | ⬜ pending |
| 06-01-02 | 01 | 1 | ACT-02 | unit | `pytest tests/test_action_double_click.py` | ❌ W0 | ⬜ pending |
| 06-01-03 | 01 | 1 | ACT-03 | unit | `pytest tests/test_action_right_click.py` | ❌ W0 | ⬜ pending |
| 06-01-04 | 01 | 1 | ACT-04 | unit | `pytest tests/test_action_drag.py` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 1 | ACT-05 | unit | `pytest tests/test_action_type.py` | ❌ W0 | ⬜ pending |
| 06-03-01 | 03 | 2 | ACT-06 | unit | `pytest tests/test_action_read.py` | ❌ W0 | ⬜ pending |
| 06-03-02 | 03 | 2 | ACT-07 | unit | `pytest tests/test_action_snip.py` | ❌ W0 | ⬜ pending |
| 06-03-03 | 03 | 2 | ACT-08 | unit | `pytest tests/test_action_select_all.py` | ❌ W0 | ⬜ pending |
| 06-04-01 | 04 | 2 | ACT-09 | unit | `pytest tests/test_action_scroll.py` | ❌ W0 | ⬜ pending |
| 06-04-02 | 04 | 2 | ACT-10 | unit | `pytest tests/test_action_wait.py` | ❌ W0 | ⬜ pending |
| 06-04-03 | 04 | 2 | ACT-11 | unit | `pytest tests/test_action_loop.py` | ❌ W0 | ⬜ pending |
| 06-04-04 | 04 | 2 | ACT-12 | unit | `pytest tests/test_action_prompt.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_action_types.py` — stubs for ACT-01 through ACT-12
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
