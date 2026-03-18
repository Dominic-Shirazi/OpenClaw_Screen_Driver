---
phase: 5
slug: routine-file-format
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-18
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_routine_format.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_routine_format.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 1 | FMT-01 | unit | `python -m pytest tests/test_routine_format.py::test_schema_structure -x` | ❌ W0 | ⬜ pending |
| 5-01-02 | 01 | 1 | FMT-02 | unit | `python -m pytest tests/test_routine_format.py::test_vlm_metadata_fields -x` | ❌ W0 | ⬜ pending |
| 5-01-03 | 01 | 1 | FMT-03 | unit | `python -m pytest tests/test_routine_format.py::test_snippet_assets -x` | ❌ W0 | ⬜ pending |
| 5-01-04 | 01 | 1 | FMT-04 | unit | `python -m pytest tests/test_routine_format.py::test_graph_roundtrip -x` | ❌ W0 | ⬜ pending |
| 5-01-05 | 01 | 1 | FMT-05 | unit | `python -m pytest tests/test_routine_format.py::test_top_level_metadata -x` | ❌ W0 | ⬜ pending |
| 5-01-06 | 01 | 1 | FMT-06 | unit | `python -m pytest tests/test_routine_format.py::test_checksum_integrity -x` | ❌ W0 | ⬜ pending |
| 5-01-07 | 01 | 1 | FMT-07 | unit | `python -m pytest tests/test_routine_format.py::test_cross_platform_paths -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_routine_format.py` — stubs for FMT-01 through FMT-07
- [ ] `tests/conftest.py` — shared fixtures (sample graph, temp routine dir)

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Human-readable JSON | FMT-01 | Subjective readability | Open routine.json in text editor, verify indent=2, clear key names |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
