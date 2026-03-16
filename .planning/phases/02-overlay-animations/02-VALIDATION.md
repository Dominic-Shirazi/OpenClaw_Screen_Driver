---
phase: 2
slug: overlay-animations
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_animation_clock.py tests/test_shimmer_layer.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_animation_clock.py tests/test_shimmer_layer.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | ANIM-05 | unit | `python -m pytest tests/test_animation_clock.py -x` | Wave 0 | ⬜ pending |
| 02-02-01 | 02 | 1 | ANIM-01 | unit | `python -m pytest tests/test_shimmer_layer.py -x` | Wave 0 | ⬜ pending |
| 02-03-01 | 03 | 2 | ANIM-02 | unit | `python -m pytest tests/test_scan_layer.py -x` | Wave 0 | ⬜ pending |
| 02-04-01 | 04 | 2 | ANIM-03 | unit | `python -m pytest tests/test_bbox_morph.py -x` | Wave 0 | ⬜ pending |
| 02-05-01 | 05 | 3 | ANIM-04 | unit | `python -m pytest tests/test_donut_cloud.py -x` | Wave 0 | ⬜ pending |
| 02-06-01 | 06 | 3 | ANIM-06 | integration | `python -m pytest tests/test_overlay_animations_integration.py -x` | Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_animation_clock.py` — stubs for ANIM-05 (clock ticking, delta-time, register/unregister)
- [ ] `tests/test_shimmer_layer.py` — stubs for ANIM-01 (shimmer creation, state color, phase property)
- [ ] `tests/test_scan_layer.py` — stubs for ANIM-02 (scan phase transitions, completion)
- [ ] `tests/test_bbox_morph.py` — stubs for ANIM-03 (morph animation, corner positions)
- [ ] `tests/test_donut_cloud.py` — stubs for ANIM-04 (cloud rendering, color transition, handles)
- [ ] `tests/test_overlay_animations_integration.py` — stubs for ANIM-06 (view integration, transitions)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| 60fps with no visible frame drops | ANIM-01 | Visual smoothness requires human eye; automated FPS logging is supplementary | Run overlay in ready state for 10s, visually confirm smooth gradient rotation |
| Mouse-reactive shimmer retreat | ANIM-01 | Organic feel of retreat is subjective | Move mouse along border edges, confirm shimmer waves retreat naturally |
| Scan animation cinematic sequence | ANIM-02 | Visual sequence quality requires human review | Trigger scan, confirm corner glow → line draw → fill → laser sequence |
| Donut cloud raindrop effect | ANIM-04 | Probability distribution visualization is visual | Trigger donut cloud, confirm raindrops appear concentrated at center |
| CPU stays under 10% during idle | ANIM-05 | System load varies; needs real monitoring | Run overlay idle for 30s, check Task Manager CPU usage |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
