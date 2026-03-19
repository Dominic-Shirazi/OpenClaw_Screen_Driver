---
phase: 05-routine-file-format
verified: 2026-03-18T00:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 5: Routine File Format Verification Report

**Phase Goal:** Routine files are portable, human-readable, and carry all data needed for reliable cross-resolution replay
**Verified:** 2026-03-18
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A Routine object serializes to a human-readable JSON dict with keys in logical order | VERIFIED | `to_dict()` confirmed key order: `$schema` first, `signature` last — 19 ordered keys. All 17 tests in test_routine_format.py pass. |
| 2 | Each step in the JSON includes element_type, label, caption, and confidence from VLM | VERIFIED | `build_v1_step` in format.py (line 83) extracts all four fields from tag_data. `_step_to_json` in record_session.py delegates to it. Live import check confirmed. |
| 3 | The NetworkX graph round-trips through to_dict/from_dict with zero data loss | VERIFIED | `OCSDGraph.to_dict()` and `OCSDGraph.from_dict()` called in format.py lines 182 and 244. `test_graph_round_trip` passes. |
| 4 | Routine metadata includes name, version, created_at, author, description, tags, category, programs, platform, theme | VERIFIED | All fields confirmed in Routine dataclass (format.py lines 144-172). `test_metadata_defaults` and `test_round_trip_all_fields` pass. |
| 5 | SHA256 checksum covers steps + graph data only (not mutable metadata) | VERIFIED | `calculate_routine_checksum(steps, graph_dict)` in checksum.py (line 35). `test_checksum_stable_across_metadata_changes` and `test_checksum_changes_with_steps` both pass. |
| 6 | Snippets are saved as {node_id}.png in snippets/ directory (not step_NN.png) | VERIFIED | record_session.py line 1090: `f"{node_id}.png"`. Live check confirmed `snippet_path: snippets/abc-123.png`. `step_{i:02d}.png` pattern absent from record_session.py. |
| 7 | Embeddings are saved as {node_id}.npy in embeddings/ directory (not step_NN.npy) | VERIFIED | record_session.py line 1098: `f"{node_id}.npy"`. Live check confirmed. |
| 8 | Storage layout is ~/.ocsd/routines/{name}/routine.json + snippets/ + embeddings/ | VERIFIED | `get_routine_dir()` returns `Path.home() / ".ocsd" / "routines"`. `Routine.save()` creates snippets/ and embeddings/ subdirectories. |
| 9 | RecordSession._save_routine() produces ocsd-routine-v1 format via Routine.save() | VERIFIED | record_session.py line 997 imports `Routine`, line 999 creates `Routine(...)`, line 1034 calls `routine.save(save_dir)`. `"$schema": "ocsd-routine-v0"` is absent. |
| 10 | v0 routines can be loaded and upgraded to v1 in memory without disk rewrite | VERIFIED | `upgrade_v0_to_v1` in migration.py (line 37) deep-copies input and returns upgraded dict. Live check confirmed: schema set to v1, node_id paths, anchors added. |
| 11 | Routine discovery scans ~/.ocsd/routines/ subdirectories for routine.json files | VERIFIED | `list_routines()` in discovery.py (line 44) iterates subdirectories checking for `routine.json`. All 4 discovery tests pass. |

**Score:** 11/11 truths verified

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|--------------|--------|---------|
| `routine/__init__.py` | — | 12 | VERIFIED | Exports Routine, VALID_CATEGORIES, calculate_routine_checksum |
| `routine/format.py` | 150 | 308 | VERIFIED | Routine dataclass with to_dict/from_dict/save/load/build_v1_step, VALID_CATEGORIES |
| `routine/checksum.py` | 20 | 59 | VERIFIED | calculate_routine_checksum with hashlib.sha256, _json_default fallback |
| `routine/migration.py` | 40 | 118 | VERIFIED | upgrade_v0_to_v1, detect_schema_version |
| `routine/discovery.py` | 30 | 91 | VERIFIED | RoutineInfo dataclass, get_routine_dir, list_routines |
| `tests/test_routine_format.py` | 100 | 284 | VERIFIED | 17 test functions — all pass |
| `tests/test_routine_migration.py` | 60 | 251 | VERIFIED | 17 test functions — all pass |
| `recorder/record_session.py` | — | (modified) | VERIFIED | _step_to_json delegates to build_v1_step; _save_routine uses Routine model |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `routine/format.py` | `mapper/graph.py` | `OCSDGraph.to_dict()` and `OCSDGraph.from_dict()` | WIRED | Lines 182 and 244 in format.py |
| `routine/format.py` | `routine/checksum.py` | `from routine.checksum import calculate_routine_checksum` | WIRED | Line 21 in format.py; called at lines 183 and 235 |
| `recorder/record_session.py` | `routine/format.py` | `from routine.format import Routine` and `build_v1_step` | WIRED | Lines 78 and 997 in record_session.py |
| `routine/migration.py` | `routine/format.py` | Sets `"$schema": "ocsd-routine-v1"` in output dict | WIRED | Line 55 in migration.py |
| `routine/discovery.py` | filesystem | Scans for `routine.json` files | WIRED | Line 69 in discovery.py |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FMT-01 | 05-01 | Routine stored as human-readable JSON with steps, bbox, region_hint, snippet/embedding paths | SATISFIED | routine/format.py Routine.to_dict() produces ordered JSON with all required fields; Routine.save() writes routine.json |
| FMT-02 | 05-01 | Each step includes VLM metadata (element_type, label, caption, confidence) | SATISFIED | build_v1_step extracts element_type, label, caption, confidence from tag_data; test_step_dict_keys and test_to_dict_key_order pass |
| FMT-03 | 05-02 | Snippets stored as +30% padded PNGs in snippets/ directory | SATISFIED | Node_id-based naming confirmed (f"{node_id}.png"); snippets/ directory created by save(). Note: the +30% padding behavior is in the recorder's crop logic, not the format — format correctly records the path |
| FMT-04 | 05-02 | CLIP embeddings stored as .npy files in embeddings/ directory | SATISFIED | Node_id-based .npy naming confirmed; embeddings/ directory created by save() |
| FMT-05 | 05-01 | NetworkX graph serializes to/from routine.json format | SATISFIED | OCSDGraph.to_dict() stored under "graph" key; OCSDGraph.from_dict() reconstructs on load; test_graph_round_trip passes |
| FMT-06 | 05-02 | Storage layout: ~/.ocsd/routines/{name}/routine.json + snippets/ + embeddings/ | SATISFIED | get_routine_dir() returns ~/.ocsd/routines/; Routine.save() creates all subdirectories |
| FMT-07 | 05-01 | Routine file includes metadata: name, version, created, author, description, tags | SATISFIED | All fields present in Routine dataclass; test_metadata_defaults confirms defaults (version="1.0.0", author=hostname, tags=[], etc.) |

All 7 requirements satisfied. No orphaned requirements detected for Phase 5 in REQUIREMENTS.md.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `routine/discovery.py` | 61 | `return []` | None | Intentional early return when directory does not exist; not a stub |

No blockers, no warnings, no placeholder patterns detected across all 8 phase-5 files.

### Human Verification Required

None — all behaviors are programmatically verifiable through tests and import checks.

The phase produces files on disk (routine.json, snippets/, embeddings/) that could optionally be inspected visually, but all structural and data integrity properties are covered by the 34-test suite.

### Test Suite Results

```
34 passed in 0.45s
  tests/test_routine_format.py   — 17/17 passed
  tests/test_routine_migration.py — 17/17 passed
```

### Commit Verification

All 5 commits referenced in SUMMARY files confirmed present in git log:

| Commit | Message |
|--------|---------|
| `44f3803` | test(05-01): add failing tests for routine file format |
| `5c9a8b6` | feat(05-01): implement routine dataclass with ordered JSON and SHA256 checksum |
| `7e93af0` | test(05-02): add failing tests for migration and discovery |
| `6c76578` | feat(05-02): implement v0-to-v1 migration and routine discovery |
| `ba22f4d` | feat(05-02): update RecordSession to produce ocsd-routine-v1 format |

### Gaps Summary

No gaps. All 11 observable truths verified, all 7 requirement IDs satisfied, all artifacts substantive and wired, all 34 tests pass, no anti-patterns found.

---

_Verified: 2026-03-18_
_Verifier: Claude (gsd-verifier)_
