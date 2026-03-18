# Phase 5: Routine File Format - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Define the authoritative `ocsd-routine-v1` schema for recorded screen automation routines. Evolve the interim `ocsd-routine-v0` format (from Phase 4) into a human-readable, cross-resolution-portable, integrity-verified format. Implement the Routine model (load/save/validate), snippet storage, and embedding storage. Ensure NetworkX graph round-trips through routine.json with no data loss.

Does NOT build the replay engine (Phase 7), action type implementations (Phase 6), or routine management operations (Phase 8). Does NOT merge with or replace `ocsd-skill-v1` — skills and routines are separate concepts.

</domain>

<decisions>
## Implementation Decisions

### Schema evolution strategy
- **Separate formats**: `ocsd-routine-v1` for recorded routines, `ocsd-skill-v1` stays for the OpenClaw skill layer. OCSD itself IS "the skill" of an OpenClaw bot — all routines are assets of that one skill. No converter/bridge needed — they operate at different abstraction levels
- **Integrity**: `$schema` version tag + SHA256 checksum over steps + graph data + empty `signature` field placeholder for future Hub trust scoring (V2)
- **V0→V1 migration**: Build the auto-upgrade path (detect schema version on load, transparently upgrade in memory) but leave it OFF by default. Don't rewrite v0 files to disk until a verification method is in place. Flip the switch later

### Graph-to-steps relationship
- **Both co-exist, cross-validated**: routine.json contains a flat `steps` array AND the full NetworkX graph dict. The trajectory is toward graph-primary, but steps serve as fallback and human readability for now
- **Conflict resolution**: If steps and graph disagree on load, warn and prefer steps (safer for the current steps-primary era)
- **Full OCSDGraph schema**: Graph section includes everything from `OCSDGraph.to_dict()` — node IDs, layers, execution stats, context_links. Ready for V2 graph-primary migration
- **1:1 step-node mapping**: Each step references a graph node by `node_id`. Step order = execution order. Graph edges link step N's node → step N+1's node
- **Linear only in V1**: Steps are a flat ordered list. No branching/looping in the format structure. The `loop` action type (Phase 6) handles that behavior-level, not format-level
- **Exact OCSDGraph.to_dict()**: Graph section is raw output of `OCSDGraph.to_dict()`. Load back with `OCSDGraph.from_dict()`. Zero translation layer

### Cross-resolution portability
- **Recording resolution stored**: Top-level `resolution` field ([w, h]) records the screen dimensions during recording
- **Anchor strategy per step**: Each step specifies which strategies the replay engine should use to find the element. Auto-determined at record time based on what data was captured — user never picks this
- **Four anchor strategies**: `visual_match` (CLIP embedding + snippet), `ocr_text` (visible text), `position_pct` (percentage coordinates), `region_hint` (narrow search to screen region). All four are always populated if data exists
- **Snippets at original resolution**: Saved as-captured from the recording screen. No normalization. Replay engine handles scaling

### Metadata and versioning
- **Semver versioning**: `version` field uses semantic versioning (major.minor.patch). Starts at "1.0.0" on first save. Update flow (Phase 8) bumps the version
- **Author = machine identity**: Auto-populated with hostname. `author_display` field for optional user-set display name (null by default)
- **Structured categories + free tags**: Separate `category` field (from a fixed list: e.g., productivity, web, system, finance, creative) plus free-form `tags` list
- **Program metadata**: `programs` list stores the target application(s) used in the routine, auto-detected from foreground app during recording
- **Platform**: `platform` field records the OS the routine was recorded on (windows/ubuntu/macos)
- **Theme**: `theme` field records dark/light/null, auto-detected from screenshot luminance during recording. Nullable
- **Auto-detect with override**: Programs, platform, and theme are auto-detected during recording. User can edit later

### File format readability
- **Pretty-printed**: 2-space indentation, `ensure_ascii=False`. Human-readable in any text editor
- **Logical field ordering**: `$schema` → identity (name, version, description) → author/timestamps → metadata (category, tags, programs, platform, theme) → start_from, resolution → steps → graph → checksum, signature

### Routine discovery
- **Directory scan**: Enumerate `~/.ocsd/routines/` subdirectories containing `routine.json`. No index file. Simple, always up-to-date
- **Subdirectory structure**: Each routine gets its own directory with routine.json + snippets/ + embeddings/

### Embedding and snippet storage
- **Node ID based naming**: Both snippets and embeddings are named by node_id (e.g., `snippets/{node_id}.png`, `embeddings/{node_id}.npy`). Element-centric, not step-index-centric. Future-proofs for V2 UI GPS where elements can be shared across routines
- **Raw .npy vectors**: One file per element's CLIP embedding. No FAISS index per routine. Replay engine loads on demand

### Error resilience
- **Graceful degradation**: Missing snippets/embeddings = skip that anchor strategy. The cascade still has OCR, VLM, and position fallback. Log a warning. Never hard-fail on missing assets
- **VLM fallback config in global config**: Optional VLM/computer-use model fallback (Gemini, GPT, Claude) configured in ocsd.yaml or .env. Global, not per-routine. Implemented in Phase 7 run flow
- **Asset requirements implicit from anchors**: If a step has anchor `visual_match`, its embedding is implicitly required. No separate asset manifest

### Claude's Discretion
- Exact SHA256 checksum implementation details (what fields to include/exclude)
- Internal Routine model class design (dataclass, Pydantic, plain dict)
- How to implement theme auto-detection from screenshot luminance
- How to detect foreground app name cross-platform (Windows vs Ubuntu)
- Migration function internals for v0→v1 upgrade
- Whether to use OrderedDict or custom JSON encoder for field ordering

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Routine format requirements
- `.planning/REQUIREMENTS.md` — FMT-01 through FMT-07 define all routine format requirements
- `.planning/ROADMAP.md` — Phase 5 success criteria (5 criteria that must be TRUE)

### Existing serialization code
- `mapper/export.py` — `ocsd-skill-v1` schema with SHA256 checksum integrity, `export_skill()`, `load_skill_from_file()`. Reference for checksum pattern — DO NOT merge with routine format
- `mapper/graph.py` — `OCSDGraph` with `to_dict()`/`from_dict()`, `add_node()` schema, `_region_hint()`. Graph section in routine.json must match this exactly

### Phase 4 interim format
- `recorder/record_session.py` — `_step_to_json()` (line 63) and `_save_routine()` (line 950) show the current v0 format. The v1 schema evolves this

### Core pipeline modules
- `core/embeddings.py` — `generate_embedding()`, `save_to_index()` for CLIP/FAISS
- `core/capture.py` — `save_snippet()` for padded PNG crops

### Project constraints
- `.planning/PROJECT.md` — "Routine auditability: JSON must be human-readable. Every step inspectable. Trust through transparency."
- `CLAUDE.md` — Code quality rules, type hints, cross-platform awareness

### Prior phase context
- `.planning/phases/04-record-flow/04-CONTEXT.md` — Save format decisions, storage location, start_from choice

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `mapper/export.py:calculate_checksum()`: SHA256 over stable JSON — reuse exact pattern for routine checksum
- `mapper/export.py:save_skill_to_file()` / `load_skill_from_file()`: File I/O + integrity check pattern — adapt for routine load/save
- `mapper/graph.py:OCSDGraph.to_dict()` / `from_dict()`: Graph serialization — use directly, no wrapper
- `recorder/record_session.py:_step_to_json()`: Current step serialization — evolve to v1 schema
- `recorder/record_session.py:_save_snippets_and_embeddings()`: Snippet + CLIP pipeline — adapt for node_id naming

### Established Patterns
- `$schema` version tag for format identification (used in both skill-v1 and routine-v0)
- SHA256 checksum for integrity verification (mapper/export.py)
- `pathlib.Path` for all file operations (CLAUDE.md requirement)
- `~/.ocsd/routines/{name}/` directory structure (already created by Phase 4)

### Integration Points
- `recorder/record_session.py:_save_routine()` — Currently saves v0; needs to save v1 instead
- `recorder/record_session.py:_step_to_json()` — Currently creates v0 step format; needs v1 step schema
- `core/config.py` — Global YAML config where VLM fallback settings would live

</code_context>

<specifics>
## Specific Ideas

- OCSD itself IS "the skill" — it is THE skill of an OpenClaw bot. All routines are assets of that one skill. Skills (skill.md) orchestrate workflows; routines are recorded sequences within OCSD
- Node ID based file naming (snippets + embeddings) sets up for V2 UI GPS where elements become shared graph nodes across routines
- Elements will eventually be GPS nodes connected by edges, relating to each other (dropdown → menu items, toolbar → buttons). The format should not prevent this evolution
- Future vision: routines as a "screen CLI" that maps agent commands to UI actions with minimal reasoning — simple JSON strings per page, massively reducing agent context usage
- Auto-detect foreground app + platform + theme during recording so the user never thinks about metadata
- Signature placeholder in the format for V2 Hub trust scoring

</specifics>

<deferred>
## Deferred Ideas

- Routine-to-skill bridge/converter — not needed; they're different abstraction levels
- FAISS index per routine — Phase 7 can build one at load time if needed for replay speed
- Per-routine VLM fallback config — global config is sufficient for V1
- SQLite routine catalog — V2, when routine count justifies it
- Embedding deduplication across routines — V2, when UI GPS shares elements
- "Screen CLI" mode (routines as agent command maps) — V2+, requires full app page mapping

</deferred>

---

*Phase: 05-routine-file-format*
*Context gathered: 2026-03-18*
