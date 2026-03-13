# OCSD Detection Pipeline Redesign

**Date:** 2026-03-13
**Status:** Draft
**Scope:** Recording detection, VLM integration, replay locate cascade, annotation reuse

## Problem Statement

The current recording pipeline uses VLM (via LiteLLM) to detect UI elements by asking
it to return bounding boxes in JSON. VLM is poor at pixel-level localization — it
returns few elements with bad coordinates that get silently filtered. Meanwhile, YOLOE
is only used for post-hoc bbox refinement after the user manually draws a box.

The result: users see nothing highlighted on screen and must manually draw every element.
Smart detection is architecturally present but functionally broken.

## Design Principles

1. **Each tool does what it's good at.** YOLOE finds boxes. VLM understands content. Neither does the other's job.
2. **VLM never gives bounding boxes.** It answers questions about cropped images and diagnoses failures.
3. **Local-first, progressively wider.** Search near the expected position first, expand scope only on failure.
4. **Don't re-boil the ocean.** If a page is already annotated, reuse those annotations.

## Architecture: 3-Layer Detection Model

### Layer 1: Areas (structural regions)

YOLOE text-prompt detection with area-level classes:

```
taskbar, browser_chrome, sidebar, dialog, content_area,
menu_bar, toolbar, system_tray, status_bar, navigation_panel
```

- **Global areas** (taskbar, system_tray) are trained once and persisted in
  `assets/global_areas.json`. These are OS-level and universal.
- **Per-skill areas** are detected per recording session and stored in the skill
  graph as `region_*` nodes.
- Areas produce large bboxes that contain elements.

### Layer 2: Elements (interactive things)

YOLOE text-prompt detection with element-level classes:

```
button, icon, textbox, checkbox, radio_button, dropdown,
scrollbar, tab, link, toggle, slider, menu_item
```

- Run per-area: crop each area bbox, run YOLOE element detection within the crop.
  This avoids overwhelming the detector with 100+ elements on a full screen.
- Each element gets a tight bbox translated back to screen-absolute coordinates.
- If single-pass (all classes at once) benchmarks faster than two-pass
  (areas then elements), use single-pass instead. Implementation must benchmark both.

### Layer 3: Identity (naming/typing via VLM)

VLM called on element crops (+ 30% buffer) for labeling:

- **Annotate mode:** batch all crops through VLM
- **Workflow mode:** VLM called on-demand only when user clicks an element

VLM prompt is engineered to return structured JSON matching TagDialog fields exactly:

```json
{
  "element_type": "button",
  "label": "Submit Order",
  "area_type": "content_area",
  "notes": "Primary action button, blue, bottom-right of form",
  "ocr_text": "Submit",
  "confidence": 0.85
}
```

The prompt explicitly lists all valid `element_type` and `area_type` values so the
VLM output maps directly to UI dropdowns without translation.

### Text Embedding Caching

`model.model.get_text_pe(class_names)` loads CLIP on first call (~300MB). Embeddings
are cached at app startup via `model.model.set_classes(names, embeddings)`. After
that, YOLOE runs inference without CLIP loaded — fast and lightweight.

Two embedding sets are maintained:
- `_area_embeddings`: for area detection classes
- `_element_embeddings`: for element detection classes

These are swapped in/out via `set_classes()` before each detection pass.

## Recording UX

### Annotate/Diagram Mode (map a whole screen)

1. Screenshot taken
2. YOLOE pass 1: detect areas → large bboxes drawn (muted, dashed borders)
3. YOLOE pass 2: detect elements per area crop → tight bboxes (bright, solid)
4. VLM batch: label all element crops → structured JSON responses
5. Auto-iterate TagDialogs: each pops up pre-populated, user confirms/edits/skips
6. User manually highlights any missed areas or elements
7. Save (Ctrl+Q)

### Workflow/Edge Mode (record a specific action sequence)

1. Overlay starts in PASSTHROUGH
2. Ctrl+R → screenshot, switch to RECORD
3. YOLOE pass 1: detect areas
4. YOLOE pass 2: detect elements per area
5. All highlights appear on overlay — NO VLM called yet (saves credits)
6. User clicks an element:
   - YOLOE refines bbox (existing refine_bbox / infer_bbox_at_point)
   - Crop + 30% buffer sent to VLM (single call)
   - TagDialog pops up pre-populated
7. Switch to PASSTHROUGH → user performs action
8. Ctrl+R → new screenshot → detect → next element...
9. Ctrl+Q → save with edges

### TagDialog Changes

New **Action Type** dropdown (all modes):
- `left_click` (default)
- `right_click`
- `double_click`
- `type_text` (shows input spec fields)
- `scroll_up` / `scroll_down`
- `hover`
- `drag_to` (future: needs second point)

Existing fields get better pre-population from VLM structured JSON.

### Annotation Reuse

At recording start, after skill name entry:

1. System scans `skills/*.json` for diagrams with overlapping name prefixes
2. If matches found, prompt: "Found existing annotations for google_search_*. Load one?"
3. If user picks existing diagram:
   - Load areas + elements as pre-existing context
   - Overlay pre-populates with those annotations (dimmed)
   - User clicks existing elements to add them to the workflow
   - Only new elements need detection + VLM labeling
4. If user picks "New": normal fresh detection flow

What gets reused: area definitions, element bboxes/labels/types, snippets, CLIP embeddings.
What doesn't: edges/workflow (skill-specific), positions treated as hints (screen may differ).

## Overlay Rendering

### Z-ordering

- **Back layer:** Area bboxes — muted colors, dashed 1px borders, semi-transparent fill
- **Front layer:** Element bboxes — bright colors per type, solid 2px borders
- **Labels:** Element name + confidence above each bbox
- **Reused annotations:** Dimmed (50% opacity) to distinguish from fresh detections

### Color Map

Existing `_TYPE_COLORS` dict in overlay.py. Areas get their own muted palette.
Elements keep the current bright color scheme.

## Replay Locate Cascade

6-stage cascade, each progressively wider:

### Stage 1: YOLOE Visual-Prompt (local)

- Use saved snippet image as visual prompt
- Search region: recorded position ± 30% buffer
- Pass if: confidence >= 0.5

### Stage 2: Area-Scoped Search

- Find parent area on current screen:
  - YOLOE text-prompt for the area type (e.g., "content_area")
  - CLIP embedding similarity against saved area snippet
  - Pass if: either >= 80% confidence, or combined average >= 70%
- Within the located area crop:
  - YOLOE visual-prompt with saved element snippet
  - CLIP embedding similarity
  - Take best match

### Stage 3: CLIP + YOLOE Full Screen

- CLIP embedding similarity scan across full screen (sliding window or grid)
- YOLOE visual-prompt on full screen (no spatial restriction)
- Compare results, take highest confidence

### Stage 4: OCR Text Match

- Full screen OCR scan
- Fuzzy match on saved `ocr_text`
- Returns region context (e.g., "found 'Submit' in menu_bar area")

### Stage 5: VLM Diagnostic (NOT locating)

- Captures full screenshot
- Asks VLM: "I'm looking for [element description]. What's on screen?
  Is this the right page? What should I try next?"
- Returns structured recovery action: retry, scroll, navigate_back, wrong_page, abort
- VLM does NOT return bounding boxes or pixel coordinates

### Stage 6: Position Fallback

- Use recorded `x_pct` / `y_pct` coordinates
- Blind click, confidence = 0.3
- Last resort

## Data Model Changes

### Node Data (additions)

```python
{
    # Existing fields preserved
    "element_type": "button",
    "label": "Submit",
    "ocr_text": "Submit",
    "relative_position": { ... },
    # New fields
    "action_type": "left_click",       # NEW: what action to perform
    "parent_area_id": "area_abc123",   # NEW: which area contains this element
    "area_type": "content_area",       # NEW: for area nodes
}
```

### Edge Data (additions)

```python
{
    "action_type": "left_click",    # Changed: now comes from TagDialog action dropdown
    "action_payload": "",
}
```

### Global Areas File

`assets/global_areas.json`:
```json
{
  "os": "windows",
  "resolution": [1920, 1080],
  "areas": [
    {"type": "taskbar", "rect": {"x": 0, "y": 1040, "w": 1920, "h": 40}},
    {"type": "system_tray", "rect": {"x": 1700, "y": 1040, "w": 220, "h": 40}}
  ]
}
```

## Implementation Waves

### Wave 1: YOLOE Detection Foundation

**Goal:** Fix "nothing highlighted" — users see detected elements on screen.

Files touched:
- `core/yoloe.py` — add `detect_all_elements()` using text-prompt mode
- `recorder/smart_detect.py` — rewire to use YOLOE instead of VLM for detection
- `recorder/overlay.py` — ensure YOLOE results render correctly
- `main.py` — update `_trigger_smart_detect` to use new pipeline

Scope:
- Single-pass YOLOE text-prompt detection (benchmark single vs two-pass)
- Text embedding caching at startup
- VLM labeling on-click only (workflow mode)
- No area hierarchy yet — flat element detection on full screen
- Existing TagDialog unchanged (no action type dropdown yet)

### Wave 2: Full Recording Pipeline

**Goal:** Area hierarchy, batch VLM labeling, auto-iterate dialogs, annotation reuse.

Files touched:
- `core/yoloe.py` — add area detection pass, per-area element detection
- `recorder/smart_detect.py` — 2-pass detection pipeline
- `recorder/dialog.py` — action type dropdown, better VLM pre-population
- `core/vision.py` — prompt-engineered VLM JSON responses
- `main.py` — annotation reuse flow, auto-iterate TagDialogs
- `recorder/overlay.py` — z-ordered area+element rendering

### Wave 3: Area-Scoped Replay

**Goal:** Smarter element location during replay using area context.

Files touched:
- `mapper/runner.py` — new 6-stage locate cascade
- `core/yoloe.py` — area-matching functions for replay
- `mapper/orchestrator.py` — integrate area-scoped recovery
- `core/vision.py` — VLM diagnostic mode (not locating)
- New: `assets/global_areas.json` — persistent OS-level areas

## Testing Strategy

Each wave includes:
- Unit tests for new functions (mocked YOLOE/VLM)
- Integration test with a local HTML page screenshot
- Manual smoke test: record a real workflow, verify overlay highlights

Wave 1 specific: benchmark YOLOE class name formulations against real screenshots
to find optimal phrasing before hardcoding.

Wave 3 specific: replay a previously-recorded skill end-to-end with the new cascade.

## Open Questions

1. **Single-pass vs two-pass:** Should be resolved empirically in Wave 1.
   If single-pass with 20 classes is fast enough, skip the area→element split
   until Wave 2.
2. **YOLOE class name tuning:** "button" vs "UI button" vs "clickable button" —
   needs empirical testing with CLIP embeddings against real UI screenshots.
3. **drag_to action:** Needs a second point. Defer to future wave or implement
   as "click element A, drag to element B" using two sequential nodes?
