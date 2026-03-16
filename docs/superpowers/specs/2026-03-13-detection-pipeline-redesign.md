# OCSD Detection Pipeline Redesign

**Date:** 2026-03-13
**Status:** Draft (rev 2 — post spec review)
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
5. **Graceful degradation.** Every AI-dependent step has a fallback. If YOLOE returns 0 elements, fall back to OCR detection. If VLM is unavailable, TagDialog still works (just without pre-population). If CLIP is unavailable, skip embedding stages.
6. **Backward compatibility.** Existing skill JSON files recorded without `action_type` or `parent_area_id` continue to work. Missing fields get sensible defaults (action_type derived from element_type, no area scoping).

## Risk: YOLOE Text-Prompt on UI Elements

YOLOE's text-prompt mode uses CLIP embeddings trained on natural images (COCO-like).
UI-specific classes ("textbox", "dropdown", "browser_chrome") are not in COCO.

**Mandatory go/no-go gate:** Wave 1 begins with a benchmark of YOLOE text-prompt
detection on 3-5 real UI screenshots (Windows desktop, browser, form page). If YOLOE
cannot reliably detect UI elements (< 50% recall on obvious elements), the fallback
plan is:

- **Fallback A:** Use YOLOE in standard detection mode (COCO classes) to find
  generic objects, then use VLM on crops to classify as UI elements. Slower but
  does not depend on text-prompt quality for UI classes.
- **Fallback B:** Use a UI-specific YOLO model (e.g., fine-tuned on UI datasets
  like RICO or WebUI) instead of YOLOE text-prompt.
- **Fallback C:** Use Windows accessibility tree (UIA) as primary detection source,
  with YOLOE visual-prompt for non-accessible elements.

The benchmark script is the first deliverable of Wave 1, before any pipeline changes.

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

**API note:** Verified working with ultralytics 8.4.21 and yoloe-26s-seg.pt on the
project machine. Text-prompt mode uses the standard `model.predict()` after
`set_classes()` — no special predictor class needed (unlike visual-prompt mode which
requires `YOLOEVPSegPredictor`). The ultralytics internal methods `get_text_pe()` and
`set_classes()` should be wrapped in a version-checked helper in case the API changes
in future ultralytics releases.

### Fallback When YOLOE Detects 0 Elements

If YOLOE returns zero detections after both passes:

- **Annotate mode:** Show overlay message "No elements detected. Draw boxes manually
  or try OCR detection." Fall back to OCR-only detection (current `smart_detect.py`
  fallback path). User can still draw boxes manually.
- **Workflow mode:** Show overlay message "No elements detected — click or draw to
  tag manually." YOLOE refinement still works on user-drawn boxes/clicks.
- Log the zero-detection event with screenshot metadata for debugging.

### Fallback When VLM Is Unavailable

If LiteLLM proxy is unreachable or returns errors:

- **Annotate mode:** YOLOE detections still appear on overlay. TagDialogs open with
  YOLOE class name as `element_type` guess and no label. User fills in manually.
  OCR text (from Tesseract) is used for label hints where available.
- **Workflow mode:** Same — click-to-tag works, VLM label is simply empty.
- Log the VLM failure once, not per-element.

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

### Backward Compatibility

Existing skill JSON files lack `action_type` and `parent_area_id`. The system handles this:

- **Missing `action_type` on node:** Derive from `element_type` using the existing
  `_action_type_for_node()` mapping in runner.py. This is the current behavior.
- **Missing `parent_area_id`:** Skip area-scoped search (Stage 2) in replay cascade.
  Fall through to Stage 3 (full-screen search).
- **Missing `action_type` on edge:** Use the derived value from the source node's
  element_type, same as current behavior.
- New `action_type` field, when present, takes precedence over the derived mapping.

### ElementType Enum Alignment

The spec's area classes must map to existing `ElementType` values:

| Spec area class | ElementType enum value |
|----------------|----------------------|
| taskbar | region_toolbar (reuse) |
| browser_chrome | region_chrome |
| sidebar | region_sidebar |
| dialog | region_modal |
| content_area | region_content |
| menu_bar | region_menu |
| toolbar | region_toolbar |
| system_tray | region_custom (with label "system_tray") |
| status_bar | region_footer (reuse) |
| navigation_panel | region_sidebar (reuse) |

No new enum values needed. YOLOE class names are distinct from ElementType values —
the mapping happens in `smart_detect.py` when building candidate dicts.

### Global Areas File

`assets/global_areas.json` — uses percentage-based coordinates for resolution independence:
```json
{
  "os": "windows",
  "areas": [
    {"type": "taskbar", "rect_pct": {"x": 0.0, "y": 0.963, "w": 1.0, "h": 0.037}},
    {"type": "system_tray", "rect_pct": {"x": 0.885, "y": 0.963, "w": 0.115, "h": 0.037}}
  ]
}
```

Percentages are resolved to pixels using the current screen resolution at runtime.

## Implementation Waves

### Wave 1: YOLOE Detection Foundation

**Goal:** Fix "nothing highlighted" — users see detected elements on screen.

**Gate:** Begins with YOLOE text-prompt benchmark. If benchmark fails (< 50% recall),
execute fallback plan before proceeding.

Files touched:
- `core/yoloe.py` — add `detect_all_elements()` using text-prompt mode
- `recorder/smart_detect.py` — rewire to use YOLOE instead of VLM for detection
- `recorder/overlay.py` — ensure YOLOE results render correctly
- `main.py` — update `_trigger_smart_detect` to use new pipeline

Scope:
- **First:** Benchmark script — test YOLOE text-prompt on 3-5 real screenshots
- Single-pass YOLOE text-prompt detection (benchmark single vs two-pass timing)
- Text embedding caching at startup
- VLM labeling on-click only (workflow mode)
- No area hierarchy yet — flat element detection on full screen
- Existing TagDialog unchanged (no action type dropdown yet)
- Fallback to OCR detection if YOLOE returns 0 elements
- Performance target: detection complete in < 2 seconds on GPU

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

## Performance Budget

| Context | Stage | Target |
|---------|-------|--------|
| Recording | YOLOE full-screen detection | < 500ms (GPU) |
| Recording | YOLOE per-area element detection | < 200ms per area |
| Recording | VLM single-crop labeling | < 3s (network dependent) |
| Recording | Total detection + render | < 2s (excluding VLM) |
| Replay | Stage 1 (YOLOE local) | < 100ms |
| Replay | Stage 2 (area-scoped) | < 500ms |
| Replay | Stage 3 (full-screen) | < 1s |
| Replay | Stage 4 (OCR) | < 500ms |
| Replay | Stage 5 (VLM diagnostic) | < 5s |

## Open Questions

1. **Single-pass vs two-pass:** Should be resolved empirically in Wave 1.
   If single-pass with 20 classes is fast enough, skip the area→element split
   until Wave 2.
2. **YOLOE class name tuning:** "button" vs "UI button" vs "clickable button" —
   needs empirical testing with CLIP embeddings against real UI screenshots.
3. **drag_to action:** Needs a second point. Defer to future wave or implement
   as "click element A, drag to element B" using two sequential nodes?
4. **Input spec visibility:** TagDialog currently shows input spec fields only for
   textbox element_type. With the new action_type dropdown, input spec should also
   appear when action_type is "type_text" regardless of element_type.
