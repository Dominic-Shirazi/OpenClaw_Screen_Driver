# Phase 3: Overlay HUD Panels - Research

**Researched:** 2026-03-16
**Domain:** PyQt6 QGraphicsScene interactive panels, frosted glass effects, form input embedding, drag handling
**Confidence:** HIGH

## Summary

Phase 3 builds two interactive HUD panels -- a tag dialog and a floating toolbar -- as QGraphicsObject items inside the existing QGraphicsScene. The primary technical challenge is embedding interactive form widgets (QLineEdit, QComboBox) into a QGraphicsScene while maintaining the overlay's transparent, always-on-top window behavior. The recommended approach uses QGraphicsProxyWidget to embed standard Qt widgets into the scene, wrapped inside custom QGraphicsObject containers that handle the frosted-glass rendering, glow animations, and positioning logic.

The frosted glass effect should use a semi-transparent dark fill with subtle blur simulation rather than real-time QGraphicsBlurEffect (which is computationally expensive and causes issues on transparent overlay windows). This aligns with the Phase 1 blocker note: "fake semi-transparent is V1 recommendation." The existing AnimationClock, ShimmerLayer avoidance rects API, and QGraphicsObject patterns from Phase 2 provide the animation and integration foundation.

The typewriter effect is a straightforward timer-driven character-by-character text insertion into QLineEdit fields, synchronized with card border glow flashes. The per-field blinking cursor and varied speeds create the "chaotic-but-purposeful" feel. The toolbar is a simpler horizontal pill with icon buttons, supporting drag via mousePressEvent/mouseMoveEvent overrides on QGraphicsObject.

**Primary recommendation:** Use QGraphicsProxyWidget to embed real Qt form widgets (QLineEdit, QComboBox, QPushButton) inside custom QGraphicsObject containers. Paint the frosted-glass background and glow effects in the container's paint() method. Register both panels as shimmer avoidance rects.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Tag dialog positioning**: Fades in near the captured element but always fully on-screen with breathing room from edges. Smart positioning using common UI conventions (prefer below/right, flip if near edge)
- **Frosted glass**: Medium frost -- semi-transparent dark panel, vaguely see what's behind. Not opaque, not see-through
- **Size**: Medium panel (~400x280px), resizes smoothly when conditional fields appear/disappear
- **Shape**: Rounded corners (14-16px radius), card-like feel
- **Fonts**: Thin white fonts on dark frost
- **Accents**: Subtle green glow -- consistent with "ready" state color language
- **Card border glow**: Same "lights shining from behind" design as screen border but at card scale -- tiny green light sources around the card perimeter shining outward parallel to the screen
- **Card glow animation**: Slow shimmer sweep at idle. During VLM typewriter fill, brightness pulses/flashes synchronized with keystrokes
- **Field styling**: Subtle dark recessed input wells for each field
- **Dismiss**: Smooth opacity fade out
- **Confirm/Cancel**: Green "Confirm" and dim "Cancel" buttons at bottom, plus Enter to confirm and Esc to cancel
- **Shimmer interaction**: Tag dialog registers as avoidance rect
- **Typewriter**: All VLM-filled fields typewrite simultaneously at different speeds (~30-50ms per character). Per-field blinking green cursors
- **Edit mode**: Pre-filled fields appear instantly with no typewriter animation
- **VLM waiting/failure**: Card border glow pulses/breathes while waiting. After timeout, fields become editable with what's filled so far
- **Action dropdown**: Two separate dropdowns (action type + element type). VLM auto-fills both but remain editable
- **Conditional fields**: Action type selection reveals/hides additional fields with smooth slide-and-fade transitions
- **Floating toolbar**: Horizontal pill bar (~250x40px), starts top-right, draggable, same frosted glass + backlit green glow
- **Context-sensitive buttons**: Recording mode [Pause][Undo Last], Tag dialog open [Confirm][Cancel][Skip], Dry-run mode [Run Step][Skip Step][Finish]
- **Hide behavior**: Both panels fully hide before any screenshot capture

### Claude's Discretion
- Exact frosted glass implementation approach (QGraphicsBlurEffect, pre-blurred snapshot, or semi-transparent dark fill)
- Smart positioning algorithm for tag dialog (prefer below-right, flip logic)
- Typewriter timing variance algorithm (how to vary per-field speeds)
- Toolbar icon design (text labels, icons, or both)
- How conditional fields animate (exact easing, duration)
- Keyboard navigation within the tag dialog

### Deferred Ideas (OUT OF SCOPE)
- "Look here" action (send crop + full screen to VLM with context prompt) -- maps to existing read/snip_and_search action types, full implementation in Phase 6
- VLM prompting best practices for element analysis -- research during Phase 4 or Phase 6

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| HUD-01 | Tag dialog -- dark frosted-glass panel slides up from element with thin fonts and glow accents | Frosted glass via semi-transparent dark fill + card border glow via radial gradients (same technique as ShimmerLayer). QGraphicsProxyWidget for form fields |
| HUD-02 | Tag dialog typewriter fill -- all VLM fields animate simultaneously on first capture | AnimationClock tick callback drives per-field character insertion at varied rates (30-50ms/char). Green cursor QGraphicsRectItem blinks at insertion point |
| HUD-03 | Tag dialog pre-fills without animation when editing existing routine steps | Conditional path in show_tag_dialog(): if edit_mode, set field values directly without typewriter registration |
| HUD-04 | Tag dialog includes action dropdown with conditional fields per action type | QComboBox via QGraphicsProxyWidget, currentIndexChanged signal triggers field show/hide with QPropertyAnimation on opacity + height |
| HUD-05 | Floating toolbar -- persistent, draggable, context-sensitive | QGraphicsObject with ItemIsMovable flag for drag, context enum for button swap, QGraphicsProxyWidget for buttons |
| HUD-06 | Floating toolbar hides during screenshot captures | Register in OverlayView.hide_for_capture() / show_after_capture() alongside existing layers |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyQt6 | 6.6+ | QGraphicsScene, QGraphicsObject, QGraphicsProxyWidget | Already used for overlay |
| PyQt6.QtCore | 6.6+ | QPropertyAnimation, QTimer, QRectF, pyqtSignal | Smooth transitions, geometry |
| PyQt6.QtWidgets | 6.6+ | QLineEdit, QComboBox, QPushButton, QCheckBox | Standard form widgets embedded via proxy |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| AnimationClock (internal) | Phase 2 | Frame-independent tick for typewriter and glow | All timed animations |
| ShimmerLayer (internal) | Phase 2 | set_avoidance_rects() API | HUD panels register their rects |
| OverlayState (internal) | Phase 1 | STATE_COLORS for green/red | Consistent color language |
| ElementType (internal) | Existing | ~25 element types in 4 categories | Element type dropdown |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| QGraphicsProxyWidget for forms | Pure QPainter text rendering | Proxy gives real keyboard focus, tab order, clipboard for free; custom painting would require reimplementing all input handling |
| Semi-transparent dark fill for frost | QGraphicsBlurEffect on captured background | Real blur is expensive per frame, causes rendering artifacts on transparent windows, and the "fake" frost looks good enough per Phase 1 decision |
| QPropertyAnimation for panel resize | Manual interpolation in tick() | QPropertyAnimation handles easing and cleanup automatically; simpler code |

## Architecture Patterns

### Recommended Project Structure
```
recorder/overlay/
    tag_dialog_panel.py     # TagDialogPanel(QGraphicsObject) -- frosted card + form
    toolbar_panel.py        # ToolbarPanel(QGraphicsObject) -- pill bar + buttons
    card_glow.py            # CardGlowMixin or helper for card-scale border glow
    typewriter_engine.py    # TypewriterEngine -- manages per-field character fill
    hud_common.py           # Shared constants: colors, font sizes, glow params
```

### Pattern 1: QGraphicsProxyWidget Inside QGraphicsObject Container
**What:** A QGraphicsObject paints the frosted background and glow, then positions child QGraphicsProxyWidgets for each form field.
**When to use:** For any HUD panel that needs both custom painting AND interactive widgets.
**Example:**
```python
# Source: Qt6 official docs + existing project patterns
class TagDialogPanel(QGraphicsObject):
    """Frosted-glass tag dialog rendered in QGraphicsScene."""

    confirmed = pyqtSignal(dict)   # Emitted with form data
    cancelled = pyqtSignal()

    def __init__(self, parent: QGraphicsObject | None = None) -> None:
        super().__init__(parent)
        self.setZValue(100)  # Above all animation layers
        self._width = 400.0
        self._height = 280.0
        self._corner_radius = 15.0
        self._opacity = 0.0  # For fade-in animation

        # Embed real Qt widgets via QGraphicsProxyWidget
        self._label_edit = QLineEdit()
        self._label_edit.setStyleSheet(
            "background: rgba(20,20,30,200); color: white; "
            "border: 1px solid rgba(50,200,50,80); border-radius: 4px; "
            "padding: 4px; font-weight: 300;"
        )
        self._label_proxy = QGraphicsProxyWidget(self)
        self._label_proxy.setWidget(self._label_edit)
        self._label_proxy.setPos(20, 60)

    def paint(self, painter, option, widget=None):
        """Paint frosted background + card border glow."""
        painter.save()
        painter.setOpacity(self._opacity)

        # Frosted background
        path = QPainterPath()
        path.addRoundedRect(
            QRectF(0, 0, self._width, self._height),
            self._corner_radius, self._corner_radius,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 15, 25, 200))  # Dark frost
        painter.drawPath(path)

        # Card border glow (radial gradients around perimeter)
        self._paint_card_glow(painter)

        painter.restore()
```

### Pattern 2: Typewriter Engine with AnimationClock
**What:** Register a tick callback that advances character positions across multiple fields simultaneously at different speeds.
**When to use:** First-capture VLM fill mode (HUD-02).
**Example:**
```python
class TypewriterEngine:
    """Drives simultaneous typewriter fill across multiple fields."""

    def __init__(self, clock: AnimationClock) -> None:
        self._clock = clock
        self._fields: list[_FieldState] = []
        self._active = False

    def start(self, fields: list[tuple[QLineEdit, str, float]]) -> None:
        """Begin typewriter fill.

        Args:
            fields: List of (widget, target_text, chars_per_second).
        """
        self._fields = [
            _FieldState(widget=w, target=t, cps=c, pos=0, accum=0.0)
            for w, t, c in fields
        ]
        self._active = True
        self._clock.register(self.tick)

    def tick(self, dt: float) -> None:
        """Advance each field by dt seconds."""
        all_done = True
        for f in self._fields:
            if f.pos >= len(f.target):
                continue
            all_done = False
            f.accum += dt
            chars_to_add = int(f.accum * f.cps)
            if chars_to_add > 0:
                new_pos = min(f.pos + chars_to_add, len(f.target))
                f.widget.setText(f.target[:new_pos])
                f.pos = new_pos
                f.accum -= chars_to_add / f.cps
        if all_done:
            self._finish()
```

### Pattern 3: Draggable QGraphicsObject (Toolbar)
**What:** Use Qt's built-in ItemIsMovable flag for drag support.
**When to use:** Floating toolbar (HUD-05).
**Example:**
```python
class ToolbarPanel(QGraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(110)  # Above tag dialog

    def itemChange(self, change, value):
        """Clamp position to screen bounds after drag."""
        if change == QGraphicsObject.GraphicsItemChange.ItemPositionChange:
            # Clamp to screen bounds
            rect = self.scene().sceneRect() if self.scene() else QRectF()
            new_pos = value
            # ... clamp logic ...
            return new_pos
        return super().itemChange(change, value)
```

### Pattern 4: Context-Sensitive Button Swap (Toolbar)
**What:** An enum-driven mode that swaps visible button proxies with fade transitions.
**When to use:** Toolbar state changes (recording vs tag-open vs dry-run).
**Example:**
```python
class ToolbarMode(Enum):
    RECORDING = auto()
    TAG_OPEN = auto()
    DRY_RUN = auto()

class ToolbarPanel(QGraphicsObject):
    def set_mode(self, mode: ToolbarMode) -> None:
        """Swap visible buttons with fade transition."""
        # Fade out current buttons
        for proxy in self._current_buttons:
            anim = QPropertyAnimation(proxy, b"opacity")
            anim.setDuration(150)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.start()
        # Fade in new buttons
        new_buttons = self._button_sets[mode]
        for proxy in new_buttons:
            proxy.show()
            anim = QPropertyAnimation(proxy, b"opacity")
            anim.setDuration(150)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.start()
        self._current_buttons = new_buttons
```

### Anti-Patterns to Avoid
- **Building custom text input from scratch in paint()**: You lose keyboard focus, clipboard, IME support, accessibility. Use QGraphicsProxyWidget with real QLineEdit/QComboBox instead.
- **Using QGraphicsBlurEffect for frosted glass on overlay**: Causes rendering artifacts on transparent windows, is GPU-expensive per frame, and the visual difference from a dark semi-transparent fill is minimal on a fullscreen overlay.
- **Calling self.update() inside paint()**: Creates infinite repaint loops. Only call update() from tick() callbacks or event handlers. This pattern is already established in Phase 2.
- **Using QDialog for the tag dialog**: The existing TagDialog is a QDialog -- rebuilding as a QGraphicsObject in the scene avoids z-order fights with the overlay's always-on-top window and enables smooth animations.
- **Forgetting to register avoidance rects**: Both panels MUST call shimmer.set_avoidance_rects() or the border glow will bleed through them.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Text input fields | Custom QPainter text rendering with keyboard handling | QLineEdit via QGraphicsProxyWidget | Keyboard focus, selection, clipboard, IME, accessibility are all built in |
| Dropdown menus | Custom painted dropdown with popup list | QComboBox via QGraphicsProxyWidget | Popup positioning, keyboard navigation, model/view all handled |
| Drag-to-move | Manual mouse tracking and position math | QGraphicsItem.ItemIsMovable flag | Qt handles mouse grab, drag, and release correctly |
| Smooth property transitions | Manual interpolation in tick() | QPropertyAnimation | Automatic easing, cleanup, sequential chaining via QSequentialAnimationGroup |
| Widget opacity fade | Manual alpha tracking per frame | QGraphicsOpacityEffect + QPropertyAnimation | Handles proxy widget visibility correctly |

**Key insight:** QGraphicsProxyWidget solves the hardest problem in this phase -- getting real, interactive form widgets inside a QGraphicsScene that's rendering a transparent overlay. Fighting Qt's widget system by reimplementing text input is a week of work for an inferior result.

## Common Pitfalls

### Pitfall 1: QGraphicsProxyWidget Focus Issues on Transparent Windows
**What goes wrong:** Embedded widgets (QLineEdit, QComboBox) don't receive keyboard focus because the overlay window has WA_ShowWithoutActivating and is configured for click-through.
**Why it happens:** The overlay is designed to not steal focus from the underlying application. When showing the tag dialog, the overlay must temporarily become focusable.
**How to avoid:** When tag dialog opens, disable click-through (same as RECORDING state), set the proxy widget's focus policy to StrongFocus, and call proxy.setFocus() explicitly. When dialog closes, restore click-through state.
**Warning signs:** Typing doesn't appear in QLineEdit fields, QComboBox doesn't open on click.

### Pitfall 2: QComboBox Popup Doesn't Appear or Appears Behind Overlay
**What goes wrong:** QComboBox popup list renders as a separate top-level window that may appear behind the always-on-top overlay.
**Why it happens:** Qt creates a new QGraphicsProxyWidget for the popup automatically, but on Windows with WS_EX_TOPMOST the popup window may not inherit the z-order.
**How to avoid:** Style the QComboBox to use a non-native popup (setMaxVisibleItems and setStyleSheet to force Qt rendering). Test on Windows early. If popups still fail, consider a custom dropdown painted in the QGraphicsObject.
**Warning signs:** Clicking a dropdown shows nothing, or the popup flickers briefly then disappears.

### Pitfall 3: Panel Resize Animation Causes Proxy Widget Clipping
**What goes wrong:** When the tag dialog resizes to show/hide conditional fields, embedded QGraphicsProxyWidgets get clipped or mispositioned.
**Why it happens:** QGraphicsProxyWidget geometry is set at creation time and doesn't automatically follow parent resize.
**How to avoid:** On resize, explicitly reposition all child proxy widgets. Use a layout helper method (_relayout_fields) called after every height change. Set proxy widget geometry relative to the panel's origin.
**Warning signs:** Fields overlap, fields extend beyond the rounded rectangle background, or fields disappear after action type change.

### Pitfall 4: Typewriter Animation Conflicts with User Editing
**What goes wrong:** The typewriter engine keeps overwriting text while the user tries to edit a field.
**Why it happens:** The typewriter tick callback doesn't know the user has started typing.
**How to avoid:** On any user keystroke in a typewriter-active field, immediately stop the typewriter for that field (unregister or mark as done). Connect QLineEdit.textEdited signal (not textChanged, which fires for programmatic changes too).
**Warning signs:** User types and text keeps getting replaced, or cursor jumps back.

### Pitfall 5: hide_for_capture() Timing with HUD Panels
**What goes wrong:** Tag dialog or toolbar partially appears in screenshots.
**Why it happens:** HUD panels with opacity animations may not be fully hidden when hide_for_capture() is called.
**How to avoid:** In hide_for_capture(), forcibly set all HUD panel opacity to 0 and call setVisible(False) before the existing hide() call. Don't rely on animation completing.
**Warning signs:** Ghost of the tag dialog appears in captured screenshots.

### Pitfall 6: Avoidance Rect Updates Lag Behind Panel Position
**What goes wrong:** The shimmer glow bleeds into the toolbar area after it's been dragged to a new position.
**Why it happens:** Avoidance rects are only updated when explicitly called, not automatically when the panel moves.
**How to avoid:** Override itemChange() on the toolbar to call _update_avoidance_rects() whenever ItemPositionHasChanged fires. For the tag dialog, update rects after show, resize, and move.
**Warning signs:** Green glow visible underneath HUD panels after position change.

## Code Examples

### Frosted Glass Background Painting
```python
# Source: Established project pattern (ShimmerLayer radial gradients) + Qt6 docs
def _paint_frost_background(self, painter: QPainter) -> None:
    """Paint semi-transparent dark frosted background with rounded corners."""
    path = QPainterPath()
    rect = QRectF(0, 0, self._width, self._height)
    path.addRoundedRect(rect, self._corner_radius, self._corner_radius)

    # Base frost fill
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(15, 15, 25, 200))  # Dark, ~78% opaque
    painter.drawPath(path)

    # Subtle inner highlight at top (simulates frosted glass gradient)
    highlight = QLinearGradient(
        QPointF(0, 0), QPointF(0, self._height * 0.3),
    )
    highlight.setColorAt(0.0, QColor(255, 255, 255, 12))
    highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
    painter.setBrush(highlight)
    painter.drawPath(path)
```

### Card Border Glow (Lights Behind Card)
```python
# Source: ShimmerLayer._build_lights() pattern adapted to card scale
def _paint_card_glow(self, painter: QPainter, brightness: float = 1.0) -> None:
    """Paint green glow lights around card perimeter, shining outward."""
    painter.save()
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

    green = QColor(50, 200, 50)
    rect = QRectF(0, 0, self._width, self._height)
    perimeter = 2 * (rect.width() + rect.height())
    light_count = 24  # More lights at card scale

    for i in range(light_count):
        frac = i / light_count
        d = frac * perimeter

        # Position on card edge + push center outward
        ex, ey, nx, ny = self._edge_point(d, rect)
        cx = ex + nx * 15.0  # 15px outside card edge
        cy = ey + ny * 15.0

        # Sweep brightness (same algorithm as ShimmerLayer)
        sweep_b = self._sweep_brightness(frac) * brightness
        if sweep_b < 0.01:
            continue

        r = 30.0  # Smaller radius than screen shimmer
        gradient = QRadialGradient(QPointF(cx, cy), r)
        core = QColor(green)
        core.setAlphaF(min(sweep_b * 0.5, 1.0))
        gradient.setColorAt(0.0, core)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    painter.restore()
```

### Embedding Form Fields via QGraphicsProxyWidget
```python
# Source: Qt6 official docs (QGraphicsProxyWidget)
def _create_field(
    self, widget: QWidget, x: float, y: float, width: float,
) -> QGraphicsProxyWidget:
    """Embed a QWidget in the scene as a child of this panel.

    Args:
        widget: The Qt widget to embed (QLineEdit, QComboBox, etc.).
        x: X position relative to panel origin.
        y: Y position relative to panel origin.
        width: Desired width.

    Returns:
        The QGraphicsProxyWidget wrapping the embedded widget.
    """
    widget.setFixedWidth(int(width))
    widget.setStyleSheet(
        "background: rgba(20, 20, 35, 220); "
        "color: rgba(240, 240, 245, 230); "
        "border: 1px solid rgba(50, 200, 50, 60); "
        "border-radius: 4px; "
        "padding: 4px 6px; "
        "font-weight: 300; "
        "font-size: 13px;"
    )
    proxy = QGraphicsProxyWidget(self)
    proxy.setWidget(widget)
    proxy.setPos(x, y)
    return proxy
```

### Smart Positioning Algorithm
```python
# Source: Existing TagDialog._clamp_to_screen() pattern, adapted for scene coords
def _compute_position(
    self,
    element_x: float,
    element_y: float,
    element_w: float,
    element_h: float,
    screen_w: float,
    screen_h: float,
) -> tuple[float, float]:
    """Compute tag dialog position near element, fully on-screen.

    Prefers below-right of the element. Flips if near screen edge.
    Always maintains breathing room from edges.

    Returns:
        (x, y) position in scene coordinates.
    """
    margin = 20.0  # Breathing room from screen edges
    gap = 12.0     # Gap between element and dialog

    # Try below-right first
    x = element_x + element_w + gap
    y = element_y

    # Flip horizontal if off right edge
    if x + self._width + margin > screen_w:
        x = element_x - self._width - gap

    # Flip vertical if off bottom edge
    if y + self._height + margin > screen_h:
        y = screen_h - self._height - margin

    # Clamp to screen bounds
    x = max(margin, min(x, screen_w - self._width - margin))
    y = max(margin, min(y, screen_h - self._height - margin))

    return x, y
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| QDialog for tag dialog | QGraphicsObject in scene | Phase 3 (now) | Eliminates z-order fights, enables smooth animations, consistent with overlay architecture |
| QGraphicsBlurEffect for frost | Semi-transparent dark fill + subtle gradient | Phase 1 decision | Avoids GPU cost and rendering artifacts on transparent windows |
| Manual drag tracking | ItemIsMovable flag + itemChange() | Qt standard | Less code, correct behavior with mouse grab/release |

**Deprecated/outdated:**
- `recorder/dialog.py` TagDialog (QDialog): Reference only for field layout and element type grouping. Being rebuilt as QGraphicsObject panel.

## Open Questions

1. **QComboBox popup z-order on Windows**
   - What we know: Qt creates popup as separate native window. On transparent overlay with WS_EX_TOPMOST, popup may not appear correctly.
   - What's unclear: Whether the popup will reliably appear in front on Windows 11 DWM.
   - Recommendation: Test early in Wave 0. If popup fails, implement custom dropdown as QPainter-drawn list within the panel, or use QListWidget in a second QGraphicsProxyWidget that appears/hides.

2. **QPropertyAnimation on QGraphicsProxyWidget opacity**
   - What we know: QGraphicsProxyWidget inherits QGraphicsObject and supports QPropertyAnimation on "opacity" property.
   - What's unclear: Whether opacity animation properly affects the embedded widget's rendering on all platforms.
   - Recommendation: Test with a simple fade-in during Wave 0. Fallback: manually set opacity in tick() callback.

3. **Keyboard focus handoff between tag dialog and overlay**
   - What we know: When tag dialog opens, click-through must be disabled so embedded widgets receive input. When it closes, click-through must be re-enabled.
   - What's unclear: Whether there's a clean way to scope focus to just the tag dialog area while keeping the rest of the overlay interactive.
   - Recommendation: When tag dialog is showing, the overlay is already in a "tag dialog open" state. Click-through is OFF (same as RECORDING). The OverlayView handles events normally. Tab order within the proxy widgets handles internal navigation.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml (Ruff) + conftest.py (pre-mocks) |
| Quick run command | `python -m pytest tests/test_tag_dialog_panel.py tests/test_toolbar_panel.py tests/test_typewriter_engine.py -x -q` |
| Full suite command | `python -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HUD-01 | Tag dialog renders frosted panel with fields, glow, rounded corners | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestInstantiation -x` | No -- Wave 0 |
| HUD-01 | Tag dialog positions near element, fully on-screen | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestPositioning -x` | No -- Wave 0 |
| HUD-01 | Tag dialog registers as shimmer avoidance rect | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestAvoidanceRect -x` | No -- Wave 0 |
| HUD-02 | Typewriter fills all fields simultaneously at different speeds | unit | `python -m pytest tests/test_typewriter_engine.py::TestSimultaneousFill -x` | No -- Wave 0 |
| HUD-02 | Card glow flashes synchronize with typewriter keystrokes | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestGlowSync -x` | No -- Wave 0 |
| HUD-03 | Edit mode pre-fills without typewriter animation | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestEditMode -x` | No -- Wave 0 |
| HUD-04 | Action dropdown shows conditional fields per action type | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestConditionalFields -x` | No -- Wave 0 |
| HUD-04 | Panel resizes smoothly when conditional fields appear/disappear | unit | `python -m pytest tests/test_tag_dialog_panel.py::TestPanelResize -x` | No -- Wave 0 |
| HUD-05 | Toolbar renders as draggable pill with context buttons | unit | `python -m pytest tests/test_toolbar_panel.py::TestInstantiation -x` | No -- Wave 0 |
| HUD-05 | Toolbar switches buttons for recording/tag-open/dry-run modes | unit | `python -m pytest tests/test_toolbar_panel.py::TestModeSwitch -x` | No -- Wave 0 |
| HUD-06 | Toolbar hides during screenshot capture | unit | `python -m pytest tests/test_toolbar_panel.py::TestHideForCapture -x` | No -- Wave 0 |
| HUD-ALL | Integration: both panels in scene, avoidance rects, hide/show | integration | `python -m pytest tests/test_hud_integration.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_tag_dialog_panel.py tests/test_toolbar_panel.py tests/test_typewriter_engine.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_tag_dialog_panel.py` -- covers HUD-01, HUD-02, HUD-03, HUD-04
- [ ] `tests/test_toolbar_panel.py` -- covers HUD-05, HUD-06
- [ ] `tests/test_typewriter_engine.py` -- covers HUD-02
- [ ] `tests/test_hud_integration.py` -- covers cross-panel integration

## Sources

### Primary (HIGH confidence)
- [QGraphicsProxyWidget Qt 6.10 docs](https://doc.qt.io/qt-6/qgraphicsproxywidget.html) -- embedding widgets in QGraphicsScene, state synchronization, popup handling
- [QGraphicsItem Qt 6.10 docs](https://doc.qt.io/qt-6/qgraphicsitem.html) -- ItemIsMovable, itemChange, mouse event handling
- [QGraphicsBlurEffect Qt 6.10 docs](https://doc.qt.io/qt-6/qgraphicsblureffect.html) -- blur capabilities and limitations
- [Graphics View Framework Qt 6.10](https://doc.qt.io/qt-6/graphicsview.html) -- event flow, focus management
- Existing codebase: `recorder/overlay/shimmer_layer.py`, `recorder/overlay/scan_layer.py`, `recorder/overlay/view.py` -- established patterns for QGraphicsObject layers, tick-based animation, avoidance rects

### Secondary (MEDIUM confidence)
- [Qt Forum: QGraphicsProxyWidget focus issues](https://forum.qt.io/topic/25531/qgraphicsproxywidget-does-not-get-focus-when-using-setfocus) -- focus workarounds
- [QtCentre: Constructing custom widgets in QGraphicsScene](https://www.qtcentre.org/threads/70522-Constructing-a-custom-widget-in-QGraphicsScene) -- custom widget patterns
- [Qt Forum: QGraphicsObject mousePressEvent](https://forum.qt.io/topic/39484/solved-qgraphicsobject-mousepressevent) -- mouse event handling patterns

### Tertiary (LOW confidence)
- None -- all claims verified against Qt official docs or existing codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- using existing PyQt6 stack, established project patterns, Qt official documentation
- Architecture: HIGH -- QGraphicsProxyWidget approach is well-documented in Qt; project patterns from Phase 1/2 are proven
- Pitfalls: HIGH -- focus issues, popup z-order, and timing gaps are well-known Qt pain points with documented workarounds
- Typewriter engine: HIGH -- straightforward timer-driven character insertion, identical pattern to existing AnimationClock tick callbacks

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (stable -- PyQt6 API unlikely to change)
