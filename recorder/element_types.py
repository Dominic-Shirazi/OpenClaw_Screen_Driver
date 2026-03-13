"""OCSD Element Type Library.

Defines the standard set of interactive, static, and structural elements
recognized by the driver. Types fall into three categories:

Interactive: Elements the runner clicks/types/drags (textbox, button, etc.)
Structural: Screen regions for visual grounding and YOLO-E training
            (region_*, landmark). Not clicked — used to locate other elements.
Meta:        Graph flow control (destination, branch_point, fingerprint).
"""

from enum import Enum


class ElementType(Enum):
    """Enumeration of recognizable element types in the UI."""

    # --- Interactive elements (runner acts on these) ---
    TEXTBOX        = "textbox"         # input field, user types into
    BUTTON         = "button"          # click, stays on page
    BUTTON_NAV     = "button_nav"      # click, navigates to new page/state
    TOGGLE         = "toggle"          # on/off, checkbox, radio
    TAB            = "tab"             # tab navigation within page
    DROPDOWN       = "dropdown"        # expands list of options
    SCROLLBAR      = "scrollbar"       # scroll target, vertical or horizontal
    LINK           = "link"            # hyperlink (text or image)
    ICON           = "icon"            # clickable icon/glyph
    DRAG_SOURCE    = "drag_source"     # drag starts here
    DRAG_TARGET    = "drag_target"     # drag ends here

    # --- Structural regions (visual grounding / YOLO-E context) ---
    REGION_CHROME  = "region_chrome"   # browser/app chrome (title bar, tabs, nav buttons)
    REGION_MENU    = "region_menu"     # menu bar or hamburger menu area
    REGION_SIDEBAR = "region_sidebar"  # sidebar / vertical nav panel
    REGION_CONTENT = "region_content"  # main content area of the page
    REGION_FORM    = "region_form"     # form area (login, signup, settings group)
    REGION_HEADER  = "region_header"   # page/section header
    REGION_FOOTER  = "region_footer"   # page/section footer
    REGION_TOOLBAR = "region_toolbar"  # toolbar (editor controls, action bar)
    REGION_MODAL   = "region_modal"    # popup/dialog container (bounding region)
    REGION_CUSTOM  = "region_custom"   # user-defined region (describe in notes)
    LANDMARK       = "landmark"        # distinctive visual anchor (logo, unique icon)

    # --- Static / read-only elements ---
    IMAGE          = "image"           # non-interactive image (may contain text)
    READ_HERE      = "read_here"       # data extraction point: read this value
    NOTIFICATION   = "notification"    # toast, alert, status message
    MODAL          = "modal"           # popup/overlay (legacy — prefer region_modal)

    # --- Meta / graph control ---
    DESTINATION    = "destination"     # success state node (page loaded, result shown)
    BRANCH_POINT   = "branch_point"    # conditional fork (if/else in the flow)
    FINGERPRINT    = "fingerprint"     # app identity checkpoint (YOLO-E verifies state)
    UNKNOWN        = "unknown"         # fallback, requires human tag