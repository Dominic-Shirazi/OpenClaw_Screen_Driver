"""
OCSD Element Type Library.
Defines the standard set of interactive and static elements recognized by the driver.
"""

from enum import Enum

class ElementType(Enum):
    """Enumeration of recognizable element types in the UI."""
    TEXTBOX        = "textbox"         # input field, user types into
    BUTTON         = "button"          # click, stays on page
    BUTTON_NAV     = "button_nav"      # click, navigates to new page/state
    TOGGLE         = "toggle"          # on/off, checkbox, radio
    TAB            = "tab"             # tab navigation within page
    DROPDOWN       = "dropdown"        # expands list of options
    SCROLLBAR      = "scrollbar"       # scroll target, vertical or horizontal
    READ_HERE      = "read_here"       # data destination: extract this value
    DRAG_SOURCE    = "drag_source"     # drag starts here
    DRAG_TARGET    = "drag_target"     # drag ends here
    IMAGE          = "image"           # non-interactive image (may contain text)
    MODAL          = "modal"           # popup/overlay container
    NOTIFICATION   = "notification"    # toast, alert, status message
    DESTINATION    = "destination"     # success state node (page loaded, result shown)
    BRANCH_POINT   = "branch_point"   # conditional fork (if/else in the flow)
    FINGERPRINT    = "fingerprint"    # app identity checkpoint (YOLO-E verifies app state)
    UNKNOWN        = "unknown"         # fallback, requires human tag