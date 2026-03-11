import logging
from typing import List, Dict, Any
from core.types import Rect

try:
    from pywinauto import Desktop
    from pywinauto.findbestmatch import MatchError
    from pywinauto.findwindows import ElementNotFoundError
except ImportError:
    logging.warning("pywinauto not installed or not supported on this OS")
    Desktop = None
    MatchError = Exception
    ElementNotFoundError = Exception

logger = logging.getLogger(__name__)

def _walk_tree(element: Any, current_depth: int, max_depth: int, result_list: List[Dict[str, Any]]) -> None:
    if current_depth > max_depth:
        return
        
    try:
        rect = element.rectangle()
        
        # Attempt to get control_type and name using UIA specific attributes, 
        # fallback to more generic pywinauto methods
        ctrl_type = getattr(element.element_info, 'control_type', None)
        if not ctrl_type:
            ctrl_type = element.friendly_class_name()
            
        name_raw = getattr(element.element_info, 'name', None)
        if name_raw is None:
            name_raw = element.window_text() or ""
            
        result_list.append({
            "rect": Rect(x=rect.left, y=rect.top, w=rect.width(), h=rect.height()),
            "control_type": str(ctrl_type),
            "name_raw": str(name_raw)
        })
        
        for child in element.children():
            _walk_tree(child, current_depth + 1, max_depth, result_list)
            
    except Exception:
        # Ignore elements that cannot be accessed or raise errors during properties read
        pass


def get_element_tree(hwnd: int, max_depth: int = 5) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with keys: rect (as Rect), control_type (str), name_raw (str).
    Walks the UIA tree from the window handle, limited to max_depth levels.
    """
    if Desktop is None:
        return []
        
    try:
        window = Desktop(backend='uia').window(handle=hwnd)
        result: List[Dict[str, Any]] = []
        _walk_tree(window.wrapper_object(), 0, max_depth, result)
        return result
    except Exception as e:
        logger.warning(f"Error getting element tree for hwnd {hwnd}: {e}")
        return []


def tab_walk(hwnd: int) -> List[Dict[str, Any]]:
    """
    Tab-order traversal, returns list of dicts with rect (as Rect) only.
    Names are omitted as they are untrusted.
    """
    if Desktop is None:
        return []
        
    try:
        window = Desktop(backend='uia').window(handle=hwnd)
        result: List[Dict[str, Any]] = []
        
        for element in window.descendants():
            try:
                # Heuristic: elements that are keyboard focusable represent tab stops
                if hasattr(element, 'is_keyboard_focusable') and element.is_keyboard_focusable():
                    rect = element.rectangle()
                    result.append({
                        "rect": Rect(x=rect.left, y=rect.top, w=rect.width(), h=rect.height())
                    })
            except Exception:
                continue
                
        return result
    except Exception as e:
        logger.warning(f"Error during tab walk for hwnd {hwnd}: {e}")
        return []


def get_focused_element() -> Dict[str, Any]:
    """
    Returns the currently focused element's rect (as Rect) and name_raw (str).
    """
    if Desktop is None:
        return {}
        
    try:
        desktop = Desktop(backend='uia')
        top_win = desktop.top_window()
        
        # Try to find the element with keyboard focus within the top window
        try:
            focused_elements = top_win.descendants(has_keyboard_focus=True)
            if focused_elements:
                element = focused_elements[0]
                rect = element.rectangle()
                name_raw = getattr(element.element_info, 'name', None)
                if name_raw is None:
                    name_raw = element.window_text() or ""
                    
                return {
                    "rect": Rect(x=rect.left, y=rect.top, w=rect.width(), h=rect.height()),
                    "name_raw": str(name_raw)
                }
        except Exception:
            pass # Fall back to the top window if focused element search fails
            
        # Fallback to the top window itself if no specific child is focused
        element = top_win.wrapper_object()
        rect = element.rectangle()
        name_raw = getattr(element.element_info, 'name', None)
        if name_raw is None:
            name_raw = element.window_text() or ""
            
        return {
            "rect": Rect(x=rect.left, y=rect.top, w=rect.width(), h=rect.height()),
            "name_raw": str(name_raw)
        }
    except Exception as e:
        logger.warning(f"Error getting focused element: {e}")
        return {}