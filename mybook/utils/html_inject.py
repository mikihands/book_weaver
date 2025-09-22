# utils/html_inject.py
from __future__ import annotations
import html
from bs4 import BeautifulSoup, Tag

def escape_html(s: str) -> str:
    return html.escape(s, quote=True)

def clamp_bbox(bbox, W, H):
    x, y, w, h = bbox
    x = max(0, min(x, W))
    y = max(0, min(y, H))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]

def inject_sources(html: str, ref_map: dict, details_map: dict | None = None) -> str:
    """
    Injects `src`, `width`, and `height` attributes into <img> tags.
    - html: The HTML string to process.
    - ref_map: A dictionary mapping a data-ref to an image URL.
    - details_map: A dictionary mapping a data-ref to details like {'width': w, 'height': h}.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        ref = img.get("data-ref") or img.get("data-image-ref") # type: ignore
        if ref and ref in ref_map:
            img["src"] = ref_map[ref] # type: ignore
            if details_map and ref in details_map:
                details = details_map[ref]
                if 'width' in details and details['width']:
                    img['width'] = details['width'] # type: ignore
                if 'height' in details and details['height']:
                    img['height'] = details['height'] # type: ignore
    return str(soup)
