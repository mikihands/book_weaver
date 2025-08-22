# mybook/templatetags/weaver_extras.py
from django import template

register = template.Library()

@register.filter
def bbox_style(bbox):
    """
    bbox = [x, y, w, h] (px)
    -> 'left:Xpx;top:Ypx;width:Wpx;height:Hpx;'
    """
    try:
        x, y, w, h = bbox
        return f"left:{float(x)}px;top:{float(y)}px;width:{float(w)}px;height:{float(h)}px;"
    except Exception:
        return ""
