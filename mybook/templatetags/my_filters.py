from django import template
from mybook.core.plans import PLAN_LEVELS

register = template.Library()

@register.simple_tag
def define_plan_levels():
    return PLAN_LEVELS

@register.simple_tag
def get_plan_level(plan_name):
    return PLAN_LEVELS.get(plan_name, 0)

@register.filter
def div(value, arg):
    """Divides the value by the arg."""
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return None

@register.filter
def mul(value, arg):
    """Multiplies the value by the arg."""
    try:
        return float(value) * float(arg)
    except (ValueError):
        return None

@register.filter
def sub(value, arg):
    """Subtracts the arg from the value."""
    return float(value) - float(arg)