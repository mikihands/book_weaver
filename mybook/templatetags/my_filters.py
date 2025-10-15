from django import template

register = template.Library()

PLAN_LEVELS = {'None': 0, 'Starter': 1, 'Growth': 2, 'Pro': 3, 'Enterprise': 4}

@register.simple_tag
def define_plan_levels():
    return PLAN_LEVELS

@register.simple_tag
def get_plan_level(plan_name):
    return PLAN_LEVELS.get(plan_name, 0)
