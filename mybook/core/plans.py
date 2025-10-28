# mybook/core/plans.py 
## DRF 인증서버에서 설정한 이름대로 나열
PLAN_LEVELS = {'Free':0, 'Starter':1, 'Growth':2, 'Pro':3, 'Enterprise':4}

def plan_level(name: str) -> int:
    return PLAN_LEVELS.get(name, 0)

def is_plan_at_least(user, required: str) -> bool:
    return plan_level(getattr(user, "plan_type", "Free")) >= plan_level(required)