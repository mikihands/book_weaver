# mybook/permissions.py
from rest_framework import permissions
from mybook.core.plans import PLAN_LEVELS
from django.utils.translation import gettext_lazy as _

class IsBookOwner(permissions.BasePermission):
    """
    요청된 객체의 소유자가 현재 인증된 사용자인지 확인합니다.
    """
    message = "You do not have permission to access this book."

    def has_object_permission(self, request, view, obj):
        return getattr(obj, "owner_id", None) == getattr(getattr(request, "user", None), "id", None)

class PlanPermission(permissions.BasePermission):
    """
    사용자의 구독 플랜 레벨이 특정 작업 수행에 필요한 최소 레벨을 충족하는지 확인합니다.
    """
    message = _("죄송합니다. 현재 기능은 상위 플랜이 필요합니다. 상위 플랜으로 업그레이드 해주세요.")

    def has_permission(self, request, view):
        # staff/superuser 우회
        if getattr(request.user, "is_staff", False) or getattr(request.user, "is_superuser", False):
            return True
        
        required_level = None
        if hasattr(view, "min_plan"): # plan name 입력방식
            required_level = PLAN_LEVELS[view.min_plan]
        elif hasattr(view, "min_plan_level"): # 숫자입력방식
            required_level = view.min_plan_level

        if required_level is None:  # 요구 레벨 없으면 통과
            return True

        user_profile = request.user # SessionAuthWithToken 클래스에서 이미 붙여주었음.

        current_level = PLAN_LEVELS.get(getattr(user_profile, "plan_type", "Free"), 0) # plan_type 없으도 터지지 않게 0 삽입

        return current_level >= required_level