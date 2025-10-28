# mybook/core/guards.py
from functools import wraps
from rest_framework.exceptions import PermissionDenied
from .plans import is_plan_at_least
from django.utils.translation import gettext

def require_plan(min_plan: str):
    def deco(func):
        @wraps(func)
        def wrapper(self, request, *args, **kwargs):
            if not is_plan_at_least(request.user, min_plan):
                raise PermissionDenied(gettext("상위 플랜이 필요합니다."))
            return func(self, request, *args, **kwargs)
        return wrapper
    return deco

# 사용예시 : 메서드별 상이한 정책 적용시
"""
class ExportPDFView(APIView):
    permission_classes = [IsAuthenticated, PlanPermission]

    @require_plan("Starter")
    def get(self, request): ...

    @require_plan("Pro")
    def post(self, request): ...
"""