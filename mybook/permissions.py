# mybook/permissions.py
from rest_framework import permissions

class IsBookOwner(permissions.BasePermission):
    """
    요청된 객체의 소유자가 현재 인증된 사용자인지 확인합니다.
    """
    message = "You do not have permission to access this book."

    def has_object_permission(self, request, view, obj):
        return getattr(obj, "owner_id", None) == getattr(getattr(request, "user", None), "id", None)
