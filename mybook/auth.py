# mybook/auth.py

from rest_framework import authentication
from rest_framework import exceptions
from django.conf import settings
from .utils.token_refresher import TokenRefresher
from .models import UserProfile
import jwt
import logging

logger = logging.getLogger(__name__)

class SessionAuthWithToken(authentication.BaseAuthentication):
    """
    세션에 저장된 JWT 토큰으로 사용자를 인증하는 커스텀 인증 클래스.
    """
    def authenticate(self, request):
        access_token = request.session.get('access_token')

        if not access_token:
            return None # 토큰이 없으면 인증 실패

        
        # 토큰 유효성 검사 및 갱신 시도
        is_valid, new_token = TokenRefresher.refresh_access_token_if_needed(request)
        
        if not is_valid:
            raise exceptions.AuthenticationFailed('Authentication failed: Invalid or expired token.')
        
        try:
            username = request.session.get("username")
            logger.debug(f"Username: {username}")  # 디버깅용 출력

            if not username:
                raise exceptions.AuthenticationFailed('Authentication failed: Username not found in session.')
            
            # Django에서 request.user에 할당할 UserProfile 객체 조회
            try:
                user_profile = UserProfile.objects.get(username=username)
                return (user_profile, new_token)
            except UserProfile.DoesNotExist:
                raise exceptions.AuthenticationFailed('No such user profile exists in local DB.')

        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Authentication failed: Invalid token format.')
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise exceptions.AuthenticationFailed('Authentication failed: An unexpected error occurred.')

