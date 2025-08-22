from django.conf import settings
#from django.core.cache import cache
from .models import UserProfile
import logging

logger = logging.getLogger(__name__)


def global_settings(request):
    """
    settings.DEBUG 값을 템플릿에 'DEBUG' 변수로 전달합니다.
    """
    return {
        'DEBUG_MODE': settings.DEBUG, # 더 명확한 변수 이름을 사용합니다.
    }

class UserSessionManager:
    def __init__(self, request):
        self.request = request
        self.access_token = self.request.session.get("access_token")
        self.username = self.request.session.get("username")

    def is_logged_in(self):
        return self.access_token is not None

    def get_user_profile(self):
        """
        세션을 이용하여 사용자 프로필을 가져오고, 캐싱하여 불필요한 DB 쿼리를 줄임.
        """
        if not self.is_logged_in() or not self.username:
            return None

        try:
            user = UserProfile.objects.get(username=self.username)
            return user
        except UserProfile.DoesNotExist:
            logger.warning(f"[CTX_PROCESSOR] 사용자 '{self.username}'을 찾을 수 없습니다.")
            return None

        # 추후 서비스 테스트 후 캐시가 필요하면 그때 활성화 하겠음
        # cache_key = f"user_profile_{self.username}"
        # user = cache.get(cache_key)

        # if not user:
        #     try:
        #         user = UserProfile.objects.get(username=self.username)
        #         cache.set(cache_key, user, 60 * 15)  # 15분 캐싱
        #         logger.debug(f"[CTX_PROCESSOR]사용자{self.username}를 특정하여 캐시에 저장함.")
        #     except UserProfile.DoesNotExist:
        #         logger.warning(f"[CTX_PROCESSOR] 사용자 '{self.username}'을 찾을 수 없습니다.")
        #         return None
        # else:
        #     # 캐시에서 데이터를 반환한 경우
        #     logger.debug("Cache hit: %s", cache_key)

        # return user

def get_user_session_manager(request):
    """
    request 객체에 저장하여 중복 쿼리를 방지하는 함수
    """
    if not hasattr(request, "_user_session_manager"):
        request._user_session_manager = UserSessionManager(request)
    return request._user_session_manager

def user_logged_in(request):
    """
    로그인 상태 여부 확인
    """
    user_manager = get_user_session_manager(request)

    return {
        "user_logged_in": user_manager.is_logged_in(),
    }

def user_profile(request):
    """
    사용자 프로필 가져오기
    """
    user_manager = get_user_session_manager(request)

    return {
        "user_profile": user_manager.get_user_profile(),
    }