import requests
import logging
from typing import Dict, Any, Tuple, Optional

from django.conf import settings
from django.utils.dateparse import parse_datetime

from common.mixins.hmac_sign_mixin import HmacSignMixin
from mybook.models import UserProfile
from mybook.utils.token_refresher import TokenRefresher

logger = logging.getLogger(__name__)

class MembershipUpdater(HmacSignMixin):
    """
    결제 서버의 Membership API를 호출하여 사용자 구독 정보를 동기화하는 유틸리티.
    """
    MEMBERSHIP_ENDPOINT = "/api/payments/membership/"

    def fetch_and_update(self, request: Any, params: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        결제 서버에서 현재 사용자의 구독 정보를 가져와 UserProfile을 업데이트합니다.
        :param request: Django의 request 객체
        :param params: GET 요청 파라미터
        :return: (성공 여부, 메시지) 튜플
        """
        user_profile = request.user
        if not isinstance(user_profile, UserProfile):
            return False, "User is not a valid UserProfile instance."

        try:
            # JWT와 HMAC 서명을 모두 포함하여 GET 요청을 보냅니다.
            resp = self.hmac_get(
                path=self.MEMBERSHIP_ENDPOINT, params=params, request=request, timeout=self.DEFAULT_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()

            # --- UserProfile 업데이트 ---
            update_fields = []
            for field in ['email', 'plan_type', 'is_paid_member', 'start_date', 'end_date', 'cancel_requested']:
                if field in data:
                    current_value = getattr(user_profile, field)
                    new_value = data[field]
                    # 날짜 필드는 datetime 객체로 변환
                    if field in ['start_date', 'end_date'] and new_value:
                        new_value = parse_datetime(new_value)

                    if current_value != new_value:
                        setattr(user_profile, field, new_value)
                        update_fields.append(field)
            
            if update_fields:
                user_profile.save(update_fields=update_fields)
                logger.info(f"'{user_profile.username}'의 구독 정보가 업데이트되었습니다: {', '.join(update_fields)}")
                return True, "구독 정보가 성공적으로 동기화되었습니다."
            
            return True, "이미 최신 구독 정보입니다."

        except requests.exceptions.RequestException as e:
            logger.error(f"'{user_profile.username}'의 구독 정보 조회 실패 (네트워크 오류): {e}")
            return False, "결제 서버와 통신 중 오류가 발생했습니다."
        except Exception as e:
            logger.exception(f"'{user_profile.username}'의 구독 정보 동기화 중 예외 발생: {e}")
            return False, "구독 정보를 처리하는 중 알 수 없는 오류가 발생했습니다."
