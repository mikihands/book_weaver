#mybook/utils/token_refresher.py
import requests
import jwt
from django.conf import settings
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

TOKEN_REFRESH_ENDPOINT = "/api/accounts/token/refresh/"

class TokenRefresher:
    @staticmethod
    def refresh_access_token(session):
        """
        Refresh access token using the refresh token in the session.
        """
        refresh_token = session.get("refresh_token")
        if not refresh_token:
            return False, "No refresh token available."

        try:
            response = requests.post(
                settings.AUTH_SERVER_URL + TOKEN_REFRESH_ENDPOINT,
                data={"refresh": refresh_token},
            )
            if response.status_code == 200:
                response_data = response.json()
                session["access_token"] = response_data["access"]
                return True, "Access token refreshed successfully."
            else:
                # 갱신 실패는 refresh_token이 만료되었거나 유효하지 않다는 의미이므로 세션을 초기화
                session.flush()
                return False, "Failed to refresh access token. Please re-login."
        except requests.exceptions.RequestException as e:
            return False, str(e)

    @staticmethod
    def is_token_valid(token):
        """
        주어진 access_token이 유효한지 검사합니다.
        만약 만료되었거나 유효하지 않다면 False를 반환합니다.
        """
        try:
            # 토큰을 디코딩하여 만료 시간 확인 (Signature는 검증하지 않음)
            decoded_token = jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256"])

            # 'exp' 필드에서 만료 시간을 가져옴
            exp_timestamp = decoded_token.get("exp")
            if exp_timestamp is None:
                return False

            # 현재 시간과 비교
            current_timestamp = timezone.now().timestamp()
            return current_timestamp < exp_timestamp
        except jwt.ExpiredSignatureError:
            # 토큰이 이미 만료된 경우
            return False
        except jwt.InvalidTokenError:
            # 잘못된 토큰 형식인 경우
            return False

    @staticmethod
    def refresh_access_token_if_needed(request):
        """
        만료된 토큰을 자동으로 갱신하고 새로운 access_token을 반환하는 함수.
        갱신이 성공하면 True와 새로운 access_token을 반환하고,
        실패하면 False와 에러 메시지를 반환한다.
        """
        access_token = request.session.get("access_token")
        refresh_token = request.session.get("refresh_token")
        # 토큰이 만료되었을 경우 갱신 시도
        if not TokenRefresher.is_token_valid(access_token):
            logger.info(
                f"access_token이 만료되었으므로 refresh를 이용하여 재발급을 시도합니다. refresh: {refresh_token}"
            )
            result, message = TokenRefresher.refresh_access_token(request.session)
            if result:
                # 갱신 성공 시 새로운 access_token이 이미 세션에 저장된 상태
                logger.info("refresh를 이용한 access_token재발급에 성공했습니다.")
                return True, request.session["access_token"]
            else:
                # 실패 시 False와 메시지 반환
                logger.warning("refresh를 이용한 access토큰 갱신에 실패하였습니다.")
                return False, message

        # 토큰이 유효한 경우, 기존의 access_token을 반환
        logger.info(f"access_token이 유효하므로 재발급없이 사용합니다.재사용토큰: {access_token}")
        return True, access_token