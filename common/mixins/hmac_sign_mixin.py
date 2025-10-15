# common/mixins/hmac_sign_mixin.py
import os, json, hmac, hashlib, time, uuid, requests, logging
from typing import Dict, Any, Optional
from django.conf import settings
from mybook.utils.token_refresher import TokenRefresher

logger = logging.getLogger(__name__)

class HmacSignMixin:
    """
    인증서버 호출에 쓰는 HMAC 서명/헤더/전송 유틸.
    - JSON 직렬화 방식을 고정(separators)하여 바이트 일치 보장
    - key rotation을 위해 key_id 지원(선택)
    - timestamp/nonce(선택)로 리플레이 방지까지 확장 가능
    """

    # 기본 설정: settings에서 주입
    HMAC_APP_ID: str = getattr(settings, "HMAC_APP_ID", "weaver")
    HMAC_SECRET: str = getattr(settings, "MAC_SECRET", "")  # HMAC_SECRET_WEAVER
    HMAC_KEY_ID: Optional[str] = getattr(settings, "HMAC_KEY_ID", None)  # "v1" 등 (선택)

    AUTH_SERVER_URL: str = getattr(settings, "AUTH_SERVER_URL", "")
    WEAVER_BASE_URL: str = getattr(settings, "WEAVER_BASE_URL", "")
    DEFAULT_TIMEOUT: int = 10

    # --- 직렬화 & 서명 ---
    @staticmethod
    def _to_body_bytes(payload: Dict[str, Any]) -> bytes:
        # 공백 제거로 바이트 일관성 보장
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _sign(self, body_bytes: bytes, secret: Optional[str] = None) -> str:
        secret_bytes = (self.HMAC_SECRET or "").encode() if not secret else secret.encode()
        return hmac.new(secret_bytes, body_bytes, hashlib.sha256).hexdigest()

    # --- 헤더 조립 ---
    def _build_headers(
        self,
        signature_hex: str,
        *,
        app_id: Optional[str] = None,
        key_id: Optional[str] = None,
        with_anti_replay: bool = False,
        ts: Optional[int] = None,
        nonce: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-Client-App": app_id or self.HMAC_APP_ID,
            "X-Client-Signature": signature_hex,
        }
        # 키 로테이션 식별용(선택)
        if key_id or self.HMAC_KEY_ID:
            key_value = key_id if key_id is not None else self.HMAC_KEY_ID
            if key_value is not None:
                headers["X-Client-Key-Id"] = key_value

        # 리플레이 방지(선택: 서버가 지원할 때만 켜세요)
        #if with_anti_replay:
        #    headers["X-Client-Timestamp"] = str(ts or int(time.time()))
        #    headers["X-Client-Nonce"] = nonce or uuid.uuid4().hex

        if extra:
            headers.update(extra)
        return headers

    # --- 전송 유틸 ---
    def hmac_post(
        self,
        path: str,
        payload: Dict[str, Any], # The data to be sent
        request: Optional[Any] = None, # Pass the Django request object to include user JWT
        *,
        timeout: Optional[int] = None,
        with_anti_replay: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """
        Signs payload with HMAC and POSTs it to the auth server.
        If a `request` object is provided, it extracts the user's JWT from the session
        and adds it to the Authorization header.
        """
        assert self.AUTH_SERVER_URL, "settings.AUTH_SERVER_URL 가 필요합니다."
        body = self._to_body_bytes(payload)
        sig = self._sign(body)

        # Start with base headers and merge any extra headers provided.
        headers_to_build = {"Referer": self.WEAVER_BASE_URL}
        if extra_headers:
            headers_to_build.update(extra_headers)

        # If request object is available, add user's JWT for end-user authentication
        if request and hasattr(request, "session"):
            # Check if token is valid and refresh if needed, before using it.
            is_valid, token_or_message = TokenRefresher.refresh_access_token_if_needed(request)
            if is_valid:
                # On success, token_or_message is the (potentially new) access token.
                headers_to_build["Authorization"] = f"Bearer {token_or_message}"
            else:
                # If token refresh fails, log it but proceed without Authorization header.
                # The auth server will then reject the request due to missing user auth.
                logger.warning(f"Could not refresh token for user. Reason: {token_or_message}. Proceeding without JWT.")
        
        headers = self._build_headers(sig, with_anti_replay=with_anti_replay, extra=headers_to_build)

        url = f"{self.AUTH_SERVER_URL.rstrip('/')}/{path.lstrip('/')}"
        return requests.post(url, headers=headers, data=body, timeout=timeout or self.DEFAULT_TIMEOUT)

    def hmac_get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        request: Optional[Any] = None,
        *,
        timeout: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        assert self.AUTH_SERVER_URL, "settings.AUTH_SERVER_URL 가 필요합니다."

        # ★ 서버 규칙에 맞게 '바디'만 서명한다. GET은 빈 바디.
        body_bytes = b""
        sig = self._sign(body_bytes)

        headers_to_build = {"Referer": self.WEAVER_BASE_URL}
        if extra_headers:
            headers_to_build.update(extra_headers)

        # (옵션) JWT 추가
        if request and hasattr(request, "session"):
            is_valid, token_or_message = TokenRefresher.refresh_access_token_if_needed(request)
            if is_valid:
                headers_to_build["Authorization"] = f"Bearer {token_or_message}"
            else:
                logger.warning(f"Could not refresh token for user. Reason: {token_or_message}. Proceeding without JWT.")

        headers = self._build_headers(sig, extra=headers_to_build)

        url = f"{self.AUTH_SERVER_URL.rstrip('/')}/{path.lstrip('/')}"
        # GET은 params로 쿼리를 전달하되, 서명 대상에는 포함하지 않는다.
        return requests.get(url, headers=headers, params=params, timeout=timeout or self.DEFAULT_TIMEOUT)


    # (선택) 공용 세션: 재시도/연결 재사용 등 붙이고 싶다면 사용
    _session: Optional[requests.Session] = None

    def get_session(self) -> requests.Session:
        if self._session is None:
            s = requests.Session()
            # 여기서 HTTPAdapter로 재시도 정책 부여 가능
            self._session = s
        return self._session

    def hmac_post_with_session(
        self, path: str, payload: Dict[str, Any], **kwargs
    ) -> requests.Response:
        body = self._to_body_bytes(payload)
        sig = self._sign(body)
        headers = self._build_headers(sig, with_anti_replay=kwargs.pop("with_anti_replay", False),
                                      extra=kwargs.pop("extra_headers", None))
        url = f"{self.AUTH_SERVER_URL.rstrip('/')}/{path.lstrip('/')}"
        return self.get_session().post(url, headers=headers, data=body, timeout=kwargs.get("timeout", self.DEFAULT_TIMEOUT))
