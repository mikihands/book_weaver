from .base import *

print(f"settings.prod activating")

DEBUG = False

ALLOWED_HOSTS = ["weaver.mikihands.com", "test.mikihands.com"]

### static과 미디어 경로 (컨테이너로 가동시)
STATIC_ROOT = env("STATIC_ROOT", default="/app/staticfiles") # type: ignore
MEDIA_ROOT  = env("MEDIA_ROOT",  default="/app/media") # type: ignore

CSRF_TRUSTED_ORIGINS = [
    "https://weaver.mikihands.com",
    "https://test.mikihands.com"
]

CORS_ALLOWED_ORIGINS = [
    "https://weaver.mikihands.com",
]

# 보안 설정 (HTTPS를 사용하는 경우)
SECURE_SSL_REDIRECT = not DEBUG
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG