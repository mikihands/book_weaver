from .base import *

print(f"settings.prod activating")

DEBUG = False

ALLOWED_HOSTS = ["weaver.mikihands.com", "test.mikihands.com"]

### static과 미디어 경로 (컨테이너로 가동시)
#STATIC_ROOT = env("STATIC_ROOT", default="/app/staticfiles") # type: ignore
#MEDIA_ROOT  = env("MEDIA_ROOT",  default="/app/media") # type: ignore
### static과 미디어 경로 (systemd 등 호스트에서 직접 가동시)
STATIC_ROOT = env("STATIC_ROOT", default="/home/jesse/book_weaver/staticfiles") # type: ignore
MEDIA_ROOT  = env("MEDIA_ROOT",  default="/home/jesse/book_weaver/media") # type: ignore

CSRF_TRUSTED_ORIGINS = [
    "https://weaver.mikihands.com",
    "https://test.mikihands.com"
]

CORS_ALLOWED_ORIGINS = [
    "https://weaver.mikihands.com",
]

# 보안 설정 (HTTPS를 사용하는 경우, not False = True )
SECURE_SSL_REDIRECT = not DEBUG
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# Celery 기본 연결 (컨테이너 배포시 바꿀것. )
CELERY_BROKER_URL = "redis://localhost:6479/0"
CELERY_RESULT_BACKEND = "redis://localhost:6479/1"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 60 * 15  # 15분 타임아웃 예시
CELERY_RESULT_EXPIRES = 3600 # 결과저장시간 기본 86400:24시간

# (옵션) Beat 스케줄 저장소(장고 DB)
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"

# (옵션) RedisBoard 설정 (Django Admin에서 /admin/redisboard/ 확인)
REDISBOARD_CONNECTIONS = {
    "default": {"HOST": "127.0.0.1", "PORT": 6479, "DB": 0},
}

#--------------- 