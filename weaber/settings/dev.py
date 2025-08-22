from .base import *
print(f"settings.dev activating")

DEBUG = True

ALLOWED_HOSTS = ["*"]

### static과 미디어 경로 (컨테이너로 가동시)
STATIC_ROOT = env("STATIC_ROOT", default="/home/jesse/book_weaver/staticfiles") # type: ignore
MEDIA_ROOT  = env("MEDIA_ROOT",  default="/home/jesse/book_weaver/media") # type: ignore

CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1",
    "https://test.mikihands.com",
]

CORS_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
]

if DEBUG:
    # Add django_browser_reload only in DEBUG mode
    INSTALLED_APPS += ['django_browser_reload']
if DEBUG:
    # Add django_browser_reload middleware only in DEBUG mode
    MIDDLEWARE += [
        "django_browser_reload.middleware.BrowserReloadMiddleware",
    ]


# Celery 기본 연결
CELERY_BROKER_URL = "redis://localhost:6479/0"
CELERY_RESULT_BACKEND = "django-db"  # django_celery_results 사용
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 60 * 15  # 15분 타임아웃 예시

# (옵션) Beat 스케줄 저장소(장고 DB)
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"

# (옵션) RedisBoard 설정 (Django Admin에서 /admin/redisboard/ 확인)
REDISBOARD_CONNECTIONS = {
    "default": {"HOST": "127.0.0.1", "PORT": 6479, "DB": 0},
}
