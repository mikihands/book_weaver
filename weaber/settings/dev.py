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

