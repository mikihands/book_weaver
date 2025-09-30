import environ
import os
import sys
from pathlib import Path
from django.utils.translation import gettext_lazy as _

BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = environ.Env()
environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

SECRET_KEY = 'django-insecure-mtub!&&klk@af)cys=%&9juoubr8c=@y3j@ai@-mz3e@cu!5)+'

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'mybook',
    'tailwind',
    'theme',
    "django_celery_results",
    "django_celery_beat",
    "redisboard",  
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    "django.middleware.locale.LocaleMiddleware",
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'weaber.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                "django.template.context_processors.i18n",
                "mybook.context_processors.global_settings",
                "mybook.context_processors.user_profile",
                "mybook.context_processors.user_logged_in",
            ],
        },
    },
]

WSGI_APPLICATION = 'weaber.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'weaverdb',
        'USER': 'jessekim80',
        'PASSWORD': 'DB_tomikikun04!!@@',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
#LANGUAGE_CODE = 'en-us'
LANGUAGE_CODE = 'ko'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

STATIC_URL = '/static/'

LANGUAGES = [
    ("en", _("English")),
    ("ko", _("Korean")),
    ("ja", _("Japanese")),
    ("es", _("Spanish")),
    ("de", _("German")),
    ("fr", _("French")),
]

# 공통 정적 파일 디렉토리 (필요시)
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "/media/"

LOCALE_PATHS = [BASE_DIR / "locale"]

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

GEMINI_API_KEY = env("GEMINI_API_KEY")
PAID_GEMINI_KEY = env("PAID_GEMINI_KEY")
GPT_OSS_KEY = env("GPT_OSS_KEY")
GPT_OSS_ENDPOINT = env("GPT_OSS_ENDPOINT", default="https://ai.mikihands.com/v1/chat/completions") # type:ignore
VLLM_API_KEY = env("VLLM_API_KEY")
VLLM_ENDPOINT = env("VLLM_ENDPOINT", default="https://ai.mikihands.com/v1/chat/completions") # type:ignore

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        }
    },
    "handlers": {
        "debug_file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "debug.log",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "formatter": "verbose",
        },
        "error_file": {
            "level": "ERROR",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "error.log",
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "formatter": "verbose",
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["error_file", "console"],
            "level": "ERROR",
            "propagate": False,
        },
        "weaber": {
            "handlers": ["debug_file", "error_file", "console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "mybook": {
            "handlers": ["debug_file", "error_file", "console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["error_file", "console"],
            "level": "ERROR",  # 요청 관련 에러만 기록
            "propagate": False,
        },
        "root": {
            "handlers": ["console"],
            "level": "DEBUG",
        },
    },
}

AUTH_SERVER_URL='https://api.mikihands.com'

TAILWIND_APP_NAME = 'theme'
INTERNAL_IPS = [
    "127.0.0.1",
    "localhost",
]

HMAC_APP_ID = "weaver"
# HMAC_KEY_ID = "v1"  # 현재 사용중인 키 식별자
MAC_SECRET=env('HMAC_SECRET_WEAVER')

REST_FRAMEWORK = {
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "contact": "5/min",  # 1분에 5회 제한(원하면 10/min, 100/day 등으로 변경)
    },
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'mybook.auth.SessionAuthWithToken',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        #'rest_framework.permissions.IsAuthenticated',
        'rest_framework.permissions.AllowAny',
    ),
}

# 이메일 서버구성
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "jessekim80@gmail.com"
EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD")
ADMIN_EMAIL_ADDRESS = "info@mikihands.com"
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER