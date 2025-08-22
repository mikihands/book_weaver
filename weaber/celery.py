# weaber/celery.py (v -> b 디렉토리 생성시 생긴 오타임)
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "weaber.settings")
app = Celery("weaber")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
