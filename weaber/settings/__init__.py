import environ
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

env = environ.Env()
environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

environment = env('DJANGO_ENV', default='dev') # type:ignore

print(f"Loading settings for: {environment}")

if environment == "prod":
    from .prod import *
else:
    from .dev import *
