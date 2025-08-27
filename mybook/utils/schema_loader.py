# mybook/utils/schema_loader.py
import json, os, time
from pathlib import Path
from django.conf import settings

_SCHEMA_CACHE = {"path": None, "mtime": 0, "data": None}

def load_weaver_schema(path: Path = Path(settings.BASE_DIR) / "mybook" / "utils" / "page.v2.json"):
    st = os.stat(path)
    if _SCHEMA_CACHE["path"] != str(path) or _SCHEMA_CACHE["mtime"] != st.st_mtime:
        with open(path, "r", encoding="utf-8") as f:
            _SCHEMA_CACHE["data"] = json.load(f)
        _SCHEMA_CACHE["path"] = str(path)
        _SCHEMA_CACHE["mtime"] = st.st_mtime
    return _SCHEMA_CACHE["data"]
