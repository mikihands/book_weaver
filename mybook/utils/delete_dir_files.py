# mybook/utils/delete_dir_files.py
import shutil
from pathlib import Path

def safe_remove(path: str | Path):
    try:
        p = Path(path)
        if p.is_file():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        # 삭제 실패는 치명적이지 않으니 무시(로그만)
        import logging
        logging.getLogger(__name__).warning("Failed to remove %s", path, exc_info=True)