# mybook/utils/original_page_image.py
from pathlib import Path
from django.conf import settings
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

VALID_DPI_MIN = 72
VALID_DPI_MAX = 384

def get_page_image_path(book_id: int, page_no: int, dpi: int = 144) -> Path:
    out_dir = Path(settings.MEDIA_ROOT) / "original" / f"book_{book_id}" / f"dpi{dpi}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"p{page_no}.png"

def ensure_page_image(pdf_path: str, book_id: int, page_no: int, dpi: int = 144) -> str:
    """
    페이지 이미지를 (존재하지 않으면) 생성하고 파일 시스템 경로 문자열을 반환.
    - JPG로 저장 (alpha=False), 품질 85
    - page_no 유효성 / dpi 범위 체크
    """
    if dpi < VALID_DPI_MIN or dpi > VALID_DPI_MAX:
        dpi = 144

    out_path = get_page_image_path(book_id, page_no, dpi)
    if out_path.exists():
        return str(out_path)

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_no - 1]
        pix = page.get_pixmap(dpi=dpi, alpha=False)   # type:ignore
        pix.save(str(out_path), jpg_quality=85)
    finally:
        doc.close()
        
    return str(out_path)