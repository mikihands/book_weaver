# mybook/utils/pdf_previews.py
import fitz, os
from pathlib import Path
from django.conf import settings

def ensure_previews(pdf_path: str, book_id: int, dpi=120, max_w=300):
    out_dir = Path(settings.MEDIA_ROOT) / "previews" / f"book_{book_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)  # type:ignore
        # 가로 리사이즈
        if pix.width > max_w:
            scale = max_w / pix.width
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale)) # type:ignore
        out = out_dir / f"p{i+1}.jpg"
        pix.save(str(out))
    doc.close()
    return str(out_dir)
