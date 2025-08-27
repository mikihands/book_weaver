# mybook/tasks.py
import json, logging, os
from pathlib import Path
from django.conf import settings
from celery import shared_task
from django.db import transaction
from mybook.models import Book, BookPage, PageImage, TranslatedPage

from .utils.faithful_prompt import build_prompt_faithful
from .utils.gemini_helper import GeminiHelper
from .utils.gemini_file_ref import GeminiFileRefManager
from .utils.extract_image import split_images_for_prompt, ImageExtractor
from .utils.schema_loader import load_weaver_schema

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "page.v2.json"

@shared_task(bind=True)
def translate_book_pages(self, book_id: int, target_lang: str):
    schema = load_weaver_schema(SCHEMA_PATH)
    book = Book.objects.get(id=book_id)

    # ✅ 파일 파트 1회 준비 (20MB 미만: inline, 20~50MB: 파일 업로드/재사용)
    file_part = GeminiFileRefManager().ensure_file_part(book)

    pages = list(BookPage.objects.filter(book=book).order_by("page_no")
                 .values("page_no", "width", "height"))
    total = len(pages)
    llm = GeminiHelper(schema=schema)

    for i, p in enumerate(pages, start=1):
        self.update_state(state="PROGRESS", meta={"current": i, "total": total})

        pno = p["page_no"]
        images = list(PageImage.objects.filter(book=book, page_no=pno).values("ref", "bbox"))
        bg, figures = split_images_for_prompt(images, p["width"], p["height"])

        page_ctx = {
            "page_no": pno,
            "size": {"w": p["width"], "h": p["height"], "units": "px"},
            "images": [{"ref": im["ref"], "bbox": im["bbox"]} for im in figures] if figures else []
        }

        sys_msg, user_msg, example_json = build_prompt_faithful(page_ctx, target_lang, total)
        logger.debug(f"[DEBUG-TASK]user_msg : {user_msg}")

        data, errs = llm.generate_page_json(
            file_part=file_part,
            sys_msg=sys_msg,
            user_msg=user_msg,
            example_json=example_json,
            max_retries=2
        )
        # data = Gemini 답변 JSON (dict)
        if data:
            page_no = data["page"]["page_no"]
            norm_w  = data["page"]["page_w"]
            norm_h  = data["page"]["page_h"]

            pdf_path = book.original_file.path
            fig_dir = os.path.join(settings.MEDIA_ROOT, "figures", f"book_{book.id}", f"page_{page_no}") # type: ignore
            os.makedirs(fig_dir, exist_ok=True)

            iex = ImageExtractor(pdf_path, norm_w, norm_h)
            ref_to_path = iex.extract_many(page_no, data.get("figures", []), output_dir=fig_dir, prefix="fig", ext="png")
            iex.close()

            for ref, abs_path in ref_to_path.items():
                rel = abs_path.replace(settings.MEDIA_ROOT, "").lstrip("/\\")
                url = settings.MEDIA_URL.rstrip("/") + "/" + rel
                data.setdefault("image_src_map", {})[ref] = url

        with transaction.atomic():
            if data:
                TranslatedPage.objects.update_or_create(
                    book=book, page_no=pno, lang=target_lang, mode="faithful",
                    defaults={"data": data, "status": "ready"}
                )
            else:
                TranslatedPage.objects.update_or_create(
                    book=book, page_no=pno, lang=target_lang, mode="faithful",
                    defaults={"data": {"error": errs}, "status": "failed"}
                )

    book.status = "ready"
    book.save(update_fields=["status"])
    return {"status": "done", "pages": total}
