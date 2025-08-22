# mybook/tasks.py
import json
from pathlib import Path
from django.conf import settings
from celery import shared_task
from django.db import transaction
from mybook.models import Book, BookPage, PageImage, TranslatedPage

from .utils.faithful_prompt import build_prompt_faithful
from .utils.gemini_helper import GeminiHelper
from .utils.gemini_file_ref import GeminiFileRefManager
from .utils.extract_image import split_images_for_prompt
from .utils.schema_loader import load_weaver_schema

SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "page.v1.json"

@shared_task(bind=True)
def translate_book_pages(self, book_id: int, target_lang: str):
    schema = load_weaver_schema()
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
            #"images": images
            "images": [{"ref": im["ref"], "bbox": im["bbox"]} for im in figures]
        }
        sys_msg, user_msg, example_json = build_prompt_faithful(page_ctx, target_lang)

        data, errs = llm.generate_page_json(
            file_part=file_part,
            sys_msg=sys_msg,
            user_msg=user_msg,
            example_json=example_json,
            max_retries=2
        )

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
