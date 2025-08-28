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
def translate_book_pages(self, book_id: int, target_lang: str, page_numbers_to_process: list[int] | None = None):
    book = Book.objects.get(id=book_id)
    try:
        schema = load_weaver_schema(SCHEMA_PATH)
        file_part = GeminiFileRefManager().ensure_file_part(book)
        llm = GeminiHelper(schema=schema)

        # 1. 처리 대상 페이지 쿼리
        if page_numbers_to_process:
            pages_qs = BookPage.objects.filter(book=book, page_no__in=page_numbers_to_process)
        else:
            pages_qs = BookPage.objects.filter(book=book)
        
        pages = list(pages_qs.order_by("page_no").values("page_no", "width", "height"))
        
        # 2. LLM에 전달할 '전체' 페이지 수와 현재 작업의 진행률을 위한 '작업' 페이지 수 분리
        total_book_pages = book.page_count
        total_job_pages = len(pages)

        prev_page_html_context = None

        for i, p in enumerate(pages, start=1):
            self.update_state(state="PROGRESS", meta={"current": i, "total": total_job_pages})

            pno = p["page_no"]

            images = list(PageImage.objects.filter(book=book, page_no=pno).values("ref", "bbox"))
            _, figures = split_images_for_prompt(images, p["width"], p["height"])

            page_ctx = {
                "page_no": pno,
                "size": {"w": p["width"], "h": p["height"], "units": "px"},
                "images": [{"ref": im["ref"], "bbox": im["bbox"]} for im in figures] if figures else []
            }

            # 3. build_prompt_faithful에는 항상 '전체' 페이지 수를 전달
            sys_msg, user_msg, example_json = build_prompt_faithful(page_ctx, target_lang, total_book_pages, book_title=book.title, book_genre=book.genre, prev_page_html=prev_page_html_context, glossary=book.glossary)
            logger.debug(f"[DEBUG-TASK]user_msg : {user_msg}")

            data, errs = llm.generate_page_json(
                file_part=file_part, sys_msg=sys_msg, user_msg=user_msg, example_json=example_json, max_retries=2
            )

            if data:
                # 4. 이미지 추출 로직 유지
                page_no_from_data = data["page"]["page_no"]
                norm_w, norm_h = data["page"]["page_w"], data["page"]["page_h"]
                pdf_path = book.original_file.path
                fig_dir = os.path.join(settings.MEDIA_ROOT, "figures", f"book_{book.id}", f"page_{page_no_from_data}") # type: ignore
                os.makedirs(fig_dir, exist_ok=True)

                iex = ImageExtractor(pdf_path, norm_w, norm_h)
                ref_to_path = iex.extract_many(page_no_from_data, data.get("figures", []), output_dir=fig_dir, prefix="fig", ext="png")
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
                    html_content = data.get("html_stage", "")
                    prev_page_html_context = (html_content[:4000] + '...') if len(html_content) > 4000 else html_content
                else:
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=pno, lang=target_lang, mode="faithful",
                        defaults={"data": {"error": errs}, "status": "failed"}
                    )
                    prev_page_html_context = None

        # 5. 모든 페이지 처리 후 최종 상태 점검
        ready_pages_count = TranslatedPage.objects.filter(book=book, lang=target_lang, mode='faithful', status='ready').count()
        if total_book_pages == ready_pages_count:
            book.status = "completed"
        else:
            book.status = "failed"
        book.save(update_fields=["status"])
        return {"status": "done", "pages_processed": total_job_pages}

    except Exception as e:
        logger.error(f"Translation task for book {book_id} failed entirely: {e}", exc_info=True)
        book.status = "failed"
        book.save(update_fields=["status"])
        raise

@shared_task(bind=True)
def retranslate_single_page(self, book_id: int, page_no: int, target_lang: str, feedback: str):
    try:
        book = Book.objects.get(id=book_id)
        page = BookPage.objects.get(book=book, page_no=page_no)
        
        prev_page_html_context = None
        if page_no > 1:
            prev_tp = TranslatedPage.objects.filter(book=book, page_no=page_no - 1, lang=target_lang, mode='faithful').first()
            if prev_tp and isinstance(prev_tp.data, dict):
                html_content = prev_tp.data.get("html_stage", "")
                prev_page_html_context = (html_content[:4000] + '...') if len(html_content) > 4000 else html_content

        # Get the current flawed translation to provide it as context
        current_issued_html = None
        current_tp = TranslatedPage.objects.filter(book=book, page_no=page_no, lang=target_lang, mode='faithful').first()
        if current_tp and isinstance(current_tp.data, dict):
            # Avoid providing the error message as context
            if "error" not in current_tp.data:
                current_issued_html = current_tp.data.get("html_stage", "")

        images = list(PageImage.objects.filter(book=book, page_no=page_no).values("ref", "bbox"))
        _, figures = split_images_for_prompt(images, page.width, page.height)
        page_ctx = {
            "page_no": page_no,
            "size": {"w": page.width, "h": page.height, "units": "px"},
            "images": [{"ref": im["ref"], "bbox": im["bbox"]} for im in figures] if figures else []
        }

        sys_msg, user_msg, example_json = build_prompt_faithful(
            page_ctx, target_lang, book.page_count, book_title=book.title, book_genre=book.genre, 
            prev_page_html=prev_page_html_context, user_feedback=feedback, current_translation_html=current_issued_html, glossary=book.glossary
        )
        logger.debug(f"[DEBUG-RETRANSLATE-TASK] user_msg: {user_msg}")

        schema = load_weaver_schema(SCHEMA_PATH)
        llm = GeminiHelper(schema=schema)
        file_part = GeminiFileRefManager().ensure_file_part(book)
        
        data, errs = llm.generate_page_json(
            file_part=file_part, sys_msg=sys_msg, user_msg=user_msg, example_json=example_json, max_retries=2
        )

        if data:
            norm_w, norm_h = data["page"]["page_w"], data["page"]["page_h"]
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

        if data:
            TranslatedPage.objects.update_or_create(
                book=book, page_no=page_no, lang=target_lang, mode="faithful",
                defaults={"data": data, "status": "ready"}
            )
        else:
            TranslatedPage.objects.update_or_create(
                book=book, page_no=page_no, lang=target_lang, mode="faithful",
                defaults={"data": {"error": errs}, "status": "failed"}
            )
        logger.info(f"Retranslation for book {book_id}, page {page_no} completed.")

    except Exception as e:
        logger.error(f"Retranslation task failed for book {book_id}, page {page_no}: {e}", exc_info=True)
        TranslatedPage.objects.filter(book_id=book_id, page_no=page_no).update(status='failed')
        raise
