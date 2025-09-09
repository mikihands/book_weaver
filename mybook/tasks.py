# mybook/tasks.py
import json, logging, os
from pathlib import Path
from django.conf import settings
from celery import shared_task
from django.db import transaction
from mybook.models import Book, BookPage, PageImage, TranslatedPage

from .utils.born_digital_prompt import build_prompt_born_digital
from .utils.faithful_prompt import build_prompt_faithful
from .utils.gemini_helper import GeminiHelper
from .utils.gemini_file_ref import GeminiFileRefManager
from .utils.extract_image import split_images_for_prompt, ImageExtractor
from .utils.schema_loader import load_weaver_schema
from .utils.born_digital import (
    collect_page_layout, spans_to_units, build_faithful_html, build_readable_html
)

from urllib.parse import urljoin

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "page.v2.json"
BORN_DIGITAL_SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "born_digital.v1.json"

def _media_url(rel_path: str) -> str:
    return urljoin(settings.MEDIA_URL, rel_path.lstrip("/\\"))

def _inject_images_from_db(layout: dict, book: Book, page_no: int) -> dict:
    # 0) 페이지 크기/메타(px) 주입
    try:
        bp = BookPage.objects.get(book=book, page_no=page_no)
        layout["size"] = {"w": float(bp.width), "h": float(bp.height), "units": "px"}
        layout["meta"] = bp.meta or {}
    except BookPage.DoesNotExist:
        layout.setdefault("size", {"w": 0.0, "h": 0.0, "units": "px"})
        layout.setdefault("meta", {})

    # 1) 이미지 주입: DB에 저장된 정규화 px 그대로
    pis = list(
        PageImage.objects
        .filter(book=book, page_no=page_no)
        .values("xref","path","bbox","clip_bbox","img_w","img_h","origin_w","origin_h","transform")
    )
    images = []
    for pi in pis:
        images.append({
            "xref": int(pi["xref"]) if pi["xref"] is not None else None,
            "path": pi["path"],
            "bbox": pi.get("bbox"),             # px, xywh
            "clip_bbox": pi.get("clip_bbox"),   # px, xywh or None
            "img_w": pi.get("img_w"),
            "img_h": pi.get("img_h"),
            "origin_w": pi.get("origin_w"),
            "origin_h": pi.get("origin_h"),
            "matrix_page": pi.get("transform"),  #px (페이지 기준)
        })
    layout["images"] = images

    # 2) xref -> URL 맵
    image_src_map = {}
    for pi in pis:
        if pi["xref"] is not None and pi["path"]:
            image_src_map[int(pi["xref"])] = _media_url(pi["path"])

    return image_src_map


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
        
        pages = list(pages_qs.order_by("page_no").values("page_no", "width", "height", "meta"))
        
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
def translate_book_pages_born_digital(self, book_id: int, target_lang: str, page_numbers_to_process: list[int] | None = None):
    book = Book.objects.get(id=book_id)
    try:
        schema = load_weaver_schema(BORN_DIGITAL_SCHEMA_PATH)
        llm = GeminiHelper(schema=schema)

        if page_numbers_to_process:
            pages_qs = BookPage.objects.filter(book=book, page_no__in=page_numbers_to_process)
        else:
            pages_qs = BookPage.objects.filter(book=book)
        
        pages = list(pages_qs.order_by("page_no"))

        total_job_pages = len(pages)
        pdf_path = book.original_file.path

        for i, page in enumerate(pages, start=1):
            pno = page.page_no
            self.update_state(state="PROGRESS", meta={"current": i, "total": total_job_pages})
            
            layout = collect_page_layout(pdf_path, pno)
            units, idx_map = spans_to_units(layout.get("spans", []))

            if not units:
                # No text on page, create empty translated page for both modes
                with transaction.atomic():
                    for mode in ["faithful", "readable"]:
                        TranslatedPage.objects.update_or_create(
                            book=book, page_no=pno, lang=target_lang, mode=mode,
                            defaults={"data": {}, "status": "ready"}
                        )
                continue

            image_src_map = _inject_images_from_db(layout, book, pno)
            logger.debug(f"[DEBUG-TASK]img_map : {image_src_map}")
            
            sys_msg, user_msg_json = build_prompt_born_digital(
                units, target_lang, book_title=book.title, book_genre=book.genre, glossary=book.glossary
            )
            logger.debug(f"[DEBUG-TASK]user_msg : {user_msg_json}")

            translated_units, errs = llm.translate_text_units(
                sys_msg=sys_msg, user_msg_json=user_msg_json, max_retries=2, expected_length=len(units)
            )

            with transaction.atomic():
                if translated_units and len(translated_units) == len(units):
                    html_faithful = build_faithful_html(
                        layout, translated_units, idx_map, 
                        image_src_map=image_src_map, 
                    )
                    logger.debug(f"[DEBUG-TASK]html_faithful : {html_faithful}")
                    html_readable = build_readable_html(layout, translated_units, idx_map)
                    logger.debug(f"[DEBUG-TASK]html_readable : {html_readable}")

                    base_db_data = {
                        "schema_version": "weaver.page.born_digital.v1",
                        "page": {
                            "page_no": pno,
                            "page_w": page.width,
                            "page_h": page.height,
                            "page_units": "px"
                        },
                        "translated_units": translated_units
                    }

                    data_faithful = {**base_db_data, "html": html_faithful}
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=pno, lang=target_lang, mode="faithful",
                        defaults={"data": data_faithful, "status": "ready"}
                    )

                    data_readable = {**base_db_data, "html": html_readable}
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=pno, lang=target_lang, mode="readable",
                        defaults={"data": data_readable, "status": "ready"}
                    )
                else:
                    db_data = {"error": errs or ["Translated unit count mismatch."]}
                    for mode in ["faithful", "readable"]:
                        TranslatedPage.objects.update_or_create(
                            book=book, page_no=pno, lang=target_lang, mode=mode,
                            defaults={"data": db_data, "status": "failed"}
                        )

        ready_pages_count = TranslatedPage.objects.filter(book=book, lang=target_lang, mode='faithful', status='ready').count()
        if book.page_count == ready_pages_count:
            book.status = "completed"
        else:
            book.status = "failed"
        book.save(update_fields=["status"])
        return {"status": "done", "pages_processed": total_job_pages}

    except Exception as e:
        logger.error(f"Born-digital translation task for book {book_id} failed: {e}", exc_info=True)
        book.status = "failed"
        book.save(update_fields=["status"])
        raise

@shared_task(bind=True)
def retranslate_single_page(self, book_id: int, page_no: int, target_lang: str, feedback: str):
    try:
        book = Book.objects.get(id=book_id)
        page = BookPage.objects.get(book=book, page_no=page_no)

        # Branch based on processing mode
        if book.processing_mode == 'born_digital':
            # --- BORN DIGITAL RETRANSLATION ---
            schema = load_weaver_schema(BORN_DIGITAL_SCHEMA_PATH)
            llm = GeminiHelper(schema=schema)
            pdf_path = book.original_file.path

            # Get current flawed translation units for context
            current_tp = TranslatedPage.objects.filter(book=book, page_no=page_no, lang=target_lang, mode='faithful').first()
            current_translated_units = None
            if current_tp and isinstance(current_tp.data, dict):
                current_translated_units = current_tp.data.get('translated_units')

            # Get original text units from PDF
            layout = collect_page_layout(pdf_path, page_no)
            units, idx_map = spans_to_units(layout.get("spans", []))

            if not units:
                logger.info(f"Retranslation skipped for born-digital page {page_no} as it has no text.")
                with transaction.atomic():
                    for mode in ["faithful", "readable"]:
                        TranslatedPage.objects.update_or_create(
                            book=book, page_no=page_no, lang=target_lang, mode=mode,
                            defaults={"data": {}, "status": "ready"}
                        )
                return

            # Build prompt with feedback
            sys_msg, user_msg_json = build_prompt_born_digital(
                units=units,
                target_lang=target_lang,
                book_title=book.title,
                book_genre=book.genre,
                glossary=book.glossary,
                user_feedback=feedback,
                current_translation=current_translated_units
            )

            # Call LLM for new translation
            translated_units, errs = llm.translate_text_units(
                sys_msg=sys_msg, user_msg_json=user_msg_json, max_retries=2, expected_length=len(units)
            )

            image_src_map = _inject_images_from_db(layout, book, page_no)
            logger.debug(f"[DEBUG-TASK]img_map : {image_src_map}")

            # Save new result
            with transaction.atomic():
                if translated_units and len(translated_units) == len(units):
                    html_faithful = build_faithful_html(
                        layout, translated_units, idx_map, 
                        image_src_map=image_src_map, 
                    )
                    html_readable = build_readable_html(layout, translated_units, idx_map)

                    base_db_data = {
                        "schema_version": "weaver.page.born_digital.v1",
                        "page": {
                            "page_no": page_no,
                            "page_w": page.width,
                            "page_h": page.height,
                            "page_units": "px"
                        },
                        "translated_units": translated_units
                    }

                    data_faithful = {**base_db_data, "html": html_faithful}
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=page_no, lang=target_lang, mode="faithful",
                        defaults={"data": data_faithful, "status": "ready"}
                    )

                    data_readable = {**base_db_data, "html": html_readable}
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=page_no, lang=target_lang, mode="readable",
                        defaults={"data": data_readable, "status": "ready"}
                    )
                else:
                    db_data = {"error": errs or ["Retranslation failed: Translated unit count mismatch."]}
                    for mode in ["faithful", "readable"]:
                        TranslatedPage.objects.update_or_create(
                            book=book, page_no=page_no, lang=target_lang, mode=mode,
                            defaults={"data": db_data, "status": "failed"}
                        )
            logger.info(f"Born-digital retranslation for book {book_id}, page {page_no} completed.")
            return # End of born-digital path

        # --- AI LAYOUT (EXISTING) RETRANSLATION ---
        
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
