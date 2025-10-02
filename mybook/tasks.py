# mybook/tasks.py
import json, logging, os, time
from bs4 import BeautifulSoup
from pathlib import Path
from django.conf import settings
from celery import shared_task
from django.db import transaction
from mybook.models import Book, BookPage, PageImage, TranslatedPage, ApiUsageLog

from .utils.born_digital_prompt import build_prompt_born_digital
from .utils.faithful_prompt import build_prompt_faithful
from .utils.gemini_helper import GeminiHelper
from .utils.gemini_file_ref import GeminiFileRefManager
from .utils.extract_image import split_images_for_prompt, ImageExtractor
from .utils.schema_loader import load_weaver_schema
from .utils.paragraphs import calculate_line_height_for_paragraph
from .utils.nonbody_detector import NonBodyDetector # NonBodyDetector 임포트
from .utils.born_digital import (
    collect_page_layout, build_faithful_html, build_readable_html
)
from PIL import Image

from urllib.parse import urljoin

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "page.v2.json"
BORN_DIGITAL_SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "born_digital.v1.json"
RETRANSLATE_SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "retranslate_schema.json"
BORN_DIGITAL_LAYOUT_TRANSLATION_SCHEMA_PATH = Path(settings.BASE_DIR) / "mybook" / "utils" / "born_digital_layout_and_translation.v1.json"

def _log_api_usage(book_id: int, request_type: str, usage_metadata, model_name: str):
    """LLM API 사용량을 별도 테이블에 기록하는 헬퍼 함수"""
    if not usage_metadata:
        return

    # [DEBUG] Log the full usage_metadata object to a file for inspection.
    try:
        # Use repr() to get a developer-friendly string representation of the object.
        # This works even if the object is not directly JSON serializable.
        metadata_repr = repr(usage_metadata)
        dump_path = Path(settings.BASE_DIR) / "test" / "usage_metadata" / f"usage_metadata_book_{book_id}_{int(time.time())}.txt"
        with open(dump_path, 'w', encoding='utf-8') as f:
            f.write(metadata_repr)
        logger.debug(f"Dumped usage_metadata for book {book_id} to {dump_path}")
    except Exception as dump_e:
        logger.warning(f"Could not dump usage_metadata for book {book_id}: {dump_e}")

    try:
        # getattr을 사용하여 필드가 없는 경우에도 안전하게 0을 기본값으로 사용합니다.
        prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
        completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
        total_tokens = getattr(usage_metadata, 'total_token_count', 0)
        cached_tokens = getattr(usage_metadata, 'cached_content_token_count', 0)
        thoughts_tokens = getattr(usage_metadata, 'thoughts_token_count', 0)

        book = Book.objects.select_related('owner').get(id=book_id)
        ApiUsageLog.objects.create(
            user=book.owner,
            book=book,
            request_type=request_type,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thoughts_tokens,
            total_tokens=total_tokens,
        )
    except Exception as e:
        logger.error(f"Failed to log API usage for book {book_id}: {e}", exc_info=True)

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
        .values("xref","path","bbox","clip_bbox","clip_path","img_w","img_h","origin_w","origin_h","transform")
    )
    images = []
    for pi in pis:
        images.append({
            "xref": int(pi["xref"]) if pi["xref"] is not None else None,
            "path": pi["path"],
            "bbox": pi.get("bbox"),             # px, xywh
            "clip_bbox": pi.get("clip_bbox"),   # px, xywh or None
            "clip_path_data_px": pi.get("clip_path"),
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
def translate_book_pages(self, book_id: int, target_lang: str, page_numbers_to_process: list[int] | None = None, model_type: str = 'standard', thinking_level: str = 'medium'):
    book = Book.objects.get(id=book_id)
    try:
        file_manager = GeminiFileRefManager()
        llm = GeminiHelper(schema=load_weaver_schema(SCHEMA_PATH), model_type=model_type, thinking_level=thinking_level)

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

            # ai_layout은 다음 페이지 컨텍스트를 위해 다음 페이지도 함께 전달합니다.
            pages_to_send = [pno]
            if pno < total_book_pages:
                pages_to_send.append(pno + 1)
            
            page_part = file_manager.get_page_parts(book, pages_to_send)
            
            images = list(PageImage.objects.filter(book=book, page_no=pno).values("ref", "bbox"))
            _, figures = split_images_for_prompt(images, p["width"], p["height"])

            page_ctx = {
                "page_no": pno,
                "size": {"w": p["width"], "h": p["height"], "units": "px"},
                "images": [{"ref": im["ref"], "bbox": im["bbox"]} for im in figures] if figures else []
            }

            # 3. build_prompt_faithful에는 항상 '전체' 페이지 수를 전달
            sys_msg, user_msg, example_json = build_prompt_faithful(page_ctx, target_lang, total_book_pages, book_title=book.title, book_genre=book.genre, prev_page_html=prev_page_html_context, glossary=book.glossary, next_page_context_available=(pno < total_book_pages))
            logger.debug(f"[DEBUG-TASK]user_msg : {user_msg}")

            data, errs, usage_metadata = llm.generate_page_json( # type: ignore
                file_parts=page_part, sys_msg=sys_msg, user_msg=user_msg, example_json=example_json, max_retries=2
            )

            _log_api_usage(book_id, 'translate_book_page', usage_metadata, llm.model_name)

            if data:
                # 4. 이미지 추출 로직 유지
                if data.get("has_figures"):
                    page_no_from_data = data["page"]["page_no"]
                    norm_w, norm_h = data["page"]["page_w"], data["page"]["page_h"]
                    pdf_path = book.original_file.path

                    # 1. HTML에서 실제 data-ref를 추출하고, 설명 레이블과 매핑합니다.
                    soup = BeautifulSoup(data.get("html_stage", ""), "html.parser")
                    html_refs = [img.get("data-ref") for img in soup.find_all("img") if img.get("data-ref")] # type: ignore

                    # 페이지 렌더링을 위한 ImageExtractor 인스턴스
                    iex = ImageExtractor(pdf_path, norm_w, norm_h)
                    page_image_for_detect = iex._render_page_raster(page_no_from_data)
                    
                    # 2단계: Bbox 탐지 API 호출
                    detected_boxes, detect_errs, usage_meta_detect = llm.detect_figures(
                        image=page_image_for_detect,
                        # labels 인자는 생성된 슬러그가 객체 탐지에 방해가 될 수 있으므로 제거합니다.
                        labels=data.get("figure_labels")
                    )

                    _log_api_usage(book_id, 'detect_figures', usage_meta_detect, llm.model_name)

                    if detected_boxes:
                        fig_dir = os.path.join(settings.MEDIA_ROOT, "figures", f"book_{book.id}", f"page_{page_no_from_data}") # type: ignore
                        os.makedirs(fig_dir, exist_ok=True)

                        normalized_figures = ImageExtractor.normalize_bboxes_from_gemini(detected_boxes, norm_w, norm_h, expand_px=3)
                        # 2. HTML에 있는 data-ref의 순서와 탐지된 객체의 순서가 일치한다고 가정하고 ref를 할당합니다.
                        if len(normalized_figures) == len(html_refs):
                            for i, fig in enumerate(normalized_figures):
                                fig['ref'] = html_refs[i]
                        else:
                            logger.warning(f"Mismatch between detected figures ({len(normalized_figures)}) and HTML refs ({len(html_refs)}) for book {book.id} page {pno}. Falling back to label-based refs.") #type:ignore
                            for i, fig in enumerate(normalized_figures):
                                fig['ref'] = fig.get('label') or f"fig_p{page_no_from_data}_{i+1}"

                        # 이미지 잘라내고 상세 정보(경로, 너비, 높이) 받기
                        ref_to_details = iex.extract_many(page_no_from_data, normalized_figures, output_dir=fig_dir, prefix="fig", ext="png")

                        # URL 생성 및 데이터 주입
                        image_src_map = {}
                        for ref, details in ref_to_details.items():
                            rel = details['path'].replace(settings.MEDIA_ROOT, "").lstrip("/\\")
                            url = settings.MEDIA_URL.rstrip("/") + "/" + rel
                            image_src_map[ref] = url
                            details['url'] = url

                        data["image_details_map"] = ref_to_details
                        data["image_src_map"] = image_src_map
                    
                    elif detect_errs:
                        logger.warning(f"Figure detection failed for book {book.id} page {pno}: {detect_errs}") #type:ignore

                    iex.close()


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

def _dump_gemini_response_for_debug(book_id, page_no, attempt, response_content, is_json=False):
    """Saves Gemini response to a debug file."""
    try:
        debug_dir = Path(settings.BASE_DIR) / "test" / "gemini_responses"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ext = "json" if is_json else "txt"
        file_path = debug_dir / f"book_{book_id}_p{page_no}_attempt_{attempt}_{timestamp}.{ext}"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response_content)
        logger.info(f"Saved Gemini debug response to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to dump Gemini debug response: {e}")

@shared_task(bind=True)
def translate_book_pages_born_digital(self, book_id: int, target_lang: str, page_numbers_to_process: list[int] | None = None, model_type: str = 'standard', thinking_level: str = 'medium'):
    book = Book.objects.get(id=book_id)
    try:
        file_manager = GeminiFileRefManager()
        llm = GeminiHelper(schema=load_weaver_schema(BORN_DIGITAL_LAYOUT_TRANSLATION_SCHEMA_PATH), model_type=model_type, thinking_level=thinking_level)

        if page_numbers_to_process:
            pages_qs = BookPage.objects.filter(book=book, page_no__in=page_numbers_to_process)
        else:
            pages_qs = BookPage.objects.filter(book=book)
        
        pages = list(pages_qs.order_by("page_no"))

        # [OPTIMIZATION] 모든 페이지의 레이아웃을 미리 읽어 메모리에 캐싱합니다.
        # 이렇게 하면 다음 페이지 컨텍스트를 위해 PDF를 반복적으로 읽는 것을 방지할 수 있습니다.
        raw_layouts = {p.page_no: collect_page_layout(book.original_file.path, p.page_no) for p in pages}
        # 이전 페이지의 번역 결과를 저장할 캐시
        translated_results_cache = {}

        total_job_pages = len(pages)
        pdf_path = book.original_file.path

        for i, page in enumerate(pages, start=1):
            pno = page.page_no
            self.update_state(state="PROGRESS", meta={"current": i, "total": total_job_pages})
            
            # 각 페이지에 해당하는 Part만 가져옵니다.
            page_part = file_manager.get_page_part(book, pno)
            
            # 1. Gemini에 전달할 원본 스팬 데이터 수집
            raw_layout_data = raw_layouts.get(pno, {})
            spans = raw_layout_data.get("spans", [])

            # 이전/다음 페이지 컨텍스트 수집
            previous_page_body_text = None
            previous_page_original_text = None
            if pno > 1:
                prev_page_result = None
                # 현재 작업에서 이미 처리된 경우 캐시에서 가져옵니다.
                if (pno - 1) in translated_results_cache:
                    prev_page_result = translated_results_cache[pno - 1]
                else:
                    # DB에서 이전 페이지 결과를 찾아봅니다. (재시도 등의 경우)
                    prev_tp = TranslatedPage.objects.filter(book=book, page_no=pno - 1, lang=target_lang, mode='faithful').first()
                    if prev_tp and isinstance(prev_tp.data, dict):
                        prev_page_result = prev_tp.data.get('gemini_result')
                
                if prev_page_result:
                    # role이 'body'인 문단의 번역 텍스트만 추출하여 컨텍스트로 사용합니다.
                    body_paras = [p for p in prev_page_result if p.get('role') == 'body']
                    if body_paras:
                        previous_page_body_text = "\n".join([p.get('translated_text', '') for p in body_paras[-3:]])
                        previous_page_original_text = body_paras[-1].get('original_text', '')

                    logger.debug(f"Previous page {pno-1} body text for context: {previous_page_body_text}")
                    logger.debug(f"Previous page {pno-1} original text for context: {previous_page_original_text}")
            next_page_span_texts = None
            # 다음 페이지의 원본 스팬 정보를 미리 읽어둔 데이터에서 가져옵니다.
            if (pno + 1) in raw_layouts:
                next_page_raw_layout = raw_layouts.get(pno + 1, {})
                # span 객체 전체가 아닌 text만 추출하여 전달합니다.
                # [FIX] 공백만 있는 스팬은 건너뛰고 의미 있는 텍스트만 수집합니다.
                next_page_spans = next_page_raw_layout.get("spans", [])
                if next_page_spans:
                    next_page_span_texts = [
                        s.get('text', '') for s in next_page_spans if s.get('text', '').strip()
                    ][:8] # 의미 있는 텍스트 8개를 수집
                    logger.debug(f"Next page {pno+1} spans for context: {next_page_span_texts}")

            # 텍스트가 없는 페이지(예: 전체 이미지 페이지) 처리 로직
            if not spans:
                logger.info(f"Book {book_id}, Page {pno}: No text spans found. Skipping Gemini call and building image-only HTML.")
                
                # 이미지만으로 faithful HTML 생성
                image_src_map = _inject_images_from_db(raw_layout_data, book, pno)
                html_faithful = build_faithful_html(raw_layout_data, [], image_src_map=image_src_map)
                
                # 가독 모드는 내용이 없으므로 빈 HTML 생성
                html_readable = build_readable_html([])

                with transaction.atomic():
                    db_data_faithful = {"schema_version": "weaver.page.born_digital.v1", "html": html_faithful}
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=pno, lang=target_lang, mode="faithful",
                        defaults={"data": db_data_faithful, "status": "ready"}
                    )
                    db_data_readable = {"schema_version": "weaver.page.born_digital.v1", "html": html_readable}
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=pno, lang=target_lang, mode="readable",
                        defaults={"data": db_data_readable, "status": "ready"}
                    )
                continue # 다음 페이지로 넘어감

            # 2. 통합 프롬프트 생성
            sys_msg, user_msg_json = build_prompt_born_digital(
                spans=spans,
                target_lang=target_lang,
                total_pages=total_job_pages,
                page_no=pno,
                book_title=book.title,
                book_genre=book.genre,
                glossary=book.glossary,
                previous_page_body_text=previous_page_body_text,
                next_page_span_texts=next_page_span_texts,
                previous_page_original_text=previous_page_original_text
            )

            # 3. Gemini API 호출
            # generate_page_json은 내부적으로 JSON 스키마 검증을 수행
            # layout_and_translation_data, errs = llm.generate_page_json(
            #     file_part=file_part, sys_msg=sys_msg, user_msg={"prompt": user_msg_json}, max_retries=2
            # )
            
            # --- Modified for Debugging ---
            raw_text, data, errs, usage_metadata = llm.generate_page_json_with_raw_response(
                file_parts=page_part, sys_msg=sys_msg, user_msg={"prompt": user_msg_json}, max_retries=2,
                debug_callback=lambda attempt, content, is_json: _dump_gemini_response_for_debug(book_id, pno, attempt, content, is_json)
            )
            layout_and_translation_data = data
            # --- End of Modification ---

            _log_api_usage(book_id, 'translate_born_digital', usage_metadata, llm.model_name)
            
            # 4. 결과 처리 및 저장
            if layout_and_translation_data:
                # 현재 페이지의 번역 결과를 다음 페이지를 위해 캐시에 저장합니다.
                translated_results_cache[pno] = layout_and_translation_data

                # 서버에서 original_text를 재구성합니다.
                for para in layout_and_translation_data:
                    indices = para.get("span_indices", [])
                    original_text = " ".join(spans[i]['text'] for i in indices if i < len(spans))
                    para['original_text'] = original_text.strip()

                # [REVISED] Gemini가 반환한 문단별 span_indices를 사용하여 직접 line-height를 계산하고 주입합니다.
                # 이렇게 하면 로컬의 휴리스틱 문단 감지 로직에 의존하지 않고 안정적으로 줄간격을 계산할 수 있습니다.
                for para_from_gemini in layout_and_translation_data:
                    para_from_gemini['line_height'] = calculate_line_height_for_paragraph(
                        span_indices=para_from_gemini.get('span_indices', []),
                        all_spans=spans
                    )

                # Gemini가 반환한 paragraph 구조를 HTML 빌더에 전달
                # build_faithful_html은 이제 번역된 텍스트를 직접 사용
                image_src_map = _inject_images_from_db(raw_layout_data, book, pno)
                html_faithful = build_faithful_html(
                    raw_layout_data, layout_and_translation_data, image_src_map=image_src_map # type: ignore
                )
                html_readable = build_readable_html(layout_and_translation_data) # type: ignore

                # DB에 저장할 데이터 구조화
                with transaction.atomic():
                    base_db_data = {
                        "schema_version": "weaver.page.born_digital.v1",
                        # LLM 응답에는 더 이상 book_id, page_no가 없으므로,
                        # 태스크 컨텍스트에서 가져와 DB에 저장합니다.
                        "page": {
                            "page_no": pno,
                            "page_w": page.width,
                            "page_h": page.height,
                            "page_units": "px"
                        },
                        # gemini_result는 이제 paragraph 객체의 리스트입니다.
                        "gemini_result": layout_and_translation_data
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
                # API 호출 실패
                with transaction.atomic():
                    db_data = {"error": errs or ["Gemini API call failed."]}
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
def retranslate_single_page(self, book_id: int, page_no: int, target_lang: str, feedback: str, model_type: str = 'standard', thinking_level: str = 'medium'):
    try:
        book = Book.objects.get(id=book_id)
        page = BookPage.objects.get(book=book, page_no=page_no)
        
        # Branch based on processing mode
        if book.processing_mode == 'born_digital':
            # --- BORN DIGITAL RETRANSLATION ---
            # The re-translation schema expects a simple array of strings.
            llm = GeminiHelper(schema=load_weaver_schema(RETRANSLATE_SCHEMA_PATH), model_type=model_type, thinking_level=thinking_level)
            file_manager = GeminiFileRefManager()

            # 1. Get the existing layout and translation data
            current_tp = TranslatedPage.objects.filter(book=book, page_no=page_no, lang=target_lang, mode='faithful').first()
            if not current_tp or not isinstance(current_tp.data, dict) or 'gemini_result' not in current_tp.data:
                raise ValueError(f"No valid previous translation data found for book {book_id}, page {page_no}.")

            # gemini_result는 이제 paragraph의 리스트입니다.
            paragraphs = current_tp.data.get('gemini_result', [])

            # --- [NEW] 이전/다음 페이지 컨텍스트 수집 ---
            previous_page_body_text = None
            previous_page_original_text = None
            if page_no > 1:
                prev_tp = TranslatedPage.objects.filter(book=book, page_no=page_no - 1, lang=target_lang, mode='faithful').first()
                if prev_tp and isinstance(prev_tp.data, dict):
                    prev_page_result = prev_tp.data.get('gemini_result')
                    if prev_page_result:
                        body_paras = [p for p in prev_page_result if p.get('role') == 'body']
                        if body_paras:
                            previous_page_body_text = "\n".join([p.get('translated_text', '') for p in body_paras[-3:]])
                            previous_page_original_text = body_paras[-1].get('original_text', '')
                        logger.debug(f"Retranslate context: Prev page {page_no-1} body text: {previous_page_body_text}")
                        logger.debug(f"Retranslate context: Prev page {page_no-1} original text: {previous_page_original_text}")

            next_page_span_texts = None
            try:
                next_page = BookPage.objects.get(book=book, page_no=page_no + 1)
                if next_page:
                    raw_layout = collect_page_layout(book.original_file.path, page_no + 1)
                    next_page_spans = raw_layout.get("spans", [])
                    if next_page_spans:
                        next_page_span_texts = [
                            s.get('text', '') for s in next_page_spans if s.get('text', '').strip()
                        ][:8]
                        logger.debug(f"Retranslate context: Next page {page_no+1} spans: {next_page_span_texts}")
            except BookPage.DoesNotExist:
                logger.debug(f"Retranslate context: Next page {page_no+1} does not exist.")
                pass
            except Exception as e:
                logger.warning(f"Error collecting next page context for re-translation: {e}")
                pass
            # --- 컨텍스트 수집 끝 ---

            if not paragraphs:
                logger.info(f"No paragraphs to re-translate for book {book_id}, page {page_no}.")
                return

            # 2. Build the re-translation prompt
            from .utils.born_digital_prompt import build_prompt_retranslate_born_digital
            sys_msg, user_msg_json = build_prompt_retranslate_born_digital(
                paragraphs=paragraphs,
                target_lang=target_lang,
                user_feedback=feedback,
                book_title=book.title,
                book_genre=book.genre,
                glossary=book.glossary,
                previous_page_body_text=previous_page_body_text,
                previous_page_original_text=previous_page_original_text,
                next_page_span_texts=next_page_span_texts
            )

            # 3. Call LLM for new translation
            newly_translated_texts, errs, usage_metadata = llm.translate_text_units( # type: ignore
                sys_msg=sys_msg, user_msg_json=user_msg_json, max_retries=2, expected_length=len(paragraphs)
            )

            _log_api_usage(book_id, 'retranslate_born_digital', usage_metadata, llm.model_name)

            # 4. Update and save the new result
            if newly_translated_texts and len(newly_translated_texts) == len(paragraphs):
                # Update the translated_text in the original gemini_result
                for i, para in enumerate(paragraphs): # paragraphs는 이제 리스트입니다.
                    para['translated_text'] = newly_translated_texts[i]
                
                # Re-build HTML with the new translations
                raw_layout_data = collect_page_layout(book.original_file.path, page_no)
                image_src_map = _inject_images_from_db(raw_layout_data, book, page_no)
                html_faithful = build_faithful_html(raw_layout_data, paragraphs, image_src_map=image_src_map)
                html_readable = build_readable_html(paragraphs)

                with transaction.atomic():
                    base_db_data = {
                        "schema_version": "weaver.page.born_digital.v1",
                        "page": {
                            "page_no": page_no,
                            "page_w": page.width,
                            "page_h": page.height,
                            "page_units": "px"
                        },
                        "gemini_result": paragraphs # Updated paragraphs list
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
                db_data = {"error": errs or ["Retranslation failed."]}
                for mode in ["faithful", "readable"]:
                    TranslatedPage.objects.update_or_create(
                        book=book, page_no=page_no, lang=target_lang, mode=mode, defaults={"data": db_data, "status": "failed"}
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
            , next_page_context_available=(page_no < book.page_count)
        )
        logger.debug(f"[DEBUG-RETRANSLATE-TASK] user_msg: {user_msg}")

        file_manager = GeminiFileRefManager()
        llm = GeminiHelper(schema=load_weaver_schema(SCHEMA_PATH), model_type=model_type, thinking_level=thinking_level)
        
        # [MODIFIED] ai_layout 재번역 시에도 다음 페이지 컨텍스트를 함께 전달합니다.
        pages_to_send = [page_no]
        if page_no < book.page_count:
            pages_to_send.append(page_no + 1)
        page_part = file_manager.get_page_parts(book, pages_to_send)
        
        data, errs, usage_metadata = llm.generate_page_json( # type: ignore
            file_parts=page_part, sys_msg=sys_msg, user_msg=user_msg, example_json=example_json, max_retries=2
        )

        _log_api_usage(book_id, 'retranslate_ai_layout', usage_metadata, llm.model_name)

        if data:
            if data.get("has_figures"):
                page_no_from_data = data["page"]["page_no"]
                norm_w, norm_h = data["page"]["page_w"], data["page"]["page_h"]
                pdf_path = book.original_file.path

                # 1. HTML에서 실제 data-ref를 추출하고, 설명 레이블과 매핑합니다.
                soup = BeautifulSoup(data.get("html_stage", ""), "html.parser")
                html_refs = [img.get("data-ref") for img in soup.find_all("img") if img.get("data-ref")] #type: ignore

                iex = ImageExtractor(pdf_path, norm_w, norm_h)
                page_image_for_detect = iex._render_page_raster(page_no_from_data)

                detected_boxes, detect_errs, usage_meta_detect = llm.detect_figures(
                    image=page_image_for_detect,
                    labels=data.get("figure_labels")
                )

                _log_api_usage(book_id, 'detect_figures_retranslate', usage_meta_detect, llm.model_name)

                if detected_boxes:
                    fig_dir = os.path.join(settings.MEDIA_ROOT, "figures", f"book_{book.id}", f"page_{page_no_from_data}") # type: ignore
                    os.makedirs(fig_dir, exist_ok=True)

                    normalized_figures = ImageExtractor.normalize_bboxes_from_gemini(detected_boxes, norm_w, norm_h, expand_px=3)
                    if len(normalized_figures) == len(html_refs):
                        for i, fig in enumerate(normalized_figures):
                            fig['ref'] = html_refs[i]
                    else:
                        logger.warning(f"Mismatch during re-translate for book {book.id} page {page_no}: detected figures ({len(normalized_figures)}) vs HTML refs ({len(html_refs)}).") #type:ignore
                        for i, fig in enumerate(normalized_figures):
                            fig['ref'] = fig.get('label') or f"fig_p{page_no_from_data}_{i+1}"
                            
                    ref_to_details = iex.extract_many(page_no_from_data, normalized_figures, output_dir=fig_dir, prefix="fig", ext="png")

                    image_src_map = {}
                    for ref, details in ref_to_details.items():
                        rel = details['path'].replace(settings.MEDIA_ROOT, "").lstrip("/\\")
                        url = settings.MEDIA_URL.rstrip("/") + "/" + rel
                        image_src_map[ref] = url
                        details['url'] = url

                    data["image_details_map"] = ref_to_details
                    data["image_src_map"] = image_src_map
                
                elif detect_errs:
                    logger.warning(f"Figure re-detection failed for book {book.id} page {page_no}: {detect_errs}") #type:ignore
                iex.close()

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
