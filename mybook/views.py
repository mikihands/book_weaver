import logging, os,hashlib, re, tempfile
from bs4 import BeautifulSoup
from pathlib import Path

from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any
import jwt

from django.conf import settings
from django.core.mail import EmailMessage
from django.core.files.base import ContentFile
from django.db import transaction, models
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
from django.core.files import File # Django File 객체를 임포트합니다.
from django.shortcuts import get_object_or_404, render, redirect
from django.utils.encoding import smart_str
from django.utils.translation import gettext as _
from django.utils.text import get_valid_filename
from django.urls import reverse
from django.http import FileResponse

from .serializers import (
    FileUploadSerializer, RegisterSerializer, LoginSerializer, 
    RetranslateRequestSerializer, StartTranslationSerializer, 
    BookSettingsSerializer, ContactSerializer, PageEditSerializer, SubscriptionSerializer, EmailVerificationStartSerializer,
    UpdateEmailSerializer, UpdatePasswordSerializer,
    BulkDeleteSerializer, PublishRequestSerializer, PaymentWebhookUpdateSerializer
)
from .models import Book, BookPage, PageImage, TranslatedPage, UserProfile
from .utils.font_scaler import FontScaler
from .utils.extract_image import extract_images_and_bboxes, is_fullpage_background
from .utils.layout_norm import normalize_pages_layout
from .utils.html_inject import inject_sources 
from .utils.pdf_sanitizer import sanitize_pdf_active_content # 새 임포트
from .utils.pdf_inspector import inspect_pdf, choose_processing_mode
from .utils.delete_dir_files import safe_remove
from .utils.membership_updater import MembershipUpdater
from .tasks import translate_book_pages, retranslate_single_page, translate_book_pages_born_digital
from .permissions import IsBookOwner
from common.mixins.hmac_sign_mixin import HmacSignMixin

logger = logging.getLogger(__name__)
AUTH_SERVER_URL=settings.AUTH_SERVER_URL
ALLOWED_CONTENT_TYPES = {"application/pdf"}  # 1차는 PDF만


def sha256_streaming(fileobj, chunk_size=1024 * 1024):
    """대용량 대응: 스트리밍으로 SHA-256 계산"""
    pos = fileobj.tell()
    fileobj.seek(0)
    h = hashlib.sha256()
    for chunk in iter(lambda: fileobj.read(chunk_size), b""):
        h.update(chunk)
    fileobj.seek(pos)  # 원위치 복구
    return h.hexdigest()


class BookUploadView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        """
        사용자로부터 PDF 파일을 받아 Book을 생성하고,
        페이지/이미지 메타를 추출한 뒤(동기),
        번역 파이프라인은 비동기로 처리 시작함.
        """
        serializer = FileUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        uploaded_file = serializer.validated_data["file"] # type: ignore
        target_language = serializer.validated_data["target_language"] # type: ignore
        title = serializer.validated_data.get("title", None) # type: ignore
        genre = serializer.validated_data.get("genre", None) # type: ignore
        user_profile = request.user

        # 0) 형식 체크 (우선 PDF만)
        ctype = getattr(uploaded_file, "content_type", None)
        if ctype not in ALLOWED_CONTENT_TYPES:
            return Response(
                {"error": f"Only PDF is supported for now. Got: {ctype}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # making safe filename
        orig_name = get_valid_filename(getattr(uploaded_file, "name", "uploaded.pdf"))

        sanitized_path = None  # finally에서 정리용    

        # 0) 업로드 스트림을 일단 메모리로 확보(대용량이면 NamedTemporaryFile을 써도 됨)
        def _read_all(fp):
            pos = fp.tell()
            fp.seek(0)
            data = fp.read()
            fp.seek(pos)
            return data
        raw_bytes = _read_all(uploaded_file)
        # 원본 기준 해시 계산 → 이 해시로만 멱등 보장
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()

        # 1) 임시파일에 쓴 뒤 Sanitize 시도
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            tmp_path = tmp.name

        try:
            sanitized_path, removed = sanitize_pdf_active_content(tmp_path) # (경로 or None, 키 리스트)
            # 2) 최종 바이트 확정: sanitize 결과 있으면 그걸로, 없으면 원본
            if sanitized_path:
                with open(sanitized_path, "rb") as f:
                    final_bytes = f.read()
                final_name = orig_name
            else:
                final_bytes = raw_bytes
                final_name = orig_name
            
            # 4) 멱등 생성
            book, created = Book.objects.get_or_create(
                owner=user_profile,
                file_hash=raw_hash,
                defaults={
                    "title": title or orig_name,
                    "genre": genre,
                    "status": "pending",
                    "original_file": ContentFile(final_bytes, name=final_name),
                },
            )

            if not created:
                return Response(
                    {"message": _("This file has already been uploaded."), "book_id": book.id}, # type:ignore
                    status=status.HTTP_200_OK,
                )

            # Set status to processing for the extraction phase
            book.status = 'processing'
            # record basic source info
            try:
                try:
                    uploaded_file.seek(0, 2)
                    size = uploaded_file.tell()
                    uploaded_file.seek(0)
                    book.source_size = size
                except Exception:
                    _path = sanitized_path if sanitized_path else tmp_path
                    book.source_size = os.path.getsize(_path)
                book.source_mime = ctype or book.source_mime
                book.save(update_fields=['status', 'source_mime', 'source_size'])
            except Exception:
                book.save(update_fields=['status'])

            # === 여기서부터는 book.original_file.path 접근 가능 ===
            # 3) PDF 검사: inspect_pdf 호출하여 메타/판정 저장
            pdf_path = book.original_file.path # 이제 이 경로는 항상 정화된 파일의 경로입니다.
            try:
                inspect_result = inspect_pdf(pdf_path)
                proc_mode = choose_processing_mode(inspect_result)

                # OCR 판정 간단 룰: 텍스트 커버리지가 높으면 OCR로 간주
                if inspect_result.text_layer_coverage is not None:
                    if inspect_result.text_layer_coverage >= 0.6:
                        book.ocr_status = 'ocr'
                    elif inspect_result.text_layer_coverage <= 0.2:
                        book.ocr_status = 'image'
                    else:
                        book.ocr_status = 'unknown'
                    book.text_coverage = float(inspect_result.text_layer_coverage)

                book.processing_mode = proc_mode
                book.avg_score = float(inspect_result.avg_score)
                book.median_score = float(inspect_result.median_score)
                book.inspection = inspect_result.asdict() # type: ignore
                # update source_size from inspector if available
                try:
                    book.source_size = int(inspect_result.file_size)
                except Exception:
                    pass
                book.save(update_fields=['ocr_status','text_coverage','processing_mode','avg_score','median_score','inspection','source_size'])
            except Exception as ie:
                # 검사 실패시에도 진행: 로깅만
                logger.exception("PDF inspection failed, continuing extraction: %s", str(ie))
                
            # 4) PDF 메타/이미지 추출 (동기)
            #    FileField가 저장됐으니 경로 접근 가능
            out_dir = os.path.join(settings.MEDIA_ROOT, "extracted_images", str(book.id)) # type: ignore
            os.makedirs(out_dir, exist_ok=True)

            extracted_pages = extract_images_and_bboxes(pdf_path, out_dir, dpi=144, media_root=settings.MEDIA_ROOT)
            # pages: [{ "page_no": int, "size": {"w":..., "h":...}, "images": [{"ref","path","bbox","xref"}] }, ...]
            norm_pages = normalize_pages_layout(extracted_pages, base_width=1200)

            # 4) DB 저장 (트랜잭션: 짧게)
            with transaction.atomic():
                # 페이지/이미지 벌크 생성
                page_objs = []
                img_objs = []
                for p in norm_pages:
                    page_objs.append(
                        BookPage(
                            book=book,
                            page_no=p["page_no"],
                            width=p["size"]["w"],
                            height=p["size"]["h"],
                            meta=p.get("meta")
                        )
                    )
                BookPage.objects.bulk_create(page_objs, ignore_conflicts=True)

                for p in norm_pages:
                    pno = p["page_no"]
                    for im in p.get("images", []):
                        img_objs.append(
                            PageImage(
                                book=book,
                                page_no=pno,
                                xref=im.get("xref", None),
                                ref=im["ref"],
                                path=im["path"],
                                bbox=im["bbox"],
                                transform=im.get("transform", None),
                                img_w=im.get("img_w"),
                                img_h=im.get("img_h"),
                                clip_bbox=im.get("clip_bbox"),
                                clip_path=im.get("clip_path_data_px"),
                                origin_w=im.get("origin_w"),
                                origin_h=im.get("origin_h"),
                            )
                        )
                if img_objs:
                    PageImage.objects.bulk_create(img_objs, ignore_conflicts=True)

                # page_count 업데이트
                book.page_count = len(norm_pages)
                book.status = "uploaded" # Extraction is done, now ready for translation prep
                book.save(update_fields=["page_count", "status"])

            prepare_url = request.build_absolute_uri(
                reverse("mybook:book_prepare", kwargs={"book_id": book.id}) # type: ignore
            )

            response_data = {
                "message": _("File processed successfully. Please prepare for translation."),
                "book_id": book.id, #type:ignore
                "prepare_url": prepare_url,
            }

            if sanitized_path:
                warning_template = _("The uploaded document contained dynamic elements (e.g., {keys}). "
                                     "To prevent potential security risks, these have been removed. "
                                     "As a result, some original interactive features or layouts may differ from the source.")
                warning_message = warning_template.format(keys=', '.join(list(set(removed))[:5]))
                response_data["sanitization_warning"] = warning_message

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            # FileField는 스토리지에 이미 올라갔을 수 있음 → 필요하면 정리 로직 추가
            logging.getLogger(__name__).exception("Upload pipeline failed")
            # 실패 표시
            try:
                if "book" in locals():
                    book.status = "failed"
                    book.save(update_fields=["status"])
            except Exception:
                pass
            return Response(
            {"error": f"An unexpected error occurred: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            # 임시파일 정리
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            if sanitized_path:
                try:
                    os.remove(sanitized_path)
                except Exception:
                    pass

import bleach
from bleach.css_sanitizer import CSSSanitizer

ALLOW_TAGS = bleach.sanitizer.ALLOWED_TAGS.union({
    "div","span","p","ul","ol","li","figure","figcaption","img",
    "table","thead","tbody","tr","th","td",
    "blockquote","code","pre","sup","sub",
    "h1","h2","h3","h4","h5","h6","a"
})

ALLOW_ATTRS = {
    **bleach.sanitizer.ALLOWED_ATTRIBUTES,
    "*": ["class","id","data-ref","data-image-ref","data-footnote-ref"],
    "img": ["src","alt","width","height","class","data-ref","data-image-ref"],
    "a": ["href","title","target","rel","class"],
    # 필요하면 table 관련 colspan/rowspan 추가 가능:
    "td": ["colspan","rowspan","class"], "th": ["colspan","rowspan","class"],
}

CSS_SAFE_PROPS = [
    # spacing & alignment
    "margin","margin-top","margin-right","margin-bottom","margin-left",
    "padding","padding-top","padding-right","padding-bottom","padding-left",
    "text-indent","text-align",
    # typography
    "line-height","letter-spacing","font-style","font-weight","text-decoration","font-size"
    # sizing (이미지/figure)
    "width","height","max-width"
]
css_sanitizer = CSSSanitizer(allowed_css_properties=CSS_SAFE_PROPS)


class BookPageView(APIView):
    permission_classes = [IsAuthenticated]  # 정책에 맞게 조정

    def get(self, request, book_id: int, page_no: int, *args, **kwargs):
        mode = request.GET.get("mode", "faithful")
        lang = request.GET.get("lang")
        bg_mode = request.GET.get("bg", "off")  # auto|on|off

        book = get_object_or_404(Book, id=book_id, owner=request.user)
        page = get_object_or_404(BookPage, book=book, page_no=page_no)

        tp_query = TranslatedPage.objects.filter(book=book, page_no=page_no, mode=mode)
        if lang:
            tp_query = tp_query.filter(lang=lang)
        tp = tp_query.first()

        # [FIX] URL에 lang 파라미터가 없을 때, DB에서 찾은 객체의 lang 값을 사용하도록 보정합니다.
        if tp and not lang:
            lang = tp.lang

        # v2 JSON 기대: schema_version=weaver.page.v2, fields: page, html_stage, figures
        data = tp.data if tp else None
        status_flag = tp.status if tp else "pending"

        # 1) 이미지 ref -> URL 매핑
        imgs_qs = PageImage.objects.filter(book=book, page_no=page_no).values("ref", "path", "bbox")
        imgs = list(imgs_qs)
        image_url_map: Dict[str, str] = {}
        for im in imgs:
            path = im["path"]
            rel = path
            if path.startswith(getattr(settings, "MEDIA_ROOT", "")):
                rel = path[len(settings.MEDIA_ROOT):].lstrip("/\\")
            try:
                url = default_storage.url(rel)
            except Exception:
                url = settings.MEDIA_URL.rstrip("/") + "/" + rel.replace("\\", "/")
            image_url_map[im["ref"]] = url

        # 2) 배경 이미지 결정 로직
        #    - auto: "페이지 전체를 덮는 이미지가 1장만 있고, 그 외 figure 후보가 없을 때"만 채택
        #    - on : 가장 큰 fullpage 후보가 있으면 사용
        #    - off: 항상 사용 안 함
        background_url = ""
        if imgs:
            # fullpage 후보 추출
            fullpage_candidates = [
                im for im in imgs if is_fullpage_background(im["bbox"], page.width, page.height)
            ]
            # fullpage 외의 일반 figure 후보
            figure_candidates = [im for im in imgs if im not in fullpage_candidates]

            use_bg = False
            if bg_mode == "on":
                use_bg = len(fullpage_candidates) >= 1
            elif bg_mode == "off":
                use_bg = False
            else:  # auto
                # 스캔 PDF 전형: fullpage 1장 & 나머지 없음 → 배경으로만 사용
                use_bg = (len(fullpage_candidates) == 1 and len(figure_candidates) == 0)

            if use_bg:
                # 여러 개라면 첫 번째만 사용(일반적으로 1개)
                bg_ref = fullpage_candidates[0]["ref"]
                background_url = image_url_map.get(bg_ref, "")

        # 3) v2.schema 처리
        html_stage_final = ""
        base_font_scale = 1.0
        page_w = page.width
        page_h = page.height

        css_vars = {}  # 기본값
        if data and data.get("schema_version") == "weaver.page.v2":
            # 안전 보정: figures bbox 클램프 , gemini_helper에서 이미 clamp 처리되어 저장됨
            W = data["page"]["page_w"]
            H = data["page"]["page_h"]

            # src 주입
            html_stage = data.get("html_stage", "")
            image_src_map = data.get("image_src_map", {})
            image_details_map = data.get("image_details_map", {})
            html_stage_final = inject_sources(html_stage, image_src_map, image_details_map)


            text_len, para_count = self._measure_html(html_stage_final)

            scaler = FontScaler(
                page_w_px=W, 
                page_h_px=H,
                default_base_font_px=18,
                lang=tp.lang if tp and tp.lang else "ko",
                density=request.GET.get("density","readable"), # readable|booky
            )

            policy = scaler.build_policy(text_len, para_count)
            css_vars = scaler.to_css_vars(policy)
            logger.debug(f"FontScalePolicy: {policy}, CSS: {css_vars}")

            # 페이지 사이즈는 v2가 주는 값 사용(정규화된 값)
            page_w = W
            page_h = H
        elif data and data.get("schema_version") == "weaver.page.born_digital.v1":
            # 'born_digital' 모드에서는 HTML이 이미 완성되어 있음
            # 'faithful' 모드는 이미지 src 주입이 필요할 수 있음
            if mode == 'faithful':
                html_stage_final = inject_sources(data.get("html", ""), image_url_map)
            else: # readable 모드는 이미지가 없음
                html_stage_final = data.get("html", "")

            page_info = data.get("page", {})
            if page_info.get("page_w") and page_info.get("page_h"):
                page_w = page_info["page_w"]
                page_h = page_info["page_h"]

        # 4) prev/next
        # Preserve query parameters like lang, mode, bg for navigation
        prev_url = next_url = None
        query_string = request.GET.urlencode()

        try:
            if page_no > 1:
                base_url = reverse("mybook:book_page", kwargs={"book_id": book.id, "page_no": page_no - 1}) #type: ignore
                prev_url = f"{base_url}?{query_string}" if query_string else base_url
            if book.page_count and page_no < book.page_count:
                base_url = reverse("mybook:book_page", kwargs={"book_id": book.id, "page_no": page_no + 1}) #type: ignore
                next_url = f"{base_url}?{query_string}" if query_string else base_url
        except Exception:
            from django.urls import NoReverseMatch
            try:
                if page_no > 1:
                    base_url = reverse("book_page", kwargs={"book_id": book.id, "page_no": page_no - 1}) #type: ignore
                    prev_url = f"{base_url}?{query_string}" if query_string else base_url
                if book.page_count and page_no < book.page_count:
                    base_url = reverse("book_page", kwargs={"book_id": book.id, "page_no": page_no + 1}) #type: ignore
                    next_url = f"{base_url}?{query_string}" if query_string else base_url
            except NoReverseMatch:
                prev_url = next_url = None

        # 5) 템플릿 컨텍스트
        ctx = {
            "book": book,
            "page_no": page_no,
            "lang": lang,
            "status": status_flag,
            "page_w": int(page_w),
            "page_h": int(page_h),
            "background_url": background_url,
            "html_stage_final": html_stage_final,  # ⬅️ 템플릿에 그대로 삽입
            "base_font_scale": base_font_scale,
            "prev_url": prev_url,
            "next_url": next_url,
            "mode": mode,
            "bg_mode": bg_mode,
        }

        ctx.update({"css_vars": css_vars})

        return render(request, "mybook/book_page_v2.html", ctx)
    
    def _measure_html(self, html: str) -> tuple[int, int]:
        soup = BeautifulSoup(html, 'html.parser')
        para_like = soup.find_all(["p","li","blockquote","h1","h2","h3","h4","h5","h6"])
        text_len = len(soup.get_text(" ", strip=True))
        return text_len, max(1, len(para_like))

class BookPageEditView(APIView):
    """
    번역된 페이지의 HTML 컨텐츠(요소의 스타일, 텍스트)를 수정하고 저장합니다.
    'born_digital' 모드에서는 span, 'ai_layout' 모드에서는 p, h1 등 블록 요소가 대상입니다.
    """
    permission_classes = [IsAuthenticated, IsBookOwner]

    def post(self, request, book_id: int, page_no: int, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book)  # 소유권 확인

        serializer = PageEditSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data

        mode = validated_data['mode']  # type:ignore
        lang = validated_data['lang']  # type:ignore
        changes = validated_data['changes']  # type:ignore

        try:
            tp = TranslatedPage.objects.get(book=book, page_no=page_no, lang=lang, mode=mode)
        except TranslatedPage.DoesNotExist:
            return Response({"error": _("해당 번역 페이지를 찾을 수 없습니다.")}, status=status.HTTP_404_NOT_FOUND)

        if not tp.data:
            return Response({"error": _("수정할 컨텐츠가 없습니다.")}, status=status.HTTP_400_BAD_REQUEST)

        # 'ai_layout' 모드는 'html_stage', 'born_digital' 모드는 'html' 키를 사용합니다.
        html_key = 'html_stage' if book.processing_mode != 'born_digital' else 'html'

        if html_key not in tp.data:
            return Response({"error": _("수정할 HTML 컨텐츠가 없습니다.")}, status=status.HTTP_400_BAD_REQUEST)

        original_html = tp.data[html_key]
        soup = BeautifulSoup(original_html, 'html.parser')

        for change in changes:
            element_id = change.get('element_id')
            if not element_id:
                continue
            element_to_edit = soup.find(id=element_id)

            if element_to_edit:
                if 'style' in change:
                    element_to_edit['style'] = change['style']  # type:ignore
                if 'text' in change:
                    element_to_edit.string = change['text']  # type:ignore
            else:
                logger.warning(f"Element with id '{element_id}' not found in page {page_no} for book {book_id}.")

        tp.data[html_key] = str(soup)
        tp.save(update_fields=['data'])

        return Response({"message": _("페이지가 성공적으로 수정되었습니다.")}, status=status.HTTP_200_OK)

class BookPrepareView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, book_id, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id, owner=request.user)
        # if book.status not in ['uploaded', 'failed']:
        #     # 이미 번역이 시작되었거나 완료된 경우, 첫 페이지로 리디렉션
        #     return redirect(reverse('mybook:book_page', kwargs={'book_id': book.id, 'page_no': 1})) # type: ignore
        return render(request, 'mybook/book_prepare.html', {'book': book})

class UpdateBookSettingsView(generics.UpdateAPIView):
    # 이 뷰가 다룰 모델의 쿼리셋을 지정.
    queryset = Book.objects.all()
    
    # 이 뷰가 사용할 시리얼라이저 클래스를 지정.
    serializer_class = BookSettingsSerializer
    
    # URL에서 객체를 식별할 필드와 키워드 인자를 지정.
    lookup_field = 'id'
    lookup_url_kwarg = 'book_id'
    
    # 권한 클래스들을 지정.
    permission_classes = [IsAuthenticated, IsBookOwner]

    def update(self, request, *args, **kwargs):
        
        super().update(request, *args, **kwargs)
        
        return Response({"message": _("설정이 성공적으로 저장되었습니다.")}, status=status.HTTP_200_OK)

class StartTranslationView(generics.GenericAPIView):
    """
    POST books/<int:book_id>/translate/
    - 본문: { "target_language": "...", "title"?: "...", "genre"?: "...", "glossary"?: "..." }
    - 소유자만 실행 가능 (IsAuthenticated + IsBookOwner)
    """
    serializer_class = StartTranslationSerializer
    queryset = Book.objects.select_related("owner") # get_object()에서 사용하기 위함
    permission_classes = [IsAuthenticated, IsBookOwner]
    lookup_field = 'id'
    lookup_url_kwarg = 'book_id'

    def post(self, request, book_id, *args, **kwargs):
        # 1) 책 객체 가져오기
        book = self.get_object() # 내부에서 self.check_object_permissions()가 호출됨
        # 2) 시리얼라이저 유효성 검사
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data
        # 3) 책 정보 업데이트
        book.title = validated_data.get('title', book.title)
        book.genre = validated_data.get('genre', book.genre)
        book.glossary = validated_data.get('glossary', book.glossary)

        book.status = 'processing'
        book.save(update_fields=['title', 'genre', 'glossary', 'status'])

        # 4) [NEW] 기존 번역이 있다면, 상태를 'pending'으로 변경하여 재번역 중임을 명시
        target_lang = validated_data['target_language']
        existing_translations = TranslatedPage.objects.filter(book=book, lang=target_lang)
        if existing_translations.exists():
            logger.info(f"Resetting status to 'pending' for {existing_translations.count()} existing translated pages for book {book.id}, lang {target_lang}.")
            # 'born_digital'은 faithful/readable 모두, 'ai_layout'은 faithful만 업데이트
            existing_translations.update(status='pending')

        # 4) 번역 작업 시작
        model_type = validated_data.get('model_type', 'standard')
        thinking_level = validated_data.get('thinking_level', 'medium')

        if book.processing_mode == 'born_digital':
            logger.debug(f"Starting born-digital translation for book {book.id} to {target_lang}")
            translate_book_pages_born_digital.delay(book.id, target_lang, model_type=model_type, thinking_level=thinking_level) # type: ignore
        else: # 'ai_layout' or 'mixed'
            logger.debug(f"Starting ai-layout translation for book {book.id} to {target_lang}")
            translate_book_pages.delay(book.id, target_lang, model_type=model_type, thinking_level=thinking_level) # type: ignore
        # 5) 응답
        page1_url = reverse('mybook:book_page', kwargs={'book_id': book.id, 'page_no': 1}) # type: ignore # type: ignore
        return Response({"message": "Translation has started.", "page1_url": page1_url}, status=status.HTTP_202_ACCEPTED)

class RetranslatePageView(generics.GenericAPIView):
    serializer_class = RetranslateRequestSerializer
    queryset = Book.objects.select_related("owner")
    permission_classes = [IsAuthenticated, IsBookOwner]
    lookup_field = 'id'
    lookup_url_kwarg = 'book_id'

    def post(self, request, book_id: int, page_no: int, *args, **kwargs):
        book = self.get_object()

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        feedback = serializer.validated_data.get('feedback', '') # type: ignore
        model_type = serializer.validated_data.get('model_type', 'standard') # type: ignore
        thinking_level = serializer.validated_data.get('thinking_level', 'medium') # type: ignore

        # Get target language from the query parameter
        target_lang = request.query_params.get('lang')

        if not target_lang:
            # If lang is not provided in the query, try to infer it.
            # This is a fallback and it's better for the client to provide it.
            distinct_langs = list(TranslatedPage.objects.filter(book=book).values_list('lang', flat=True).distinct())
            if len(distinct_langs) == 1:
                target_lang = distinct_langs[0]
            elif len(distinct_langs) == 0:
                 return Response({"error": "No translated pages found for this book. Cannot determine language."}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response(
                    {"error": f"Multiple translations exist ({', '.join(distinct_langs)}). Please specify which language to retranslate using the 'lang' query parameter."},
                    status=status.HTTP_400_BAD_REQUEST
                )

        # Set the specific page status to 'pending' to indicate re-translation.
        if book.processing_mode == 'born_digital':
            # For 'born_digital' books, both 'faithful' and 'readable' modes must be updated.
            TranslatedPage.objects.filter(
                book=book, page_no=page_no, lang=target_lang, mode__in=['faithful', 'readable']
            ).update(status='pending')
        else: # For 'ai_layout' or other modes, only 'faithful' mode is affected.
            TranslatedPage.objects.filter(
                book=book, page_no=page_no, lang=target_lang, mode='faithful'
            ).update(status='pending')

        retranslate_single_page.delay(book.id, page_no, target_lang, feedback, model_type=model_type, thinking_level=thinking_level) # type: ignore

        return Response({"message": f"Retranslation for page {page_no} ({target_lang.upper()}) has been queued."}, status=status.HTTP_202_ACCEPTED)


class RetryTranslationView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated, IsBookOwner]
    queryset = Book.objects.select_related("owner")
    lookup_field = 'id'
    lookup_url_kwarg = 'book_id'

    def post(self, request, book_id: int, *args, **kwargs):
        book = self.get_object()

        if book.status not in ['failed', 'completed']:
            return Response({"error": f"Cannot retry translation for a book with status '{book.status}'."}, status=status.HTTP_400_BAD_REQUEST)

        # Get target language
        first_tp = TranslatedPage.objects.filter(book=book).first()
        if not first_tp:
            # If no pages were translated at all, restart from scratch
            target_lang = request.data.get('target_language', 'ko') # Or get it from somewhere else
            pages_to_process = None # Process all pages
        else:
            target_lang = first_tp.lang
            # Find missing or failed pages
            all_page_nos = set(BookPage.objects.filter(book=book).values_list('page_no', flat=True))
            ready_page_nos = set(TranslatedPage.objects.filter(book=book, lang=target_lang, mode='faithful', status='ready').values_list('page_no', flat=True))
            pages_to_process = sorted(list(all_page_nos - ready_page_nos))

            if not pages_to_process:
                book.status = 'completed'
                book.save(update_fields=['status'])
                return Response({"message": "All pages are already translated successfully."}, status=status.HTTP_200_OK)

        book.status = 'processing'
        book.save(update_fields=['status'])
        
        translate_book_pages.delay(book.id, target_lang, page_numbers_to_process=pages_to_process) # type: ignore
        # For retry, use the book's current model_type and thinking_level
        model_type = book.model_type
        thinking_level = book.thinking_level
        translate_book_pages.delay(book.id, target_lang, page_numbers_to_process=pages_to_process, model_type=model_type, thinking_level=thinking_level)

        return Response({"message": f"Retry for {len(pages_to_process) if pages_to_process else 'all'} pages has been queued."}, status=status.HTTP_202_ACCEPTED)
    

class TranslatedPageStatusView(APIView):
    """
    특정 책 페이지의 번역 상태를 확인하는 API.
    GET /books/<int:book_id>/pages/<int:page_no>/status/
    """
    permission_classes = [IsAuthenticated, IsBookOwner]

    def get(self, request, book_id: int, page_no: int, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book) # IsBookOwner 권한 검사

        mode = request.GET.get("mode", "faithful")
        lang = request.GET.get("lang")
        
        tp_query = TranslatedPage.objects.filter(
            book=book, 
            page_no=page_no, 
            mode=mode
        )
        if lang:
            tp_query = tp_query.filter(lang=lang)
        tp_status = tp_query.values_list('status', flat=True).first()

        # TranslatedPage 객체가 아직 생성되지 않았다면 'pending'으로 간주
        current_status = tp_status if tp_status else "pending"

        return Response({"status": current_status})
    
class UploadPage(APIView):
    permission_classes =[AllowAny]
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/upload_page.html')


class RegisterView(HmacSignMixin, APIView):
    permission_classes = [AllowAny]
    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        AUTH_REGISTER_ENDPOINT = "/api/accounts/register/"
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            resp = self.hmac_post(AUTH_REGISTER_ENDPOINT, serializer.validated_data) # type: ignore
        except Exception as e:
            return Response(
                {"error": f"Failed to connect to authentication server: {e}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if resp.status_code == status.HTTP_201_CREATED:
            auth_data = resp.json()
            # 멱등 필요 시 get_or_create 권장
            UserProfile.objects.create(
                username=auth_data["user"]["username"],
                email=auth_data["user"]["email"],
                secret_key=auth_data["user"]["secret_key"],
            )
            return Response({"message": "User registered successfully."}, status=status.HTTP_201_CREATED)

        # 실패를 그대로 전달
        try:
            return Response(resp.json(), status=resp.status_code)
        except ValueError:
            return Response({"error": "Auth server returned non-JSON error."}, status=resp.status_code)


class StartEmailVerificationView(HmacSignMixin, APIView):
    """
    회원가입을 위한 이메일 검증을 시작합니다.
    POST /auth/email-verification/start/
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = EmailVerificationStartSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        endpoint = "/api/accounts/email/verification/start"
        payload = {
            "email": serializer.validated_data['email'],
            "service_name": "weaver"
        }

        try:
            resp = self.hmac_post(endpoint, payload)
            resp.raise_for_status()
            return Response(resp.json(), status=resp.status_code)
        except Exception as e:
            logger.error(f"Failed to start email verification. Error: {e}")
            try:
                return Response(e.response.json(), status=e.response.status_code) # type: ignore
            except:
                return Response({"error": _("인증 서버와 통신 중 오류가 발생했습니다.")}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class CheckEmailVerificationStatusView(HmacSignMixin, APIView):
    """
    이메일 검증 상태를 확인합니다.
    GET /auth/email-verification/status/?request_id=...
    """
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        request_id = request.query_params.get('request_id')
        endpoint = f"/api/accounts/email/verification/status?request_id={request_id}"
        resp = self.hmac_get(endpoint)
        return Response(resp.json(), status=resp.status_code)
    
class RegisterSuccessView(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/register_success.html')
    
class RegisterPage(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/register_page.html')
    
class LoginView(HmacSignMixin, APIView):
    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)
        AUTH_TOKEN_ENDPOINT = "/api/accounts/token/"

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            resp = self.hmac_post(AUTH_TOKEN_ENDPOINT, serializer.validated_data) # type: ignore
        except Exception as e:
            return Response(
                {"error": f"Failed to connect to authentication server: {e}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        if resp.status_code == status.HTTP_200_OK:
            auth_data = resp.json()
            
            # JWT 토큰을 세션에 저장
            request.session['access_token'] = auth_data.get("access")
            request.session['refresh_token'] = auth_data.get("refresh")
            request.session['username'] = serializer.validated_data['username'] #type:ignore
            
            # JWT refresh 토큰 만료일을 세션 만료일로 설정
            refresh_exp = auth_data.get("refresh_exp")
            if refresh_exp:
                try:
                    refresh_exp_datetime = datetime.fromisoformat(refresh_exp.replace("Z", "+00:00"))
                    expire_in_seconds = int((refresh_exp_datetime - datetime.now()).total_seconds())
                    request.session.set_expiry(expire_in_seconds)
                except ValueError:
                    logger.error("Invalid expiration timestamp received from auth server.")

            return Response({
                "message": "Login successful.",
                "redirect_url": "/" # 로그인 성공 후 메인 페이지로 리디렉션
            }, status=status.HTTP_200_OK)
        
        # 실패를 그대로 전달
        try:
            return Response(resp.json(), status=resp.status_code)
        except ValueError:
            return Response({"error": "Auth server returned non-JSON error."}, status=resp.status_code)

class LoginPage(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/login_page.html')
    
class PricingPage(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/pricing.html')

class FeaturesPage(APIView):
    def get(self, request, *args, **kwargs):
        return render(request, 'mybook/features.html')

class BookshelfView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        # 사용자가 소유한 모든 책을 가져옵니다.
        # Book 모델을 기준으로 LEFT JOIN하여 각 책에 연결된 모든 번역 언어 정보를 가져옵니다.
        # 이로 인해 한 책이 여러 언어로 번역된 경우 (Book, lang1), (Book, lang2)처럼 여러 행으로 나타나고,
        # 번역이 아직 없는 책은 lang이 null인 한 개의 행으로 나타납니다.
        books_with_langs = Book.objects.filter(
            owner=request.user
        ).values(
            'id',
            'title',
            'created_at',
            'status',
            'page_count',
            'translated_pages__lang'  # This performs the LEFT JOIN
        ).distinct().order_by('-created_at', 'translated_pages__lang')

        # Book.status의 display 값을 미리 준비합니다.
        status_display_map = dict(Book.PROCESSING_STATUS)

        # 템플릿에서 사용하기 편한 형태로 데이터를 재구성합니다.
        books_for_template = []
        for item in books_with_langs:
            books_for_template.append({
                'id': item['id'],
                'title': item['title'],
                'created_at': item['created_at'],
                'status': item['status'],
                'get_status_display': status_display_map.get(item['status'], item['status']),
                'page_count': item['page_count'],
                'target_lang': item['translated_pages__lang'], # 템플릿 변수명에 맞춤
            })

        return render(request, 'mybook/bookshelf.html', {'books': books_for_template})

class ProfileView(HmacSignMixin, APIView):
    """
    사용자 프로필(계정 관리) 페이지를 렌더링합니다.
    구독 정보, 결제 내역 등을 표시합니다.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        # --- 최신 구독 정보 동기화 ---
        updater = MembershipUpdater()
        success, message = updater.fetch_and_update(request)
        if not success:
            # 동기화 실패 시 사용자에게 알릴 수 있지만, 우선은 로깅만 처리합니다.
            logger.warning(f"'{request.user.username}'의 프로필 페이지에서 구독 정보 동기화 실패: {message}")
        # --------------------------

        user_profile = request.user

        # --- 결제 내역 조회 ---
        payment_history = []
        try:
            history_endpoint = "/api/payments/history/"
            params = {"service_name": "weaver"}
            resp = self.hmac_get(history_endpoint, request=request, params=params)
            resp.raise_for_status()
            payment_history = resp.json()
        except Exception as e:
            logger.error(
                f"Failed to fetch payment history for user {user_profile.username}. Error: {e}"
            )
            # API 호출에 실패해도 페이지는 정상적으로 렌더링되도록 빈 리스트를 전달합니다.
        #logger.debug(f"Payment history for user {user_profile.username}: {payment_history}")
        context = {
            'user_profile': user_profile,
            'payment_history': payment_history,
        }
        return render(request, 'mybook/profile.html', context)

class CancelSubscriptionView(HmacSignMixin, APIView):
    """
    사용자의 구독 해지를 처리합니다.
    POST /api/payments/cancel-subscription/
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user_profile = request.user
        if not user_profile.is_paid_member:
            return Response({"error": _("활성화된 구독이 없습니다.")}, status=status.HTTP_400_BAD_REQUEST)
        
        if user_profile.cancel_requested:
            return Response({"error": _("이미 구독 해지가 신청된 상태입니다.")}, status=status.HTTP_400_BAD_REQUEST)

        endpoint = "/api/payments/cancel-subscription/"
        payload = {"service_name": "weaver"}

        try:
            resp = self.hmac_post(endpoint, payload, request=request)
            resp.raise_for_status()
            response_data = resp.json()

            # 결제 서버로부터 받은 만료일로 UserProfile 업데이트
            if response_data.get('success') and response_data.get('end_date'):
                user_profile.end_date = response_data['end_date']
                user_profile.cancel_requested = True
                # is_paid_member는 구독 만료 웹훅을 통해 False로 변경되므로 여기서는 변경하지 않습니다.
                user_profile.save(update_fields=['end_date', 'cancel_requested'])
                
                logger.info(f"Subscription cancellation requested for user {user_profile.username}. Service remains active until {response_data['end_date']}.")
                return Response({
                    "message": _("구독 해지 신청이 완료되었습니다. 서비스는 만료일까지 유지됩니다."),
                    "end_date": response_data['end_date']
                }, status=status.HTTP_200_OK)
            else:
                raise Exception(response_data.get('error', 'Unknown error from payment server.'))

        except Exception as e:
            logger.exception(f"Failed to cancel subscription for user {user_profile.username}. Error: {e}")
            error_message = _("구독 해지 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            return Response({"error": error_message}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


def user_logout(request):
    """
    동일한 브라우저에서 다른 아이디로 로그인했을 때 세션을 공유하는 일이 없도록 완전히 flush
    """
    request.session.flush()
    # 로그아웃 후 메인 페이지로 이동
    return redirect("mybook:upload_page")


class BookSearchAPIView(APIView):
    """
    책 전체에서 번역된 텍스트를 검색하는 API.
    GET /books/<int:book_id>/search/?q=<query>
    """
    permission_classes = [IsAuthenticated, IsBookOwner]

    def get(self, request, book_id: int, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book)

        query = request.GET.get('q', '').strip()
        if not query or len(query) < 2:
            return Response({"error": "검색어는 2글자 이상 입력해주세요."}, status=status.HTTP_400_BAD_REQUEST)

        if book.processing_mode == 'born_digital':
            search_key = 'html'
        else:
            search_key = 'html_stage'

        # PostgreSQL의 JSONB 필드에 대한 icontains 쿼리 활용
        filter_kwargs = {
            'book': book,
            'status': 'ready',
            f'data__{search_key}__icontains': query
        }
        matching_pages = TranslatedPage.objects.filter(
            **filter_kwargs
        ).order_by('page_no')

        results = []
        processed_pages = set()

        for page in matching_pages:
            if page.page_no in processed_pages: continue
            html_content = page.data.get(search_key, '')
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(" ", strip=True)

            # Create snippet from the first match
            match = re.search(re.escape(query), text, re.IGNORECASE)
            if match:
                start_index = max(0, match.start() - 40)
                end_index = min(len(text), match.end() + 40)
                
                snippet_text = text[start_index:end_index]
                
                highlighted_snippet = re.sub(
                    f'({re.escape(query)})', 
                    r'<mark class="bg-yellow-200 rounded-sm px-1">\1</mark>', 
                    snippet_text, 
                    flags=re.IGNORECASE
                )

                if start_index > 0: highlighted_snippet = "..." + highlighted_snippet
                if end_index < len(text): highlighted_snippet = highlighted_snippet + "..."

                results.append({'page_no': page.page_no, 'snippet': highlighted_snippet})
                processed_pages.add(page.page_no)

        return Response(results)

def _client_ip(request):
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "")

class ContactAPIView(APIView):
    """
    POST /api/contact/
    - JSON Body: {name, email, subject, message, website?}
    - 웹/모바일 공용. (웹은 CSRF 필요, 모바일은 추후 토큰/JWT 적용 권장)
    """
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "contact"  # settings에서 rate 지정

    def post(self, request, *args, **kwargs):
        serializer = ContactSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        is_bot = bool(data.get("website"))  # honeypot # type: ignore
        ip = _client_ip(request) or "unknown"

        if not is_bot:
            name = data["name"] # type: ignore
            user_email = data["email"] # type: ignore
            subject = data["subject"] # type: ignore
            message = data["message"] # type: ignore

            admin_to = getattr(settings, "ADMIN_EMAIL_ADDRESS", None) or settings.EMAIL_HOST_USER
            from_addr = getattr(settings, "DEFAULT_FROM_EMAIL", settings.EMAIL_HOST_USER)

            email_subject = f"[Weaver Contact] {subject}"
            email_body = (
                f"From : {name} <{user_email}>\n"
                f"IP   : {ip}\n"
                f"-----\n\n"
                f"{message}\n"
            )

            mail = EmailMessage(
                subject=email_subject,
                body=email_body,
                from_email=from_addr,
                to=[admin_to],
                reply_to=[user_email],
            )
            mail.send(fail_silently=False)

        # 봇이든 아니든 동일 응답(탐지 회피)
        return Response({"ok": True}, status=status.HTTP_201_CREATED)

# -------- Bookshelf 개선작업 ------------    
class BulkDeleteBooksAPIView(APIView):
    permission_classes = [IsAuthenticated, IsBookOwner]

    def post(self, request):
        """
        요청 바디 예:
        { "ids": [3, 5, 9] }
        """
        ser = BulkDeleteSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        ids = ser.validated_data["ids"] # type:ignore

        # 소유자 한정
        qs = Book.objects.filter(id__in=ids, owner=request.user)  # UserProfile를 인증 user로 매핑해둔 전제
        found_ids = list(qs.values_list("id", flat=True))

        # 실제 삭제: on_delete=CASCADE로 DB는 정리되지만, 파일 정리는 직접 해야 함
        # - 원본 PDF: book.original_file.path
        # - figures: MEDIA_ROOT/figures/book_{id}
        # - 기타 생성물(추후 확장)
        deleted_count = 0
        with transaction.atomic():
            for book in qs:
                # 이미지페이지 제거 ( OCR 처리 안된 파일 이미지화 )
                figures_dir = Path(settings.MEDIA_ROOT) / "figures" / f"book_{book.id}" #type:ignore
                safe_remove(figures_dir)
                # 출판물 제거
                published_pdf = Path(settings.MEDIA_ROOT) / "published" / f"book_{book.id}.pdf" #type:ignore
                safe_remove(published_pdf)
                # 본문내 이미지 추출파일 제거
                extracted_images = Path(settings.MEDIA_ROOT) / "extracted_images" / f"{book.id}" #type:ignore
                safe_remove(extracted_images)
                # 원본파일
                original_files = Path(settings.MEDIA_ROOT) / "original" / f"book_{book.id}" #type:ignore
                safe_remove(original_files)

                # 마지막으로 DB 레코드 삭제
                book.delete()
                deleted_count += 1

        logger.info(f"Book삭제됨: user: {request.user.username}, count: {len(ids)}, deleted: {deleted_count}")
        return Response(
            {
                "deleted": deleted_count,
                "requested_ids": ids,
                "actually_deleted_ids": found_ids,
            },
            status=status.HTTP_200_OK,
        )


class PublishBookAPIView(APIView):
    """
    단일 책 PDF 출판.
    - faithful/readable 둘 다 지원.
    - 기본: faithful.
    - 즉시 동기 PDF 생성(WeasyPrint 설치 권장). 미설치 시 400 반환.
    """
    permission_classes = [IsAuthenticated, IsBookOwner]

    def _get_publish_params(self, request_data, book_id):
        ser = PublishRequestSerializer(data=request_data)
        ser.is_valid(raise_exception=True)
        validated_data = ser.validated_data
        lang = validated_data.get("lang") or self._guess_lang(book_id) #type:ignore
        mode = validated_data.get("mode", "faithful") #type:ignore
        return lang, mode

    def get(self, request, book_id: int):
        # 쿼리파라미터로 lang, mode 허용 ?lang=ko&mode=faithful
        lang, mode = self._get_publish_params(request.query_params, book_id)
        return self._publish_and_stream(book_id, lang, mode)

    def post(self, request, book_id: int):
        # POST body로 lang, mode
        lang, mode = self._get_publish_params(request.data, book_id)
        return self._publish_and_stream(book_id, lang, mode)

    # --------- 내부 헬퍼 ---------
    def _guess_lang(self, book_id: int) -> str:
        # 번역된 페이지 중 가장 많은 lang을 추정(간단 추정)
        agg = (
            TranslatedPage.objects
            .filter(book_id=book_id, status="ready")
            .values_list("lang", flat=True)
        )
        # 첫 값 또는 기본 ko
        return agg[0] if agg else "ko"

    def _collect_pages_html(self, book_id: int, lang: str, mode: str) -> list[dict]:
        """
        태스크 구현 차이에 따라 data 키가 다름:
        - born_digital 파이프라인: data['html'] 에 최종 HTML 저장됨 
        - NO OCR faithful 파이프라인: data['html_stage'] 를 사용
        
        Returns a list of dictionaries, each containing html, page_no, width, and height.
        """
        pages_qs = (
            TranslatedPage.objects
            .filter(book_id=book_id, lang=lang, mode=mode, status="ready")
            .order_by("page_no")
        )

        # Get all BookPage dimensions for this book in one query to avoid N+1
        page_dims = {
            p.page_no: (p.width, p.height) 
            for p in BookPage.objects.filter(book_id=book_id)
        }

        results: list[dict] = []
        for tp in pages_qs:
            data = tp.data or {}
            html = data.get("html") or data.get("html_stage") or ""
            width, height = page_dims.get(tp.page_no, (None, None))
            
            results.append({
                "html": html,
                "page_no": tp.page_no,
                "width": width,
                "height": height,
            })
        return results

    def _render_pdf(self, merged_html: str, dynamic_css: str, out_path: Path) -> bool:
        """
        WeasyPrint 기반 렌더링. 미설치/에러 시 False.
        """
        try:
            from weasyprint import HTML, CSS  # type: ignore
        except Exception:
            return False

        # 기본 CSS(여백/페이지 브레이크)
        base_css = """
            /* Default page size for pages without specific dimensions */
            @page { size: A4; margin: 16mm; }
            section.weaver-page { width: 100%; height: 100%; } /* Fill the page area */
            img { max-width: 100%; height: auto; }
        """
        final_css = base_css + "\n" + dynamic_css
        HTML(string=merged_html, base_url=str(settings.MEDIA_ROOT)).write_pdf(
            target=str(out_path),
            stylesheets=[CSS(string=final_css)],
        )
        return True

    def _publish_and_stream(self, book_id: int, lang: str, mode: str):
        book = get_object_or_404(Book, id=book_id, owner=self.request.user)

        # 상태 점검: completed가 아니라도(부분 성공 등) 출판을 허용할지 정책 결정
        # 1차 버전: 'ready' 페이지가 1장 이상이면 생성 시도
        html_pages_data = self._collect_pages_html(book_id, lang, mode)
        if not html_pages_data:
            return Response(
                {"detail": "출판 가능한 번역 페이지가 없습니다. (lang/mode 확인)"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Dynamic CSS and HTML sections generation
        css_rules = []
        html_sections = []
        unique_sizes = set()

        for page_data in html_pages_data:
            width = page_data.get('width')
            height = page_data.get('height')
            
            if width and height:
                w_int, h_int = int(width), int(height)
                page_size_name = f"s_{w_int}x{h_int}"
                
                if (w_int, h_int) not in unique_sizes:
                    css_rules.append(f"@page {page_size_name} {{ size: {w_int}px {h_int}px; margin: 0; }}")
                    unique_sizes.add((w_int, h_int))
                
                html_sections.append(f'<section class="weaver-page" style="page: {page_size_name}; page-break-after: always;">{page_data["html"]}</section>')
            else:
                # Fallback for pages without dimensions, uses default @page rule
                html_sections.append(f'<section class="weaver-page" style="page-break-after: always;">{page_data["html"]}</section>')


        # HTML 합치기
        doc_html = f"""
        <!doctype html>
        <html lang={lang}>
        <head>
          <meta charset="utf-8" />
          <title>{smart_str(book.title or 'Weaver Book')}</title>
        </head>
        <body>
          {''.join(html_sections)}
        </body>
        </html>
        """

        # WeasyPrint를 위해 이미지 경로를 URL에서 파일 시스템 경로로 변환
        soup = BeautifulSoup(doc_html, 'html.parser')
        for img in soup.find_all('img'):
            src = img.get('src') #type:ignore
            if src and src.startswith(settings.MEDIA_URL): #type:ignore
                # URL 경로(/media/...)를 파일 시스템 절대 경로로 변환
                # 예: /media/figures/book_1/fig.png -> /app/media/figures/book_1/fig.png
                relative_path = src[len(settings.MEDIA_URL):].lstrip('/\\') #type:ignore
                fs_path = os.path.join(settings.MEDIA_ROOT, relative_path)
                
                if os.path.exists(fs_path):
                    # WeasyPrint가 인식할 수 있도록 file:// URI로 변환
                    img['src'] = Path(fs_path).as_uri() #type:ignore
                else:
                    logger.warning(f"Image not found for PDF generation: {fs_path}")

        # 출력 경로
        pub_dir = Path(settings.MEDIA_ROOT) / "published"
        pub_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = pub_dir / f"book_{book.id}.pdf" #type:ignore

        # PDF 렌더
        final_html_for_pdf = str(soup)
        dynamic_css = "\n".join(css_rules)
        ok = self._render_pdf(final_html_for_pdf, dynamic_css, out_pdf)
        if not ok:
            # WeasyPrint가 없으면 에러 안내
            return Response(
                {"detail": "PDF 엔진(weasyprint) 미설치 또는 렌더 오류"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 성공 → 파일 스트리밍
        response = FileResponse(open(out_pdf, "rb"), as_attachment=True, filename=out_pdf.name)
        return response

class CreateSubscriptionView(HmacSignMixin, APIView):
    """
    사용자가 선택한 요금제(plan_type)로 결제 서버에 결제 페이지를 요청하고,
    반환된 URL로 사용자를 리디렉션 시킵니다.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = SubscriptionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        plan_type = serializer.validated_data['plan_type'] # type: ignore
        username = request.user.username

        # LANGUAGE_CODE에 따라 결제 게이트웨이 분기
        lang_code = request.LANGUAGE_CODE.split('-')[0]
        if lang_code == 'ko':
            endpoint = "/api/payments/toss-payment-page/"
        else:
            endpoint = "/api/payments/paypal-payment-page/"

        payload = {
            "username": username,
            "plan_type": plan_type,
            "service_name": "weaver"
        }

        try:
            resp = self.hmac_post(endpoint, payload, request=request)
            resp.raise_for_status() # 2xx 이외의 상태 코드에 대해 예외 발생
            
            response_data = resp.json()
            approval_url = response_data.get("approval_url")

            if not approval_url:
                logger.error(f"Approval URL not found in response from {endpoint}. Response: {response_data}")
                return Response({"error": _("결제 페이지 URL을 받지 못했습니다.")}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({"approval_url": approval_url}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception(f"Failed to create subscription for user {username} with plan {plan_type}. Error: {e}")
            err_msg = _("결제 서버와 통신 중 오류가 발생했습니다: {e}").format(e=e)
            return Response({"error":err_msg }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class TossPaymentSuccessView(APIView):
    """
    Toss 결제 성공 후 최종 리디렉션되는 페이지.
    GET /payment/toss/success/?pt=...&sd=...&ed=...
    결제 서버에서 전달된 쿼리 파라미터를 바탕으로 사용자에게 성공 결과를 보여줍니다.
    실제 사용자 정보 업데이트는 웹훅을 통해 비동기로 처리될 것을 기대합니다.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        plan_type = request.GET.get('pt')
        end_date = request.GET.get('ed')

        context = {
            'status': 'success',
            'message': _("구독이 성공적으로 활성화되었습니다!"),
            'plan_type': plan_type,
            'end_date': end_date,
        }
        # 사용자가 계정 페이지로 돌아가서 최신 상태를 확인할 수 있도록 안내합니다.
        # --- 최신 구독 정보 동기화 ---
        updater = MembershipUpdater()
        success, message = updater.fetch_and_update(request)
        if not success:
            # 동기화 실패 시 사용자에게 알릴 수 있지만, 우선은 로깅만 처리합니다.
            logger.warning(f"'{request.user.username}'의 프로필 페이지에서 구독 정보 동기화 실패: {message}")
        # --------------------------
        return render(request, 'mybook/payment_result.html', context)


class TossPaymentFailView(APIView):
    """
    Toss 결제 실패 후 최종 리디렉션되는 페이지.
    GET /payment/toss/fail/?code=...&message=...
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        error_code = request.GET.get('code')
        error_message = request.GET.get('message', _("알 수 없는 오류가 발생했습니다."))

        context = {
            'status': 'error',
            'message': _("결제에 실패했습니다."),
            'error_code': error_code,
            'error_message': error_message,
        }
        return render(request, 'mybook/payment_result.html', context)


class PayPalSuccessView(HmacSignMixin, APIView):
    """
    PayPal 결제 성공 후 리디렉션되는 URL.
    GET /payment/paypal_success/?info=...
    PayPal에서 받은 쿼리 파라미터를 결제 서버로 전달하여 최종 승인을 요청합니다.
    성공 시 UserProfile을 업데이트하고 결과 페이지를 렌더링합니다.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        info = request.query_params.get('info')
        if not info:
            logger.error("PayPal success redirect but 'info' query parameter is missing.")
            context = {
                'status': 'error',
                'message': _("결제 정보를 확인하는데 필요한 정보가 누락되었습니다.")
            }
            return render(request, 'mybook/payment_result.html', context)

        endpoint = "/api/payments/paypal-success/"
        payload = {"info": info}

        try:
            resp = self.hmac_post(endpoint, payload, request=request)
            resp.raise_for_status()
            response_data = resp.json()

            if not response_data.get('success'):
                raise Exception(response_data.get('message', 'Unknown error from payment server.'))

            # --- UserProfile 업데이트 ---
            try:
                user_profile = request.user
                user_profile.plan_type = response_data.get('plan_type', user_profile.plan_type)
                user_profile.is_paid_member = True
                user_profile.start_date = response_data.get('start_date')
                user_profile.end_date = response_data.get('end_date')
                user_profile.save(update_fields=['plan_type', 'is_paid_member', 'start_date', 'end_date'])
                
                logger.info(f"PayPal payment executed and UserProfile updated for {user_profile.username}.")

                context = {
                    'status': 'success',
                    'message': _("구독이 성공적으로 활성화되었습니다!"),
                    'plan_type': user_profile.plan_type,
                    'end_date': user_profile.end_date.strftime('%Y-%m-%d') if user_profile.end_date else None,
                }
                return render(request, 'mybook/payment_result.html', context)

            except Exception as db_error:
                logger.exception(f"DB update failed for {request.user.username} after successful payment. Error: {db_error}")
                # 결제는 성공했으나 서비스 활성화에 실패한 심각한 경우입니다.
                # 사용자에게는 고객센터 문의를 유도합니다.
                context = {
                    'status': 'error',
                    'message': _("결제는 완료되었으나 구독 정보를 업데이트하는 데 실패했습니다. 즉시 고객센터로 문의해주세요.")
                }
                return render(request, 'mybook/payment_result.html', context, status=500)

        except Exception as e:
            logger.exception(f"PayPal payment execution failed for user {request.user.username}. Error: {e}")
            
            error_message = _("결제를 최종 승인하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            context = {
                'status': 'error',
                'message': error_message
            }
            return render(request, 'mybook/payment_result.html', context)


class PayPalCancelView(HmacSignMixin, APIView):
    """
    PayPal 결제 취소 후 리디렉션되는 URL.
    GET /payment/paypal/cancel/?info=...
    사용자가 결제를 취소했음을 결제 서버에 알립니다.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        info = request.query_params.get('info')

        if info:
            endpoint = "/api/payments/paypal-cancel/"
            payload = {"info": info}
            try:
                # 결제 서버에 취소 사실을 알립니다. 실패해도 사용자 흐름은 계속 진행합니다.
                resp = self.hmac_post(endpoint, payload, request=request)
                resp.raise_for_status()
                logger.info(f"Successfully notified payment server of cancellation for user {request.user.username}.")
            except Exception as e:
                logger.error(f"Failed to notify payment server of cancellation for user {request.user.username}. Error: {e}")
        
        context = {
            'status': 'cancel',
            'message': _("결제가 취소되었습니다.")
        }
        return render(request, 'mybook/payment_result.html', context)

class UpgradeSubscriptionView(HmacSignMixin, APIView):
    """
    플랜 업그레이드를 시작하고 결제 서버에 차액 결제 페이지를 요청합니다.
    POST /api/payments/upgrade-subscription/
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        new_plan = request.data.get('new_plan')
        if not new_plan:
            return Response({"error": _("업그레이드할 플랜을 선택해주세요.")}, status=status.HTTP_400_BAD_REQUEST)

        # API 변경 사항: 최종 성공/취소 URL을 생성하여 전달합니다.
        final_success_url = request.build_absolute_uri(reverse('mybook:upgrade_complete'))
        final_cancel_url = request.build_absolute_uri(reverse('mybook:upgrade_failed'))

        endpoint = "/api/payments/upgrade-subscription/"
        payload = {
            "service_name": "weaver",
            "new_plan": new_plan,
            "final_success_url": final_success_url,
            "final_cancel_url": final_cancel_url,
        }

        try:
            resp = self.hmac_post(endpoint, payload, request=request)
            resp.raise_for_status()
            response_data = resp.json()

            # Toss Payments (동기 처리) vs PayPal (비동기 처리) 분기
            if response_data.get("payment_approval_url"):
                # PayPal: 승인 URL로 리디렉션
                return Response({
                    "payment_approval_url": response_data["payment_approval_url"]
                }, status=status.HTTP_200_OK)
            elif response_data.get("success"):
                # Toss Payments: 즉시 업그레이드 완료
                return Response({
                    "upgraded": True,
                    "message": response_data.get("message", _("플랜이 성공적으로 업그레이드되었습니다."))
                }, status=status.HTTP_200_OK)
            else:
                raise Exception("Invalid response from payment server")

        except Exception as e:
            logger.exception(f"Failed to upgrade subscription for user {request.user.username}. Error: {e}")
            error_message = _("플랜 업그레이드 중 오류가 발생했습니다.")
            try:
                error_detail = e.response.json().get('error', '') # type: ignore
                if error_detail:
                    error_message = f"{error_message} ({error_detail})"
            except:
                pass
            return Response({"error": error_message}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

class UpgradeCompleteView(APIView):
    """
    플랜 업그레이드 최종 성공 후 리디렉션되는 페이지.
    GET /payment/upgrade-complete/
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        # TODO: 필요 시, 쿼리 파라미터로 전달된 정보를 바탕으로 UserProfile을 한번 더 동기화할 수 있습니다.
        # 현재는 웹훅을 통해 최종 상태가 업데이트될 것을 기대합니다.
        context = {
            'status': 'success',
            'message': _("플랜 업그레이드가 성공적으로 완료되었습니다. 잠시 후 계정 정보가 업데이트됩니다.")
        }
        return render(request, 'mybook/payment_result.html', context)

class UpgradeFailedView(APIView):
    """
    플랜 업그레이드 최종 취소/실패 후 리디렉션되는 페이지.
    GET /payment/upgrade-failed/
    """
    def get(self, request, *args, **kwargs):
        context = {
            'status': 'cancel', # 또는 'error'
            'message': _("플랜 업그레이드가 취소되었거나 실패했습니다.")
        }
        return render(request, 'mybook/payment_result.html', context)

class UpdateEmailView(HmacSignMixin, APIView):
    """
    사용자의 이메일 주소를 변경합니다.
    """
    permission_classes = [IsAuthenticated]

    ENDPOINT = "/api/accounts/update-email/"

    def post(self, request, *args, **kwargs):
        serializer = UpdateEmailSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        new_email = serializer.validated_data['email'] # type: ignore

        # API 사양 변경: 성공/실패 시 리디렉션될 프론트엔드 URL을 생성하여 전달
        success_url = request.build_absolute_uri(reverse('mybook:email_change_result'))
        fail_url = request.build_absolute_uri(reverse('mybook:email_change_result'))

        payload = {
            "email": new_email,
            "success_url": success_url,
            "fail_url": fail_url,
        }

        try:
            resp = self.hmac_post(self.ENDPOINT, payload, request=request)
            resp.raise_for_status()

            # API 사양 변경: 이제 즉시 이메일을 변경하지 않고, 확인 메일 발송 요청만 보냅니다.
            # 로컬 DB의 이메일은 사용자가 링크를 클릭한 후, 프로필 페이지에 다시 방문했을 때
            # MembershipUpdater와 유사한 메커니즘을 통해 동기화되어야 합니다. (추후 구현)
            # 우선은 인증 서버의 응답을 그대로 반환합니다.
            logger.info(f"Email change verification sent for user {request.user.username} to {new_email}.")

            return Response({"message": _("새 이메일 주소로 확인 메일이 발송되었습니다. 메일을 확인하여 변경을 완료해주세요.")}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Failed to update email for user {request.user.username}. Error: {e}")
            try:
                return Response(e.response.json(), status=e.response.status_code) # type: ignore
            except:
                return Response({"error": _("이메일 변경 중 오류가 발생했습니다.")}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

class EmailChangeResultView(APIView):
    """
    이메일 변경 확인 링크 클릭 후 리디렉션되는 페이지.
    성공/실패 결과를 쿼리 파라미터로 받아 표시합니다.
    """
    permission_classes = [AllowAny] # 누구나 접근 가능해야 함

    def get(self, request, *args, **kwargs):
        # API 서버가 `message`나 `new_email`을 주면 성공, `error`를 주면 실패로 간주합니다.
        new_email = request.GET.get('new_email')
        error_message = request.GET.get('error')
        status = 'success' if new_email or request.GET.get('message') else 'fail'

        # 사용자가 로그인 상태라면, 최신 프로필 정보를 가져오기 위해 동기화를 시도합니다.
        if request.user.is_authenticated:
            updater = MembershipUpdater()
            updater.fetch_and_update(request, params={'req_fields':'email'})

        context = {
            'status': status,
            'new_email': new_email,
            'error_message': error_message,
        }
        return render(request, 'mybook/email_change_result.html', context)


class UpdatePasswordView(HmacSignMixin, APIView):
    """
    사용자의 비밀번호를 변경합니다.
    """
    permission_classes = [IsAuthenticated]
    ENDPOINT = "/api/accounts/update-password/"
    
    def post(self, request, *args, **kwargs):
        logger.debug("-----updatepassword view ------")
        serializer = UpdatePasswordSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
       
        try:
            logger.debug(f"serializer.validated_data: {serializer.validated_data}")
            logger.debug(f"request.data: {request.data}")
            resp = self.hmac_post(self.ENDPOINT, serializer.validated_data, request=request) # type: ignore
            resp.raise_for_status()
            return Response(resp.json(), status=resp.status_code)
        except Exception as e:
            logger.error(f"Failed to update password for user {request.user.username}. Error: {e}")
            try:
                return Response(e.response.json(), status=e.response.status_code) # type: ignore
            except:
                return Response({"error": _("비밀번호 변경 중 오류가 발생했습니다.")}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

class DeleteAccountView(HmacSignMixin, APIView):
    """
    사용자 계정을 삭제합니다.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user_profile = request.user
        
        # 보안을 위해 요청 본문에 username이 포함되어 있는지 한번 더 확인 (선택사항)
        # if request.data.get('username') != user_profile.username:
        #     return Response({"error": "Invalid request."}, status=status.HTTP_400_BAD_REQUEST)

        endpoint = "/api/accounts/delete/"
        payload = {"username": user_profile.username}

        try:
            # DELETE 메소드를 사용해야 하지만, hmac_post는 내부적으로 username을 추가해주므로 사용
            # 인증 서버 API가 DELETE 메소드를 요구한다면 hmac_delete 메소드를 만들어야 함
            # 여기서는 POST로 가정하고 진행
            resp = self.hmac_post(endpoint, payload, request=request)
            resp.raise_for_status()

            # 인증 서버에서 성공적으로 삭제되면 로컬 DB에서도 삭제
            user_profile.delete()
            request.session.flush() # 세션 정보 완전히 삭제
            logger.info(f"User account {user_profile.username} deleted successfully.")
            
            return Response(resp.json(), status=resp.status_code)
        except Exception as e:
            logger.error(f"Failed to delete account for user {user_profile.username}. Error: {e}")
            return Response({"error": _("계정 삭제 중 오류가 발생했습니다.")}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

class PaymentWebhookView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        # 1) username로 대상 인스턴스 식별
        username = request.data.get("username")
        if not username:
            return Response(
                {"username": ["This field is required."]},
                status=status.HTTP_400_BAD_REQUEST,
            )

        profile = get_object_or_404(UserProfile, username=username)

        # 2) 부분 업데이트
        serializer = PaymentWebhookUpdateSerializer(
            instance=profile,
            data=request.data,
            partial=True,                 # 온 것만 반영
            context={"request": request}, # 헤더 검증용
        )
        serializer.is_valid(raise_exception=True)
        #logger.debug(f"serializer.validated_data: {serializer.validated_data}")

        updated_fields = serializer.save()  # update_fields 리스트를 반환하도록 구현해 둠
        logger.debug(f"[VIEW-웹훅 수신] user:{username}, updated_fields: {updated_fields}")

        return Response(status=status.HTTP_204_NO_CONTENT)