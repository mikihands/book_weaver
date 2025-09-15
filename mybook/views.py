import logging, os, json, hashlib, re
from bs4 import BeautifulSoup
from pathlib import Path

from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from datetime import datetime
from typing import Dict

from django.conf import settings
from django.core.mail import EmailMessage
from django.db import transaction, models
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
from django.shortcuts import get_object_or_404, render, redirect
from django.utils.encoding import smart_str
from django.utils.translation import gettext as _
from django.urls import reverse
from django.http import FileResponse

from .serializers import (
    FileUploadSerializer, RegisterSerializer, LoginSerializer, 
    RetranslateRequestSerializer, StartTranslationSerializer, 
    BookSettingsSerializer, ContactSerializer, PageEditSerializer,
    BulkDeleteSerializer, PublishRequestSerializer
)
from .models import Book, BookPage, PageImage, TranslatedPage, UserProfile
from .utils.font_scaler import FontScaler
from .utils.gemini_helper import GeminiHelper
from .utils.extract_image import extract_images_and_bboxes, is_fullpage_background
from .utils.layout_norm import normalize_pages_layout
from .utils.html_inject import inject_sources, escape_html
from .utils.pdf_inspector import inspect_pdf, choose_processing_mode
from .utils.delete_dir_files import safe_remove
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

        try:
            # 1) 파일 해시(스트리밍) → 중복 업로드 방지
            file_hash = sha256_streaming(uploaded_file)
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            # 2) Book get_or_create (파일 저장 포함)
            #    파일 저장은 DB 트랜잭션 바깥에서 수행해도 OK
            book, created = Book.objects.get_or_create(
                owner=user_profile,
                file_hash=file_hash,
                defaults={
                    "title": title or uploaded_file.name,
                    "genre": genre,
                    "original_file": uploaded_file,
                    "status": "pending", # Start with pending
                },
            )
            if not created:
                # 이미 같은 파일이 존재 → 기존 book_id 반환
                return Response(
                    {"message": "This file has already been uploaded.", "book_id": book.id}, # type: ignore
                    status=status.HTTP_200_OK,
                )

            # Set status to processing for the extraction phase
            book.status = 'processing'
            # record basic source info
            try:
                book.source_mime = ctype or book.source_mime
                # uploaded_file may have size attribute or seek/read to get size
                try:
                    uploaded_file.seek(0, 2)
                    size = uploaded_file.tell()
                    uploaded_file.seek(0)
                    book.source_size = size
                except Exception:
                    # fallback to file object on disk after save
                    pass
                book.save(update_fields=['status', 'source_mime', 'source_size'])
            except Exception:
                book.save(update_fields=['status'])

            # === 여기서부터는 book.original_file.path 접근 가능 ===
            # 3) PDF 검사: inspect_pdf 호출하여 메타/판정 저장
            pdf_path = book.original_file.path
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

            return Response(
                {"message": "File processed successfully. Please prepare for translation.", "book_id": book.id, "prepare_url": prepare_url}, # type: ignore
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            # FileField는 스토리지에 이미 올라갔을 수 있음 → 필요하면 정리 로직 추가
            # 여기서는 로깅만
            import logging
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
            #html_stage = bleach.clean(html_stage, tags=ALLOW_TAGS, attributes=ALLOW_ATTRS, strip=True) #type:ignore
            html_stage_final = inject_sources(data["html_stage"], data.get("image_src_map", {}))


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
    번역된 페이지의 HTML 컨텐츠(span의 스타일, 텍스트)를 수정하고 저장합니다.
    """
    permission_classes = [IsAuthenticated, IsBookOwner]

    def post(self, request, book_id: int, page_no: int, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book) # 소유권 확인

        serializer = PageEditSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data

        mode = validated_data['mode'] # type:ignore
        lang = validated_data['lang'] # type:ignore
        changes = validated_data['changes'] # type:ignore

        try:
            tp = TranslatedPage.objects.get(book=book, page_no=page_no, lang=lang, mode=mode)
        except TranslatedPage.DoesNotExist:
            return Response({"error": _("해당 번역 페이지를 찾을 수 없습니다.")}, status=status.HTTP_404_NOT_FOUND)

        if not tp.data or 'html' not in tp.data:
            return Response({"error": _("수정할 HTML 컨텐츠가 없습니다.")}, status=status.HTTP_400_BAD_REQUEST)

        original_html = tp.data['html']
        soup = BeautifulSoup(original_html, 'html.parser')

        for change in changes:
            span_id = change['span_id']
            span_to_edit = soup.find('span', id=span_id)

            if span_to_edit:
                if 'style' in change:
                    span_to_edit['style'] = change['style'] # type:ignore
                if 'text' in change:
                    span_to_edit.string = change['text'] # type:ignore
            else:
                logger.warning(f"Span with id '{span_id}' not found in page {page_no} for book {book_id}.")

        tp.data['html'] = str(soup)
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
        book.title = validated_data.get('title', book.title) # type: ignore
        book.genre = validated_data.get('genre', book.genre) # type: ignore
        book.glossary = validated_data.get('glossary', book.glossary) # type: ignore
        book.status = 'processing'
        book.save(update_fields=['title', 'genre', 'glossary', 'status'])
        # 4) 번역 작업 시작
        target_lang = validated_data['target_language'] # type: ignore
        if book.processing_mode == 'born_digital':
            logger.debug(f"Starting born-digital translation for book {book.id} to {target_lang}")
            translate_book_pages_born_digital.delay(book.id, target_lang)
        else: # 'ai_layout' or 'mixed'
            logger.debug(f"Starting ai-layout translation for book {book.id} to {target_lang}")
            translate_book_pages.delay(book.id, target_lang)
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

        retranslate_single_page.delay(book.id, page_no, target_lang, feedback) # type: ignore

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
            print(f"Server response status code: {resp.status_code}")
            print(f"Server response text: {resp.text}")
            return Response(resp.json(), status=resp.status_code)
        except ValueError:
            return Response({"error": "Auth server returned non-JSON error."}, status=resp.status_code)

    
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

        # PostgreSQL의 JSONB 필드에 대한 icontains 쿼리 활용
        matching_pages = TranslatedPage.objects.filter(
            book=book,
            status='ready',
            data__html_stage__icontains=query
        ).order_by('page_no')

        results = []
        processed_pages = set()

        for page in matching_pages:
            if page.page_no in processed_pages:
                continue

            html_content = page.data.get('html_stage', '')
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