import logging, os, json, hashlib
from bs4 import BeautifulSoup
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime
from typing import Dict

from django.conf import settings
from django.db import transaction
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse

from .serializers import FileUploadSerializer, RegisterSerializer, LoginSerializer, RetranslateRequestSerializer, StartTranslationSerializer, BookSettingsSerializer
from .models import Book, BookPage, PageImage, TranslatedPage, UserProfile
from .utils.font_scaler import FontScaler
from .utils.gemini_helper import GeminiHelper
from .utils.extract_image import extract_images_and_bboxes, is_fullpage_background
from .utils.layout_norm import normalize_pages_layout
from .utils.html_inject import inject_sources
from .tasks import translate_book_pages, retranslate_single_page
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

            # 2) Book get_or_create (파일 저장 포함)
            #    파일 저장은 DB 트랜잭션 바깥에서 수행해도 OK
            book, created = Book.objects.get_or_create(
                file_hash=file_hash,
                defaults={
                    "title": title or uploaded_file.name,
                    "genre": genre,
                    "original_file": uploaded_file,
                    "owner": user_profile,
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
            book.save(update_fields=['status'])
            # 3) PDF 메타/이미지 추출 (동기)
            #    FileField가 저장됐으니 경로 접근 가능
            pdf_path = book.original_file.path
            out_dir = os.path.join(settings.MEDIA_ROOT, "extracted_images", str(book.id)) # type: ignore
            os.makedirs(out_dir, exist_ok=True)

            extracted_pages = extract_images_and_bboxes(pdf_path, out_dir, dpi=144)
            # pages: [{ "page_no": int, "size": {"w":..., "h":...}, "images": [{"ref","path","bbox"}] }, ...]
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
                                ref=im["ref"],
                                path=im["path"],
                                bbox=im["bbox"],
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
        """
        단일 페이지 렌더(정밀 모드)
        URL 예: /books/<book_id>/pages/<page_no>/?mode=faithful&bg=auto|on|off
        """
        mode = request.GET.get("mode", "faithful")
        bg_mode = request.GET.get("bg", "off")  # auto|on|off

        book = get_object_or_404(Book, id=book_id)
        page = get_object_or_404(BookPage, book=book, page_no=page_no)
        tp = TranslatedPage.objects.filter(book=book, page_no=page_no, mode=mode).first()

        # v2 JSON 기대: schema_version=weaver.page.v2, fields: page, html_stage, figures, styles?
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

        # 4) prev/next
        prev_url = next_url = None
        try:
            if page_no > 1:
                prev_url = reverse("mybook:book_page", kwargs={"book_id": book.id, "page_no": page_no - 1}) #type: ignore
            if book.page_count and page_no < book.page_count:
                next_url = reverse("mybook:book_page", kwargs={"book_id": book.id, "page_no": page_no + 1}) #type: ignore
        except Exception:
            from django.urls import NoReverseMatch
            try:
                if page_no > 1:
                    prev_url = reverse("book_page", kwargs={"book_id": book.id, "page_no": page_no - 1}) #type: ignore
                if book.page_count and page_no < book.page_count:
                    next_url = reverse("book_page", kwargs={"book_id": book.id, "page_no": page_no + 1}) #type: ignore
            except NoReverseMatch:
                prev_url = next_url = None

        # 5) 템플릿 컨텍스트
        ctx = {
            "book": book,
            "page_no": page_no,
            "status": status_flag,
            "page_w": int(page_w),
            "page_h": int(page_h),
            "background_url": background_url,
            "html_stage_final": html_stage_final,  # ⬅️ 템플릿에 그대로 삽입
            "base_font_scale": base_font_scale,
            "prev_url": prev_url,
            "next_url": next_url,
            "bg_mode": bg_mode,
        }

        ctx.update({"css_vars": css_vars})

        return render(request, "mybook/book_page_v2.html", ctx)
    
    def _measure_html(self, html: str) -> tuple[int, int]:
        soup = BeautifulSoup(html, 'html.parser')
        para_like = soup.find_all(["p","li","blockquote","h1","h2","h3","h4","h5","h6"])
        text_len = len(soup.get_text(" ", strip=True))
        return text_len, max(1, len(para_like))


class BookPrepareView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, book_id, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id, owner=request.user)
        if book.status not in ['uploaded', 'failed']:
            # 이미 번역이 시작되었거나 완료된 경우, 첫 페이지로 리디렉션
            return redirect(reverse('mybook:book_page', kwargs={'book_id': book.id, 'page_no': 1}))
        return render(request, 'mybook/book_prepare.html', {'book': book})


class UpdateBookSettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, book_id, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id, owner=request.user)
        serializer = BookSettingsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        validated_data = serializer.validated_data
        book.title = validated_data.get('title', book.title) # type: ignore
        book.genre = validated_data.get('genre', book.genre) # type: ignore
        book.glossary = validated_data.get('glossary', book.glossary) # type: ignore
        book.save(update_fields=['title', 'genre', 'glossary'])
        
        return Response({"message": "설정이 성공적으로 저장되었습니다."}, status=status.HTTP_200_OK)


class StartTranslationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, book_id, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id, owner=request.user)
        serializer = StartTranslationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        validated_data = serializer.validated_data
        book.title = validated_data.get('title', book.title) # type: ignore
        book.genre = validated_data.get('genre', book.genre) # type: ignore
        book.glossary = validated_data.get('glossary', book.glossary) # type: ignore
        book.status = 'processing'
        book.save()

        translate_book_pages.delay(book.id, validated_data['target_language']) # type: ignore
        
        page1_url = reverse('mybook:book_page', kwargs={'book_id': book.id, 'page_no': 1}) # type: ignore
        return Response({"message": "Translation has started.", "page1_url": page1_url}, status=status.HTTP_202_ACCEPTED)

class RetranslatePageView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, book_id: int, page_no: int, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id, owner=request.user)
        
        serializer = RetranslateRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        feedback = serializer.validated_data.get('feedback', '') # type: ignore

        # Get target language from the first available translated page
        first_tp = TranslatedPage.objects.filter(book=book).first()
        if not first_tp:
            return Response({"error": "No translated pages found for this book."}, status=status.HTTP_400_BAD_REQUEST)
        target_lang = first_tp.lang

        # Set the specific page status to 'pending' to indicate re-translation
        TranslatedPage.objects.filter(book=book, page_no=page_no, lang=target_lang, mode='faithful').update(status='pending')
        
        # Asynchronously call the re-translation task
        retranslate_single_page.delay(book_id, page_no, target_lang, feedback) # type: ignore

        return Response({"message": f"Retranslation for page {page_no} has been queued."}, status=status.HTTP_202_ACCEPTED)


class RetryTranslationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, book_id: int, *args, **kwargs):
        book = get_object_or_404(Book, id=book_id, owner=request.user)

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

def user_logout(request):
    """
    동일한 브라우저에서 다른 아이디로 로그인했을 때 세션을 공유하는 일이 없도록 완전히 flush
    """
    request.session.flush()
    # 로그아웃 후 메인 페이지로 이동
    return redirect("mybook:upload_page")