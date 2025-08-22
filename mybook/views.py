import logging, os, json, hashlib
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

from .serializers import FileUploadSerializer, RegisterSerializer, LoginSerializer
from .models import Book, BookPage, PageImage, TranslatedPage, UserProfile
from .utils.gemini_helper import GeminiHelper
from .utils.extract_image import extract_images_and_bboxes, is_fullpage_background
from .utils.faithful_prompt import build_prompt_faithful
from .utils.render_json_to_html import render_json_to_html  
from .tasks import translate_book_pages
from common.mixins.hmac_sign_mixin import HmacSignMixin
# from .tasks import translate_book_pages 

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
                    "title": uploaded_file.name,
                    "original_file": uploaded_file,
                    "owner": user_profile,
                    "status": "processing",
                },
            )
            if not created:
                # 이미 같은 파일이 존재 → 기존 book_id 반환
                return Response(
                    {"message": "This file has already been uploaded.", "book_id": book.id}, # type: ignore
                    status=status.HTTP_200_OK,
                )

            # 3) PDF 메타/이미지 추출 (동기)
            #    FileField가 저장됐으니 경로 접근 가능
            pdf_path = book.original_file.path
            out_dir = os.path.join(settings.MEDIA_ROOT, "extracted_images", str(book.id)) # type: ignore
            os.makedirs(out_dir, exist_ok=True)

            pages = extract_images_and_bboxes(pdf_path, out_dir, dpi=144)
            # pages: [{ "page_no": int, "size": {"w":..., "h":...}, "images": [{"ref","path","bbox"}] }, ...]

            # 4) DB 저장 (트랜잭션: 짧게)
            with transaction.atomic():
                # 페이지/이미지 벌크 생성
                page_objs = []
                img_objs = []
                for p in pages:
                    page_objs.append(
                        BookPage(
                            book=book,
                            page_no=p["page_no"],
                            width=p["size"]["w"],
                            height=p["size"]["h"],
                        )
                    )
                BookPage.objects.bulk_create(page_objs, ignore_conflicts=True)

                for p in pages:
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
                book.page_count = len(pages)
                book.status = "processing"
                book.save(update_fields=["page_count", "status"])

            # 5) 번역 파이프라인 시작 (Celery를 통한 비동기 작업)
            translate_book_pages.delay(book.id, target_language) #type: ignore

            page1_url = request.build_absolute_uri(
                reverse("mybook:book_page", kwargs={"book_id": book.id, "page_no": 1})  # type: ignore
            )

            return Response(
                {"message": "Upload accepted. Processing started.", "book_id": book.id, "page1_url": page1_url}, # type: ignore
                status=status.HTTP_202_ACCEPTED,
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
    
# class BookPageView(APIView):
#     """GEMIMNI 버젼"""
#     def get(self, request, book_id, page_number, *args, **kwargs):
#         translated_page = get_object_or_404(
#             TranslatedPage, 
#             book__id=book_id, 
#             page_no=page_number
#         )
        
#         # 페이지에 포함된 이미지 정보 가져오기
#         page_images_info = list(PageImage.objects.filter(book__id=book_id, page_no=page_number).values('ref', 'path', 'bbox'))
        
#         # JSON 데이터를 HTML로 렌더링
#         rendered_html = render_json_to_html(translated_page.data, page_images_info)
        
#         context = {
#             'translated_html': rendered_html,
#             'page_number': translated_page.page_no,
#             'book_title': translated_page.book.title,
#             'page_width': translated_page.data['page']['size']['w'],
#             'page_height': translated_page.data['page']['size']['h'],
#         }
#         return render(request, 'mybook/book_page.html', context)

# (선택) 표 HTML sanitize를 하고 싶다면:
try:
    import bleach
    ALLOW_TAGS = ["table","thead","tbody","tfoot","tr","th","td","caption","colgroup","col","b","i","em","strong","u","sub","sup","span","p","ul","ol","li","br","hr"]
    ALLOW_ATTRS = {"*": ["colspan","rowspan","align","valign"]}
except Exception:
    bleach = None

class BookPageView(APIView):
    permission_classes = [IsAuthenticated]  # 정책에 맞게 조정

    def get(self, request, book_id: int, page_no: int, *args, **kwargs):
        """
        단일 페이지 렌더(정밀 모드)
        URL 예: /books/<book_id>/pages/<page_no>/?mode=faithful&bg=auto|on|off
        """
        mode = request.GET.get("mode", "faithful")
        bg_mode = request.GET.get("bg", "auto")  # auto|on|off

        book = get_object_or_404(Book, id=book_id)
        page = get_object_or_404(BookPage, book=book, page_no=page_no)
        tp = TranslatedPage.objects.filter(book=book, page_no=page_no, mode=mode).first()

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

        # 2) JSON 데이터 + (선택) 표 sanitize
        data = tp.data if tp else None
        status_flag = tp.status if tp else "pending"
        if data and bleach:
            for b in data.get("blocks", []):
                if b.get("type") == "table" and "content" in b and "html" in b["content"]:
                    b["content"]["html"] = bleach.clean(
                        b["content"]["html"], tags=ALLOW_TAGS, attributes=ALLOW_ATTRS, strip=True
                    )

        # 3) 배경 이미지 결정 로직
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

        # 4) prev/next
        prev_url = next_url = None
        try:
            if page_no > 1:
                prev_url = reverse("mybook:book_page", kwargs={"book_id": book.id, "page_number": page_no - 1}) #type: ignore
            if book.page_count and page_no < book.page_count:
                next_url = reverse("mybook:book_page", kwargs={"book_id": book.id, "page_number": page_no + 1}) #type: ignore
        except Exception:
            # 네임스페이스를 쓰지 않는 프로젝트라면 이름을 "book_page"로 변경
            from django.urls import NoReverseMatch
            try:
                if page_no > 1:
                    prev_url = reverse("book_page", kwargs={"book_id": book.id, "page_number": page_no - 1}) #type: ignore
                if book.page_count and page_no < book.page_count:
                    next_url = reverse("book_page", kwargs={"book_id": book.id, "page_number": page_no + 1}) #type: ignore
            except NoReverseMatch:
                prev_url = next_url = None

        # 5) 템플릿 컨텍스트
        ctx = {
            "book": book,
            "page_no": page_no,
            "data": data,
            "status": status_flag,
            "background_url": background_url,  # ✅ 배경 URL (없으면 빈 문자열)
            "image_url_map": image_url_map,    # figure 렌더용
            "prev_url": prev_url,
            "next_url": next_url,
        }
        return render(request, "mybook/book_page.html", ctx)

    
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