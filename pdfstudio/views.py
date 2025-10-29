# pdfstudio/views.py
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404, render
from django.conf import settings
from django.http import FileResponse, Http404
from django.urls import reverse
from pathlib import Path
from mybook.models import Book, BookPage
from mybook.permissions import IsBookOwner
from .serializers import ReorderSerializer, DeleteRestoreSerializer
from mybook.utils.pdf_previews import ensure_previews
import fitz, os, shutil, tempfile

MEDIA = settings.MEDIA_ROOT
MEDIA_URL = settings.MEDIA_URL

def _manifest(book: Book) -> dict:
    m = book.edit_manifest or {}
    m.setdefault("version", 1)
    m.setdefault("order", list(range(1, (book.page_count or 0) + 1)))
    m.setdefault("deleted", [])
    return m

class EditorPage(APIView):
    permission_classes = [IsAuthenticated, IsBookOwner]
    def get(self, request, book_id):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book)
        # 썸네일 없으면 생성
        pdf_path = book.edited_file.path if book.edited_file else book.original_file.path
        if not book.previews_ready:
            ensure_previews(pdf_path, book.id) # type:ignore
            book.previews_ready = True
            book.save(update_fields=["previews_ready"])
        return render(request, "pdfstudio/editor.html", {"book": book})

class ListPagesAPI(APIView):
    permission_classes = [IsAuthenticated, IsBookOwner]
    def get(self, request, book_id):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book)
        man = _manifest(book)

        # 썸네일 경로
        pdir = Path(MEDIA) / "previews" / f"book_{book.id}" #type:ignore
        pages = []
        # BookPage에서 폭/높이 재사용
        dims = {p.page_no: (p.width, p.height) for p in BookPage.objects.filter(book=book)}
        for pno in man["order"]:
            jpg = pdir / f"p{pno}.jpg"
            url = MEDIA_URL.rstrip("/") + f"/previews/book_{book.id}/p{pno}.jpg" #type:ignore
            w, h = dims.get(pno, (None, None))
            pages.append({
                "page_no": pno,
                "width": w or 0,
                "height": h or 0,
                "preview_url": str(url),
                "deleted": (pno in man["deleted"]),
            })
        return Response({"pages": pages})

class ReorderPagesAPI(APIView):
    permission_classes = [IsAuthenticated, IsBookOwner]
    def patch(self, request, book_id):
        ser = ReorderSerializer(data=request.data); ser.is_valid(raise_exception=True)
        order = ser.validated_data["order"] # type:ignore
        book = get_object_or_404(Book, id=book_id); self.check_object_permissions(request, book)
        if len(order) != book.page_count or sorted(order) != list(range(1, book.page_count+1)):
            return Response({"detail":"order 길이/구성이 원본과 다릅니다."}, status=400)
        man = _manifest(book); man["order"] = order; book.edit_manifest = man # type:ignore
        book.save(update_fields=["edit_manifest"])
        return Response({"ok": True})

class DeletePagesAPI(APIView):
    permission_classes = [IsAuthenticated, IsBookOwner]
    def post(self, request, book_id):
        ser = DeleteRestoreSerializer(data=request.data); ser.is_valid(raise_exception=True)
        book = get_object_or_404(Book, id=book_id); self.check_object_permissions(request, book)
        man = _manifest(book)
        for p in ser.validated_data["pages"]: # type:ignore
            if p not in man["deleted"]:
                man["deleted"].append(p)
        book.edit_manifest = man # type:ignore
        book.save(update_fields=["edit_manifest"])
        return Response({"ok": True, "deleted": man["deleted"]})

class RestorePagesAPI(APIView):
    permission_classes = [IsAuthenticated, IsBookOwner]
    def post(self, request, book_id):
        ser = DeleteRestoreSerializer(data=request.data); ser.is_valid(raise_exception=True)
        book = get_object_or_404(Book, id=book_id); self.check_object_permissions(request, book)
        man = _manifest(book)
        man["deleted"] = [p for p in man["deleted"] if p not in ser.validated_data["pages"]] # type:ignore
        book.edit_manifest = man # type:ignore
        book.save(update_fields=["edit_manifest"])
        return Response({"ok": True, "deleted": man["deleted"]})

class CommitEditsAPI(APIView):
    """
    edit_manifest의 order/deleted를 반영하여 새 PDF를 생성해 edited_file에 저장.
    이후 BookPage/PageImage는 재생성(기존 파이프라인 재사용).
    """
    permission_classes = [IsAuthenticated, IsBookOwner]
    def post(self, request, book_id):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book)
        man = _manifest(book)
        # 최종 유지할 페이지 시퀀스
        final_seq = [p for p in man["order"] if p not in set(man["deleted"])]
        if not final_seq:
            return Response({"detail":"모든 페이지가 삭제될 수는 없습니다."}, status=400)

        src_pdf = book.edited_file.path if book.edited_file else book.original_file.path
        doc = fitz.open(src_pdf)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        out = fitz.open()
        for p in final_seq:
            out.insert_pdf(doc, from_page=p-1, to_page=p-1)
        out.save(tmp.name); out.close(); doc.close()

        # 파일 배치
        dest_dir = Path(book.original_file.storage.location) / f"original/book_{book.id}" # type:ignore
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / "edited.pdf"
        shutil.move(tmp.name, dest_path)

        # 모델 업데이트
        book.edited_file.name = str(dest_path).replace(str(Path(settings.MEDIA_ROOT))+"/", "")
        # 이제 edited_file 기준으로 썸네일 재생성
        ensure_previews(str(dest_path), book.id) # type:ignore
        book.previews_ready = True
        # 페이지 카운트 갱신(나중에 재추출 과정에서 정확한 값으로 또 갱신됨)
        book.page_count = len(final_seq)
        book.save(update_fields=["edited_file","previews_ready","page_count","edit_manifest"])

        # 기존 추출물 정리 & 재추출(동기 or Celery)
        # - 간단히 태스크만 큐잉하고, 결과가 오면 status='uploaded'로 고정
        from mybook.utils.extract_image import extract_images_and_bboxes
        from mybook.utils.layout_norm import normalize_pages_layout
        import os
        pdf_path = str(dest_path)
        out_dir = Path(settings.MEDIA_ROOT) / "extracted_images" / f"{book.id}" # type:ignore
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pages = extract_images_and_bboxes(pdf_path, str(out_dir), dpi=144, media_root=settings.MEDIA_ROOT)
        norm_pages = normalize_pages_layout(pages, base_width=1200)

        # BookPage 갱신(간단화: 싹 지우고 채우기)
        BookPage.objects.filter(book=book).delete()
        BookPage.objects.bulk_create([
            BookPage(book=book, page_no=p["page_no"], width=p["size"]["w"], height=p["size"]["h"], meta=p.get("meta"))
            for p in norm_pages
        ])
        book.page_count = len(norm_pages)
        book.status = "uploaded"  # 다시 번역 대기 상태
        book.save(update_fields=["page_count","status"])

        return Response({
            "ok": True,
            "page_count": book.page_count,
            "prepare_url": reverse("mybook:book_prepare", kwargs={"book_id": book.id}) # type:ignore
        }, status=200)

class ResetEditsAPI(APIView):
    permission_classes = [IsAuthenticated, IsBookOwner]
    def post(self, request, book_id):
        book = get_object_or_404(Book, id=book_id); self.check_object_permissions(request, book)
        book.edit_manifest = None
        book.edited_file = None # type:ignore
        book.previews_ready = False
        book.save(update_fields=["edit_manifest","edited_file","previews_ready"])
        return Response({"ok": True})

class StreamOriginalForEditorView(APIView):
    """PDF.js가 열어볼 원본 PDF 스트림(소유자만)"""
    permission_classes = [IsAuthenticated, IsBookOwner]
    def get(self, request, book_id):
        book = get_object_or_404(Book, id=book_id)
        self.check_object_permissions(request, book)
        path = book.edited_file.path if book.edited_file else book.original_file.path
        if not os.path.exists(path):
            raise Http404
        return FileResponse(open(path, "rb"), content_type="application/pdf")
