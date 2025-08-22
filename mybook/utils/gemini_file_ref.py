# mybook/utils/gemini_file_ref.py
import os, logging
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MAX_INLINE = 20 * 1024 * 1024      # 20MB
MAX_FILEAPI = 50 * 1024 * 1024     # 50MB
MIME_PDF = "application/pdf"

class GeminiFileRefManager:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _inline_part(self, file_path: str):
        with open(file_path, "rb") as f:
            data = f.read()
        return types.Part.from_bytes(data=data, mime_type=MIME_PDF)

    def _upload_once(self, book, file_path: str):
        """20~50MB: 최초 1회 업로드하고 Book에 메타 보관."""
        with open(file_path, "rb") as f:
            uploaded = self.client.files.upload(file=f, config=dict(mime_type=MIME_PDF)) # type: ignore
            logger.debug(f"File uploaded: {uploaded}")
        # 업로드 확인(선택) - files.get으로 메타 체크
        try:
            info = self.client.files.get(name=uploaded.name) # type: ignore
            logger.debug(f"File uploaded check: {info}")
            uri = getattr(info, "uri", None) or getattr(uploaded, "uri", None)
        except Exception:
            uri = getattr(uploaded, "uri", None)

        now = timezone.now()
        book.gemini_file_name = uploaded.name
        book.gemini_file_uri = uri
        book.gemini_file_uploaded_at = now
        # 48시간 보관 → 여유 있게 47시간 캐시
        book.gemini_file_expires_at = now + timedelta(hours=47)
        book.save(update_fields=[
            "gemini_file_name", "gemini_file_uri",
            "gemini_file_uploaded_at", "gemini_file_expires_at"
        ])
        return uploaded

    def ensure_file_part(self, book):
        """
        반환값: 'generate_content(contents=[<<<이걸 첫 파트로>>> , ...])' 에 넣을 수 있는 객체
        - <20MB: inline Part
        - 20~50MB: File API 업로드 파일 객체(1회 업로드 후 재사용)
        - >50MB: 예외(현재는 미지원)
        """
        file_path = book.original_file.path
        size = book.source_size or os.path.getsize(file_path)
        book.source_size = size
        book.source_mime = MIME_PDF
        book.save(update_fields=["source_size", "source_mime"])

        if size < MAX_INLINE:
            return self._inline_part(file_path)

        if size > MAX_FILEAPI:
            raise ValueError("PDF가 50MB를 초과합니다. (현재: %.2f MB)" % (size / (1024*1024)))

        # 20~50MB: 기존 업로드가 유효하면 재사용
        if book.gemini_file_name and book.gemini_file_expires_at and book.gemini_file_expires_at > timezone.now():
            try:
                # 재확인 후 그 객체를 contents에 사용
                return self.client.files.get(name=book.gemini_file_name)
            except Exception as e:
                logger.warning(f"files.get 실패. 재업로드 시도: {e}")

        # 없거나 만료/오류 → 새로 업로드
        return self._upload_once(book, file_path)
