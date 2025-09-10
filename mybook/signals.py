from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Book
import logging, os
from uuid import uuid4
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)

@receiver(post_save, sender=Book)
def move_original_into_book_dir(sender, instance: "Book", created, **kwargs):
    """
    PK 부여 후에도 original_file이 _tmp 경로에 있다면
    original/book_{id}/ 로 안전하게 이동
    """
    if not instance.original_file:
        return

    current_path = instance.original_file.name  # storage 상대경로
    if not current_path.startswith("original/_tmp/"):
        return

    filename = os.path.basename(current_path)
    final_relpath = f"original/book_{instance.pk}/{filename}"

    if default_storage.exists(final_relpath):
        # 충돌 시 파일명에 suffix 추가
        stem, ext = os.path.splitext(filename)
        final_relpath = f"original/book_{instance.pk}/{stem}_{uuid4().hex[:8]}{ext}"

    # 같은 Storage(FileSystemStorage 가정)에서 "이동" 구현
    # 1) 원본을 열어 새 경로로 저장
    with default_storage.open(current_path, "rb") as f:
        default_storage.save(final_relpath, f)
    # 2) 원본 삭제
    try:
        default_storage.delete(current_path)
    except Exception:
        pass

    # 3) 모델 필드 갱신
    instance.original_file.name = final_relpath
    # 무한 루프 방지: update_fields 사용
    instance.save(update_fields=["original_file"])