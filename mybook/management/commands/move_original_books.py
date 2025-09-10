# mybook/management/commands/move_original_books.py
import os
from uuid import uuid4
from pathlib import Path
from django.core.management.base import BaseCommand
from django.core.files.storage import default_storage
from mybook.models import Book

class Command(BaseCommand):
    help = "Move existing Book.original_file into original/book_{id}/ folder"

    def handle(self, *args, **options):
        moved = 0
        skipped = 0
        for b in Book.objects.exclude(original_file=""):
            rel = b.original_file.name
            if not rel:
                skipped += 1
                continue
            # 이미 목표 폴더면 스킵
            if rel.startswith(f"original/book_{b.pk}/"):
                skipped += 1
                continue
            # tmp도 최종으로
            filename = os.path.basename(rel)
            target = f"original/book_{b.pk}/{filename}"
            if default_storage.exists(target):
                stem, ext = os.path.splitext(filename)
                target = f"original/book_{b.pk}/{stem}_{uuid4().hex[:8]}{ext}"
            # 이동
            with default_storage.open(rel, "rb") as f:
                default_storage.save(target, f)
            try:
                default_storage.delete(rel)
            except Exception:
                pass
            b.original_file.name = target
            b.save(update_fields=["original_file"])
            moved += 1

        self.stdout.write(self.style.SUCCESS(f"Moved: {moved}, Skipped: {skipped}"))
