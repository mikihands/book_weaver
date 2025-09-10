from django.db import models
import hashlib
from uuid import uuid4

class UserProfile(models.Model):
    username = models.CharField(max_length=150, unique=True, help_text="인증 서버에서 제공받은 username")
    email = models.EmailField(unique=True, help_text="인증 서버에서 제공받은 email")
    secret_key = models.CharField(max_length=256, unique=True, help_text="인증 서버와의 통신에서 사용할 secret_key, 인증서버에서 제공함")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    plan_type = models.CharField(max_length=50, default='None', help_text='사용자의 서비스 요금제 유형') # 인증서버에서 받아오는 값을 그대로 저장
    is_paid_member = models.BooleanField(default=False, help_text='유료 회원 여부')
    start_date = models.DateTimeField(null=True, blank=True)
    end_date = models.DateTimeField(null=True, blank=True, help_text='유료 구독 만료일')
    cancel_requrest = models.BooleanField(default=False)
    pdf_download_count = models.IntegerField(
        default=0,
        help_text='PDF 다운로드 횟수 (무료 회원 제한용)'
    )
    total_uploaded_files = models.IntegerField(
        default=0,
        help_text='총 업로드 파일 수'
    )
    last_activity = models.DateTimeField(
        auto_now=True,
        help_text='마지막 활동 시간'
    )

    @property
    def is_authenticated(self):
        return True 

    def __str__(self):
        return self.username

def book_upload_to(instance, filename):
    # PK가 있으면 최종 경로
    if instance.pk:
        return f"original/book_{instance.pk}/{filename}"
    # PK가 아직 없으면 임시 경로(뒤에서 post_save에서 이동)
    return f"original/_tmp/{uuid4().hex}/{filename}"


class Book(models.Model):
    PROCESSING_STATUS = [
        ('pending', '업로드 처리 대기 중'),
        ('uploaded', '번역 대기 중'),
        ('processing', '번역 중'),
        ('completed', '완료'),
        ('failed', '실패'),
    ]

    OCR_STATUS = [
        ('unknown', '판별 안됨'),
        ('ocr', 'OCR 처리됨'),
        ('image', 'OCR 미처리 (스캔본)'),
    ]

    title = models.CharField(max_length=255, blank=True, help_text="업로드된 책의 제목")
    genre = models.CharField(max_length=100, blank=True, null=True, help_text="책의 장르")
    glossary = models.TextField(blank=True, null=True, help_text="번역에 사용할 사용자 정의 용어집")
    original_file = models.FileField(upload_to=book_upload_to, help_text="사용자가 업로드한 원본 파일")
    file_hash = models.CharField(
        max_length=64, 
        db_index=True, 
        help_text="파일 내용의 해시값 (중복 업로드 방지용)"
    )
    page_count = models.IntegerField(default=0, help_text="원본 문서의 총 페이지 수")
    status = models.CharField(
        max_length=16,
        choices=PROCESSING_STATUS,
        default='pending',
        help_text='문서 전체 처리 상태'
    )
    owner = models.ForeignKey(UserProfile, on_delete=models.CASCADE, help_text="파일을 업로드한 사용자")
    created_at = models.DateTimeField(auto_now_add=True)

    source_mime = models.CharField(max_length=64, default="application/pdf")
    source_size = models.BigIntegerField(default=0)

    # ✅ OCR 여부 판별 결과
    ocr_status = models.CharField(
        max_length=16,
        choices=OCR_STATUS,
        default='unknown',
        help_text="PDF가 OCR로 처리된 텍스트 기반인지, 이미지 기반인지"
    )
    text_coverage = models.FloatField(
        null=True, blank=True,
        help_text="페이지 내 텍스트 커버리지 비율(0.0~1.0). OCR 여부 추정 근거"
    )

    # Gemini File API 메타(20~50MB 구간에서 사용)
    gemini_file_name = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    gemini_file_uri = models.CharField(max_length=512, null=True, blank=True)
    gemini_file_uploaded_at = models.DateTimeField(null=True, blank=True)
    gemini_file_expires_at = models.DateTimeField(null=True, blank=True)

    # === inspector 결과 저장 및 처리 모드 ===
    processing_mode = models.CharField(
        max_length=32, null=True, blank=True,
        help_text="pdf_inspector가 권장하는 처리 모드 (born_digital | ai_layout | mixed)"
    )
    inspection = models.JSONField(
        null=True, blank=True,
        help_text="pdf_inspector.inspect_pdf() 결과의 직렬화된 dict"
    )
    avg_score = models.FloatField(null=True, blank=True)
    median_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.title or self.original_file.name

    class Meta:
        verbose_name = "Book"
        verbose_name_plural = "Books"
        constraints = [
            models.UniqueConstraint(fields=['file_hash', 'owner'], name='unique_file_hash_owner')
        ]

    def save(self, *args, **kwargs):
        if not self.file_hash and self.original_file:
            file_content = self.original_file.read()
            self.file_hash = hashlib.sha256(file_content).hexdigest()
            self.original_file.seek(0)  # 중요! 안 하면 파일 포인터 끝에 머뭄
        super().save(*args, **kwargs)

class BookPage(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="pages")
    page_no = models.IntegerField(help_text="페이지 번호 (1부터 시작)")
    width = models.FloatField(help_text="페이지 너비 (px)")
    height = models.FloatField(help_text="페이지 높이 (px)")
    meta = models.JSONField(null=True, blank=True, help_text="페이지 normalize 이후의 메타정보")

    def __str__(self):
        return f"Page {self.page_no} of {self.book.title or self.book.original_file.name}"
    
    class Meta:
        unique_together = ('book', 'page_no')

class PageImage(models.Model):
    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="images")
    page_no = models.IntegerField(help_text="이미지가 속한 페이지 번호")
    xref = models.IntegerField(db_index=True, help_text=" OCR처리된 PDF 내 이미지 객체의 xref 번호", null=True, blank=True)
    ref = models.CharField(max_length=64, db_index=True, help_text="이미지 참조 ID (예: img_p1_1)")
    path = models.CharField(max_length=512, help_text="서버에 저장된 이미지 파일 경로")
    bbox = models.JSONField(help_text="페이지 내 이미지 위치 및 크기 [x, y, w, h]")
    transform = models.JSONField(null=True, blank=True,
        help_text="PDF CTM matrix [a,b,c,d,e,f] for positioning/cropping")
    clip_bbox = models.JSONField(null=True, blank=True, help_text="페이지내 이미지가 crop되었을때 클리핑마스크 bbox")
    img_w = models.IntegerField(null=True, blank=True, help_text="서버에 저장된 이미지 너비")
    img_h = models.IntegerField(null=True, blank=True, help_text="서버에 저장된 이미지 높이")
    origin_w = models.IntegerField(null=True, blank=True, help_text="원본 이미지 너비")
    origin_h = models.IntegerField(null=True, blank=True, help_text="원본 이미지 높이")

    def __str__(self):
        return f"Image '{self.ref}' on page {self.page_no} of {self.book.title or self.book.original_file.name}"
    
    class Meta:
        unique_together = ('book', 'ref')

class TranslatedPage(models.Model):
    TRANSLATION_MODES = [
        ("faithful", "정밀 모드"),
        ("readable", "가독 모드"),
    ]
    TRANSLATION_STATUS = [
        ("pending", "대기 중"),
        ("ready", "준비 완료"),
        ("failed", "실패"),
    ]

    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="translated_pages")
    page_no = models.IntegerField(help_text="번역된 페이지 번호")
    lang = models.CharField(max_length=10, help_text="번역된 언어")
    mode = models.CharField(max_length=16, choices=TRANSLATION_MODES, default="faithful", help_text="번역 모드")
    data = models.JSONField(help_text="weaver.page.v1 JSON 형식의 번역 데이터")
    status = models.CharField(max_length=16, choices=TRANSLATION_STATUS, default="pending", help_text="번역 상태")
    tokens_used = models.IntegerField(null=True, blank=True, help_text="번역에 사용된 토큰 수")
    duration_ms = models.IntegerField(null=True, blank=True, help_text="번역에 걸린 시간 (ms)")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Translated Page {self.page_no} ({self.lang}, {self.mode}) of {self.book.title or self.book.original_file.name}"

    class Meta:
        unique_together = ('book', 'page_no', 'lang', 'mode')