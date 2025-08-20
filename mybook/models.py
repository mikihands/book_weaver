from django.db import models
from django.conf import settings

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

    def __str__(self):
        return self.username


class UploadedFile(models.Model):
    user = models.ForeignKey(
        'auth.User', 
        on_delete=models.CASCADE,
        help_text='파일을 업로드한 사용자'
    )
    original_file = models.FileField(
        upload_to='uploads/',
        help_text='사용자가 업로드한 원본 파일'
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text='파일 업로드 시간'
    )
    
    def __str__(self):
        return f'{self.original_file.name} by {self.user.username}'

class TranslatedPage(models.Model):
    uploaded_file = models.ForeignKey(
        UploadedFile, 
        on_delete=models.CASCADE,
        help_text='원본 파일'
    )
    page_number = models.IntegerField(
        help_text='페이지 번호'
    )
    translated_html = models.TextField(
        help_text='번역된 HTML 콘텐츠'
    )
    html_url = models.CharField(
        max_length=255,
        help_text='HTML 콘텐츠 URL'
    )
    # translated_lang = models.CharField(
    #     max_length=10,
    #     help_text='번역된 언어'
    # )


    def __str__(self):
        return f'Page {self.page_number} of {self.uploaded_file.original_file.name}'