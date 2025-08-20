from django.urls import path
from .views import (
    BookUploadView, 
    BookPageView, 
    UploadPage, 
    RegisterView, 
    RegisterSuccessView, 
    RegisterPage
)

app_name = 'mybook'

urlpatterns = [
    path('', UploadPage.as_view(), name='upload_page'), # 메인 페이지 URL 추가
    path('upload/', BookUploadView.as_view(), name='book_upload'),
    path('books/<int:book_id>/pages/<int:page_number>/', BookPageView.as_view(), name='book_page'),
    path('auth/register/', RegisterView.as_view(), name='auth_register'),
    path('auth/register/success/', RegisterSuccessView.as_view(), name='register_success'),
    path('auth/register/page/', RegisterPage.as_view(), name='register_page'),
]