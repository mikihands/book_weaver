from django.urls import path
from .views import (
    BookUploadView, 
    BookPageView, 
    UploadPage, 
    RegisterView, 
    RegisterSuccessView, 
    RegisterPage,
    LoginView,
    LoginPage,
    user_logout
)

app_name = 'mybook'

urlpatterns = [
    path('', UploadPage.as_view(), name='upload_page'), # 메인 페이지 URL 추가
    path('upload/', BookUploadView.as_view(), name='book_upload'),
    path('books/<int:book_id>/pages/<int:page_no>/', BookPageView.as_view(), name='book_page'),
    path('auth/register/', RegisterView.as_view(), name='auth_register'),
    path('auth/register/success/', RegisterSuccessView.as_view(), name='register_success'),
    path('auth/register/page/', RegisterPage.as_view(), name='register_page'),
    path('auth/login/', LoginView.as_view(), name='auth_login'),
    path('auth/login/page/', LoginPage.as_view(), name='login_page'),
    path('auth/logout/', user_logout, name='logout'),  # 로그아웃 URL 추가
]