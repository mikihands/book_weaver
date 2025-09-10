from django.urls import path
from django.views.generic import TemplateView
from .views import (
    BookUploadView, 
    BookPageView, 
    RetranslatePageView,
    BookPrepareView,
    UpdateBookSettingsView,
    StartTranslationView,
    RetryTranslationView,
    TranslatedPageStatusView,
    UploadPage, 
    RegisterView, 
    RegisterSuccessView, 
    RegisterPage,
    LoginView,
    LoginPage,
    PricingPage,
    FeaturesPage,
    user_logout,
    BookshelfView,
    BookSearchAPIView,
    ContactAPIView,
    BulkDeleteBooksAPIView,
    PublishBookAPIView,
)

app_name = 'mybook'

urlpatterns = [
    path('', UploadPage.as_view(), name='upload_page'), # 메인 페이지 URL 추가
    path('upload/', BookUploadView.as_view(), name='book_upload'),
    path('books/<int:book_id>/search/', BookSearchAPIView.as_view(), name='book_search'),
    path('books/<int:book_id>/pages/<int:page_no>/', BookPageView.as_view(), name='book_page'),
    path('books/<int:book_id>/pages/<int:page_no>/status/', TranslatedPageStatusView.as_view(), name='translated_page_status'),
    path('books/<int:book_id>/prepare/', BookPrepareView.as_view(), name='book_prepare'),
    path('books/<int:book_id>/update_settings/', UpdateBookSettingsView.as_view(), name='update_book_settings'),
    path('books/<int:book_id>/translate/', StartTranslationView.as_view(), name='start_translation'),
    path('books/<int:book_id>/pages/<int:page_no>/retranslate/', RetranslatePageView.as_view(), name='retranslate_page'),
    path('books/<int:book_id>/retry/', RetryTranslationView.as_view(), name='retry_translation'),
    path('auth/register/', RegisterView.as_view(), name='auth_register'),
    path('auth/register/success/', RegisterSuccessView.as_view(), name='register_success'),
    path('auth/register/page/', RegisterPage.as_view(), name='register_page'),
    path('auth/login/', LoginView.as_view(), name='auth_login'),
    path('auth/login/page/', LoginPage.as_view(), name='login_page'),
    path('pricing/', PricingPage.as_view(), name='pricing'),
    path('features/', FeaturesPage.as_view(), name='features'),
    path('auth/logout/', user_logout, name='logout'),  # 로그아웃 URL 추가
    path('bookshelf/', BookshelfView.as_view(), name='bookshelf'),
    path("api/contact/", ContactAPIView.as_view(), name="api_contact"),
    path("contact/", TemplateView.as_view(template_name="contact_us.html"), name="contact"),
    path("books/bulk-delete/", BulkDeleteBooksAPIView.as_view(), name="bulk_delete_books"),
    path("books/<int:book_id>/publish/", PublishBookAPIView.as_view(), name="publish_book"),
]