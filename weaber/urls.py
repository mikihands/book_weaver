from django.contrib import admin
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path("i18n/", include("django.conf.urls.i18n")),  # 공통적인 언어 변경 URL 추가
    path("", include("mybook.api_urls")),  # i18n이 적용되지 않는 API/Webhook URL
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += [
        path("__reload__/", include("django_browser_reload.urls")),
    ]

urlpatterns += i18n_patterns(
    path("", include("mybook.urls", namespace="mybook")),  # mybook 앱의 URL 포함
)