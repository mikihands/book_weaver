from django.contrib import admin
from .models import Book, BookPage, PageImage, TranslatedPage, ApiUsageLog

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'owner', 'status', 'page_count', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('title', 'owner__username')
    readonly_fields = ('file_hash', 'created_at')
    
@admin.register(BookPage)
class BookPageAdmin(admin.ModelAdmin):
    list_display = ('id', 'book', 'page_no', 'width', 'height')
    list_filter = ('book__title',)
    search_fields = ('book__title',)
    
@admin.register(PageImage)
class PageImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'book', 'page_no', 'ref', 'path')
    list_filter = ('book__title', 'page_no')
    search_fields = ('ref', 'book__title')
    
@admin.register(TranslatedPage)
class TranslatedPageAdmin(admin.ModelAdmin):
    list_display = ('id', 'book', 'page_no', 'lang', 'mode', 'status', 'created_at')
    list_filter = ('book__title', 'lang', 'mode', 'status')
    search_fields = ('book__title', 'lang')
    readonly_fields = ('duration_ms', 'created_at')

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        # 데이터가 많은 JSON 필드는 직접 편집하지 않도록 읽기 전용으로 설정
        if 'data' in form.base_fields: #type: ignore
            form.base_fields['data'].widget.attrs['readonly'] = True #type: ignore
        return form

@admin.register(ApiUsageLog)
class ApiUsageLogAdmin(admin.ModelAdmin):
    list_display = ('user', 'book', 'request_type','model_name' ,'prompt_tokens', 'completion_tokens', 'cached_tokens', 'thinking_tokens', 'total_tokens', 'created_at')
    list_filter = ('request_type', 'created_at', 'user', 'book')
    search_fields = ('user__username', 'book__title', 'request_type')
    readonly_fields = (
        'user', 'book', 'request_type', 'model_name', 'prompt_tokens',
        'completion_tokens', 'cached_tokens', 'thinking_tokens', 'total_tokens', 'created_at'
    )
    date_hierarchy = 'created_at'

    def has_add_permission(self, request):
        # 로그는 시스템에서만 생성되므로 수동 추가를 막습니다.
        return False

    def has_change_permission(self, request, obj=None):
        # 로그는 수정할 수 없도록 합니다.
        return False
