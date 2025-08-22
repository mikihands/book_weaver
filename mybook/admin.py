from django.contrib import admin
from .models import Book, BookPage, PageImage, TranslatedPage

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
    readonly_fields = ('tokens_used', 'duration_ms', 'created_at')

    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        # 데이터가 많은 JSON 필드는 직접 편집하지 않도록 읽기 전용으로 설정
        if 'data' in form.base_fields: #type: ignore
            form.base_fields['data'].widget.attrs['readonly'] = True #type: ignore
        return form
