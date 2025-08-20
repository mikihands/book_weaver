from django.contrib import admin
from .models import UploadedFile, TranslatedPage

@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'original_file', 'uploaded_at')
    list_display_links = ('id', 'original_file')
    list_filter = ('uploaded_at', 'user')
    search_fields = ('user__username', 'original_file')
    readonly_fields = ('uploaded_at',)

@admin.register(TranslatedPage)
class TranslatedPageAdmin(admin.ModelAdmin):
    list_display = ('id', 'uploaded_file', 'page_number', 'html_url')
    list_display_links = ('id', 'uploaded_file')
    list_filter = ('uploaded_file',)
    search_fields = ('uploaded_file__original_file__name',)
    
    def get_form(self, request, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        # 번역된 HTML은 내용이 길어서 직접 편집하지 않도록 텍스트 영역을 비활성화
        if 'translated_html' in form.base_fields: #type:ignore
            form.base_fields['translated_html'].widget.attrs['readonly'] = True #type:ignore
        return form
