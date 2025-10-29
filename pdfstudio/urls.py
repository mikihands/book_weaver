# pdfstudio/urls.py
from django.urls import path
from .views import (
    EditorPage,                     # GET: 편집 UI (템플릿)
    ListPagesAPI,                   # GET: 썸네일/상태
    ReorderPagesAPI,                # PATCH: 순서 반영
    DeletePagesAPI, RestorePagesAPI,# POST: 삭제/복원
    CommitEditsAPI, ResetEditsAPI,  # POST: 커밋/초기화
    StreamOriginalForEditorView,    # GET: 에디터에서 PDF.js로 볼 때 원본 스트림(소유자 보호)
)

app_name = "pdfstudio"

urlpatterns = [
    path("books/<int:book_id>/editor/", EditorPage.as_view(), name="editor"),
    path("api/books/<int:book_id>/pages/", ListPagesAPI.as_view(), name="list_pages"),
    path("api/books/<int:book_id>/pages/reorder/", ReorderPagesAPI.as_view(), name="reorder"),
    path("api/books/<int:book_id>/pages/delete/", DeletePagesAPI.as_view(), name="delete"),
    path("api/books/<int:book_id>/pages/restore/", RestorePagesAPI.as_view(), name="restore"),
    path("api/books/<int:book_id>/commit/", CommitEditsAPI.as_view(), name="commit"),
    path("api/books/<int:book_id>/reset/", ResetEditsAPI.as_view(), name="reset"),
    path("api/books/<int:book_id>/original.pdf", StreamOriginalForEditorView.as_view(), name="stream_original"),
]
