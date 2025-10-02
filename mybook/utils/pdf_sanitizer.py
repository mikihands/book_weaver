#mybook/utils/pdf_sanitizer.py
import fitz  # PyMuPDF
import logging
import os
import tempfile
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# PDF 내에서 잠재적으로 위험한 "active" 요소를 나타내는 키워드 목록
# 이 목록은 필요에 따라 확장될 수 있습니다.
DANGEROUS_PDF_KEYS = [
    "/JavaScript",
    "/OpenAction",
    "/AA",  # Additional Actions
    "/Launch",
    "/EmbeddedFiles",
    "/RichMedia",
    #"/URI", # URI 액션도 외부 링크로 간주될 수 있어 포함 (필요에 따라 조절)
    "/SubmitForm",
    "/GoToR", # Remote GoTo action
    "/GoToE", # Embedded GoTo action
    "/Sound",
    "/Movie",
    "/3D",
]

def sanitize_pdf_active_content(original_pdf_path: str) -> tuple[Optional[str], List[str]]:
    """
    PDF 파일에서 잠재적으로 위험한 active 콘텐츠(예: JavaScript, OpenAction 등)를 제거합니다.
    원본 파일을 수정하지 않고, 정화된 내용을 임시 파일로 저장한 후 해당 경로를 반환합니다.

    :param original_pdf_path: 원본 PDF 파일의 경로.
    :return: 튜플 (정화된 임시 PDF 파일 경로 | None, 제거된 키 목록).
    """
    doc = None
    try:
        doc = fitz.open(original_pdf_path)
        removed_keys: List[str] = []
        modified = False

        # 1. 문서 카탈로그(Document Catalog)에서 active 요소 제거
        # 문서 카탈로그는 PDF의 최상위 객체로, 문서 전체에 영향을 미치는 액션을 포함할 수 있습니다.
        catalog_xref = doc.pdf_catalog()
        if catalog_xref > 0:
            for key in DANGEROUS_PDF_KEYS:
                if doc.xref_get_key(catalog_xref, key):
                    doc.xref_set_key(catalog_xref, key, "null")
                    removed_keys.append(key)
                    logger.warning(f"Removed '{key}' from document catalog (xref: {catalog_xref}) in {original_pdf_path}")
                    modified = True

        # 2. 각 페이지에서 active 요소 제거
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            page_xref = page.xref

            # 페이지 딕셔너리에서 active 요소 제거
            for key in DANGEROUS_PDF_KEYS:
                if doc.xref_get_key(page_xref, key):
                    doc.xref_set_key(page_xref, key, "null")
                    removed_keys.append(key)
                    logger.warning(f"Removed '{key}' from page {pno+1} dictionary (xref: {page_xref}) in {original_pdf_path}")
                    modified = True

            # 어노테이션(링크, 필드 등)에서 active 요소 제거
            # 어노테이션은 /A (Action) 또는 /AA (Additional Actions) 키를 가질 수 있습니다.
            for annot in page.annots():
                annot_xref = annot.xref
                if annot_xref > 0:
                    for key in ["/A", "/AA"]: # 어노테이션에 특화된 액션 키
                        if doc.xref_get_key(annot_xref, key):
                            doc.xref_set_key(annot_xref, key, "null")
                            removed_keys.append(f"{key} in Annotation")
                            logger.warning(f"Removed '{key}' from annotation (xref: {annot_xref}) on page {pno+1} in {original_pdf_path}")
                            modified = True

        if modified:
            # 변경 사항이 있다면 임시 파일로 저장
            # NamedTemporaryFile을 사용하여 파일 시스템에 저장하고 경로를 얻습니다.
            # delete=False로 설정하여 함수 종료 후에도 파일이 유지되도록 합니다.
            # 호출자가 이 파일을 사용한 후 명시적으로 삭제해야 합니다.
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_path = temp_file.name
            temp_file.close() # NamedTemporaryFile은 열린 상태로 반환되므로 닫아줍니다.

            doc.save(temp_path, garbage=3, deflate=True) # 최적화하여 저장
            logger.info(f"PDF active content sanitized. Saved to temporary file: {temp_path}")
            return temp_path, list(set(removed_keys))
        else:
            logger.info(f"No active content found or removed in {original_pdf_path}. No new file created.")
            return None, []

    except Exception as e:
        logger.error(f"Error sanitizing PDF '{original_pdf_path}': {e}", exc_info=True)
        return None, []
    finally:
        # doc 객체가 성공적으로 열렸다면 항상 닫아줍니다.
        if doc:
            doc.close()