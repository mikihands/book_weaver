# mybook/utils/gemini_file_ref.py
import logging
import io
from django.conf import settings
from google import genai
from google.genai import types
import pymupdf as fitz
from typing import List

logger = logging.getLogger(__name__)
"""
과거 코드: 20~50MB 구간에서 Gemini File API를 사용하여 PDF를 업로드하고
페이지 단위로 참조하는 기능을 담당.
현재는 inline Part 방식을 사용하여, 페이지 단위로 PDF를 추출하고
바이트 스트림으로 Gemini API에 전달하는 방식으로 변경됨.
"""
MIME_PDF = "application/pdf"

class GeminiFileRefManager:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def _create_pdf_part_from_pages(self, original_pdf_path: str, page_numbers: List[int]) -> types.Part:
        """
        Extracts specific pages from a PDF, creates a new in-memory PDF,
        and returns it as a Gemini `Part` object.
        """
        if not page_numbers:
            raise ValueError("Page numbers list cannot be empty.")

        try:
            source_doc = fitz.open(original_pdf_path)
            new_doc = fitz.open()  # Create a new, empty PDF in memory

            for page_num in page_numbers:
                # page_num is 1-based, PyMuPDF is 0-based
                if 0 <= page_num - 1 < source_doc.page_count:
                    new_doc.insert_pdf(source_doc, from_page=page_num - 1, to_page=page_num - 1)
            
            if new_doc.page_count == 0:
                raise ValueError(f"Could not extract any of the requested pages {page_numbers} from the source PDF.")

            # Save the new PDF to a byte stream
            pdf_bytes = new_doc.write()
            new_doc.close()
            source_doc.close()

            return types.Part.from_bytes(data=pdf_bytes, mime_type=MIME_PDF)

        except Exception as e:
            logger.error(f"Failed to create PDF part for pages {page_numbers} from {original_pdf_path}: {e}")
            raise

    def get_page_part(self, book, page_no: int) -> types.Part:
        """
        Extracts a single page from the book's PDF and returns it as a `Part`.
        """
        logger.debug(f"Creating inline Part for book {book.id}, page {page_no}")
        return self._create_pdf_part_from_pages(book.original_file.path, [page_no])

    def get_page_parts(self, book, page_nos: List[int]) -> types.Part:
        """
        Extracts multiple pages from the book's PDF, combines them into a
        single new PDF, and returns it as a `Part`.
        """
        logger.debug(f"Creating combined inline Part for book {book.id}, pages {page_nos}")
        return self._create_pdf_part_from_pages(book.original_file.path, page_nos)
