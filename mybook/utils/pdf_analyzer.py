import fitz  # PyMuPDF
from typing import List, Dict, Any

class PdfAnalyzer:
    """
    PyMuPDF를 사용하여 PDF 문서를 분석하는 클래스입니다.
    OCR 처리 여부를 판별하고, 텍스트, 이미지, 폰트 등 다양한 정보를 추출합니다.
    """

    def __init__(self, pdf_path: str):
        """
        클래스 초기화 시 PDF 파일을 엽니다.

        :param pdf_path: 분석할 PDF 파일의 경로
        """
        try:
            self.doc = fitz.open(pdf_path)
            self.path = pdf_path
            print(f"✅ '{pdf_path}' 파일을 성공적으로 열었습니다.")
        except Exception as e:
            self.doc = None
            print(f"❌ 파일 열기 실패: {e}")
            raise

    def is_ocr_processed(self, text_length_threshold: int = 100) -> bool:
        """
        PDF에 텍스트 레이어가 있는지 확인하여 OCR 처리 여부를 판별합니다.
        페이지의 총 텍스트 길이가 임계값보다 크면 OCR 처리된 것으로 간주합니다.

        :param text_length_threshold: OCR 처리로 판단할 최소 텍스트 길이 (기본값: 100)
        :return: OCR 처리되었으면 True, 아니면 False
        """
        if not self.doc:
            return False
        
        total_text_length = 0
        # 문서의 모든 페이지를 순회하며 텍스트 길이를 합산
        for page in self.doc:
            total_text_length += len(page.get_text("text").strip())
            if total_text_length > text_length_threshold:
                return True
        return False

    def get_text_data(self) -> List[Dict[str, Any]]:
        """
        문서의 모든 페이지에서 텍스트 블록과 위치 정보를 추출합니다.
        OCR 처리가 되어있지 않으면 빈 리스트를 반환합니다.

        :return: 페이지별 텍스트 블록 정보 리스트
        """
        if not self.doc or not self.is_ocr_processed():
            print("⚠️ 텍스트를 추출할 수 없습니다. OCR 처리가 안 된 파일일 수 있습니다.")
            return []

        all_pages_data = []
        for page_num, page in enumerate(self.doc):
            # "dict" 옵션을 사용하여 구조화된 데이터 추출
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
            page_data = {
                "page_number": page_num + 1,
                "blocks": blocks
            }
            all_pages_data.append(page_data)
        return all_pages_data

    def get_image_data(self) -> List[Dict[str, Any]]:
        """
        문서의 모든 페이지에서 이미지 정보를 추출합니다. (xref, 크기 등)

        :return: 페이지별 이미지 정보 리스트
        """
        if not self.doc:
            return []
            
        all_images_data = []
        for page_num, page in enumerate(self.doc):
            images = page.get_images(full=True)
            if images:
                page_images = {
                    "page_number": page_num + 1,
                    "images": [
                        {
                            "xref": img[0],
                            "width": img[2],
                            "height": img[3],
                            "bpc": img[4], # bits per component
                            "colorspace": img[5],
                            "name": img[7]
                        }
                        for img in images
                    ]
                }
                all_images_data.append(page_images)
        return all_images_data

    def get_font_data(self) -> List[Dict[str, Any]]:
        """
        문서에서 사용된 모든 폰트 정보를 추출합니다.

        :return: 폰트 정보 리스트 (xref, type, basefont, name)
        """
        if not self.doc:
            return []
            
        fonts = self.doc.get_fonts(full=True)
        return [
            {
                "xref": font[0],
                "type": font[1],
                "basefont": font[2],
                "name": font[3]
            }
            for font in fonts
        ]

    def close(self):
        """
        PDF 문서를 닫습니다.
        """
        if self.doc:
            self.doc.close()
            print(f"'{self.path}' 파일을 닫았습니다.")

# --- 클래스 사용 예시 ---
if __name__ == '__main__':
    # 여기에 실제 PDF 파일 경로를 입력하세요.
    # ocr_pdf_path = "path/to/your/ocr_document.pdf"
    # image_pdf_path = "path/to/your/image_only_document.pdf"
    
    # 예시를 위해 임시 PDF 파일을 생성합니다.
    # 1. OCR 처리된 PDF 생성 (텍스트 포함)
    doc_ocr = fitz.new_doc()
    page_ocr = doc_ocr.new_page()
    page_ocr.insert_text((50, 72), "This is a test document with OCR-like text.", fontsize=12)
    doc_ocr.save("ocr_example.pdf")
    doc_ocr.close()
    
    # 2. 이미지만 있는 PDF 생성
    doc_img = fitz.new_doc()
    doc_img.new_page()
    doc_img.save("image_example.pdf") # 내용은 없지만 구조는 PDF
    doc_img.close()
    
    print("--- 1. OCR 처리된 PDF 분석 ---")
    try:
        analyzer_ocr = PdfAnalyzer("ocr_example.pdf")

        # OCR 여부 판별
        if analyzer_ocr.is_ocr_processed():
            print("결과: OCR 처리된 문서입니다. 👍")
            
            # 정보 추출
            text_info = analyzer_ocr.get_text_data()
            print("\n[추출된 텍스트 정보 (첫 페이지만)]")
            print(text_info[0] if text_info else "없음")
            
            font_info = analyzer_ocr.get_font_data()
            print("\n[추출된 폰트 정보]")
            print(font_info)
            
        else:
            print("결과: 이미지만 있는 문서입니다. 📄")
        
        analyzer_ocr.close()

    except Exception as e:
        print(f"오류 발생: {e}")

    print("\n" + "="*50 + "\n")

    print("--- 2. 이미지만 있는 PDF 분석 ---")
    try:
        analyzer_img = PdfAnalyzer("image_example.pdf")

        # OCR 여부 판별
        if analyzer_img.is_ocr_processed():
            print("결과: OCR 처리된 문서입니다. 👍")
        else:
            print("결과: 이미지만 있는 문서입니다. 📄")
            
            # 정보 추출 시도
            text_info_img = analyzer_img.get_text_data()
            print("\n[텍스트 정보 추출 시도]")
            print(text_info_img)
            
        analyzer_img.close()
    except Exception as e:
        print(f"오류 발생: {e}")