import json
from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings
from django.test import TestCase


class PdfDebuggingAnalysisTest(TestCase):
    """
    이 테스트는 특정 PDF 파일들을 분석하여 PyMuPDF(fitz)의 주요 메서드들이
    어떤 원시 데이터를 반환하는지 확인하고, 그 결과를 JSON 파일로 저장합니다.

    목적:
    - PDF 렌더링, 특히 텍스트 색상 추출과 관련된 문제 디버깅.
    - PDF 파일마다 `get_text("dict")`, `get_text("rawdict")`, `get_drawings()` 등의
      출력 차이를 명확히 이해하기 위함.
    - 예를 들어, 텍스트 색상이 `rawdict`에 포함되는지, 아니면 별도의 `drawing` 객체로
      처리되는지 등을 파악할 수 있습니다.

    사용법:
    1. `pdf_files_to_test` 리스트에 분석하고 싶은 PDF 파일의 절대 경로를 추가합니다.
       (프로젝트 루트에 `test_pdfs` 폴더를 만들고 그 안에 파일을 두는 것을 권장합니다.)
    2. `./manage.py test mybook.tests.PdfDebuggingAnalysisTest` 명령으로 테스트를 실행합니다.
    3. 실행 후, 프로젝트 루트의 `logs/` 디렉토리에 `[파일명]_page_[번호]_analysis.json`
       형식의 분석 결과 파일들이 생성됩니다.
    4. 생성된 JSON 파일을 열어 각 메서드의 반환 값을 확인하고, `born_digital.py`의
       로직과 비교하며 문제를 추적합니다.
    """

    def test_analyze_pdfs_for_debugging(self):
        # 결과 JSON 파일을 저장할 디렉토리
        logs_dir = Path(settings.BASE_DIR) / "logs"
        logs_dir.mkdir(exist_ok=True)

        # --- 테스트용 샘플 PDF 생성 ---
        # 이 PDF는 검은색 텍스트, 빨간색 텍스트, 그리고 파란색 테두리와 초록색 채우기를 가진 사각형을 포함합니다.
        sample_doc = fitz.open()
        page = sample_doc.new_page() # type:ignore
        page.insert_text((50, 72), "This is a standard black text.", color=(0, 0, 0))
        page.insert_text((50, 92), "This is a red text.", color=(1, 0, 0))
        rect = fitz.Rect(50, 110, 250, 140)
        page.draw_rect(rect, color=(0, 0, 1), fill=(0, 1, 0), width=1.5)
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(50, 72, 200, 82), "uri": "https://example.com"})
        sample_pdf_path = logs_dir / "sample_for_analysis.pdf"
        sample_doc.save(str(sample_pdf_path))
        sample_doc.close()
        # --- 샘플 PDF 생성 끝 ---

        # 분석할 PDF 파일 목록. 여기에 디버깅하고 싶은 파일의 경로를 추가하세요.
        # 예: Path(settings.BASE_DIR) / 'test_pdfs' / '사원총회_의사록_양식.pdf'
        pdf_files_to_test = [
            #sample_pdf_path,
            #logs_dir / "사원총회 의사록 양식.pdf" ,
            logs_dir / "weekly_news.pdf" ,
            # 여기에 분석하고 싶은 PDF 파일 경로를 추가하세요.
            # Path(settings.BASE_DIR) / '사원총회_의사록_양식.pdf',
        ]

        for pdf_path in pdf_files_to_test:
            if not pdf_path.exists():
                print(f"경고: 파일을 찾을 수 없습니다. 건너뜁니다: {pdf_path}")
                continue

            self.assertTrue(pdf_path.exists(), f"테스트 파일이 존재해야 합니다: {pdf_path}")
            doc = fitz.open(pdf_path)
            print(f"\n--- PDF 분석 중: {pdf_path.name} ({doc.page_count} 페이지) ---")

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_index = page_num + 1

                # born_digital.py에서 사용하는 주요 fitz 메서드들의 결과를 수집
                analysis_results = {
                    "metadata": {
                        "filename": pdf_path.name,
                        "page_index": page_index,
                        "page_rect": list(page.rect),
                    },
                    "get_text_dict": page.get_text("dict"), # type:ignore
                    "get_text_rawdict": page.get_text("rawdict"), # type:ignore
                    "get_drawings": page.get_drawings(),
                    "get_images_full": page.get_images(full=True),
                    "get_links": page.get_links(), # type:ignore
                }

                # 결과 파일명 생성
                output_filename = f"{pdf_path.stem}_page_{page_index}_analysis.json"
                output_path = logs_dir / output_filename

                # JSON 파일로 저장
                with open(output_path, "w", encoding="utf-8") as f:
                    # PyMuPDF가 반환하는 객체 중 직렬화 불가능한 객체(e.g., fitz.Rect)를 처리
                    def default_serializer(o):
                        if isinstance(o, (fitz.Rect, fitz.Point)):
                            return list(o)
                        if isinstance(o, bytes):
                            return o.decode('utf-8', 'replace')
                        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

                    json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=default_serializer)

                self.assertTrue(output_path.exists())
                print(f"  - 페이지 {page_index} 분석 완료. 결과 저장: {output_path}")

            doc.close()
