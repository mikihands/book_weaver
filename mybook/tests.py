import json
import os
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from pathlib import Path
from django.conf import settings 
from django.test import TestCase
import base64

from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from unittest.mock import patch, MagicMock
from mybook.utils.gemini_helper import GeminiHelper
import unittest
from mybook.utils.layout_prompt import SYS_MSG, create_layout_user_prompt
from mybook.utils.born_digital import _collect_spans_from, collect_page_layout
from mybook.utils.paragraphs import UnitsBuilder

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
        test_dir = Path(settings.BASE_DIR) / "test"
        test_dir.mkdir(exist_ok=True)

        # --- 테스트용 샘플 PDF 생성 ---
        # 이 PDF는 검은색 텍스트, 빨간색 텍스트, 그리고 파란색 테두리와 초록색 채우기를 가진 사각형을 포함합니다.
        sample_doc = fitz.open()
        page = sample_doc.new_page() # type:ignore
        page.insert_text((50, 72), "This is a standard black text.", color=(0, 0, 0))
        page.insert_text((50, 92), "This is a red text.", color=(1, 0, 0))
        rect = fitz.Rect(50, 110, 250, 140)
        page.draw_rect(rect, color=(0, 0, 1), fill=(0, 1, 0), width=1.5)
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(50, 72, 200, 82), "uri": "https://example.com"})
        sample_pdf_path = test_dir / "sample_for_analysis.pdf"
        sample_doc.save(str(sample_pdf_path))
        sample_doc.close()
        # --- 샘플 PDF 생성 끝 ---

        # 분석할 PDF 파일 목록. 여기에 디버깅하고 싶은 파일의 경로를 추가하세요.
        # 예: Path(settings.BASE_DIR) / 'test_pdfs' / '사원총회_의사록_양식.pdf'
        pdf_files_to_test = [
            #sample_pdf_path,
            test_dir / "one-step-from-reality-chapter1_2.pdf" ,
            # 여기에 분석하고 싶은 PDF 파일 경로를 추가하세요.
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
                    "get_text_dict": page.get_text("dict",flags=fitz.TEXTFLAGS_DICT | fitz.TEXT_PRESERVE_IMAGES), # type:ignore
                    #"get_text_rawdict": page.get_text("rawdict"), # type:ignore
                    #"get_text_html": page.get_text("html"), # type:ignore
                    #"get_text_xhtml": page.get_text("xhtml"), # type:ignore
                    #"get_text_blocks": page.get_text("blocks"), # type:ignore
                    "get_drawings": page.get_drawings(extended=True), # type:ignore
                    #"get_drawings": page.get_drawings(), # type:ignore
                    #"get_cdrawings":page.get_cdrawings(extended=True), # get_drawings 에서 빈속성을 뺀 신속 간단 버젼
                    "get_images_full": page.get_images(full=True),
                    #"get_links": page.get_links(), # type:ignore,
                    "get_image_info": page.get_image_info(xrefs=True), # type:ignore
                    #"get_xobjects":page.get_xobjects(),
                }

                # 결과 파일명 생성
                out_dir = Path(settings.BASE_DIR) / "test" / "testbook2"
                out_dir.mkdir(exist_ok=True)
                
                output_filename = f"{pdf_path.stem}_page_{page_index}_analysis.json"
                output_path = out_dir / output_filename

                # JSON 파일로 저장
                with open(output_path, "w", encoding="utf-8") as f:
                    # PyMuPDF가 반환하는 객체 중 직렬화 불가능한 객체(e.g., fitz.Rect)를 처리
                    def default_serializer(o):
                        if isinstance(o, (fitz.Rect, fitz.Point)):
                            return list(o) # type:ignore
                        if isinstance(o, bytes):
                            return o.decode('utf-8', 'replace')
                        if isinstance(o, (set, tuple)):
                            return list(o)
                        if isinstance(o, fitz.Quad):
                            return {
                                "ul": list(o.ul), "ur": list(o.ur), "lr": list(o.lr), "ll": list(o.ll), # type:ignore
                                "bbox": list(o.rect)  # type:ignore
                            }
                        if isinstance(o, fitz.Matrix):
                            return [o.a, o.b, o.c, o.d, o.e, o.f]
                        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

                    json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=default_serializer)

                self.assertTrue(output_path.exists())
                print(f"  - 페이지 {page_index} 분석 완료. 결과 저장: {output_path}")

            doc.close()

class GeminiImageAnalysisTest(TestCase):
    """
    이 테스트는 특정 이미지 파일을 Gemini API로 전송하여, 이미지 내의 주요 객체(figure)를
    탐지하고, 반환된 경계 상자(bounding box)를 사용하여 해당 부분을 잘라내(crop)
    별도의 파일로 저장하는 과정을 검증합니다.

    목적:
    - Gemini의 이미지 객체 탐지 및 JSON 출력 기능 테스트.
    - 반환된 정규화된 좌표를 실제 픽셀 좌표로 변환하는 로직 검증.
    - PIL(Pillow) 라이브러리를 사용하여 이미지를 자르고 저장하는 기능 확인.

    사용법:
    1. `logs/` 디렉토리에 `test_image_with_figures.png` 라는 이름으로 분석하고 싶은
       이미지 파일을 위치시킵니다. (테스트 실행 시 샘플 이미지가 자동 생성됩니다.)
    2. `./manage.py test mybook.tests.GeminiImageAnalysisTest` 명령으로 테스트를 실행합니다.
    3. 실행 후, `logs/` 디렉토리에 `cropped_figure_[인덱스].png` 형식으로
       잘라낸 이미지 파일들이 생성됩니다.
    """
    def setUp(self):
        """테스트에 필요한 디렉토리와 샘플 이미지 파일을 준비합니다."""
        self.logs_dir = Path(settings.BASE_DIR) / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # 테스트용 샘플 이미지 생성
        self.image_path = self.logs_dir / "test_image_with_figures.jpeg"
        if not self.image_path.exists():
            self._create_sample_image()

    def _create_sample_image(self):
        """테스트용 샘플 이미지를 생성합니다."""
        img = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(img)
        # Gemini가 탐지할 만한 사각형(figure)을 그립니다.
        draw.rectangle([200, 150, 600, 450], fill='red', outline='black', width=2)
        # 이미지에 텍스트를 추가합니다.
        draw.text((50, 50), "This is a sample image with text and a figure.", fill="black")
        img.save(self.image_path)
        print(f"\n테스트용 샘플 이미지를 생성했습니다: {self.image_path}")

    def test_detect_and_crop_figures_from_image(self):
        """Gemini를 사용하여 이미지에서 객체를 탐지하고 잘라내어 저장합니다."""
        self.assertTrue(self.image_path.exists(), f"테스트 이미지가 존재해야 합니다: {self.image_path}")

        try:
            # settings.py에 GEMINI_API_KEY가 설정되어 있어야 합니다.
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
        except Exception as e:
            self.fail(f"Gemini 클라이언트 초기화에 실패했습니다. GEMINI_API_KEY 설정을 확인하세요. 오류: {e}")

        prompt = """
        Detect all prominent items in the image. Do not include text.

        - **CRITICAL RULE**: If one detected item is completely inside another (e.g., an icon inside a map), you MUST only return the bounding box for the larger, containing item. Do not return nested items.
        - The response must be a JSON array where each object has a 'label' and a 'box_2d'.
        - The 'box_2d' must be `[ymin, xmin, ymax, xmax]` normalized to 0-1000.
        """

        image = Image.open(self.image_path)

        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=[image, prompt],
                config=config
            )
        except Exception as e:
            self.fail(f"Gemini API 호출에 실패했습니다: {e}")

        width, height = image.size
        
        try:
            raw_text = response.text
            json_start = raw_text.find('[')
            if json_start == -1:
                json_start = raw_text.find('{')
            
            json_end = raw_text.rfind(']')
            if json_end == -1:
                json_end = raw_text.rfind('}')
            
            if json_start != -1 and json_end != -1:
                json_str = raw_text[json_start:json_end+1]
                bounding_boxes = json.loads(json_str)
                if isinstance(bounding_boxes, dict):
                    bounding_boxes = [bounding_boxes]
            else:
                self.fail(f"응답에서 유효한 JSON을 찾을 수 없습니다: {raw_text}")

        except (json.JSONDecodeError, AttributeError) as e:
            self.fail(f"Gemini 응답을 파싱하는 데 실패했습니다: {e}\nResponse text: {response.text}")

        self.assertIsInstance(bounding_boxes, list, "JSON 응답은 리스트여야 합니다.")
        
        print(f"\n--- Gemini 응답 (탐지된 Bounding Boxes) ---")
        print(json.dumps(bounding_boxes, indent=2))
        print("------------------------------------------")

        cropped_files = []
        for i, bounding_box in enumerate(bounding_boxes):
            self.assertIn("box_2d", bounding_box, "각 bounding box에는 'box_2d' 키가 있어야 합니다.")
            box_2d = bounding_box["box_2d"]
            self.assertEqual(len(box_2d), 4, "'box_2d'는 4개의 값을 가져야 합니다. (ymin, xmin, ymax, xmax)")

            abs_y1 = int(box_2d[0] / 1000 * height)
            abs_x1 = int(box_2d[1] / 1000 * width)
            abs_y2 = int(box_2d[2] / 1000 * height)
            abs_x2 = int(box_2d[3] / 1000 * width)
            
            cropped_image = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
            
            output_path = self.logs_dir / f"cropped_figure_{i+1}.png"
            cropped_image.save(output_path)
            
            self.assertTrue(output_path.exists(), f"잘라낸 이미지가 저장되어야 합니다: {output_path}")
            cropped_files.append(output_path)
            print(f"  - Figure {i+1} 저장 완료: {output_path} (bbox: {[abs_x1, abs_y1, abs_x2, abs_y2]})")

        self.assertGreater(len(cropped_files), 0, "하나 이상의 figure를 탐지하고 잘라내야 합니다.")



class TestExtractImagesFromHTML(TestCase):
    """page.get_text('html') 결과에서 <img> 태그를 찾아 이미지 파일로 저장"""

    def setUp(self):
        # 테스트할 PDF 파일 리스트 (BASE_DIR 기준 상대경로)
        self.pdf_files = [
            Path(settings.BASE_DIR) / "test" / "all_cat_test.pdf",
            # Path(settings.BASE_DIR) / "test" / "another.pdf",
        ]
        self.test_dir = Path(settings.BASE_DIR) / "test"
        self.test_dir.mkdir(exist_ok=True)

    def _extract_images_from_page(self, page, output_prefix: str):
        """한 페이지의 HTML에서 img 태그를 찾아 저장"""
        html = page.get_text("html")
        soup = BeautifulSoup(html, "html.parser")
        imgs = soup.find_all("img")

        if not imgs:
            print(f"[INFO] No <img> tags found on page {page.number}")
            return

        for i, img in enumerate(imgs):
            src = img.get("src")
            style = img.get("style", "")
            # 저장 경로 생성
            outfile = self.test_dir / f"{output_prefix}_p{page.number}_img{i}.png"

            if src and src.startswith("data:image"):
                # Base64 디코딩 후 저장
                b64data = src.split(",", 1)[1]
                image_bytes = base64.b64decode(b64data)
                with open(outfile, "wb") as f:
                    f.write(image_bytes)
                print(f"[SAVED] {outfile}")
            else:
                # src 없는 경우 → clip 영역으로 렌더링해서 저장
                print(f"[INFO] Page {page.number} img[{i}] has no src, rendering clip")
                # style에서 위치, 크기 추출
                clip_rect = self._parse_clip_from_style(style, page)
                pix = page.get_pixmap(clip=clip_rect, alpha=True)
                pix.save(outfile)
                print(f"[RENDERED] {outfile}")

    def _parse_clip_from_style(self, style: str, page):
        """
        style 문자열에서 left, top, width, height 값을 추출하여 Rect 반환
        예: style="position:absolute; left:0pt; top:20pt; width:419pt; height:255pt"
        """
        import re
        nums = {}
        for key in ["left", "top", "width", "height"]:
            m = re.search(rf"{key}:\s*([0-9.]+)pt", style)
            if m:
                nums[key] = float(m.group(1))

        # 기본값은 페이지 전체
        left = nums.get("left", 0)
        top = nums.get("top", 0)
        width = nums.get("width", page.rect.width)
        height = nums.get("height", page.rect.height)

        # PyMuPDF Rect(x0, y0, x1, y1)
        return fitz.Rect(left, top, left + width, top + height)

    def test_extract_images_from_html(self):
        for pdf_path in self.pdf_files:
            if not pdf_path.exists():
                print(f"[WARN] File not found: {pdf_path}")
                continue

            with fitz.open(pdf_path) as doc:
                output_prefix = pdf_path.stem
                for page in doc:
                    self._extract_images_from_page(page, output_prefix)


class ImageMaskExtractionTests(TestCase):
    """
    - BASE_DIR/test/test_11page.pdf 등에서 이미지를 찾는다.
    - 각 이미지마다:
        * has-mask=True 이고 smask xref가 있으면: Pixmap 결합(알파 포함)으로 추출
        * 아니면: page.get_pixmap(clip=이미지 bbox, alpha=True)로 '보이는 그대로' 폴백 렌더
    - 결과는 BASE_DIR/test/img_output/ 에 PNG로 저장
    """

    @classmethod
    def setUpTestData(cls):
        cls.test_dir = Path(settings.BASE_DIR) / "test"
        cls.out_dir = cls.test_dir / "img_output"
        cls.out_dir.mkdir(parents=True, exist_ok=True)

        # 여기에 분석하고 싶은 PDF 파일 경로를 추가
        cls.pdf_files_to_test = [
            cls.test_dir / "one-step-from-reality-chapter1_2.pdf",
        ]

    def _extract_with_smask_or_render(self, doc, page, pdf_stem: str) -> int:
        """
        현재 페이지의 모든 이미지에 대해 결과 PNG 생성.
        반환값: 생성된 파일 수
        """
        # 1) 페이지 이미지 목록과 smask 매핑(xref -> smask_xref) 확보
        smask_map = {}
        for tpl in page.get_images(full=True):
            # 튜플 형식: (xref, smask, width, height, bpc, colorspace, alt_cs, name, filter, referencer)
            xref, smask_xref = tpl[0], tpl[1]
            if smask_xref:
                smask_map[xref] = smask_xref

        # 2) get_image_info로 각 이미지 메타( bbox / has-mask / xref ) 순회
        outputs = 0
        infos = page.get_image_info(xrefs=True)
        for idx, info in enumerate(infos, 1):
            xref = info.get("xref")
            has_mask = bool(info.get("has-mask"))
            bbox = info.get("bbox")
            tag = f"{pdf_stem}_p{page.number + 1:03d}_img{idx:02d}"

            try:
                if has_mask and smask_map.get(xref):
                    # ---- SMask 결합 추출 ----
                    base_pix = fitz.Pixmap(doc, xref)
                    # [FIX] base_pix에 알파 채널이 있으면 제거합니다.
                    # fitz.Pixmap(base, mask)를 호출하려면 base에 알파가 없어야 합니다.
                    if base_pix.alpha:
                        base_pix = fitz.Pixmap(fitz.csRGB, base_pix)

                    mask_pix = fitz.Pixmap(doc, smask_map[xref])
                    combined = fitz.Pixmap(base_pix, mask_pix)  # 알파 합성
                    out_path = self.out_dir / f"{tag}_withalpha.png"
                    combined.save(out_path.as_posix())
                    print(f"[SMASK] {tag} (xref={xref}, smask={smask_map[xref]}) -> {out_path.name}")
                    outputs += 1
                else:
                    # ---- 폴백: 보이는 그대로 렌더링 ----
                    if not has_mask:
                        print(f"[FALLBACK:no-has-mask] {tag} (xref={xref})")
                    else:
                        print(f"[FALLBACK:smask-missing] {tag} (xref={xref}, has-mask=True, smask_map=0)")

                    rect = fitz.Rect(*bbox)
                    # 해상도가 너무 낮지 않도록 DPI 약간 올림
                    pix = page.get_pixmap(clip=rect, alpha=True, dpi=144)
                    out_path = self.out_dir / f"{tag}_render.png"
                    pix.save(out_path.as_posix())
                    outputs += 1

            except Exception as e:  # pragma: no cover
                print(f"처리 실패: {tag} (xref={xref}): {e}")

        return outputs

    def test_extract_images_with_smask_or_render(self):
        total_outputs = 0
        any_pdf_found = False

        for pdf_path in self.pdf_files_to_test:
            if not pdf_path.exists():
                print("테스트 PDF가 없습니다: %s (스킵)", pdf_path)
                continue

            any_pdf_found = True
            with fitz.open(pdf_path.as_posix()) as doc:
                pdf_stem = pdf_path.stem
                for page in doc:
                    total_outputs += self._extract_with_smask_or_render(doc, page, pdf_stem)

        if not any_pdf_found:
            self.skipTest("테스트용 PDF가 없어 스킵합니다. BASE_DIR/test/ 에 파일을 두세요.")

        # 최소 한 장이라도 생성되었는지 확인 (문서에 이미지가 전혀 없다면 실패)
        self.assertGreater(total_outputs, 0, "결과 PNG가 생성되지 않았습니다. 테스트 PDF와 페이지 내 이미지 유무를 확인하세요.")


# This is a sample response that Gemini might return, conforming to the schema.
# We'll use this to mock the API call in the test.
MOCK_GEMINI_RESPONSE = {
    "book_id": "test-book-123",
    "page_no": 1,
    "paragraphs": [
        {
            "role": "title",
            "span_indices": [0, 1],
            "alignment": "center",
            "flow": "left_to_right",
            "font_changes": []
        },
        {
            "role": "body",
            "span_indices": [2, 3, 4],
            "alignment": "justify",
            "flow": "left_to_right",
            "font_changes": [
                {
                    "text": "important concept",
                    "styles": ["bold"]
                }
            ]
        },
        {
            "role": "pagination",
            "span_indices": [5],
            "alignment": "center",
            "flow": "left_to_right",
            "font_changes": []
        }
    ]
}


class GeminiLayoutAnalysisTest(TestCase):
    def setUp(self):
        # Load the JSON schema
        schema_path = os.path.join(settings.BASE_DIR, 'mybook', 'utils', 'born_digital_weaver_layout.v1.json')
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        self.test_dir = Path(settings.BASE_DIR) / "test"
        self.pdf_path = self.test_dir / "testbook2-4_5page.pdf" # The user will provide this file.

    @patch('mybook.utils.gemini_helper.genai.Client')
    def test_layout_analysis_request_with_real_pdf(self, mock_genai_client):
        """
        Tests sending a layout analysis request using a real PDF file
        from the 'test' directory.
        """
        # Skip the test if the user-provided PDF is not found.
        if not self.pdf_path.exists():
            self.skipTest(f"Test PDF not found at {self.pdf_path}. Please place a single-page PDF there to run this test.")

        # --- 1. Setup Mocks ---
        mock_response = MagicMock()
        mock_response.text = json.dumps(MOCK_GEMINI_RESPONSE)
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_client_instance = MagicMock()
        mock_client_instance.models = mock_model_instance
        mock_genai_client.return_value = mock_client_instance

        # --- 2. Prepare Inputs from Real PDF ---
        book_id = "gemini-layout-test"
        page_no = 1

        # Open the PDF and extract spans
        doc = fitz.open(self.pdf_path)
        self.assertEqual(doc.page_count, 1, "The test PDF should be a single page.")
        page = doc.load_page(0) # 0-indexed
        page_dict = page.get_text("dict")
        spans = _collect_spans_from(page_dict, page_no)
        doc.close()

        self.assertGreater(len(spans), 0, "Spans should be extracted from the PDF.")

        # Create the file part for Gemini
        with open(self.pdf_path, "rb") as f:
            pdf_bytes = f.read()
        file_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        # Create the user message
        user_msg = create_layout_user_prompt(book_id, page_no, spans)

        # --- 3. Instantiate Helper and Call ---
        gemini_helper = GeminiHelper(schema=self.schema)
        result, errors = gemini_helper.generate_page_json(
            file_part=file_part,
            sys_msg=SYS_MSG,
            user_msg=user_msg,
            example_json=json.dumps(self.schema.get("examples", [])[0])
        )

        # --- 4. Assertions ---
        mock_model_instance.generate_content.assert_called_once()
        # Check that the call was made with the correct parts
        call_args, call_kwargs = mock_model_instance.generate_content.call_args
        contents = call_kwargs['contents']
        self.assertEqual(contents[0], file_part)
        self.assertIn('"task": "Analyze the layout', contents[1])
        self.assertIn(f'"book_id": "{book_id}"', contents[1])
        self.assertIn('"spans":', contents[1])
        self.assertTrue(len(json.loads(contents[1])['spans']) > 0)

        # Check the result (which is from the mock response)
        self.assertIsNotNone(result, "The result should not be None on success.")
        self.assertIsNone(errors, "Errors should be None on success.")
        self.assertEqual(result['book_id'], MOCK_GEMINI_RESPONSE['book_id'])
        self.assertEqual(len(result['paragraphs']), 3)

    @unittest.skipUnless(os.getenv('RUN_REAL_API_TESTS') == 'true', "Skipping real API test. Set RUN_REAL_API_TESTS=true to run.")
    def test_real_layout_analysis_with_gemini_api(self):
        """
        Sends a REAL request to the Gemini API and prints/saves the result.
        This test will only run if the environment variable RUN_REAL_API_TESTS is set to 'true'.
        
        Usage:
        $ RUN_REAL_API_TESTS=true ./manage.py test mybook.tests.GeminiLayoutAnalysisTest.test_real_layout_analysis_with_gemini_api
        """
        if not self.pdf_path.exists():
            self.skipTest(f"Test PDF not found at {self.pdf_path}. Please place a single-page PDF there to run this test.")

        # --- 1. Prepare Inputs from Real PDF ---
        book_id = "gemini-layout-test-real"
        page_no = 1

        doc = fitz.open(self.pdf_path)
        page = doc.load_page(0)
        page_dict = page.get_text("dict")
        spans = _collect_spans_from(page_dict, page_no)
        doc.close()

        with open(self.pdf_path, "rb") as f:
            pdf_bytes = f.read()
        file_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

        user_msg = create_layout_user_prompt(book_id, page_no, spans)

        # --- 2. Instantiate Helper and Call (REAL API CALL) ---
        from mybook.utils.gemini_helper import GeminiHelper
        gemini_helper = GeminiHelper(schema=self.schema)
        result, errors = gemini_helper.generate_page_json(
            file_part=file_part,
            sys_msg=SYS_MSG,
            user_msg=user_msg,
            example_json=json.dumps(self.schema.get("examples", [])[0])
        )

        # --- 3. Check and Save the Result ---
        print("\n--- Gemini API Real Response ---")
        if errors:
            print("🚨 Errors occurred:")
            print(json.dumps(errors, indent=2, ensure_ascii=False))
        else:
            print("✅ Success! Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # Save the result to a file for inspection
            result_path = self.test_dir / "layout_test_result.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n📄 Result saved to: {result_path}")

        self.assertIsNone(errors, "The real API call should not produce schema or parsing errors.")
        self.assertIsNotNone(result, "The real API call should return a result.")
        if result:
            self.assertEqual(result['book_id'], book_id)
