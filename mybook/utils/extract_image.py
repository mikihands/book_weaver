#mybook/utils/extract_image.py
import pymupdf as fitz
from pathlib import Path
from typing import List, Dict, Any
import os
from PIL import Image
from dataclasses import dataclass
from typing import Tuple
from io import BytesIO


def _rect_to_xywh(rect) -> List[float]:
    # rect: fitz.Rect (x0,y0,x1,y1) in pt
    x, y, x2, y2 = rect
    return [float(x), float(y), float(x2 - x), float(y2 - y)]

def extract_images_and_bboxes(
    pdf_path: str,
    out_dir: str,
    dpi: int = 144,   # 원하는 출력 DPI (px 변환용)
) -> List[Dict[str, Any]]:
    """
    returns: [
      {
        "page_no": 1,
        "size": {"w": <px>, "h": <px>},
        "images": [
          {"ref": "img_p1_1", "path": ".../img_p1_1.png", "bbox": [x,y,w,h]}
        ]
      }, ...
    ]
    """
    doc = fitz.open(pdf_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scale = dpi / 72.0  # pt -> px 변환 스케일
    results = []

    for pno in range(len(doc)):
        page = doc[pno]
        # 페이지 크기(pt) -> px 변환
        pw_pt, ph_pt = page.rect.width, page.rect.height
        pw_px, ph_px = pw_pt * scale, ph_pt * scale

        page_info = {
            "page_no": pno + 1,
            "size": {"w": pw_px, "h": ph_px},
            "images": []
        }

        # 1) 이미지 메타(xref 리스트)
        # get_images(full=True) -> [(xref, smask, width, height, bpc, colorspace, alt, name, ...), ...]
        xref_seen = set()
        for (xref, *_rest) in page.get_images(full=True): # type: ignore
            if xref in xref_seen:
                continue
            xref_seen.add(xref)

            # 2) 이 이미지가 배치된 모든 위치의 Rect 얻기
            rects = page.get_image_rects(xref)  # [Rect(...), ...] # type: ignore
            if not rects:
                continue  # 혹시라도 배치 정보가 없으면 스킵

            # 3) 이미지 파일 추출 (한 번만 저장, 여러 rect가 같은 파일을 참조)
            img = doc.extract_image(xref)
            ext = img.get("ext", "png")
            ref_base = f"img_p{pno+1}_{xref}"
            img_path = Path(out_dir) / f"{ref_base}.{ext}"
            with open(img_path, "wb") as f:
                f.write(img["image"])

            # 4) 배치된 각 위치마다 bbox 레코드 생성
            for idx, rect in enumerate(rects, start=1):
                xywh_pt = _rect_to_xywh(rect)               # pt 좌표
                # pt -> px 스케일 적용
                x_px = xywh_pt[0] * scale
                y_px = xywh_pt[1] * scale
                w_px = xywh_pt[2] * scale
                h_px = xywh_pt[3] * scale

                ref = f"{ref_base}_{idx}" if len(rects) > 1 else ref_base
                page_info["images"].append({
                    "ref": ref,
                    "path": str(img_path),
                    "bbox": [x_px, y_px, w_px, h_px],
                    "xref": int(xref)  # 디버깅/추적용(옵션)
                })

        results.append(page_info)

    return results

def is_fullpage_background(bbox, page_w: float, page_h: float, *, area_thresh=0.88, margin=8) -> bool:
    """bbox가 페이지 전체를 사실상 덮는지 휴리스틱으로 판정"""
    try:
        x, y, w, h = bbox
    except Exception:
        return False
    if page_w <= 0 or page_h <= 0:
        return False
    area_ratio = (w * h) / (page_w * page_h + 1e-6)
    near_left   = x <= margin
    near_top    = y <= margin
    near_right  = abs((x + w) - page_w) <= margin
    near_bottom = abs((y + h) - page_h) <= margin
    return (area_ratio >= area_thresh) and near_left and near_top and near_right and near_bottom

def split_images_for_prompt(images, page_w, page_h):
    """Split images into background and figure categories."""
    bg = []
    figures = []
    for im in images:
        if is_fullpage_background(im["bbox"], page_w, page_h):
            bg.append(im)
        else:
            figures.append(im)
    return bg, figures


@dataclass
class FigureBBox:
    x: int
    y: int
    w: int
    h: int

class ImageExtractor:
    """
    - 페이지를 지정된 폭(norm_w)에 맞춰 한 번 렌더(캐시)
    - 렌더된 래스터에서 bbox 픽셀 영역을 잘라 PNG로 저장
    - bbox는 [x, y, w, h] (정규화 좌표계: width=norm_w, height=norm_h) 기준
    """
    def __init__(self, pdf_path: str, norm_w: int, norm_h: int | None = None):
        self.pdf_path = pdf_path
        try:
            self.doc = fitz.open(pdf_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to open PDF file: {e}")
        self.norm_w = int(norm_w)
        self.norm_h = int(norm_h) if norm_h else None
        self._page_cache: Dict[int, Image.Image] = {}  # page_no -> PIL Image

    def close(self):
        if self.doc:
            self.doc.close()

    # --- 내부 유틸 ---
    def _render_page_raster(self, page_no: int) -> Image.Image:
        """
        page_no(1-base)를 norm_w 폭에 맞춰 렌더.
        norm_h가 주어져도 aspect 유지 후 scale만 맞추면 됨(세로는 자동).
        """
        if page_no in self._page_cache:
            return self._page_cache[page_no]

        page = self.doc[page_no - 1]
        rect = page.rect  # PDF points(72dpi) 좌표계
        # 원하는 출력 폭
        target_w = self.norm_w
        # zoomX = target_w / page.width_in_pixels_at_1.0? -> points 기반이므로 아래처럼
        # 1.0 zoom일 때 너비 픽셀 ≈ rect.width * 1.333... (96dpi 가정)이지만
        # PyMuPDF는 matrix로 배율을 직접 지정하면 픽셀 출력이 됨.
        zoom = target_w / float(rect.width)  # rect.width는 points, matrix가 곱해져 비율만 맞으면 OK
        mat = fitz.Matrix(zoom, zoom)
        pix = fitz.utils.get_pixmap(page, matrix=mat, alpha=False)  # 전체 페이지 렌더
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        self._page_cache[page_no] = img
        return img

    @staticmethod
    def _clamp_bbox(x:int,y:int,w:int,h:int,W:int,H:int) -> Tuple[int,int,int,int]:
        x = max(0, min(x, W))
        y = max(0, min(y, H))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return x, y, w, h

    # --- 공개 API ---
    def extract_figure(
        self,
        page_no: int,
        bbox: Dict[str, int] | List[int] | Tuple[int,int,int,int],
        output_path: str,
        fmt: str = "PNG",
        quality: int = 90
    ) -> bool:
        """
        bbox: {"x":..,"y":..,"w":..,"h":..} 또는 [x,y,w,h]
        """
        try:
            # 페이지 렌더(캐시)
            page_img = self._render_page_raster(page_no)
            W, H = page_img.width, page_img.height

            # bbox 통일
            if isinstance(bbox, dict):
                x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
            else:
                x, y, w, h = map(int, bbox)

            x, y, w, h = self._clamp_bbox(x, y, w, h, W, H)

            # crop
            crop = page_img.crop((x, y, x+w, y+h))

            # 저장 디렉토리
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if fmt.upper() == "PNG":
                crop.save(output_path, "PNG")
            elif fmt.upper() in ("JPG","JPEG"):
                crop = crop.convert("RGB")
                crop.save(output_path, "JPEG", quality=quality, optimize=True)
            elif fmt.upper() == "WEBP":
                crop.save(output_path, "WEBP", quality=quality, method=6)
            else:
                crop.save(output_path)  # 포맷 추정
            return True
        except Exception as e:
            print(f"[ImageExtractor] Error extracting p{page_no} bbox={bbox}: {e}")
            return False

    def extract_many(
        self,
        page_no: int,
        figures: List[Dict],
        output_dir: str,
        prefix: str = "fig",
        ext: str = "png"
    ) -> Dict[str, str]:
        """
        figures: [{"ref":"img_pN_1", "bbox":[x,y,w,h], ... }, ...]
        return: { ref: absolute_path }
        """
        os.makedirs(output_dir, exist_ok=True)
        out: Dict[str,str] = {}
        for i, f in enumerate(figures, 1):
            ref = f.get("ref") or f"img_p{page_no}_{i}"
            bbox = f.get("bbox") or [f.get("bbox_x"), f.get("bbox_y"), f.get("bbox_w"), f.get("bbox_h")]
            if not bbox or len(bbox) != 4:
                raise ValueError(f"Invalid bbox for figure {ref}: {bbox}")
            out_path = os.path.join(output_dir, f"{prefix}_p{page_no}_{i}.{ext}")
            ok = self.extract_figure(page_no, bbox, out_path, fmt=ext.upper()) #type:ignore
            if ok:
                out[ref] = out_path
        return out