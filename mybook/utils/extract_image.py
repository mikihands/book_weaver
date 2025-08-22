#mybook/utils/extract_image.py
import pymupdf as fitz
from pathlib import Path
from typing import List, Dict, Any

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
