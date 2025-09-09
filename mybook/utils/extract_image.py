#mybook/utils/extract_image.py
import pymupdf as fitz
from pathlib import Path
from typing import List, Dict, Any
import os
from PIL import Image
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# def _rect_to_xywh(rect) -> List[float]:
#     # rect: fitz.Rect (x0,y0,x1,y1) in pt
#     x, y, x2, y2 = rect
#     return [float(x), float(y), float(x2 - x), float(y2 - y)]

# def extract_images_and_bboxes(
#     pdf_path: str,
#     out_dir: str,
#     dpi: int = 144,   # 원하는 출력 DPI (px 변환용)
#     *,
#     media_root: str
# ) -> List[Dict[str, Any]]:
#     """
#     returns: [
#       {
#         "page_no": 1,
#         "size": {"w": <px>, "h": <px>},
#         "images": [
#           {"ref": "img_p1_1", "path": ".../img_p1_1.png", "bbox": [x,y,w,h], "xref": int}
#         ]
#       }, ...
#     ]
#     """
#     doc = fitz.open(pdf_path)
#     Path(out_dir).mkdir(parents=True, exist_ok=True)

#     scale = dpi / 72.0  # pt -> px 변환 스케일
#     results = []

#     for pno in range(len(doc)):
#         page = doc[pno]
#         # 페이지 크기(pt) -> px 변환
#         pw_pt, ph_pt = page.rect.width, page.rect.height
#         pw_px, ph_px = pw_pt * scale, ph_pt * scale

#         page_info = {
#             "page_no": pno + 1,
#             "size": {"w": pw_px, "h": ph_px},
#             "images": []
#         }

#         # 1) 이미지 메타(xref 리스트)
#         # get_images(full=True) -> [(xref, smask, width, height, bpc, colorspace, alt, name, ...), ...]
#         xref_seen = set()
#         for (xref, *_rest) in page.get_images(full=True): # type: ignore
#             if xref in xref_seen:
#                 continue
#             xref_seen.add(xref)

#             # 2) 이 이미지가 배치된 모든 위치의 Rect 얻기
#             rects = page.get_image_rects(xref)  # [Rect(...), ...] # type: ignore
#             if not rects:
#                 continue  # 혹시라도 배치 정보가 없으면 스킵

#             # 3) 이미지 파일 추출 (한 번만 저장, 여러 rect가 같은 파일을 참조)
#             img = doc.extract_image(xref)
#             ext = img.get("ext", "png")
#             ref_base = f"img_p{pno+1}_{xref}"
#             img_path = Path(out_dir) / f"{ref_base}.{ext}"
#             with open(img_path, "wb") as f:
#                 f.write(img["image"])

#             # ★ MEDIA_ROOT 상대경로 계산
#             rel_path = os.path.relpath(str(img_path), media_root).replace("\\", "/")

#             # 4) 배치된 각 위치마다 bbox 레코드 생성
#             for idx, rect in enumerate(rects, start=1):
#                 xywh_pt = _rect_to_xywh(rect)               # pt 좌표
#                 # pt -> px 스케일 적용
#                 x_px = xywh_pt[0] * scale
#                 y_px = xywh_pt[1] * scale
#                 w_px = xywh_pt[2] * scale
#                 h_px = xywh_pt[3] * scale

#                 ref = f"{ref_base}_{idx}" if len(rects) > 1 else ref_base
#                 page_info["images"].append({
#                     "ref": ref,
#                     "path": rel_path, # 상대경로 저장
#                     "bbox": [x_px, y_px, w_px, h_px],
#                     "xref": int(xref)  # 디버깅/추적용(옵션)
#                 })

#         results.append(page_info)

#     return results

# --------------------------------------------------------------
### (A) 클립 수집: collect_clips() + clip_items_to_path_and_bbox()
# The function `clip_items_to_path_and_bbox` was too restrictive, only handling
# simple rectangular clip paths. It has been removed. The `scissor` rectangle
# provided by PyMuPDF for any clip operation is sufficient and more robust.

def collect_clips(page: fitz.Page):
    """
    extended=True로 드로잉을 받아 'clip' 노드만 뽑아냄.
    The 'scissor' rect is the bounding box of the clip path, which is what we need.
    return: list[{"index":int, "level":int, "scissor":Rect, "bbox_xywh_pt":[x,y,w,h] or None}]
    """
    out = []
    draws = page.get_drawings(extended=True)
    for idx, d in enumerate(draws):
        if d.get("type") == "clip":
            scissor_rect = d.get("scissor")
            bbox_xywh = None
            # We only care about valid, non-empty rectangles.
            if scissor_rect and scissor_rect.is_valid and not scissor_rect.is_empty:
                bbox_xywh = [scissor_rect.x0, scissor_rect.y0, scissor_rect.width, scissor_rect.height]

            out.append({
                "index": idx,
                "level": d.get("level", 0),
                "scissor": scissor_rect,
                "bbox_xywh_pt": bbox_xywh,
            })
    return out

### (B) 이미지 수집: collect_images()

def collect_images(page: fitz.Page, doc: fitz.Document, out_dir: str, media_root: str):
    infos = page.get_image_info(xrefs=True) # type:ignore
    occ = []
    saved = {}
    for order, info in enumerate(infos):
        xref = info.get("xref")
        if not xref:
            continue
        tr = info.get("transform") or [1,0,0,1,0,0]
        bbox = info.get("bbox")     # [x0,y0,x1,y1] (pt)
        origin_image_w = int(info["width"])  # original image width (int)
        origin_image_h = int(info["height"]) # original image height (int)

        # xref 파일 저장(한 번만)
        if xref not in saved:
            raw = doc.extract_image(xref)
            ext = raw.get("ext","png")
            p = Path(out_dir) / f"xref_{xref}.{ext}"
            if not p.exists():
                p.write_bytes(raw["image"])
            rel = os.path.relpath(str(p), media_root).replace("\\","/")
            saved[xref] = {
                "path": rel,
                "img_w": int(raw.get("width") or 1), #서버에 저장된 이미지의 해상도 px
                "img_h": int(raw.get("height") or 1),
            }

        meta = saved[xref]
        occ.append({
            "order": order,
            "xref": int(xref),
            "transform_pt": list(tr), # pt
            "bbox_pt": [bbox[0], bbox[1], bbox[2], bbox[3]],  # xyxy (pt)
            "path": meta["path"],
            "img_w": meta["img_w"], #px
            "img_h": meta["img_h"], #px
            "origin_w": origin_image_w, #px
            "origin_h": origin_image_h, #px
        })
    return occ

### (C) 매칭: iou_xyxy() + match_clips_to_images()

def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0,bx0), max(ay0,by0)
    ix1, iy1 = min(ax1,bx1), min(ay1,by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    areaA = (ax1-ax0)*(ay1-ay0)
    areaB = (bx1-bx0)*(by1-by0)
    return inter / (areaA + areaB - inter + 1e-6)

def match_clips_to_images(clips, images):
    """
    clips: collect_clips()
    images: collect_images()
    return: dict[xref] -> {"clip_bbox_pt":[x,y,w,h] or None}
    (간단화: 사각형 clip만 지원)
    """
    # clip 외접 박스(xyxy)
    clip_rects = []
    for c in clips:
        r = c.get("scissor")
        if not r: 
            continue
        clip_rects.append({
            "index": c["index"],
            "xyxy": [r.x0, r.y0, r.x1, r.y1],
            "bbox_xywh_pt": c.get("bbox_xywh_pt"),  # 사각형이면 존재
        })

    out = {}
    for im in images:
        ibox = im["bbox_pt"]  # xyxy
        best = None; best_iou = 0.0
        for c in clip_rects:
            # Find the clip with the highest IoU with the image's bounding box.
            # The original ordering heuristic was flawed as it compared indices from two different lists.
            i = iou_xyxy(ibox, c["xyxy"])
            if i > best_iou:
                best_iou, best = i, c
        if best and best_iou >= 0.10:
            out[im["xref"]] = {"clip_bbox_pt": best.get("bbox_xywh_pt")}
        else:
            out[im["xref"]] = {"clip_bbox_pt": None}
    return out

### (D) 최종 추출기: extract_images_and_bboxes()

def extract_images_and_bboxes(
    pdf_path: str,
    out_dir: str,
    dpi: int = 144,
    *,
    media_root: str
) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0  # pt→px
    results = []

    for pno in range(len(doc)):
        page = doc[pno]
        pw_pt, ph_pt = page.rect.width, page.rect.height
        page_info = {
            "page_no": pno + 1,
            "size": {"w": pw_pt*scale, "h": ph_pt*scale},  # 페이지 크기 px로 반환(기존과 동일).
            "pt_to_px": scale,
            "images": []
        }

        # 1) 수집
        clips   = collect_clips(page)
        logger.debug(f"[EXTRACT-IMG-Clips] : {clips}")
        images  = collect_images(page, doc, out_dir, media_root)  # transform_pt, bbox_pt, path, img_w/h
        logger.debug(f"[EXTRACT-IMG-Images] : {images}")
        matches = match_clips_to_images(clips, images)
        logger.debug(f"[EXTRACT-IMG-Matches] : {matches}")

        # 2) 페이지 이미지 기록으로 변환 (px 좌표로 저장)
        for im in images:
            x0,y0,x1,y1 = im["bbox_pt"]
            logger.debug(f"[EXTRACT-IMG-Bbox-pt] : {x0,y0,x1,y1}")
            bbox_px = [x0*scale, y0*scale, (x1-x0)*scale, (y1-y0)*scale]  # xywh(px)
            logger.debug(f"[EXTRACT-IMG-Bbox-px] : {bbox_px}")
            m = matches.get(im["xref"], {})
            cb_pt = m.get("clip_bbox_pt")  # [x,y,w,h] (pt) or None
            logger.debug(f"[EXTRACT-IMG-Clip] : {cb_pt}")
            clip_bbox_px = [cb_pt[0]*scale, cb_pt[1]*scale, cb_pt[2]*scale, cb_pt[3]*scale] if cb_pt else None
            logger.debug(f"[EXTRACT-IMG-Clip-px] : {clip_bbox_px}")
            tf_pt = im.get("transform_pt")
            logger.debug(f"[EXTRACT-IMG-Transform] : {tf_pt}")
            transform_px = [tf_pt[0]*scale, tf_pt[1]*scale, tf_pt[2]*scale, tf_pt[3]*scale, tf_pt[4]*scale, tf_pt[5]*scale]  # pt -> px
            logger.debug(f"[EXTRACT-IMG-Transform-px] : {transform_px}")

            # 픽셀화 할 수 있는 것은 모두 픽셀처리 해둠. 다음 normalize에서는 스케일만 적용.
            page_info["images"].append({
                "ref": f"img_p{pno+1}_{im['xref']}",
                "xref": im["xref"],
                "path": im["path"],                         # MEDIA_ROOT 상대경로
                "bbox": bbox_px,                            # [x,y,w,h] px
                "transform": transform_px,                  # [a,b,c,d,e,f] px
                "img_w": im["img_w"],                       # collect_images에서 이미 px
                "img_h": im["img_h"],                       # collect_images에서 이미 px
                "clip_bbox": clip_bbox_px,                  # [x,y,w,h] px or None
                "origin_w": im["origin_w"],                 # collect_images에서 이미 px
                "origin_h": im["origin_h"],                 # collect_images에서 이미 px
            })

        results.append(page_info)
    return results

## -------------------------------------

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