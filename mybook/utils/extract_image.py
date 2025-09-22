#mybook/utils/extract_image.py
import pymupdf as fitz
from pathlib import Path
from typing import List, Dict, Any
import os, hashlib, re
from bs4 import BeautifulSoup
import base64
from io import BytesIO
from PIL import Image, ImageOps
from dataclasses import dataclass
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def _svg_path_from_items(items):
    """Converts PyMuPDF drawing 'items' into an SVG path data string, ending with 'Z'."""
    parts = []
    current_pos = None

    for it in items or []:
        op = it[0]
        if op == "m":  # move to
            if len(it) < 2: continue
            p = it[1]
            parts.append(f"M {p[0]} {p[1]}")
            current_pos = p
        elif op == "l":  # line to
            if len(it) < 3: continue
            p1, p2 = it[1], it[2]
            # If the path is not continuous, start a new subpath with moveto.
            if current_pos is None or (current_pos[0] != p1[0] or current_pos[1] != p1[1]):
                parts.append(f"M {p1[0]} {p1[1]}")
            parts.append(f"L {p2[0]} {p2[1]}")
            current_pos = p2
        elif op == "re":  # rectangle
            if len(it) < 2: continue
            x0, y0, x1, y1 = it[1]
            # Start a new subpath for the rectangle
            parts.append(f"M {x0} {y0} L {x1} {y0} L {x1} {y1} L {x0} {y1} Z")
            current_pos = None  # A closed path resets the current position
        elif op == "c":  # bezier curve
            if len(it) < 5: continue
            p_start, p_c1, p_c2, p_end = it[1], it[2], it[3], it[4]
            # If path is not continuous, start a new subpath with moveto.
            if current_pos is None or (current_pos[0] != p_start[0] or current_pos[1] != p_start[1]):
                parts.append(f"M {p_start[0]} {p_start[1]}")
            parts.append(f"C {p_c1[0]} {p_c1[1]} {p_c2[0]} {p_c2[1]} {p_end[0]} {p_end[1]}")
            current_pos = p_end
    path_data = " ".join(parts)

    # Per user request, ensure the path string for this clip object ends with a 'Z'.
    # This will be used as a delimiter for splitting into multiple <path> tags later.
    if path_data and not path_data.strip().endswith('Z'):
        path_data += " Z"

    return path_data

def _scale_svg_path(path_data: str, scale: float) -> str:
    """
    SVG 경로 데이터 문자열의 모든 숫자 값을 스케일링합니다.
    과학적 표기법을 올바르게 처리하고 매우 작은 값은 0으로 처리합니다.
    """
    if not path_data:
        return ""

    # PyMuPDF가 생성할 수 있는 '1.23e-12.0'과 같은 잘못된 과학 표기법을 '1.23e-12'로 수정합니다.
    path_data = re.sub(r'([eE][-+]?\d+)\.0+\b', r'\1', path_data)

    def scale_match(m):
        val = float(m.group(0))
        scaled_val = val * scale
        # 0.1 미만의 매우 작은 값은 0으로 처리하여 불필요한 정밀도를 제거합니다.
        if abs(scaled_val) < 0.1:
            return "0"
        # 소수점 둘째 자리까지 반올림하여 문자열로 반환합니다.
        return f"{scaled_val:.2f}"

    # 정수, 부동소수점, 과학 표기법을 포함한 모든 숫자를 찾는 정규식입니다.
    number_regex = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    return re.sub(number_regex, scale_match, path_data)

# --------------------------------------------------------------
### (A) 클립 수집: collect_clips()

def collect_clips(page: fitz.Page):
    """
    extended=True로 드로잉을 받아 'clip' 노드만 뽑아냅니다.
    'scissor'는 클리핑 경로의 바운딩 박스이며, 비사각형 클립을 위해 원본 경로 데이터도 추출합니다.
    return: list[{"index":int, "level":int, "scissor":Rect, "bbox_xywh_pt":[x,y,w,h] or None, "path_data_pt": str or None}]
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

            # 비정형 클립을 위해 실제 경로 데이터 추출
            path_data_pt = None
            items = d.get("items")
            if items:
                path_data_pt = _svg_path_from_items(items)

            out.append({
                "index": idx,
                "level": d.get("level", 0),
                "scissor": scissor_rect,
                "bbox_xywh_pt": bbox_xywh,
                "path_data_pt": path_data_pt,
            })
    return out

### (B) 매칭: iou_xyxy() + match_clips_to_images()

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
    Associates images with their corresponding clipping paths.
    An image can be associated with multiple clips. This function finds all clips
    that have a significant overlap (IoU >= 0.1) with the image's bounding box.
    It then aggregates these clips into a single definition by:
    1. Concatenating all SVG path data into one string.
    2. Calculating the union of all clip bounding boxes to form a single container bbox.

    clips: list of clip dicts from collect_clips()
    images: list of image dicts from an extractor (e.g., AdvancedImageExtractor)
    return: dict mapping image xref to a dict with aggregated clip info:
            {"clip_bbox_pt":[x,y,w,h] or None, "clip_path_data_pt": str or None}
    """
    # clip 외접 박스(xyxy)
    clip_rects = []
    for c in clips:
        r = c.get("scissor")
        if not r or not r.is_valid or r.is_empty:
            continue
        clip_rects.append({
            "index": c["index"],
            "xyxy": [r.x0, r.y0, r.x1, r.y1],
            "bbox_xywh_pt": c.get("bbox_xywh_pt"),  # 사각형이면 존재
            "path_data_pt": c.get("path_data_pt"),
        })

    out = {}
    for im in images:
        ibox = im["bbox_pt"]  # xyxy
        matching_clips = []
        for c in clip_rects:
            i = iou_xyxy(ibox, c["xyxy"])
            if i >= 0.10:  # Collect all clips with IoU above threshold
                matching_clips.append(c)

        if matching_clips:
            # Aggregate all matching clips into a single clip definition.
            all_paths = [mc.get("path_data_pt") for mc in matching_clips if mc.get("path_data_pt")]
            final_path = " ".join(all_paths) if all_paths else None

            all_bboxes = [mc.get("bbox_xywh_pt") for mc in matching_clips if mc.get("bbox_xywh_pt")]
            final_bbox = None
            if all_bboxes:
                min_x = min(b[0] for b in all_bboxes)
                min_y = min(b[1] for b in all_bboxes)
                max_x2 = max(b[0] + b[2] for b in all_bboxes)
                max_y2 = max(b[1] + b[3] for b in all_bboxes)
                final_bbox = [min_x, min_y, max_x2 - min_x, max_y2 - min_y]

            out[im["xref"]] = {
                "clip_bbox_pt": final_bbox,
                "clip_path_data_pt": final_path
            }
        else:
            out[im["xref"]] = {"clip_bbox_pt": None, "clip_path_data_pt": None}
    return out

### (C) 새로운 이미지 추출 클래스 (New Image Extraction Classes)
class AdvancedImageExtractor:
    """
    get_image_info()와 get_text("html")의 순서가 일치한다는 규칙을 기반으로,
    표준 이미지는 직접 추출(※ SMask 결합 포함)하고,
    인라인 이미지는 HTML에서 데이터를 가져와 매핑하는 최종 추출기.
    """
    def __init__(self, page: fitz.Page, doc: fitz.Document, pno: int, out_dir: str, media_root: str):
        self.page = page
        self.doc = doc
        self.pno = pno
        self.out_dir = Path(out_dir)
        self.media_root = media_root
        self.saved_files = {}  # {unique_key: {"path": ..., "img_w": ..., "img_h": ...}}

    def _save_image(self, key: Any, image_bytes: bytes, ext: str) -> Dict[str, Any]:
        """바이트를 파일로 저장하고 {path,img_w,img_h} 메타를 캐시."""
        if key in self.saved_files:
            return self.saved_files[key]

        filename_key = f"p{self.pno}_{key}"
        p = self.out_dir / f"img_{filename_key}.{ext}"
        p.write_bytes(image_bytes)
        
        rel_path = os.path.relpath(str(p), self.media_root).replace("\\", "/")
        img_size = Image.open(BytesIO(image_bytes)).size
        
        self.saved_files[key] = {
            "path": rel_path,
            "img_w": img_size[0],
            "img_h": img_size[1],
        }
        return self.saved_files[key]
    
    def _pixmap_to_png_bytes(self, pix: fitz.Pixmap) -> bytes:
        """Pixmap → PNG 바이트 (버전별 호환 처리)"""
        try:
            return pix.tobytes("png")
        except Exception as e:
            raise RuntimeError(f"Pixmap → PNG 직렬화 실패: {e}")

    def _build_smask_map(self) -> Dict[int, int]:
        """
        page.get_images(full=True)에서 (image xref → smask xref) 매핑 생성.
        smask 없으면 0.
        """
        smask_map: Dict[int, int] = {}
        for tpl in self.page.get_images(full=True):
            # (xref, smask, width, height, bpc, colorspace, alt_cs, name, filter, referencer)
            xref, smask_xref = tpl[0], tpl[1]
            smask_map[xref] = smask_xref or 0
        return smask_map

    def _extract_standard_image(self, info: Dict[str, Any], smask_map: Dict[int, int]) -> Dict[str, Any] | None:
        """
        표준 이미지(xref>0) 하나를 추출.
        - has-mask=True & smask_xref>0 → Pixmap 결합 후 PNG로 저장
        - 그 외 → 기존 extract_image(xref) 그대로 저장
        """
        xref = int(info["xref"])
        number = int(info["number"])
        has_mask = bool(info.get("has-mask"))
        smask_xref = int(smask_map.get(xref, 0))
        transform = info.get("transform", [1, 0, 0, 1, 0, 0])
        bbox = info.get("bbox", [0, 0, 0, 0])

        try:
            if has_mask and smask_xref > 0:
                # ---- SMask 결합 경로: PNG(alpha)로 저장 ----
                base_pix = fitz.Pixmap(self.doc, xref)
                mask_pix = fitz.Pixmap(self.doc, smask_xref)
                combined = fitz.Pixmap(base_pix, mask_pix)  # 알파 합성
                image_bytes = self._pixmap_to_png_bytes(combined)
                saved_meta = self._save_image(f"{xref}_smask", image_bytes, "png")
                logger.debug(f"[SMASK] xref={xref}, smask={smask_xref} PNG(alpha) 저장")

                return {
                    "xref": xref,
                    "number": number,
                    "transform_pt": transform,
                    "bbox_pt": bbox,
                    **saved_meta,
                    "origin_w": int(info.get("width", 0)),
                    "origin_h": int(info.get("height", 0)),
                }

            # ---- 기본 경로: 원본 바이트 추출 ----
            raw = self.doc.extract_image(xref)
            if not raw or not raw.get("image"):
                return None
            ext = raw.get("ext", "png")
            saved_meta = self._save_image(xref, raw["image"], ext)

            return {
                "xref": xref,
                "number": number,
                "transform_pt": transform,
                "bbox_pt": bbox,
                **saved_meta,
                "origin_w": int(info.get("width", 0)),
                "origin_h": int(info.get("height", 0)),
            }

        except Exception as e:
            logger.warning(f"표준 이미지 xref {xref} 처리 중 오류: {e}")
            return None

    # ------------ 메인 ---------------

    def extract(self) -> List[Dict[str, Any]]:
        logger.debug(f"---------------[AdvancedImageExtractor p.{self.pno}]호출됨--------------")
        
        # 1. 메타데이터의 기준점: get_image_info()
        info_list = self.page.get_image_info(xrefs=True) #type:ignore
        if not info_list:
            return []

        extracted_images: List[Dict[str, Any]] = []
        num_images_to_process = len(info_list)

        # smask 맵을 미리 계산 (표준 이미지에만 사용)
        smask_map = self._build_smask_map()
        # 인라인 이미지 매핑을 위해 HTML은 '필요할 때' 최초 1회만 파싱하기위해 for 외부에 변수설정해둠
        html_soup = None
        html_img_tags: List[Any] = []

        # 4. 순서(index)를 기준으로 매핑하여 이미지 추출
        for i in range(num_images_to_process):
            info = info_list[i]
            xref = int(info["xref"])
            number = int(info["number"])

            if xref > 0:
                # ---- 표준 이미지: SMask 우선 결합, 아니면 기존 추출 ----
                meta = self._extract_standard_image(info, smask_map)
                if meta:
                    extracted_images.append(meta)
                continue

            # ---- 인라인 이미지 (xref == 0): 기존 매핑 유지 ----
            if html_soup is None:
                html_content = self.page.get_text("html")
                html_soup = BeautifulSoup(html_content, "html.parser")
                html_img_tags = html_soup.find_all("img")

                if len(info_list) != len(html_img_tags):
                    logger.warning(
                        f"Page {self.pno}: Mismatch get_image_info({len(info_list)}) vs HTML img tags({len(html_img_tags)})"
                    )

            if i >= len(html_img_tags):
                logger.warning(f"인라인 이미지 매핑 실패: index {i} / img_tags {len(html_img_tags)}")
                continue

            img_tag = html_img_tags[i]
            src = img_tag.get("src", "")

            if src and src.startswith("data:image"):
                try:
                    b64data = src.split(",", 1)[1]
                    image_bytes = base64.b64decode(b64data)

                    file_key = f"inline_{number}"
                    saved_meta = self._save_image(file_key, image_bytes, "png")
                    unique_xref = 10000 * self.pno + number  # 내부 고유키

                    extracted_images.append({
                        "xref": unique_xref,
                        "number": number,
                        "transform_pt": info.get("transform", [1, 0, 0, 1, 0, 0]),
                        "bbox_pt": info.get("bbox", [0, 0, 0, 0]),
                        **saved_meta,
                        "origin_w": int(info.get("width", 0)),
                        "origin_h": int(info.get("height", 0)),
                    })
                except Exception as e:
                    logger.warning(f"인라인 이미지 base64 처리 실패(index={i}): {e}")
            else:
                logger.warning("img 태그는 있으나 src가 없거나 data URI 형식이 아닙니다.")

        return extracted_images

class TextDictImageExtractorV2:
    """
    page.get_text("dict")의 image block을 보강 수집.
    - image + mask → RGBA로 합성하여 PNG로 저장
    - collect_images()가 만드는 스키마와 동일하게 반환
      (transform_pt, bbox_pt(xyxy, pt), path, img_w/h, origin_w/h)
    """
    def __init__(self, page: fitz.Page, pno: int, out_dir: str, media_root: str):
        self.page = page
        self.pno = pno  # 1-based
        self.out_dir = Path(out_dir)
        self.media_root = media_root
        self.saved_hashes: set[str] = set()
        self.order_base = 1000

    # --- utils ---
    @staticmethod
    def _ensure_bytes(blob):
        if blob is None:
            return None
        if isinstance(blob, bytes):
            return blob
        if isinstance(blob, str):
            return blob.encode("latin-1", errors="ignore")
        raise TypeError(f"Unsupported blob type: {type(blob)}")

    @staticmethod
    def _stable_hash(b: bytes) -> str:
        return hashlib.sha1(b).hexdigest()

    @staticmethod
    def _open_img(b: bytes) -> Image.Image:
        return Image.open(BytesIO(b))

    @staticmethod
    def _normalize_mask(mask_img: Image.Image) -> Image.Image:
        if mask_img.mode not in ("1", "L"):
            mask_img = mask_img.convert("L")
        hist = mask_img.histogram()
        if hist:
            black_ratio = hist[0] / max(1, sum(hist))
            if black_ratio > 0.8:
                mask_img = ImageOps.invert(mask_img)
        return mask_img

    def _compose_rgba(self, image_bytes: bytes, mask_bytes: bytes | None) -> Image.Image:
        base = self._open_img(image_bytes).convert("RGBA")
        if mask_bytes:
            m = self._open_img(mask_bytes)
            m = self._normalize_mask(m)
            if m.size != base.size:
                m = m.resize(base.size, Image.Resampling.NEAREST)
            base.putalpha(m)
        return base

    def _save_png(self, img: Image.Image, name: str) -> str:
        p = self.out_dir / f"{name}.png"
        os.makedirs(p.parent, exist_ok=True)
        img.save(str(p), "PNG")
        return os.path.relpath(str(p), self.media_root).replace("\\", "/")

    # --- main ---
    def extract(self) -> List[Dict[str, Any]]:
        logger.debug("---------------[TextDictImageExtractorV2]호출됨--------------")
        out_images: List[Dict[str, Any]] = []
        text_dict = self.page.get_text("dict", flags = fitz.TEXTFLAGS_DICT | fitz.TEXT_PRESERVE_IMAGES)  #type:ignore
        logger.debug(f"[textdict] : {len(text_dict)}")
        logger.debug(f"[textdict.blocks 갯수] : {len(text_dict.get("blocks"))}")
        
        text_dict_images = [ td_image for td_image in text_dict.get("blocks", []) if td_image.get("type") == 1 ]
        logger.debug(f"[textdict 이미지 갯수] : {len(text_dict_images)}")
        current_order = self.order_base
        for block in text_dict_images:
            if block.get("type") != 1:
                continue

            img_b = self._ensure_bytes(block.get("image"))
            if not img_b:
                logger.debug(f"[textdict 이미지 없음]")
                continue
            mask_b = self._ensure_bytes(block.get("mask"))
            if not mask_b:
                logger.debug(f"[textdict 마스크 없음]")

            # 중복 방지
            sig = self._stable_hash(img_b + (mask_b or b""))
            if sig in self.saved_hashes:
                continue
            self.saved_hashes.add(sig)

            # 합성
            composed = self._compose_rgba(img_b, mask_b)
            if not composed:
                logger.debug(f"[textdict 이미지 합성 실패]")

            # 파일명: textdict_p{pno}_{blockNo}
            base_name = f"textdict_p{self.pno}_{block['number']}"
            rel_path = self._save_png(composed, base_name)
            logger.debug(f"[textdict이미지 저장 상대경로] : {rel_path}")

            # 메인 스키마에 맞춰 반환
            # - bbox_pt: xyxy (points)
            # - transform_pt: [a,b,c,d,e,f] (points)
            x0, y0, x1, y1 = block["bbox"]
            tf = block.get("transform", [1,0,0,1,0,0])
            logger.debug(f"[textdict 이미지 tf] : {tf}")

            # img_w/h: 저장된 이미지 픽셀 크기
            img_w, img_h = composed.width, composed.height
            origin_w = int(block.get("width") or img_w)
            origin_h = int(block.get("height") or img_h)

            out_images.append({
                "order": current_order,
                "xref": - (100000 * self.pno + block["number"]),  # 음수 xref로 충돌 회피
                "number": block['number'],
                "transform_pt": list(tf),
                "bbox_pt": [x0, y0, x1, y1],
                "path": rel_path,           # MEDIA_ROOT 상대경로
                "img_w": img_w,            # px
                "img_h": img_h,            # px
                "origin_w": origin_w,      # px
                "origin_h": origin_h,      # px
            })
            current_order += 1

        return out_images


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
        page_no = pno + 1
        pw_pt, ph_pt = page.rect.width, page.rect.height
        page_info = {
            "page_no": pno + 1,
            "size": {"w": pw_pt*scale, "h": ph_pt*scale},  # 페이지 크기 px로 반환(기존과 동일).
            "pt_to_px": scale,
            "images": []
        }

        # --- MODIFIED: 'number' 기반의 새로운 병합 로직 ---
        merged_images_by_key = {}

        # 1. AdvancedExtractor 결과를 먼저 추가
        adv_extractor = AdvancedImageExtractor(page, doc, page_no, out_dir, media_root)
        for img in adv_extractor.extract():
            key = (page_no, img['number'])
            merged_images_by_key[key] = img

        # 2. TextDictExtractor 결과로 덮어쓰기 (우선적용)
        textdict_extractor = TextDictImageExtractorV2(page, page_no, out_dir, media_root)
        for img in textdict_extractor.extract():
            key = (page_no, img['number'])
            merged_images_by_key[key] = img
        
        all_images = list(merged_images_by_key.values())
        # --- End of MODIFIED section ---

        # --- 수집된 이미지 후처리 ---
        clips   = collect_clips(page)
        matches = match_clips_to_images(clips, all_images)
        logger.debug(f"Page {page_no} [Total images found]: {len(all_images)}")

        # 2) 페이지 이미지 기록으로 변환 (px 좌표로 저장)
        for im in all_images:
            x0,y0,x1,y1 = im["bbox_pt"]
            logger.debug(f"[EXTRACT-IMG-Bbox-pt] : {x0,y0,x1,y1}")
            bbox_px = [x0*scale, y0*scale, (x1-x0)*scale, (y1-y0)*scale]  # xywh(px)
            logger.debug(f"[EXTRACT-IMG-Bbox-px] : {bbox_px}")

            m = matches.get(im["xref"], {})
            cb_pt = m.get("clip_bbox_pt")  # [x,y,w,h] (pt) or None
            logger.debug(f"[EXTRACT-IMG-Clip] : {cb_pt}")
            clip_bbox_px = [cb_pt[0]*scale, cb_pt[1]*scale, cb_pt[2]*scale, cb_pt[3]*scale] if cb_pt else None
            logger.debug(f"[EXTRACT-IMG-Clip-px] : {clip_bbox_px}")

            clip_path_data_pt = m.get("clip_path_data_pt")
            logger.debug(f"[EXTRACT-IMG-Clip-Path] : {clip_path_data_pt}")
            clip_path_data_px = _scale_svg_path(clip_path_data_pt, scale) if clip_path_data_pt else None
            logger.debug(f"[EXTRACT-IMG-Clip-Path-px] : {clip_path_data_px}")

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
                "clip_path_data_px": clip_path_data_px,     # pt 단위의 SVG 경로 데이터를 px로 변환
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
    ) -> Image.Image | None:
        """
        bbox: {"x":..,"y":..,"w":..,"h":..} 또는 [x,y,w,h]
        Returns the cropped PIL Image on success, otherwise None.
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
            return crop
        except Exception as e:
            print(f"[ImageExtractor] Error extracting p{page_no} bbox={bbox}: {e}")
            return None

    def extract_many(
        self,
        page_no: int,
        figures: List[Dict],
        output_dir: str,
        prefix: str = "fig",
        ext: str = "png"
    ) -> Dict[str, Dict[str, Any]]:
        """
        figures: [{"ref":"img_pN_1", "bbox":[x,y,w,h], ... }, ...].
        The 'bbox' is expected to be in page pixel coordinates, as prepared by normalize_bboxes_1000.
        return: { ref: {"path": absolute_path, "width": int, "height": int} }
        """
        os.makedirs(output_dir, exist_ok=True)
        out: Dict[str, Dict[str, Any]] = {}
        for i, f in enumerate(figures, 1):
            ref = f.get("ref") or f.get("label") or f"img_p{page_no}_{i}"
            bbox = f.get("bbox")  # This should be present after normalization
            # The 'bbox' must be present and valid after the normalization step.
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(v is None for v in bbox):
                logger.warning(f"Skipping figure {ref} in page {page_no} due to missing or invalid 'bbox'. Figure data: {f}")
                continue
            out_path = os.path.join(output_dir, f"{prefix}_p{page_no}_{i}.{ext}")
            cropped_image = self.extract_figure(page_no, bbox, out_path, fmt=ext.upper())  # type:ignore
            if cropped_image:
                out[ref] = {
                    "path": out_path,
                    "width": cropped_image.width,
                    "height": cropped_image.height,
                }
        return out
    
    @staticmethod
    def normalize_bboxes_1000(figures: List[Dict], page_w: int, page_h: int, expand_px: int = 3) -> List[Dict]:
        """
        Gemini는 무조건 주어진 이미지의 bbox를 가로 & 세로 0~1000의 좌표계에서 추론하여 반환하므로,
        반환값을 실제 이미지에 적용할 수 있도록 페이지 좌표계(px)로 변환합니다.
        - 입력: Gemini가 준 bbox_x, bbox_y, bbox_w, bbox_h (0-1000 grid)
        - 출력: 페이지 px(top-left) 기준의 xywh 정수 bbox로 변환 + 여유 expand + 경계 클램프
        """
        if not figures:
            return []

        sx = page_w / 1000.0
        sy = page_h / 1000.0

        out = []
        for fg in figures:
            # Gemini가 반환하는 1000단위 좌표계의 bbox 값들을 가져옵니다.
            x_1000 = fg.get("bbox_x",0)
            y_1000 = fg.get("bbox_y",0)
            w_1000 = fg.get("bbox_w",0)
            h_1000 = fg.get("bbox_h",0)

            if any(v is None for v in [x_1000, y_1000, w_1000, h_1000]):
                # 필요한 키가 없으면 원본을 그대로 추가하고 건너뜁니다.
                out.append(fg)
                continue

            # 1000 그리드 좌표를 실제 페이지 픽셀 좌표로 변환합니다.
            x_px = x_1000 * sx
            y_px = y_1000 * sy
            w_px = w_1000 * sx
            h_px = h_1000 * sy

            # 이미지 경계에 약간의 여유(padding)를 줍니다.
            x = x_px - expand_px
            y = y_px - expand_px
            w = w_px + 2 * expand_px
            h = h_px + 2 * expand_px

            # 페이지 경계를 벗어나지 않도록 bbox를 조정(clamp)합니다.
            clamped_bbox = ImageExtractor._clamp_bbox(int(x), int(y), int(w), int(h), page_w, page_h)

            # 변환된 bbox를 'bbox' 키에 저장하여 figure 딕셔너리를 업데이트합니다.
            out.append({**fg, "bbox": list(clamped_bbox)})
        return out

    @staticmethod
    def normalize_bboxes_from_gemini(bounding_boxes: List[Dict], page_w: int, page_h: int, expand_px: int = 3) -> List[Dict]:
        """
        Converts bounding boxes from Gemini's normalized 0-1000 format to page pixel coordinates.
        - Input: `bounding_boxes` from Gemini, e.g., [{'label': '...', 'box_2d': [ymin, xmin, ymax, xmax]}]
        - Output: A list of figure dicts, e.g., [{'label': '...', 'bbox': [x, y, w, h]}] in page pixels.
        """
        out = []
        for box_data in bounding_boxes:
            box_2d = box_data.get("box_2d")
            if not box_2d or len(box_2d) != 4:
                continue

            ymin, xmin, ymax, xmax = box_2d
            
            # Convert 0-1000 grid to absolute pixel coordinates
            abs_y1 = int(ymin / 1000 * page_h)
            abs_x1 = int(xmin / 1000 * page_w)
            abs_y2 = int(ymax / 1000 * page_h)
            abs_x2 = int(xmax / 1000 * page_w)

            # Add padding and convert to xywh format
            x = abs_x1 - expand_px
            y = abs_y1 - expand_px
            w = (abs_x2 - abs_x1) + 2 * expand_px
            h = (abs_y2 - abs_y1) + 2 * expand_px

            clamped_bbox = ImageExtractor._clamp_bbox(int(x), int(y), int(w), int(h), page_w, page_h)
            
            new_data = {key: val for key, val in box_data.items() if key != 'box_2d'}
            new_data['bbox'] = list(clamped_bbox)
            out.append(new_data)
        return out