# mybook/utils/born_digital.py
import logging
import fitz 
from .html_inject import escape_html
from .pick_best_path import pick_final_clip
from typing import Dict, Any, List, Tuple
from .paragraphs import Paragraph # Paragraph 타입 힌트를 위해 임포트
from .nonbody_detector import NonBodyLabel # NonBodyLabel 타입 힌트를 위해 임포트
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def _is_light_color(rgb_float_tuple: tuple[float, float, float] | None, threshold: float = 0.9) -> bool:
    """
    주어진 색상이 흰색에 가까운 밝은 색상인지 확인합니다.
    배경색을 텍스트 색상으로 잘못 추정하는 것을 방지하기 위해 사용됩니다.
    휘도(Luminance)를 계산하여 임계값(threshold)보다 높으면 밝은 색으로 간주합니다.
    이 헬퍼는 get_text()로 추출한 color가 모두 0일때만 호출합니다. 
    """
    if not rgb_float_tuple or len(rgb_float_tuple) != 3:
        return True  # 색상 정보가 없으면 배경으로 간주하여 무시
    r, g, b = rgb_float_tuple
    # 휘도(Luminance) 공식. 1.0에 가까울수록 흰색.
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance > threshold

def _merge_span_colors(base_spans, color_spans):
    """(block,line,span) 키로 rawdict 색상을 base(dict) 스팬에 주입"""
    # color_spans: rawdict에서 뽑은 spans (block,line,span,color)
    color_map = {}
    for sp in color_spans:
        key = (sp.get("block"), sp.get("line"), sp.get("span"))
        col = sp.get("color")
        if col is not None:
            color_map[key] = col
    for s in base_spans:
        key = (s.get("block"), s.get("line"), s.get("span"))
        if "color" not in s or not s.get("color"):
            if key in color_map:
                s["color"] = color_map[key]
    return base_spans

def _collect_spans_from(mode_dict, page_no):
    spans = []
    for bi, blk in enumerate(mode_dict.get("blocks", [])):
        if blk.get("type") == 0:
            for li, line in enumerate(blk.get("lines", [])):
                for si, sp in enumerate(line.get("spans", [])):
                    spans.append({
                        "text": sp.get("text", ""),
                        "bbox": sp.get("bbox", []),
                        "font": sp.get("font", ""),
                        "size": sp.get("size", 0.0),
                        "flags": sp.get("flags", 0),
                        "color": sp.get("color", 0),  # dict일 땐 0일 수도
                        "dir": sp.get("dir", None),
                        "block": bi, "line": li, "span": si,
                        "origin": sp.get("origin", []),
                    })
    return spans

def _rgb_tuple_to_hex(rgb):
    if not rgb:            # None, (), []
        return None        # ★ 절대 기본색으로 치환하지 말 것
    r, g, b = rgb
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def collect_page_layout(pdf_path: str, page_no: int) -> Dict[str, Any]:
    """
    page_no: 1-based
    반환:
    {
      "page_no": int,
      "size": {"w": float, "h": float},
      "spans": [ { "text": str, "bbox": [x0,y0,x1,y1], "font": str, "size": float, "flags": int, "block":int, "line":int, "span":int }, ... ],
      "images": [ { "bbox":[...], "xref":int, "path":str (있다면) }, ...],
      "links": [ ... ]  # 선택
    }
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_no - 1)
    w, h = page.rect.width, page.rect.height

    # 1) base: dict (텍스트 안정)
    base = page.get_text("dict") # type: ignore
    spans = _collect_spans_from(base, page_no)

    # 2) enrich: rawdict 색상 병합
    raw = page.get_text("rawdict") # type: ignore
    color_spans = _collect_spans_from(raw, page_no) if raw else []
    if color_spans:
        spans = _merge_span_colors(spans, color_spans)

    # 3) 색상 전부 0이면 드로잉 레이어에서 보정 시도(헤딩/컬러 텍스트 대응)
    if spans and not any(int(s.get("color", 0) or 0) for s in spans):
        try:
            drawings = page.get_drawings()
            # 간단 추정: 각 스팬 bbox와 가장 면적이 겹치는 fill 색을 채택
            for s in spans:
                sx0, sy0, sx1, sy1 = s["bbox"]
                s_area = max((sx1 - sx0) * (sy1 - sy0), 1e-6)
                best = None
                best_overlap = 0.0
                for d in drawings:
                    fill = d.get("fill")  # (r,g,b) 0..1
                    # 배경색으로 추정되는 매우 밝은 색상은 텍스트 색상으로 간주하지 않습니다.
                    # 이 로직이 없으면 텍스트 주변의 밝은 배경 사각형 색상을
                    # 텍스트 색상으로 잘못 지정하는 문제가 발생합니다.
                    if not fill or _is_light_color(fill):
                        continue
                    # path bbox
                    bx0, by0, bx1, by1 = d.get("rect") or (None,)*4
                    if bx0 is None:
                        continue
                    ox0, oy0 = max(sx0, bx0), max(sy0, by0) # type: ignore
                    ox1, oy1 = min(sx1, bx1), min(sy1, by1) # type: ignore
                    if ox1 <= ox0 or oy1 <= oy0:
                        continue
                    overlap = (ox1 - ox0) * (oy1 - oy0) / s_area
                    if overlap > best_overlap:
                        r, g, b = [int(255*x) for x in fill]
                        best = (r << 16) | (g << 8) | b
                        best_overlap = overlap
                if best is not None:
                    s["color"] = best
        except Exception:
            pass

    # 4) 이미지 xref 보강
    images: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for (xref, *_rest) in page.get_images(full=True):
        if xref in seen:
            continue
        seen.add(xref)
        rects = page.get_image_rects(xref)  # 배치된 위치들 (여러 번 참조될 수 있음) # type:ignore
        if not rects:
            continue
        for idx, r in enumerate(rects, start=1):
            # bbox는 PDF 사용자 좌표계(pt)의 [x0,y0,x1,y1]로 저장 권장
            images.append({
                "ref": f"img_p{page_no}_{xref}_{idx}" if len(rects) > 1 else f"img_p{page_no}_{xref}",
                "xref": int(xref),                       # ★ 정수 xref
                "bbox": [r.x0, r.y0, r.x1, r.y1],        # pt 좌표
                # 여기서는 path를 넣지 않음 (저장은 업로드 단계 extract_images_and_bboxes에서)
            })

    # ✅ NEW: vector drawings
    drawings = []
    for d in page.get_drawings():
        drawings.append({
            "rect": d.get("rect"), # [x0,y0,x1,y1]
            "items": d.get("items"), # path commands
            "stroke": _rgb_tuple_to_hex(d.get("color")),
            "fill": _rgb_tuple_to_hex(d.get("fill")),
            "stroke_raw": d.get("color"),   # ★ 추가
            "fill_raw": d.get("fill"),      # ★ 추가
            "width": d.get("width") or 0.0,
            "opacity": d.get("opacity") or 1.0,
            "fill_opacity": d.get("fill_opacity"),
            "stroke_opacity": d.get("stroke_opacity"),
            "even_odd": bool(d.get("even_odd")),
            "lineCap": d.get("lineCap"),
            "lineJoin": d.get("lineJoin"),
            "dashes": d.get("dashes"),
        })
    
    # ★ 링크 사각형 수집 (pt 좌표)
    link_rects = []
    try:
        for L in page.get_links() or []: #type:ignore
            r = L.get("from")
            if r:
                link_rects.append([float(r.x0), float(r.y0), float(r.x1), float(r.y1)])
    except Exception:
        pass

    # logger.debug(f"[DEBUG-UTILS-COLLECT]page {page_no} spans: {spans}, images: {len(images)}, drawings: {drawings}")
    # # drawings 디버깅
    # for di, d in enumerate(drawings):
    #     logger.debug(f"[DEBUG-UTILS-COLLECT] drawing {di}: " + ", ".join(f"{k}={v}" for k,v in d.items() if v))

    return {"page_no": page_no, "size_pt": {"w": w, "h": h}, "spans": spans, "images": images, "drawings": drawings, "links": link_rects,}


# 도우미: PyMuPDF color(int) → #RRGGBB
def _int_color_to_hex(c: int) -> str:
    # PyMuPDF는 0xRRGGBB 정수로 오는 케이스가 일반적
    r = (c >> 16) & 255
    g = (c >> 8) & 255
    b = (c >> 0) & 255
    return f"#{r:02x}{g:02x}{b:02x}"

def _guess_bold_italic(font_name: str, flags: int) -> tuple[bool, bool]:
    # flags는 PDF마다 다르게 쓰이는 케이스가 있어 폰트명도 함께 체크
    name = (font_name or "").lower()
    bold = ("bold" in name) or bool(flags & 2)   # 경험적: 2가 bold인 PDF가 있음
    italic = ("italic" in name) or ("oblique" in name) or bool(flags & 1)
    return bold, italic

def _map_font_family(font_name: str, font_map: dict[str,str] | None) -> str:
    if font_map and font_name in font_map:
        return font_map[font_name]
    # 기본 폴백
    if "times" in font_name.lower():
        return '"Times New Roman", Times, serif'
    if "arial" in font_name.lower() or "helvetica" in font_name.lower():
        return 'Arial, Helvetica, sans-serif'
    if "cinzel" in font_name.lower():
        return '"Cinzel", serif'
    return 'system-ui, sans-serif'

def _svg_path_from_items(items):
    # items from get_drawings(): [ ... ('l', p1, p2), ('re', rect), ('c', p1, c1, c2, p2), ... ]
    parts = []
    current_pos = None

    for it in items or []:
        op = it[0]
        if op == "m":  # move to
            if len(it) < 2: continue
            p = it[1]
            parts.append(f"M{p[0]} {p[1]}")
            current_pos = p
        elif op == "l":  # line to
            # PyMuPDF's 'l' is ('l', p1, p2), a self-contained line from p1 to p2.
            if len(it) < 3: continue
            p1, p2 = it[1], it[2]
            # If the path is not continuous, start a new subpath with moveto.
            if current_pos is None or (current_pos[0] != p1[0] or current_pos[1] != p1[1]):
                parts.append(f"M{p1[0]} {p1[1]} L{p2[0]} {p2[1]}")
            parts.append(f"L{p2[0]} {p2[1]}")
            current_pos = p2
        elif op == "re":  # rectangle
            if len(it) < 2: continue
            x0,y0,x1,y1 = it[1]
            parts.append(f"M{x0} {y0} L{x1} {y0} L{x1} {y1} L{x0} {y1} Z")
            current_pos = (x0, y1) # 마지막 꼭짓점으로 업데이트
        elif op == "c":  # bezier curve
            # PyMuPDF get_drawings() "c" is ('c', p_start, p_c1, p_c2, p_end)
            if len(it) < 5: continue
            p_start, p_c1, p_c2, p_end = it[1], it[2], it[3], it[4]

            # If path is not continuous, start a new subpath with moveto.
            if current_pos is None or (current_pos[0] != p_start[0] or current_pos[1] != p_start[1]):
                parts.append(f"M{p_start[0]} {p_start[1]}")

            parts.append(f"C{p_c1[0]} {p_c1[1]} {p_c2[0]} {p_c2[1]} {p_end[0]} {p_end[1]}")
            current_pos = p_end
        # 'h', 'v' 등 다른 경로 명령도 필요시 추가 가능

    return " ".join(parts)

def _effective_linewidth_pt(d: Dict[str, Any]) -> float:
    """
    get_drawings() 항목 d에서 실제 페이지 좌표계 기준의 선두께(pt)를 계산.
    - 일부 PDF에서 width가 밀리포인트(1/1000 pt)로 들어오는 케이스 → 휴리스틱 처리
    - transform(CTM)이 있으면 스케일 반영
    - linewidth=0(헤어라인) 보정 (SVG엔 헤어라인 개념이 없어 아주 얇은 pt로 치환)
    """
    w_raw = float(d.get("width") or 0.0)

    # (A) 밀리포인트 휴리스틱: 현실적으로 20pt↑ 선두께는 비정상 → 1/1000로 가정
    #   예) 381 -> 0.381pt, 1143 -> 1.143pt
    w_pt = (w_raw / 1000.0) if w_raw > 20.0 else w_raw

    # (B) CTM 스케일 반영 (등방성 근사: (sx+sy)/2)
    m = d.get("transform")
    if m and isinstance(m, (list, tuple)) and len(m) >= 4:
        a, b, c, d_ = float(m[0]), float(m[1]), float(m[2]), float(m[3])
        sx = (a*a + b*b) ** 0.5
        sy = (c*c + d_*d_) ** 0.5
        w_pt *= (sx + sy) / 2.0

    # (C) 헤어라인 보정 (PDF의 0pt는 "디바이스 최소선": SVG에선 거의 안 보임)
    if w_pt == 0.0:
        w_pt = 0.25  # ≈ 0.25pt (임의값, 필요시 조정)

    return w_pt

def _parse_dashes_field(dashes_val):
    """
    PyMuPDF가 '[] 0' 같은 문자열이나 ([array], phase) 형태로 반환하는 경우 모두 처리.
    반환: (dash_list: List[float] | None, phase: float | None)
    """
    if not dashes_val:
        return None, None

    # 문자열 케이스: "[3 2] 0" / "[] 0"
    if isinstance(dashes_val, str):
        try:
            parts = dashes_val.strip().split()
            if len(parts) >= 2:
                arr_str = parts[0].strip()
                phase = float(parts[1])
                arr_str = arr_str.strip("[]")
                dash_list = [float(x) for x in arr_str.split()] if arr_str else []
                return dash_list, phase
        except Exception:
            return None, None

    # (list|tuple) 케이스: ([dash...], phase)
    if isinstance(dashes_val, (list, tuple)) and len(dashes_val) == 2:
        arr, phase = dashes_val
        try:
            dash_list = list(arr) if isinstance(arr, (list, tuple)) else None
            phase_val = float(phase)
            return dash_list, phase_val
        except Exception:
            return None, None

    return None, None

class IsTextHighlight:
    """
    PDF 드로잉 객체가 텍스트를 가리는 '보이지 않는' 사각형인지 판별합니다.
    
    일부 PDF는 텍스트 선택이나 메타데이터 추가를 위해 텍스트와 동일하거나 매우 유사한 색상의
    사각형을 텍스트 위에 그리는 경우가 있습니다. 이런 사각형은 시각적인 하이라이트가 아니므로
    렌더링에서 제외함
    
    이 클래스는 드로잉의 채우기 색상이 그 아래 텍스트의 색상과 '유사한' 경우 (색상 거리가 임계값 미만),
    이를 True로 식별하여 호출자가 렌더링을 건너뛸 수 있도록 합니다.
    """
    def __init__(
        self,
        drawing_obj: Dict[str, Any],
        spans: List[Dict[str, Any]],
        color_threshold: int = 40,
    ) -> None:
        self.d = drawing_obj
        self.spans = spans
        self.color_threshold = color_threshold

    def _color_distance(self, c1_rgb: Tuple[int, int, int], c2_rgb: Tuple[int, int, int]) -> float:
        """두 RGB 색상(0-255) 간의 유클리드 거리를 계산합니다."""
        r1, g1, b1 = c1_rgb
        r2, g2, b2 = c2_rgb
        return ((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)**0.5

    def _int_to_rgb(self, c: int) -> Tuple[int, int, int]:
        """PyMuPDF 정수 색상을 (R, G, B) 튜플로 변환합니다."""
        r = (c >> 16) & 255
        g = (c >> 8) & 255
        b = c & 255
        return r, g, b

    def _float_tuple_to_rgb(self, c: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """(0-1) float 튜플 색상을 (0-255) 정수 튜플로 변환합니다."""
        r, g, b = c
        return int(r * 255), int(g * 255), int(b * 255)

    def is_highlight(self) -> bool:
        """
        드로잉 객체가 텍스트와 색상이 유사한 '보이지 않는' 사각형인지 여부를 반환합니다.
        
        :return: 유사하면 True, 그렇지 않으면 False.
        """
        # 1. 하이라이트는 보통 획(stroke)이 없는 채워진 사각형입니다.
        fill_raw = self.d.get("fill_raw")
        if not fill_raw or self.d.get("stroke_raw"):
            return False

        # 2. 드로잉이 단일 사각형으로 구성되어 있는지 확인합니다.
        items = self.d.get("items")
        if not items or len(items) != 1 or items[0][0] != 're':
            return False

        d_rect = self.d.get("rect")
        if not d_rect or d_rect.is_empty:
            return False

        # NEW: 작은 드로잉은 텍스트 하이라이트가 아닐 가능성이 높음.
        # 점선 등의 구성요소일 수 있으므로 필터링에서 제외.
        # 임계값(예: 4.0 pt^2, 2x2 pt 크기)은 실험을 통해 조정 가능.
        # 사용자께서 제공해주신 예시의 점 크기는 약 0.36 pt^2 (0.6 * 0.6) 입니다.
        if d_rect.width * d_rect.height < 4.0:
            return False

        d_color_rgb = self._float_tuple_to_rgb(fill_raw)

        # 3. 이 사각형과 겹치는 텍스트 스팬을 찾습니다.
        for span in self.spans:
            s_rect = fitz.Rect(span["bbox"])
            if s_rect.is_empty or not d_rect.intersects(s_rect):
                continue

            # 4. 겹치는 스팬의 색상이 사각형의 채우기 색상과 유사한지 확인합니다.
            span_color_int = span.get("color")
            if not isinstance(span_color_int, int) or span_color_int == 0:
                continue

            span_color_rgb = self._int_to_rgb(span_color_int)

            logger.debug(f"rect색상 : {d_color_rgb}, span색상 : {span_color_rgb}")

            if self._color_distance(d_color_rgb, span_color_rgb) < self.color_threshold:
                # 색상 차이가 임계값보다 작으면, 텍스트를 가리는 보이지 않는 사각형일 가능성이 높으므로 True를 반환합니다.
                # 호출자는 이 결과를 바탕으로 해당 드로잉을 그리지 않을 수 있습니다.
                return True

        return False

class IsLinkUnderline:
    """
    PDF 드로잉 객체가 하이퍼링크의 시각적인 밑줄인지 판단하는 클래스입니다.
    얇은 수평선을 감지하고, 이 선이 하이퍼링크의 경계 바로 아래에 위치하는지 확인합니다.
    """
    def __init__(
        self,
        drawing_obj: Dict[str, Any],
        link_rects: List[List[float]],
        ) -> None:
        """초기화
            :param drawing_obj: 밑줄인지 검사할 PDF 드로잉 객체.
            :param link_rects: 문서 내 모든 하이퍼링크의 경계 상자 목록.
        """
        self.drawing_obj = drawing_obj
        self.link_rects = link_rects

    def _is_underline_for_rect(self, underline_cand_rect: List[float], text_rect: List[float]) -> bool:
        """
        후보 사각형이 텍스트 사각형에 대한 그럴듯한 밑줄인지 확인.
        """
        u_x0, u_y0, u_x1, u_y1 = underline_cand_rect
        t_x0, t_y0, t_x1, t_y1 = text_rect

        # 1. 수평 위치 검사: 밑줄 후보가 링크의 수평 범위 내에 있는지 확인합니다.
        #    PDF에서 밑줄이 여러 조각으로 나뉘어 그려지는 경우가 있으므로,
        #    전체 너비 비율 대신 각 조각이 링크 범위 안에 있는지를 검사하는 것이 더 강건합니다.
        #    약간의 오차(2.0pt)를 허용합니다.
        if not (u_x0 >= t_x0 - 4.0 and u_x1 <= t_x1 + 4.0):
            return False

        # 2. 수직 위치: 밑줄은 텍스트 기준선 바로 아래에 있어야 함.
        # 밑줄의 상단(u_y0)이 텍스트의 하단(t_y1)에 가까워야 함.
        # PyMuPDF 좌표는 Y가 아래로 갈수록 증가.
        vertical_gap = u_y0 - t_y1
        
        # 텍스트 상자 안쪽(-4.0pt)부터 약간 아래(+4.0pt)까지 허용함.
        # 링크의 bbox는 실제 텍스트보다 클 수 있으므로, 밑줄이 bbox 하단보다 더 위에 있을 수 있습니다.
        if not (-4.0 < vertical_gap < 4.0):
            return False
            
        return True

    def _is_hairline_horizontal(self, hairline_pt: float = 1.5, min_width_pt: float = 10.0) -> bool:
        """
        PDF pt(포인트) 기준. 높이가 아주 얇고 폭이 충분히 크면 True를 반환합니다. items[0][0] == 'l' 이면 무조건 라인임.
        """
        # drawing_obj의 rect는 fitz.Rect객체임 
        r = self.drawing_obj.get("rect")
        if not r: return False
        
        items = self.drawing_obj.get("items")
        if items and items[0][0] == "l":
            return True

        # fitz.Rect 속성을 사용하여 안전하게 너비와 높이를 가져옴.
        w = r.width
        h = r.height

        # 얇은 수평선인지 확인.
        return h <= hairline_pt and w >= min_width_pt

    def link_underline(self) -> bool:
        """
        드로잉 객체가 링크의 밑줄인지 최종적으로 판단합니다.
        """
        if not self.link_rects or not self.drawing_obj:
            return False
        
        # 1. 드로잉 자체가 얇은 수평선 모양인지 확인.
        if not self._is_hairline_horizontal():
            return False

        r = self.drawing_obj.get("rect")
        if not r:
            return False
        
        # fitz.Rect를 float 리스트로 변환.
        a = [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]

        # 2. 이 선이 링크 사각형 중 하나의 그럴듯한 밑줄인지 확인.
        for link_rect in self.link_rects:
            if self._is_underline_for_rect(a, link_rect):
                return True
                
        return False

def _should_draw(d, page_area_pt: float, link_rects: list, spans: list) -> bool:
    """현재 C 단계에서 강제로 4% 설정했으나, 상황을 봐서 조정
    """
    # (a) 링크 밑줄이면 그리지 않음
    ilu = IsLinkUnderline(d,link_rects)
    if ilu.link_underline():
        return False
    
    # (b) 텍스트와 동일계열 색상박스는 그리지 않음
    ith = IsTextHighlight(d, spans, color_threshold=350)
    if ith.is_highlight():
        logger.debug(f"[DEBUG-DRAW] Skipping drawing {d.get('rect')} due to IsTextHighlight")
        return False

    # (c) stroke와 fill이 모두 없음 → 그리지 않음 (클립/레이아웃 가능성 큼)
    if (d.get("stroke") is None or d.get("width", 0) == 0) and d.get("fill") is None:
        return False

    # (d) stroke 폭 0 → 사실상 화면에 영향 없음 → 생략 (PDF hairline은 별도 처리)
    if (d.get("width", 0) == 0) and (d.get("fill") is None):
        return False
    
    it = d.get("items") or []
    # (e) re 아이템 중 채움이 투명하고 stroke_opacity 가 null일 때 그리지 않음
    if it and it[0][0] == "re" and (d.get("fill_opacity") == 0.0 and not d.get("stroke_opacity")):
        return False
    # (f) 면적의 4%, fill=흰색/검정 & stroke 없음 → 배경/마진 상자일 확률
    if len(it) == 1 and it[0][0] == "re" and d.get("stroke") is None:
        (x0,y0,x1,y1) = d["rect"]
        area = max((x1-x0)*(y1-y0), 1.0)
        if area >= 0.04 * page_area_pt and (d.get("fill") in ("#ffffff", "#000000", None)):
            return False

    return True

def _build_drawings_svg(drawings, w_pt, h_pt, scale=1.0, link_rects=None, spans=None):
    if not drawings:
        return ""
    link_rects = link_rects or []
    spans = spans or []
    page_area_pt = w_pt * h_pt
    out = [f'<svg class="pdf-vectors" width="{int(w_pt*scale)}" height="{int(h_pt*scale)}" '
           f'viewBox="0 0 {w_pt} {h_pt}" style="position:absolute;left:0;top:0;">']

    for d in drawings:
        if not _should_draw(d, page_area_pt, link_rects, spans):
            logger.debug(f"[DEBUG-DRAW] Skipping drawing {d.get('rect')} due to _should_draw condition")
            continue

        stroke = d.get("stroke")
        fill = d.get("fill")
        lw_pt = _effective_linewidth_pt(d) * scale

        # dash
        dash_list, dash_phase = _parse_dashes_field(d.get("dashes"))
        dash_attr = ""
        if dash_list is not None:
            dash_attr = f' stroke-dasharray="{" ".join(str(x*scale) for x in dash_list)}"'
            if dash_phase:
                dash_attr += f' stroke-dashoffset="{dash_phase*scale}"'

        path_d = _svg_path_from_items(d.get("items"))
        if not path_d and d.get("rect"):
            x0,y0,x1,y1 = d["rect"]
            path_d = f"M{x0} {y0} L{x1} {y0} L{x1} {y1} L{x0} {y1} Z"
        if not path_d:
            continue

        # None → 'none'
        if stroke is None or lw_pt <= 0:
            stroke = "none"
        if fill is None:
            fill = "none"

        opacity = d.get("opacity") or 1.0
        out.append(
            f'<path d="{path_d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{lw_pt}" fill-opacity="{opacity}" '
            f'stroke-opacity="{opacity}"{dash_attr} />'
        )

    out.append("</svg>")
    return "".join(out)


def build_faithful_html(
    raw_layout: Dict[str, Any],
    translated_paragraphs: List[Dict[str, Any]],
    *,
    image_src_map: dict[int, str] | None = None,
    font_map: dict[str, str] | None = None,
) -> str:
    # 1) 페이지 px 크기
    size = raw_layout.get("size") or {}
    page_w_px = int(round(float(size.get("w", 0))))
    page_h_px = int(round(float(size.get("h", 0))))

    image_html_parts: List[str] = []
    svg_defs: List[str] = []

    # 2) PT→PX 스케일(m): 텍스트/드로잉용
    meta = raw_layout.get("meta") or {}
    pt_to_px = float((meta.get("norm") or {}).get("pt_to_px", 1.0) or 1.0)
    norm_scale = float((meta.get("norm") or {}).get("scale", 1.0) or 1.0)
    m = pt_to_px * norm_scale
    logger.debug(f"[DEBUG-UTILS-BUILD]: m={m}")

    # 3) 이미지
    for im in raw_layout.get("images", []):
        src = image_src_map.get(int(im.get("xref") or -1)) if image_src_map else None
        if not src:
            continue

        # 컨테이너는 clip_bbox가 있으면 그걸 쓰고, 없으면 bbox를 씁니다.
        # 복잡한 클립의 경우, 컨테이너는 클립 경로의 바운딩 박스여야 합니다.
        box = im.get("clip_bbox") or im.get("bbox")
        cx, cy, cw, ch = box
        
        logger.debug(f"[DEBUG-UTILS-BUILD]: box={box}")

        container_style = (
            f'position:absolute;left:{cx:.2f}px;top:{cy:.2f}px;'
            f'width:{cw:.2f}px;height:{ch:.2f}px;overflow:hidden;'
        )

        # 비정형 클리핑 경로 처리
        clip_path_data_px : str = im.get("clip_path_data_px")
        if clip_path_data_px:
            clip_id = f"clip-path-for-img-{im.get('xref', 'unknown')}"

            # 여러 클립 경로가 합쳐져 있을 경우, 가장 적합한 경로 하나를 선택합니다.
            final_clip_path = pick_final_clip(clip_path_data_px)

            if final_clip_path:
                # 선택된 단일 경로로 clipPath를 생성합니다.
                clip_path_element = f'<path transform="translate({-cx:.2f}, {-cy:.2f})" d="{final_clip_path}" />'

                svg_defs.append(
                    f'<clipPath id="{clip_id}">{clip_path_element}</clipPath>'
                )
                container_style = f'position:absolute;left:{cx:.2f}px;top:{cy:.2f}px;width:{cw:.2f}px;height:{ch:.2f}px;clip-path:url(#{clip_id});'

        # get_text("dict")로 추출된 이미지(xref < 0)는 벡터 드로잉에 가려질 수 있으므로
        # z-index를 높여서 항상 위에 오도록 보장합니다.
        xref = im.get("xref")
        if xref is not None and xref < 0:
            container_style += "z-index:1;"

        # 이미지 자체의 스타일 (transform)
        a,b,c,d,e,f = im.get("matrix_page")
        origin_w, origin_h = im.get("origin_w"), im.get("origin_h")

        # 노멀라이즈 clip기준으로 matrix 계산
        A = a / origin_w
        B = b / origin_w
        C = c / origin_h
        D = d / origin_h
        E_local = e - cx
        F_local = f - cy

        matrix_css = f"matrix({A:.6f},{B:.6f},{C:.6f},{D:.6f},{E_local:.2f},{F_local:.2f})"
        
        logger.debug(f"[DEBUG-UTILS-BUILD]: matrix_css={matrix_css}")
        
        img_style = (
            f'position:absolute;left:0;top:0;'
            f'width:{origin_w}px;height:{origin_h}px;'
            f'transform:{matrix_css};transform-origin:0 0;max-width:none;max-height:none;'
        )

        image_html_parts.append(
            f'<div style="{container_style}">'
            f'  <img src="{escape_html(src)}" style="{img_style}" alt="">'
            f'</div>'
        )

    # --- 모든 HTML 조각들을 조립 ---
    html: List[str] = [f'<div class="page" style="position:relative;width:{page_w_px}px;height:{page_h_px}px;overflow:hidden;">']

    # 생성된 클립 경로가 있다면 SVG <defs> 블록을 추가합니다.
    if svg_defs:
        html.append(f'<svg width="0" height="0" style="position:absolute;pointer-events:none;"><defs>{"".join(svg_defs)}</defs></svg>')

    # 이미지 div들을 추가합니다.
    html.extend(image_html_parts)

    # 4) 드로잉(SVG) — 헤더 배경/수평선/세로 컬러바가 여기서 렌더됨
    spans = raw_layout.get("spans", []) # 5) 텍스트에서도 활용됨. _should_draw()를 판정하기 위해서 미리 가져옴.

    drawings = raw_layout.get("drawings", [])
    w_pt, h_pt = raw_layout["size_pt"]["w"], raw_layout["size_pt"]["h"] #페이지 사이즈 (pt 좌표)
    link_rects = raw_layout.get("links", [])  # pt 좌표
    logger.debug(f"[DEBUG-UTILS-BUILD]: w_pt={w_pt}, h_pt={h_pt}")
    if w_pt <= 0: w_pt = 1 # prevent division by zero
    html.append(_build_drawings_svg(drawings, w_pt, h_pt, scale=m, link_rects=link_rects, spans=spans))

    # 5) 텍스트 스팬 (Gemini가 반환한 문단 구조와 번역 사용)
    def apply_styles_from_markers(text: str, para_spans: List[Dict[str, Any]], all_spans: List[Dict[str, Any]]) -> str:
        """Parses special markers in the translated text and replaces them with HTML style tags."""
        import re

        # 1. HTML 이스케이프를 먼저 수행하여 마커가 손상되지 않도록 합니다.
        text = escape_html(text)

        # 2. 스타일 마커를 HTML 태그로 변환합니다.
        # Bold: §b§...§/b§ -> <span style="font-weight:bold;">...</span>
        text = re.sub(r'§b§(.*?)§/b§', r'<span style="font-weight:bold;">\1</span>', text)

        # Italic: §i§...§/i§ -> <span style="font-style:italic;">...</span>
        text = re.sub(r'§i§(.*?)§/i§', r'<span style="font-style:italic;">\1</span>', text)

        # Color: §c§...§/c§ -> <span style="color:...">...</span>
        # 색상 마커는 실제 색상 값을 찾아야 하므로 콜백 함수를 사용합니다.
        def color_replacer(match):
            # match.group(1)은 마커 안의 내용물입니다. 여기에는 다른 스타일 태그가 포함될 수 있습니다.
            # 예: §c§<span style="font-weight:bold;">colored bold text</span>§/c§
            inner_html = match.group(1)
            
            # 원본 스팬에서 이 텍스트 조각을 포함하는 스팬을 찾아 색상을 가져옵니다.
            # 스타일 태그를 제거하여 순수 텍스트만으로 원본 스팬을 검색합니다.
            soup = BeautifulSoup(inner_html, "html.parser")
            unescaped_fragment = soup.get_text()
            
            for span in para_spans:
                if unescaped_fragment in span.get("text", ""):
                    color_hex = _int_color_to_hex(span.get("color", 0))
                    # 기존에 있던 태그(예: bold)를 유지하면서 color 스타일만 덧씌웁니다.
                    # 만약 inner_html에 이미 span 태그가 있다면, 그 태그에 style을 추가합니다.
                    # 여기서는 간단하게 외부를 감싸는 방식으로 처리합니다.
                    return f'<span style="color:{color_hex};">{inner_html}</span>'

            # 색상을 찾지 못하면 스타일 없이 텍스트만 반환
            return inner_html

        # §c§ 마커를 가장 나중에 처리하여 다른 스타일 태그가 포함된 경우에도 동작하도록 합니다.
        text = re.sub(r'§c§(.*?)§/c§', color_replacer, text, flags=re.DOTALL)

        return text.replace("\n", "<br>")

    for para_idx, para in enumerate(translated_paragraphs):
        span_indices = para.get("span_indices", [])
        if not span_indices:
            continue

        # 문단의 첫 번째 스팬을 기준으로 위치와 스타일을 결정
        first_span_idx = span_indices[0]
        if first_span_idx >= len(spans):
            continue
        
        s = spans[first_span_idx]
        
        # 문단 전체의 bbox 계산
        para_spans = [spans[i] for i in span_indices if i < len(spans)]
        if not para_spans: continue
        
        x0 = min(sp["bbox"][0] for sp in para_spans)
        y0 = min(sp["bbox"][1] for sp in para_spans)
        x1 = max(sp["bbox"][2] for sp in para_spans)
        y1 = max(sp["bbox"][3] for sp in para_spans)

        x0, y0, x1, y1 = x0*m, y0*m, x1*m, y1*m
        font_size = float(s.get("size", 12)) * m
        color_int = int(s.get("color", 0))
        color_css = _int_color_to_hex(color_int)
        family = _map_font_family(s.get("font", ""), font_map)
        role = para.get("role", "body")
        alignment = para.get("alignment", "left")

        style = (
            f"position:absolute;left:{x0}px;top:{y0}px;"
            f"font-size:{font_size:.2f}px;"
            f"color:{color_css};"
            f"font-family:{escape_html(family)};"
            f"width:{(x1 - x0):.2f}px;white-space:pre-wrap;"
            f"line-height:1.2;text-align:{alignment};"
        )

        # 단일 스팬으로 구성된 문단일 경우, 원본 높이를 data 속성으로 전달
        data_attrs = ""
        if len(para_spans) == 1:
            data_attrs = f' data-allowed-height="{(y1 - y0):.2f}"'

        # 번역된 텍스트의 마커를 파싱하여 스타일 적용
        final_text = apply_styles_from_markers(
            para.get("translated_text", ""), para_spans, spans
        )

        html.append(f'<div id="para-{para_idx}" data-role="{role}" style="{style}"{data_attrs}>{final_text}</div>')

    html.append("</div>")
    return "".join(html)


def build_readable_html(translated_paragraphs: List[Dict[str, Any]]) -> str:
    """
    가독 모드: 단락 중심. 간단 버전은 unit별 <p>로 나열.
    (실 서비스에서는 컬럼 분리/헤딩 추정/리스트 감지 등 확장)
    """
    parts = ["<article>"]
    if not translated_paragraphs:
        parts.append("</article>")
        return "".join(parts)
    for para in translated_paragraphs:
        text = para.get("translated_text", "")
        t = escape_html(text.strip())
        if t:
            parts.append(f"<p>{t}</p>")
    parts.append("</article>")
    return "".join(parts)
