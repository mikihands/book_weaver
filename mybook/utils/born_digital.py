# mybook/utils/born_digital.py
import logging
import fitz 
from .html_inject import escape_html
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

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
                    if not fill:
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

    logger.debug(f"[DEBUG-UTILS-COLLECT]page {page_no} spans: {spans}, images: {len(images)}, drawings: {drawings}")
    # drawings 디버깅
    for di, d in enumerate(drawings):
        logger.debug(f"[DEBUG-UTILS-COLLECT] drawing {di}: " + ", ".join(f"{k}={v}" for k,v in d.items() if v))

    return {"page_no": page_no, "size_pt": {"w": w, "h": h}, "spans": spans, "images": images, "drawings": drawings, "links": link_rects,}


def spans_to_units(spans: List[Dict[str, Any]]) -> Tuple[List[str], List[List[int]]]:
    """
    스팬 → 번역 단위(문장 또는 라인 단위)로 묶고, 역매핑 인덱스(idx_map)를 유지.
    단순 버전: 같은 line의 스팬을 합쳐 1개 유닛으로.
    반환:
      units: ["...", "...", ...]
      idx_map: [[span_idx, ...], ...]  # unit i가 어떤 스팬들로 구성되는지
    """
    units, idx_map = [], []
    if not spans:
        return units, idx_map

    # 라인 번호 기준 그룹핑
    from itertools import groupby
    # The original implementation with `spans.index(s)` is inefficient and can be incorrect
    # if there are duplicate span dictionaries in the list.
    # A more robust and efficient approach is to enumerate the spans first to preserve their original index.
    keyfn = lambda item: (item[1]["block"], item[1]["line"])

    # Sort (index, span_dict) tuples by block and line number.
    sorted_indexed_spans = sorted(enumerate(spans), key=keyfn)

    for _, group in groupby(sorted_indexed_spans, key=keyfn):
        g = list(group)
        text = "".join(item[1]["text"] for item in g).strip()
        if text:
            units.append(text)
            idx_map.append([item[0] for item in g])  # Append original indices
    logger.debug(f"[DEBUG-UTILS-SPANS]units : {units}")
    logger.debug(f"[DEBUG-UTILS-SPANS]idx_map : {idx_map}")
    return units, idx_map

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
    # items: [ ... ('l', p1, p2), ('re', rect), ('c', cp1, cp2, end), ... ] 형태
    parts = []
    current_pos = None

    for it in items or []:
        op = it[0]
        if op == "m":  # move to
            p = it[1]
            parts.append(f"M{p[0]} {p[1]}")
            current_pos = p
        elif op == "l":  # line to
            p1, p2 = it[1], it[2]
            # PyMuPDF의 'l'은 (p1, p2)로 오지만, 실제로는 p1에서 p2로의 선.
            # 이미 'm'으로 시작점을 잡았다면 'L'만 사용.
            if current_pos is None:
                parts.append(f"M{p1[0]} {p1[1]} L{p2[0]} {p2[1]}")
            else:
                parts.append(f"L{p2[0]} {p2[1]}")
            current_pos = p2
        elif op == "re":  # rectangle
            x0,y0,x1,y1 = it[1]
            parts.append(f"M{x0} {y0} L{x1} {y0} L{x1} {y1} L{x0} {y1} Z")
            current_pos = (x0, y1) # 마지막 꼭짓점으로 업데이트
        elif op == "c":  # bezier curve
            x1, y1 = it[1]
            x2, y2 = it[2]
            x3, y3 = it[3]
            parts.append(f"C{x1} {y1} {x2} {y2} {x3} {y3}")
            current_pos = (x3, y3)
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

def _should_draw(d, page_area_pt: float, link_rects: list) -> bool:
    # (0) 링크 밑줄이면 그리지 않음
    ilu = IsLinkUnderline(d,link_rects)
    if ilu.link_underline():
        return False

    # (a) stroke와 fill이 모두 없음 → 그리지 않음 (클립/레이아웃 가능성 큼)
    if (d.get("stroke") is None or d.get("width", 0) == 0) and d.get("fill") is None:
        return False

    # (b) stroke 폭 0 → 사실상 화면에 영향 없음 → 생략 (PDF hairline은 별도 처리)
    if (d.get("width", 0) == 0) and (d.get("fill") is None):
        return False

    # (c) 아주 큰 사각형인데 fill=흰색/검정 & stroke 없음 → 배경/마진 상자일 확률
    it = d.get("items") or []
    if len(it) == 1 and it[0][0] == "re" and d.get("stroke") is None:
        (x0,y0,x1,y1) = d["rect"]
        area = max((x1-x0)*(y1-y0), 1.0)
        if area >= 0.40 * page_area_pt and (d.get("fill") in ("#ffffff", "#000000", None)):
            return False

    return True

def _build_drawings_svg(drawings, w_pt, h_pt, scale=1.0, link_rects=None):
    if not drawings:
        return ""
    link_rects = link_rects or []
    page_area_pt = w_pt * h_pt
    out = [f'<svg class="pdf-vectors" width="{int(w_pt*scale)}" height="{int(h_pt*scale)}" '
           f'viewBox="0 0 {w_pt} {h_pt}" style="position:absolute;left:0;top:0;">']

    for d in drawings:
        if not _should_draw(d, page_area_pt, link_rects):
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
    layout: Dict[str, Any],
    translated_units: List[str] | None,
    idx_map: List[List[int]],
    *,
    image_src_map: dict[int, str] | None = None,
    font_map: dict[str, str] | None = None,
) -> str:
    # 1) 페이지 px 크기
    size = layout.get("size") or {}
    page_w_px = int(round(float(size.get("w", 0))))
    page_h_px = int(round(float(size.get("h", 0))))
    html: List[str] = [f'<div class="page" style="position:relative;width:{page_w_px}px;height:{page_h_px}px;">']

    # 2) PT→PX 스케일(m): 텍스트/드로잉용
    meta = layout.get("meta") or {}
    pt_to_px = float((meta.get("norm") or {}).get("pt_to_px", 1.0) or 1.0)
    norm_scale = float((meta.get("norm") or {}).get("scale", 1.0) or 1.0)
    m = pt_to_px * norm_scale
    logger.debug(f"[DEBUG-UTILS-BUILD]: m={m}")

    # 3) 이미지
    for im in layout.get("images", []):
        src = image_src_map.get(int(im.get("xref") or -1)) if image_src_map else None
        if not src:
            continue

        # 1) 컨테이너는 clip_px 있으면 그걸, 없으면 bbox_px
        box = im.get("clip_bbox") or im.get("bbox")
        cx, cy, cw, ch = box
        
        logger.debug(f"[DEBUG-UTILS-BUILD]: box={box}")

        container_style = (
            f'position:absolute;left:{cx:.2f}px;top:{cy:.2f}px;'
            f'width:{cw:.2f}px;height:{ch:.2f}px;overflow:hidden;'
        )

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

        html.append(
            f'<div style="{container_style}">'
            f'  <img src="{escape_html(src)}" style="{img_style}" alt="">'
            f'</div>'
        )

    # 4) 드로잉(SVG) — 헤더 배경/수평선/세로 컬러바가 여기서 렌더됨
    drawings = layout.get("drawings", [])
    w_pt, h_pt = layout["size_pt"]["w"], layout["size_pt"]["h"] #페이지 사이즈 (pt 좌표)
    link_rects = layout.get("links", [])  # pt 좌표
    logger.debug(f"[DEBUG-UTILS-BUILD]: w_pt={w_pt}, h_pt={h_pt}")
    if w_pt <= 0: w_pt = 1 # prevent division by zero
    html.append(_build_drawings_svg(drawings, w_pt, h_pt, scale=m, link_rects=link_rects))

    if not translated_units:
        html.append("</div>")
        return "".join(html)

    # 5) 텍스트 스팬
    spans = layout.get("spans", [])
    span_texts = [s["text"] for s in spans]
    for unit_idx, span_indices in enumerate(idx_map):
        for j, si in enumerate(span_indices):
            if unit_idx < len(translated_units) and si < len(span_texts):
                span_texts[si] = translated_units[unit_idx] if j == 0 else ""

    for s, text in zip(spans, span_texts):
        if not text:
            continue
        x0, y0, x1, y1 = s["bbox"]
        x0, y0 = x0*m, y0*m
        font_size = float(s.get("size", 12)) * m
        color_int = int(s.get("color", 0))
        color_css = _int_color_to_hex(color_int)
        bold, italic = _guess_bold_italic(s.get("font", ""), int(s.get("flags", 0)))
        family = _map_font_family(s.get("font", ""), font_map)

        style = (
            f"position:absolute;left:{x0}px;top:{y0}px;"
            f"font-size:{font_size:.2f}px;"
            f"color:{color_css};"
            f"font-family:{family};"
            f"white-space:pre;line-height:1.1;"
        )
        if bold:   style += "font-weight:700;"
        if italic: style += "font-style:italic;"

        html.append(f'<span style="{style}">{escape_html(text)}</span>')

    html.append("</div>")
    return "".join(html)


def build_readable_html(layout: Dict[str, Any], translated_units: List[str] | None, idx_map: List[List[int]]) -> str:
    """
    가독 모드: 단락 중심. 간단 버전은 unit별 <p>로 나열.
    (실 서비스에서는 컬럼 분리/헤딩 추정/리스트 감지 등 확장)
    """
    parts = ["<article>"]
    if not translated_units:
        parts.append("</article>")
        return "".join(parts)
    for text in translated_units:
        t = escape_html(text.strip())
        if t:
            parts.append(f"<p>{t}</p>")
    parts.append("</article>")
    return "".join(parts)

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

        # 1. 수평 정렬: 밑줄은 텍스트와 수평으로 정렬되어함.
        # 약간의 여유를 허용.
        horizontal_overlap = max(0, min(u_x1, t_x1) - max(u_x0, t_x0))
        text_width = t_x1 - t_x0
        
        # 텍스트 너비의 80% 이상 겹쳐야 함
        if text_width > 0 and (horizontal_overlap / text_width) < 0.8:
            return False

        # 2. 수직 위치: 밑줄은 텍스트 기준선 바로 아래에 있어야 함.
        # 밑줄의 상단(u_y0)이 텍스트의 하단(t_y1)에 가까워야 함.
        # PyMuPDF 좌표는 Y가 아래로 갈수록 증가.
        vertical_gap = u_y0 - t_y1
        
        # 텍스트 상자 안쪽(-2.0pt)부터 약간 아래(+4.0pt)까지 허용함.
        if not (-2.0 < vertical_gap < 4.0):
            return False
            
        return True

    def _is_hairline_horizontal(self, hairline_pt: float = 1.5, min_width_pt: float = 10.0) -> bool:
        """
        PDF pt(포인트) 기준. 높이가 아주 얇고 폭이 충분히 크면 True를 반환합니다.
        """
        r = self.drawing_obj.get("rect")
        if not r: return False
        
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