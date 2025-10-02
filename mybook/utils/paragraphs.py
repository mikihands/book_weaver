# mybook/utils/paragraphs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import statistics

"""
tasks.py의 translate_book_pages_born_digital() 내부에서 spans_to_units() 대신하여 사용할 것임.
ub = UnitsBuilder(use_paragraphs=True)
units, idx_map = ub.build_units(layout.get("spans", []))
"""

@dataclass
class Line:
    idxs: List[int]               # 포함된 span 인덱스들
    text: str
    x0: float; y0: float; x1: float; y1: float
    origin_x: Optional[float]
    size: Optional[float]
    block: int
    line_no: int

@dataclass
class Paragraph:
    line_indices: List[int]       # lines 배열의 인덱스들
    span_indices: List[int]       # 원본 spans 인덱스들(flat)
    bbox: Tuple[float, float, float, float]
    text: str
    line_height: Optional[float] = None # 문단 내 상대적 줄간격 (예: 1.5)

class ParagraphDetector:
    """
    page.get_text('dict')에서 추출해 둔 spans 리스트를 입력으로 받아
    문단 단위로 묶어냅니다.
    - 규칙:
      * 공백 라인 => 강한 문단 경계
      * 들여쓰기 변화(origin_x) => 후보 경계 (허용오차 epsilon)
      * 줄간격 gap > median_leading * GAP_FACTOR => 후보 경계
      * 스타일 급변(size 차이 큼) => 후보 경계
    - 다단(컬럼): 간단히 x0 분포를 클러스터링(히스토그램)해서 열별로 처리(옵션)
    """
    def __init__(
        self,
        spans: List[Dict[str, Any]],
        enable_column_split: bool = True,
        indent_eps: float = 1.2,         # origin.x 허용오차(포인트)
        gap_factor: float = 1.6,         # median leading 대비 경계 배수
        size_jump_ratio: float = 0.12,   # 폰트크기 급변(12%) 이상이면 경계 후보
        min_para_chars: int = 1,         # 매우 짧은 라인만 있는 문단 병합 방지용
    ):
        self.spans = spans
        self.enable_column_split = enable_column_split
        self.indent_eps = indent_eps
        self.gap_factor = gap_factor
        self.size_jump_ratio = size_jump_ratio
        self.min_para_chars = min_para_chars

        self.lines: List[Line] = self._build_lines()

    # ---------- public API ----------
    def detect_paragraphs(self) -> List[Paragraph]:
        if not self.lines:
            return []
        clusters = self._split_columns(self.lines) if self.enable_column_split else [list(range(len(self.lines)))]
        paragraphs: List[Paragraph] = []
        for cluster in clusters:
            paragraphs.extend(self._detect_paragraphs_in_cluster(cluster))
        return paragraphs

    # ---------- internals ----------
    def _build_lines(self) -> List[Line]:
        # spans는 이미 block/line/… 필드가 있다고 가정(collect_page_layout에서 만들던 구조)
        # block, line 기준으로 묶는다.
        indexed = list(enumerate(self.spans))
        indexed.sort(key=lambda t: (t[1]["block"], t[1]["line"], t[1]["span"]))
        lines: List[Line] = []
        cur_key = None
        cur_idxs: List[int] = []
        cur_spans: List[Dict[str, Any]] = []

        def flush():
            if not cur_spans:
                return
            text = "".join(s["text"] for s in cur_spans)
            # bbox 집계(안전한 min/max)
            x0 = min(s["bbox"][0] for s in cur_spans)
            y0 = min(s["bbox"][1] for s in cur_spans)
            x1 = max(s["bbox"][2] for s in cur_spans)
            y1 = max(s["bbox"][3] for s in cur_spans)
            # 대표 origin/size
            origin_x = None
            size = None
            for s in cur_spans:
                ox = s.get("origin", [None, None])[0]
                if ox is not None:
                    origin_x = float(ox); break
            for s in cur_spans:
                if s.get("size") is not None:
                    size = float(s["size"]); break
            block = cur_spans[0]["block"]
            line_no = cur_spans[0]["line"]
            lines.append(Line(cur_idxs.copy(), text, x0, y0, x1, y1, origin_x, size, block, line_no))

        for i, s in indexed:
            key = (s["block"], s["line"])
            if key != cur_key:
                flush()
                cur_key = key
                cur_idxs = []
                cur_spans = []
            cur_idxs.append(i)
            cur_spans.append(s)
        flush()
        # y 증가 방향이 위/아래냐에 따라 정렬이 달라질 수 있으므로 y0 기준 재정렬
        lines.sort(key=lambda L: (L.block, L.y0, L.x0))
        return lines

    def _split_columns(self, line_indices: List[Line] | List[int]) -> List[List[int]]:
        # 매우 간단한 열 분리: x0 값 히스토그램 피크로 2~3개 클러스터 추정
        # 안정성 위해 같은 block 내에서만 열 분리
        # 입력이 int 인덱스 리스트면 Line로 변환
        if line_indices and isinstance(line_indices[0], int):
            indices: List[int] = line_indices  # type: ignore
        else:
            indices = list(range(len(self.lines)))

        # block별로 그룹 후, 각 그룹을 x0 분포로 다시 나눈다
        from collections import defaultdict
        by_block: Dict[int, List[int]] = defaultdict(list)
        for i in indices:
            by_block[self.lines[i].block].append(i)

        clusters: List[List[int]] = []
        for _, idxs in by_block.items():
            xs = [self.lines[i].x0 for i in idxs]
            if len(set(round(x) for x in xs)) <= 1 or len(idxs) < 6:
                clusters.append(sorted(idxs, key=lambda i: (self.lines[i].y0, self.lines[i].x0)))
                continue
            # 2-피크만 시도(대부분 2단)
            cut = self._find_x_cut(xs)
            if cut is None:
                clusters.append(sorted(idxs, key=lambda i: (self.lines[i].y0, self.lines[i].x0)))
            else:
                left = [i for i in idxs if self.lines[i].x0 <= cut]
                right = [i for i in idxs if self.lines[i].x0 > cut]
                clusters.append(sorted(left, key=lambda i: (self.lines[i].y0, self.lines[i].x0)))
                clusters.append(sorted(right, key=lambda i: (self.lines[i].y0, self.lines[i].x0)))
        return clusters

    def _find_x_cut(self, xs: List[float]) -> Optional[float]:
        # 1D Otsu 비슷한 간단 컷: 정렬 후 중간 큰 간격을 컷으로
        xs2 = sorted(xs)
        gaps = [(xs2[i+1]-xs2[i], i) for i in range(len(xs2)-1)]
        if not gaps: return None
        g, i = max(gaps, key=lambda t: t[0])
        return (xs2[i] + xs2[i+1]) / 2.0 if g > 10.0 else None  # 10pt 이상이면 열 경계 후보
    
    def _detect_paragraphs_in_cluster(self, cluster: List[int]) -> List[Paragraph]:
        Ls = [self.lines[i] for i in cluster]
        if len(Ls) == 0:
            return []
        # leading 통계
        gaps = []
        for i in range(1, len(Ls)):
            gaps.append(max(0.0, Ls[i].y0 - Ls[i-1].y1))
        median_leading = statistics.median(gaps) if gaps else 0.0

        paras: List[Paragraph] = []
        cur_line_idxs: List[int] = []

        def is_blank(line: Line) -> bool:
            return len(line.text.strip()) == 0

        def new_paragraph(prev: Line, cur: Line) -> bool:
            # 1) 빈 줄
            if is_blank(prev) or is_blank(cur):
                return True
            # 2) 줄간격 경계
            gap = max(0.0, cur.y0 - prev.y1)
            if median_leading > 0 and gap > median_leading * self.gap_factor:
                return True
            # 3) 들여쓰기 변화(허용오차 포함)
            # BUGFIX: 이전 로직은 들여쓰기가 늘어나는 경우(양수)만 분리했음.
            # x좌표가 크게 바뀌면 방향에 상관없이 분리해야 함.
            if prev.origin_x is not None and cur.origin_x is not None:
                if abs(cur.origin_x - prev.origin_x) > self.indent_eps:
                    return True
            # 4) 폰트 사이즈 급변
            if prev.size and cur.size:
                if abs(cur.size - prev.size) / max(prev.size, 0.1) > self.size_jump_ratio:
                    return True
            return False

        for i, line in enumerate(Ls):
            if i == 0:
                cur_line_idxs.append(cluster[i])
                continue
            prev = Ls[i-1]
            if new_paragraph(prev, line):
                self._flush(paras, cur_line_idxs)
                cur_line_idxs = []
            cur_line_idxs.append(cluster[i])
        self._flush(paras, cur_line_idxs)

        # 너무 짧은 문단은 이웃과 병합(옵션)
        paras = self._merge_tiny(paras, min_chars=self.min_para_chars)
        return paras

    def _flush(self, paras: List[Paragraph], line_indices: List[int]):
        if not line_indices:
            return
        # span 인덱스 평탄화
        span_idxs = []
        texts = []
        xs0, ys0, xs1, ys1 = [], [], [], []
        for li in line_indices:
            L = self.lines[li]
            span_idxs.extend(L.idxs)
            texts.append(L.text)
            xs0.append(L.x0); ys0.append(L.y0); xs1.append(L.x1); ys1.append(L.y1)

        # --- NEW: Calculate line height ---
        line_height_ratio = None
        para_lines = [self.lines[li] for li in line_indices]
        if len(para_lines) > 1:
            line_spacings = []
            font_sizes = [l.size for l in para_lines if l.size and l.size > 0]

            if font_sizes:
                median_font_size = statistics.median(font_sizes)
                for i in range(len(para_lines) - 1):
                    spacing = para_lines[i+1].y0 - para_lines[i].y0
                    if spacing > 0:
                        line_spacings.append(spacing)

                if line_spacings and median_font_size > 0:
                    median_spacing = statistics.median(line_spacings)
                    line_height_ratio = round(median_spacing / median_font_size, 2)

        text = self._merge_text_with_hyphen(texts)
        bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
        paras.append(Paragraph(line_indices.copy(), span_idxs, bbox, text, line_height=line_height_ratio))

    def _merge_text_with_hyphen(self, lines: List[str]) -> str:
        out = []
        for i, t in enumerate(lines):
            t = t.rstrip()
            if i < len(lines)-1:
                nxt = lines[i+1].lstrip()
                if t.endswith("-") and (nxt[:1].islower() or nxt[:1].isdigit()):
                    out.append(t[:-1])  # 하이픈 제거 후 연결
                    continue
            out.append(t + (" " if i < len(lines)-1 else ""))
        return "".join(out).strip()

    def _merge_tiny(self, paras: List[Paragraph], min_chars: int) -> List[Paragraph]:
        if not paras: return paras
        merged: List[Paragraph] = []
        for p in paras:
            if merged and len(p.text) < min_chars:
                # 이전과 합치기
                prev = merged[-1]
                new_text = (prev.text + " " + p.text).strip()
                new_bbox = (min(prev.bbox[0], p.bbox[0]), min(prev.bbox[1], p.bbox[1]),
                            max(prev.bbox[2], p.bbox[2]), max(prev.bbox[3], p.bbox[3]))
                merged[-1] = Paragraph(prev.line_indices + p.line_indices,
                                       prev.span_indices + p.span_indices,
                                       new_bbox, new_text,
                                       line_height=prev.line_height)
            else:
                merged.append(p)
        return merged


class UnitsBuilder:
    """
    기존 spans_to_units(line 단위)와 동일한 반환 형태를 유지하되,
    문단 단위로 units / idx_map을 만들어 준다.
    실패/자신 없으면 라인 단위로 폴백 가능.
    """
    def __init__(self, use_paragraphs: bool = True):
        self.use_paragraphs = use_paragraphs
        self.paragraphs: List[Paragraph] = []
        self.detector: Optional[ParagraphDetector] = None

    def build_units(self, spans: List[Dict[str, Any]]) -> Tuple[List[str], List[List[int]]]:
        if not spans:
            return [], []

        if self.use_paragraphs:
            self.detector = ParagraphDetector(spans)
            self.paragraphs = self.detector.detect_paragraphs()
            # 문단이 너무 적거나 모든 문단이 한 줄뿐이면 이득이 적으니 폴백
            if len(self.paragraphs) >= 1 and any(len(p.text) > 0 for p in self.paragraphs):
                units = [p.text for p in self.paragraphs]
                idx_map = [p.span_indices for p in self.paragraphs]
                return units, idx_map

        # ---- 폴백: 기존 라인 단위 그룹핑 ----
        from itertools import groupby
        keyfn = lambda item: (item[1]["block"], item[1]["line"])
        sorted_indexed_spans = sorted(enumerate(spans), key=keyfn)
        units, idx_map = [], []
        for _, group in groupby(sorted_indexed_spans, key=keyfn):
            g = list(group)
            text = "".join(item[1]["text"] for item in g).strip()
            if text:
                units.append(text)
                idx_map.append([item[0] for item in g])
        return units, idx_map


def calculate_line_height_for_paragraph(span_indices: List[int], all_spans: List[Dict[str, Any]]) -> Optional[float]:
    """
    Gemini가 반환한 특정 문단(span_indices)에 대한 상대적 줄간격(line-height)을 계산합니다.
    이 함수는 ParagraphDetector의 내부 상태에 의존하지 않습니다.

    :param span_indices: 문단을 구성하는 스팬들의 원본 인덱스 리스트.
    :param all_spans: 페이지의 모든 스팬 정보 리스트.
    :return: 계산된 상대적 줄간격 (e.g., 1.5) 또는 계산 불가 시 None.
    """
    if not span_indices or not all_spans or len(span_indices) < 2:
        return None

    para_spans = [all_spans[i] for i in span_indices if i < len(all_spans)]

    # 1. 스팬들을 (block, line) 키로 그룹화하여 라인 단위로 재구성합니다.
    lines_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for span in para_spans:
        key = (span.get("block", 0), span.get("line", 0))
        if key not in lines_map:
            lines_map[key] = []
        lines_map[key].append(span)

    # 2. 각 라인의 y0, size를 계산하고 y0 기준으로 정렬합니다.
    para_lines = []
    for spans_in_line in lines_map.values():
        if not spans_in_line:
            continue
        y0 = min(s['bbox'][1] for s in spans_in_line)
        # 라인의 대표 폰트 크기를 찾습니다.
        size = next((s.get('size') for s in spans_in_line if s.get('size')), None)
        para_lines.append({'y0': y0, 'size': size})

    # y-좌표 순으로 라인 정렬
    para_lines.sort(key=lambda x: x['y0'])

    if len(para_lines) < 2:
        return None

    # 3. 라인 간 간격(spacing)과 폰트 크기의 중간값을 계산합니다.
    line_spacings = [para_lines[i+1]['y0'] - para_lines[i]['y0'] for i in range(len(para_lines) - 1)]
    font_sizes = [l['size'] for l in para_lines if l.get('size') and l['size'] > 0]

    if not line_spacings or not font_sizes:
        return None

    median_spacing = statistics.median(line_spacings)
    median_font_size = statistics.median(font_sizes)

    if median_font_size > 0:
        return round(median_spacing / median_font_size, 2)

    return None
