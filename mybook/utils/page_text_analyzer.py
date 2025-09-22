# mybook/utils/page_text_analyzer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import re, statistics as st

@dataclass
class Line:
    span_idxs: List[int]
    text: str
    x0: float; y0: float; x1: float; y1: float
    origin_x: Optional[float]
    size: Optional[float]
    block: int
    line_no: int

@dataclass
class Paragraph:
    line_indices: List[int]     # index in self.lines
    span_indices: List[int]
    bbox: Tuple[float, float, float, float]
    text: str

@dataclass
class NonBodyLabel:
    span_idx: int
    role: str        # "title"|"subtitle"|"header"|"footer"|"pagenum"|"floating"
    align: str       # "left"|"right"|"center"
    confidence: float
    reason: str

_TITLE_HINTS = re.compile(r"(chapter|chapitre|cap[íi]tulo|章|제\s*\d+\s*장|prologue|epilogue)", re.I)
_ROMAN_NUM  = re.compile(r"^(?=[ivxlcdm]+$)i{1,3}|iv|vi{0,3}|ix|x{1,3}$", re.I)
_PAGE_NUM   = re.compile(r"^\d{1,4}$")

class PageTextAnalyzer:
    def __init__(
        self,
        spans: List[Dict[str, Any]],
        page_w: float,
        page_h: float,
        *,
        column_split: bool = True,
        x_cut_gap_pt: float = 10.0,
        gap_factor: float = 1.6,
        indent_eps_pt: float = 1.2,
        size_jump_ratio: float = 0.12,
        nb_width_ratio_max: float = 0.55,
        nb_lr_eps_pt: float = 3.0,
        nb_top_band_ratio: float = 0.18,
        nb_bottom_band_ratio: float = 0.18,
        nb_singleton_only: bool = True,
    ):
        self.spans = spans
        self.page_w = page_w
        self.page_h = page_h

        # thresholds
        self.column_split = column_split
        self.x_cut_gap_pt = x_cut_gap_pt
        self.gap_factor = gap_factor
        self.indent_eps = indent_eps_pt
        self.size_jump_ratio = size_jump_ratio

        # non-body thresholds
        self.nb_width_ratio_max = nb_width_ratio_max
        self.nb_lr_eps = nb_lr_eps_pt
        self.nb_top_band_ratio = nb_top_band_ratio
        self.nb_bottom_band_ratio = nb_bottom_band_ratio
        self.nb_singleton_only = nb_singleton_only

        # caches
        self._lines: Optional[List[Line]] = None
        self._clusters: Optional[List[List[int]]] = None
        self._median_leading: Optional[float] = None
        self._p80_size: Optional[float] = None
        self._line_counts: Optional[Dict[Tuple[int,int], int]] = None

        self._build_common_features()

    # -------- public API --------

    def detect_paragraphs(self) -> List[Paragraph]:
        paras: List[Paragraph] = []
        for cluster in self._clusters or [list(range(len(self._lines or [])))]:
            paras.extend(self._detect_paragraphs_in_cluster(cluster))
        return paras

    def infer_alignment_for_paragraph(self, para: Paragraph, eps: float = 3.0) -> str:
        lines = [self._lines[i] for i in para.line_indices]
        if not lines:
            return "left"
        core = lines[:-1] if len(lines) > 1 else lines
        Ls = [ln.x0 for ln in core]; Rs = [ln.x1 for ln in core]
        Lm, Rm = st.median(Ls), st.median(Rs)
        slackL = [abs(x - Lm) for x in Ls]
        slackR = [abs(Rm - x) for x in Rs]
        medL, medR = st.median(slackL), st.median(slackR)
        centers = [(ln.x0 + ln.x1)/2 for ln in core]
        center_stability = st.pstdev(centers) if len(centers) > 1 else 0.0

        if medL < eps and medR < eps: return "justify"
        if medL < eps and medR > eps*2: return "left"
        if medR < eps and medL > eps*2: return "right"
        if abs(medL - medR) < eps and center_stability < eps: return "center"
        return "left"

    def detect_nonbody_spans(self) -> List[NonBodyLabel]:
        if not self.spans:
            return []
        spans = self.spans
        page_w, page_h = self.page_w, self.page_h
        line_counts = self._line_counts or {}

        sizes = [float(s.get("size") or 0.0) for s in spans if s.get("size") is not None]
        p80 = self._percentile(sizes, 80) if sizes else None

        labels: List[NonBodyLabel] = []
        for i, s in enumerate(spans):
            text = (s.get("text") or "").strip()
            if not text:
                continue
            x0,y0,x1,y1 = s.get("bbox", (0,0,0,0))
            w = max(0.0, x1 - x0)
            width_ratio = w / max(page_w, 1e-6)
            if width_ratio > self.nb_width_ratio_max:
                continue
            if self.nb_singleton_only and line_counts.get((int(s.get("block") or 0), int(s.get("line") or 0)), 0) > 1:
                continue

            L = x0; R = page_w - x1; diff = abs(L - R)
            if diff <= self.nb_lr_eps:
                align, a_score = "center", 1.0 - min(1.0, diff/(self.nb_lr_eps+1e-6))
            elif L + self.nb_lr_eps < R:
                align, a_score = "left", min(1.0, (R-L)/max(page_w*0.5,1.0))
            elif R + self.nb_lr_eps < L:
                align, a_score = "right", min(1.0, (L-R)/max(page_w*0.5,1.0))
            else:
                align, a_score = "left", 0.3

            midy = (y0 + y1)/2
            top_band = page_h * self.nb_top_band_ratio
            bottom_band = page_h * (1.0 - self.nb_bottom_band_ratio)
            size = float(s.get("size") or 0.0)
            size_boost = 0.2 if (p80 and size >= p80) else 0.0

            if midy <= top_band:
                if _TITLE_HINTS.search(text):
                    role, r_score, base = "title", 0.9 + size_boost, "top & title-key"
                else:
                    role, r_score, base = "header", 0.6 + size_boost, "top"
            elif midy >= bottom_band:
                if _PAGE_NUM.match(text) or _ROMAN_NUM.match(text):
                    role, r_score, base = "pagenum", 0.9, "bottom & pagenum"
                else:
                    role, r_score, base = "footer", 0.6, "bottom"
            else:
                if _TITLE_HINTS.search(text):
                    role, r_score, base = "subtitle", 0.65 + size_boost, "mid & title-key"
                else:
                    role, r_score, base = "floating", 0.5 + size_boost*0.5, "mid floating"

            conf = round(0.4*a_score + 0.6*r_score, 3)
            reason = f"{base}; width_ratio={width_ratio:.3f}, L={L:.1f}, R={R:.1f}"
            labels.append(NonBodyLabel(span_idx=i, role=role, align=align, confidence=conf, reason=reason))
        return labels

    def build_units(self, use_paragraphs: bool = True) -> Tuple[List[str], List[List[int]]]:
        if use_paragraphs:
            paras = self.detect_paragraphs()
            if paras:
                return [p.text for p in paras], [p.span_indices for p in paras]
        # fallback: line 단위
        units, idx_map = [], []
        from itertools import groupby
        indexed = sorted(enumerate(self.spans), key=lambda t: (t[1]["block"], t[1]["line"], t[1]["span"]))
        for _, grp in groupby(indexed, key=lambda t: (t[1]["block"], t[1]["line"])):
            g = list(grp)
            text = "".join(item[1]["text"] for item in g).strip()
            if text:
                units.append(text)
                idx_map.append([item[0] for item in g])
        return units, idx_map

    # -------- internals (공통 전처리) --------

    def _build_common_features(self):
        # 1) 라인 구성
        indexed = list(enumerate(self.spans))
        indexed.sort(key=lambda t: (t[1]["block"], t[1]["line"], t[1]["span"]))
        lines: List[Line] = []
        cur_key = None
        cur_idx: List[int] = []
        cur_sp: List[Dict[str, Any]] = []

        def flush():
            if not cur_sp: return
            text = "".join(s.get("text","") for s in cur_sp)
            x0 = min(s["bbox"][0] for s in cur_sp)
            y0 = min(s["bbox"][1] for s in cur_sp)
            x1 = max(s["bbox"][2] for s in cur_sp)
            y1 = max(s["bbox"][3] for s in cur_sp)
            origin_x = next((float(s.get("origin",[None,None])[0]) for s in cur_sp if s.get("origin")), None)
            size = next((float(s["size"]) for s in cur_sp if s.get("size") is not None), None)
            block = int(cur_sp[0]["block"]); line_no = int(cur_sp[0]["line"])
            lines.append(Line(cur_idx.copy(), text, x0,y0,x1,y1, origin_x, size, block, line_no))

        for i, s in indexed:
            key = (s["block"], s["line"])
            if key != cur_key:
                flush(); cur_key = key; cur_idx=[]; cur_sp=[]
            cur_idx.append(i); cur_sp.append(s)
        flush()

        # 표준 순서
        lines.sort(key=lambda L: (L.block, L.y0, L.x0))
        self._lines = lines

        # 2) 리딩(median gap)
        gaps=[]
        for i in range(1, len(lines)):
            gap = max(0.0, lines[i].y0 - lines[i-1].y1)
            gaps.append(gap)
        self._median_leading = st.median(gaps) if gaps else 0.0

        # 3) 칼럼 분할
        if self.column_split:
            self._clusters = self._split_columns(lines)
        else:
            self._clusters = [list(range(len(lines)))]

        # 4) 폰트 퍼센타일 / 라인 내 스팬 수
        sizes = [ln.size for ln in lines if ln.size]
        self._p80_size = self._percentile(sizes, 80) if sizes else None
        from collections import Counter
        line_counts = Counter((int(s.get("block") or 0), int(s.get("line") or 0)) for s in self.spans)
        self._line_counts = dict(line_counts)

    def _split_columns(self, lines: List[Line]) -> List[List[int]]:
        from collections import defaultdict
        by_block = defaultdict(list)
        for i, ln in enumerate(lines):
            by_block[ln.block].append(i)
        clusters: List[List[int]] = []
        for _, idxs in by_block.items():
            xs = [lines[i].x0 for i in idxs]
            if len(set(round(x) for x in xs)) <= 1 or len(idxs) < 6:
                clusters.append(sorted(idxs, key=lambda i: (lines[i].y0, lines[i].x0)))
                continue
            xs2 = sorted(xs)
            gaps = [(xs2[j+1]-xs2[j], j) for j in range(len(xs2)-1)]
            g, j = max(gaps, key=lambda t: t[0]) if gaps else (0.0, 0)
            if g > self.x_cut_gap_pt:
                cut = (xs2[j] + xs2[j+1]) / 2.0
                left = [i for i in idxs if lines[i].x0 <= cut]
                right= [i for i in idxs if lines[i].x0 >  cut]
                clusters.append(sorted(left, key=lambda i: (lines[i].y0, lines[i].x0)))
                clusters.append(sorted(right, key=lambda i: (lines[i].y0, lines[i].x0)))
            else:
                clusters.append(sorted(idxs, key=lambda i: (lines[i].y0, lines[i].x0)))
        return clusters

    def _detect_paragraphs_in_cluster(self, cluster: List[int]) -> List[Paragraph]:
        Ls = [self._lines[i] for i in cluster]
        paras: List[Paragraph] = []
        med_lead = self._median_leading or 0.0

        def is_blank(L: Line) -> bool: return len(L.text.strip())==0

        def new_para(prev: Line, cur: Line) -> bool:
            if is_blank(prev) or is_blank(cur): return True
            gap = max(0.0, cur.y0 - prev.y1)
            if med_lead>0 and gap > med_lead*self.gap_factor: return True
            if prev.origin_x is not None and cur.origin_x is not None:
                if abs(cur.origin_x - prev.origin_x) > self.indent_eps:
                    if (cur.origin_x - prev.origin_x) > self.indent_eps*0.9:
                        return True
            if prev.size and cur.size:
                if abs(cur.size - prev.size)/max(prev.size, 0.1) > self.size_jump_ratio:
                    return True
            return False

        cur_idx: List[int] = []
        for k, i in enumerate(cluster):
            if k==0:
                cur_idx.append(i); continue
            if new_para(self._lines[cluster[k-1]], self._lines[i]):
                self._flush_para(paras, cur_idx); cur_idx=[]
            cur_idx.append(i)
        self._flush_para(paras, cur_idx)
        return paras

    def _flush_para(self, paras: List[Paragraph], line_indices: List[int]):
        if not line_indices: return
        span_idxs: List[int] = []
        texts: List[str] = []
        xs0, ys0, xs1, ys1 = [], [], [], []
        for li in line_indices:
            L = self._lines[li]
            span_idxs.extend(L.span_idxs)
            texts.append(L.text)
            xs0.append(L.x0); ys0.append(L.y0); xs1.append(L.x1); ys1.append(L.y1)
        text = self._merge_text_with_hyphen(texts)
        bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
        paras.append(Paragraph(line_indices.copy(), span_idxs, bbox, text))

    def _merge_text_with_hyphen(self, lines: List[str]) -> str:
        out=[]
        for i,t in enumerate(lines):
            t=t.rstrip()
            if i < len(lines)-1:
                nxt = lines[i+1].lstrip()
                if t.endswith("-") and (nxt[:1].islower() or nxt[:1].isdigit()):
                    out.append(t[:-1]); continue
            out.append(t + (" " if i < len(lines)-1 else ""))
        return "".join(out).strip()

    def _percentile(self, arr: List[float], p: int) -> float:
        if not arr: return 0.0
        arr2 = sorted(arr)
        k = max(0, min(len(arr2)-1, int(round((p/100.0)*(len(arr2)-1)))))
        return arr2[k]
