# mybook/utils/nonbody_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import statistics as st

_TITLE_HINTS = re.compile(
    r"(chapter|chapitre|capitulo|capítulo|章|제\s*\d+\s*장|prologue|epilogue)",
    re.IGNORECASE
)
_ROMAN_NUM = re.compile(r"^(?=[ivxlcdm]+$)i{1,3}|iv|vi{0,3}|ix|x{1,3}$", re.IGNORECASE)
_PAGE_NUM = re.compile(r"^\d{1,4}$")

@dataclass
class SpanInfo:
    idx: int
    text: str
    bbox: Tuple[float, float, float, float]  # x0,y0,x1,y1
    size: float
    block: int
    line: int

@dataclass
class NonBodyLabel:
    span_idx: int
    role: str          # "title" | "subtitle" | "header" | "footer" | "pagenum" | "floating"
    align: str         # "left" | "right" | "center"
    confidence: float  # 0..1
    reason: str

class NonBodyDetector:
    """
    페이지의 개별 span들 중 '본문이 아닌' 단일 요소들을 휴리스틱으로 식별.
    - 입력: spans(dict 목록), page_w, page_h
    - 출력: NonBodyLabel 목록 (역할/정렬/신뢰도)
    """
    def __init__(
        self,
        width_ratio_max: float = 0.55,  # page_w * 0.55 이하만 후보
        lr_eps_pt: float = 3.0,         # 좌우 균형 판정 허용오차(포인트)
        top_band_ratio: float = 0.18,   # 상단 밴드(헤더/제목) 판정
        bottom_band_ratio: float = 0.18,# 하단 밴드(푸터/쪽번호) 판정
        singleton_only: bool = True,    # 라인 내 다수 span이면 제외(문단 내 인라인일 가능성↑)
        min_chars: int = 1,             # 너무 짧은 텍스트(공백)는 제외
    ):
        self.width_ratio_max = width_ratio_max
        self.lr_eps_pt = lr_eps_pt
        self.top_band_ratio = top_band_ratio
        self.bottom_band_ratio = bottom_band_ratio
        self.singleton_only = singleton_only
        self.min_chars = min_chars

    # ---- public -------------------------------------------------------------

    def detect(self, spans: List[Dict[str, Any]], page_w: float, page_h: float) -> List[NonBodyLabel]:
        if not spans:
            return []

        sinfos = self._collect_span_infos(spans)

        # 라인 내 span 개수 통계(단일성 판정)
        line_counts = self._line_span_counts(sinfos)

        # 폰트 크기 퍼센타일(큰 폰트 가점)
        sizes = [s.size for s in sinfos if s.size]
        p80 = self._percentile(sizes, 80) if sizes else None

        labels: List[NonBodyLabel] = []
        for s in sinfos:
            text = s.text.strip()
            if len(text) < self.min_chars:
                continue

            x0, y0, x1, y1 = s.bbox
            w = max(0.0, x1 - x0)
            width_ratio = w / max(page_w, 1e-6)
            left_gap = x0
            right_gap = max(0.0, page_w - x1)
            lr_diff = abs(left_gap - right_gap)

            # (1) 후보 필터링
            if width_ratio > self.width_ratio_max:
                continue
            if self.singleton_only and line_counts[(s.block, s.line)] > 1:
                continue

            # (2) 정렬 추정
            if lr_diff <= self.lr_eps_pt:
                align = "center"
                align_score = 1.0 - min(1.0, lr_diff / (self.lr_eps_pt + 1e-6))
            elif left_gap + self.lr_eps_pt < right_gap:
                align = "left"
                align_score = min(1.0, (right_gap - left_gap) / max(page_w*0.5, 1.0))
            elif right_gap + self.lr_eps_pt < left_gap:
                align = "right"
                align_score = min(1.0, (left_gap - right_gap) / max(page_w*0.5, 1.0))
            else:
                align = "left"
                align_score = 0.3

            # (3) 위치 밴드로 역할 추정
            role, role_score, role_reason = self._infer_role_by_position(
                text, s.size, p80, y0, y1, page_h
            )

            # (4) 제목/쪽번호 패턴 가점
            pattern_boost, p_reason = self._pattern_boost(text, role)
            role_score = min(1.0, role_score + pattern_boost)

            # (5) 최종 신뢰도
            conf = 0.4 * align_score + 0.6 * role_score

            labels.append(NonBodyLabel(
                span_idx=s.idx,
                role=role,
                align=align,
                confidence=round(conf, 3),
                reason=self._build_reason(role_reason, p_reason, width_ratio, left_gap, right_gap)
            ))
        return labels

    # ---- internals ----------------------------------------------------------

    def _collect_span_infos(self, spans: List[Dict[str, Any]]) -> List[SpanInfo]:
        infos: List[SpanInfo] = []
        for i, s in enumerate(spans):
            text = s.get("text", "") or ""
            bbox = tuple(s.get("bbox", (0,0,0,0)))  # type: ignore
            size = float(s.get("size") or 0.0)
            block = int(s.get("block") or 0)
            line = int(s.get("line") or 0)
            infos.append(SpanInfo(i, text, bbox, size, block, line))
        return infos

    def _line_span_counts(self, sinfos: List[SpanInfo]) -> Dict[Tuple[int,int], int]:
        from collections import Counter
        c = Counter((s.block, s.line) for s in sinfos)
        return dict(c)

    def _percentile(self, arr: List[float], p: int) -> float:
        if not arr: return 0.0
        arr2 = sorted(arr)
        k = max(0, min(len(arr2)-1, int(round((p/100.0)*(len(arr2)-1)))))
        return arr2[k]

    def _infer_role_by_position(
        self, text: str, size: float, p80: Optional[float], y0: float, y1: float, page_h: float
    ) -> Tuple[str, float, str]:
        top_band = page_h * self.top_band_ratio
        bottom_band = page_h * (1.0 - self.bottom_band_ratio)
        mid_y = (y0 + y1) / 2.0

        size_boost = 0.2 if (p80 and size >= p80) else 0.0

        # 상단: 제목/헤더 후보
        if mid_y <= top_band:
            if _TITLE_HINTS.search(text):
                return "title", 0.9 + size_boost, "top-band & title keyword"
            return ("header", 0.6 + size_boost, "top-band")

        # 하단: 푸터/쪽번호 후보
        if mid_y >= bottom_band:
            if _PAGE_NUM.match(text) or _ROMAN_NUM.match(text):
                return "pagenum", 0.9, "bottom-band & page number pattern"
            return ("footer", 0.6, "bottom-band")

        # 중간: 좁은 폭 단일 라인 → 부제/플로팅 텍스트
        if _TITLE_HINTS.search(text):
            return "subtitle", 0.65 + size_boost, "mid & title keyword"
        # 기본 플로팅
        return "floating", 0.5 + size_boost*0.5, "mid floating"

    def _pattern_boost(self, text: str, role: str) -> Tuple[float, str]:
        if role in ("title", "subtitle"):
            if _TITLE_HINTS.search(text):
                return 0.1, "title-hint"
        if role in ("pagenum", "footer"):
            if _PAGE_NUM.match(text) or _ROMAN_NUM.match(text):
                return 0.1, "page-hint"
        return 0.0, ""

    def _build_reason(self, base: str, extra: str, wr: float, L: float, R: float) -> str:
        parts = [base]
        if extra: parts.append(extra)
        parts.append(f"width_ratio={wr:.3f}, left={L:.1f}, right={R:.1f}")
        return "; ".join(parts)
