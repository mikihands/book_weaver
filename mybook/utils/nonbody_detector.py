# mybook/utils/nonbody_detector.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from .paragraphs import Paragraph, Line # Paragraph, Line 임포트
import re
import statistics as st

_TITLE_HINTS = re.compile(
    r"(chapter|chapitre|capitulo|capítulo|章|제\s*\d+\s*장|prologue|epilogue)",
    re.IGNORECASE
)
_ROMAN_NUM = re.compile(r"^(?=[ivxlcdm]+$)i{1,3}|iv|vi{0,3}|ix|x{1,3}$", re.IGNORECASE)
_PAGE_NUM = re.compile(r"^\d{1,4}$")

@dataclass
class NonBodyLabel:
    span_idx: int
    role: str          # "title" | "subtitle" | "header" | "footer" | "pagenum" | "floating"
    align: str         # "left" | "right" | "center"
    confidence: float  # 0..1
    reason: str

class NonBodyDetector:
    """
    페이지의 문단(Paragraph)들 중 '본문이 아닌' 요소를 휴리스틱으로 식별.
    - 입력: paragraphs, lines (ParagraphDetector 결과), page_w, page_h
    - 출력: NonBodyLabel 목록 (역할/정렬/신뢰도)
    """
    def __init__(
        self,
        width_ratio_max: float = 0.55,  # page_w * 0.55 이하만 후보
        lr_eps_pt: float = 5.0,         # 좌우 균형 판정 허용오차(포인트)
        top_band_ratio: float = 0.18,   # 상단 밴드(헤더/제목) 판정
        bottom_band_ratio: float = 0.18,# 하단 밴드(푸터/쪽번호) 판정
        min_chars: int = 2,             # 너무 짧은 텍스트(1글자)는 제외
    ):
        self.width_ratio_max = width_ratio_max
        self.lr_eps_pt = lr_eps_pt
        self.top_band_ratio = top_band_ratio
        self.bottom_band_ratio = bottom_band_ratio
        self.min_chars = min_chars

    # ---- public -------------------------------------------------------------

    def detect(self, paragraphs: List[Paragraph], lines: List[Line], page_w: float, page_h: float) -> List[NonBodyLabel]:
        if not paragraphs or not lines:
            return []

        # 본문 폰트 크기 통계 (중위값, 80퍼센타일)
        body_median_size, body_p80_size = self._get_body_font_stats(paragraphs, lines)

        labels: List[NonBodyLabel] = []
        for para in paragraphs:
            # --- 핵심 필터링 ---
            # 1. 여러 줄로 된 문단은 본문으로 간주하고 건너뜀.
            if len(para.line_indices) > 1:
                continue

            # 2. 너무 짧은 텍스트는 무시.
            text = para.text.strip()
            if len(text) < self.min_chars:
                continue
            
            # --- 이제 한 줄짜리 문단만 분석 ---
            line = lines[para.line_indices[0]]
            if not line.idxs: continue # 스팬이 없는 라인은 무시

            x0, y0, x1, y1 = line.x0, line.y0, line.x1, line.y1
            w = max(0.0, x1 - x0)
            width_ratio = w / max(page_w, 1e-6)
            left_gap = x0
            right_gap = max(0.0, page_w - x1)
            lr_diff = abs(left_gap - right_gap)

            # (A) 후보 필터링
            if width_ratio > self.width_ratio_max:
                continue

            # (B) 정렬 추정
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
            
            # (C) 폰트 크기 기반 가점
            size_boost = 0.0
            if body_p80_size and line.size and line.size >= body_p80_size:
                size_boost = 0.2

            # (D) 위치 밴드로 역할 추정
            role, role_score, role_reason = self._infer_role_by_position(
                line.size, body_median_size, y0, y1, page_h
            )

            # (E) 제목/쪽번호 패턴으로 역할 구체화 및 점수 재조정
            new_role, new_score, p_reason = self._refine_role_by_pattern(
                text, role, role_score, size_boost
            )
            role, role_score = new_role, new_score

            # (F) 최종 신뢰도 및 오탐지 제거
            # 'floating'으로 분류되었지만 폰트 크기가 본문과 유사하면, 그냥 짧은 본문일 가능성이 높음.
            if role == 'floating' and body_median_size and line.size and abs(line.size - body_median_size) < 1.0:
                continue # 오탐지로 보고 건너뜀

            conf = 0.4 * align_score + 0.6 * role_score

            labels.append(NonBodyLabel(
                span_idx=line.idxs[0], # 대표 스팬 인덱스
                role=role,
                align=align,
                confidence=round(conf, 3),
                reason=self._build_reason(role_reason, p_reason, width_ratio, left_gap, right_gap)
            ))
        return labels

    # ---- internals ----------------------------------------------------------

    def _get_body_font_stats(
        self, paragraphs: List[Paragraph], lines: List[Line]
    ) -> Tuple[Optional[float], Optional[float]]:
        """본문으로 간주되는 문단들(2줄 이상)의 폰트 크기 통계를 계산합니다."""
        body_line_sizes = []
        for p in paragraphs:
            if len(p.line_indices) > 1:  # 2줄 이상 문단을 본문으로 가정
                for line_idx in p.line_indices:
                    line_size = lines[line_idx].size
                    if line_size and line_size > 0:
                        body_line_sizes.append(line_size)
        
        if not body_line_sizes: # 2줄 이상 문단이 없으면, 모든 라인을 대상으로 계산
            body_line_sizes = [l.size for l in lines if l.size and l.size > 0]

        if not body_line_sizes:
            return None, None

        return st.median(body_line_sizes), self._percentile(body_line_sizes, 80)  

    def _percentile(self, arr: List[float], p: int) -> float:
        if not arr: return 0.0
        arr2 = sorted(arr)
        k = max(0, min(len(arr2)-1, int(round((p/100.0)*(len(arr2)-1)))))
        return arr2[k]

    def _infer_role_by_position(
        self, size: Optional[float], body_median_size: Optional[float], y0: float, y1: float, page_h: float
    ) -> Tuple[str, float, str]:
        top_band = page_h * self.top_band_ratio
        bottom_band = page_h * (1.0 - self.bottom_band_ratio)
        mid_y = (y0 + y1) / 2.0

        size_boost = 0.0
        if size and body_median_size and size > body_median_size * 1.2:
            size_boost = 0.2 # 본문보다 크면 제목/헤더일 가능성

        # 상단: 헤더 후보
        if mid_y <= top_band:
            return "header", 0.6 + size_boost, "top-band"

        # 하단: 푸터 후보
        if mid_y >= bottom_band:
            return "footer", 0.6, "bottom-band"

        # 중간: 플로팅 텍스트
        return "floating", 0.5 + size_boost * 0.5, "mid-band"

    def _refine_role_by_pattern(
        self, text: str, role: str, score: float, size_boost: float
    ) -> Tuple[str, float, str]:
        """패턴 매칭으로 역할을 더 구체화하고 점수를 조정합니다."""
        if role == "header" and _TITLE_HINTS.search(text):
            return "title", 0.9 + size_boost, "title-keyword"
        if role == "footer" and (_PAGE_NUM.match(text) or _ROMAN_NUM.match(text)):
            return "pagenum", 0.9, "pagenum-pattern"
        if role == "floating" and _TITLE_HINTS.search(text):
            return "subtitle", 0.65 + size_boost, "subtitle-keyword"
        return role, score, ""  # 변경 없음

    def _build_reason(self, base: str, extra: str, wr: float, L: float, R: float) -> str:
        parts = [base]
        if extra: parts.append(extra)
        parts.append(f"width_ratio={wr:.3f}, left={L:.1f}, right={R:.1f}")
        return "; ".join(parts)
