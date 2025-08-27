# mybook/utils/font_scaler.py
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class FontScalePolicy:
    base_px: int
    leading: float
    para_gap_px: int
    content_w_px: int
    content_h_px: int
    top_margin: int
    pad_px: int

class FontScaler:
    """
    OCR 없이 페이지 크기 + 텍스트 분량 기반으로 폰트 정책 추정.
    - 문자폭 계수로 avg_chars_per_line을 동적 계산 (라틴/CJK 대응)
    - 클램프 + 완충(EMA) 지원
    """
    def __init__(
        self,
        page_w_px: int,
        page_h_px: int,
        margin_x_ratio: float = 0.09,
        margin_y_ratio: float = 0.09,
        default_base_font_px: int = 18,
        lang: str = "ko",            # "ko","ja","zh"=CJK, 그 외 라틴
        density: str = "booky",   # "readable" | "booky"
        ema_alpha: float = 0.4       # 연속 페이지 매끈 보정(0=무시, 1=즉시 반영)
    ):
        self.page_w_px = page_w_px
        self.page_h_px = page_h_px
        self.base_content_w_px = int(round(page_w_px * (1 - 2 * margin_x_ratio)))
        self.base_content_h_px = int(round(page_h_px * (1 - 2 * margin_y_ratio)))
        self.default_base_font_px = default_base_font_px
        self.lang = lang
        self.density = density
        self.ema_alpha = ema_alpha
        self._ema_base_px = None  # 내부 상태(책 단위로 유지 추천)

    @property
    def _char_width_em(self) -> float:
        # 대략치: 라틴 0.52em, CJK 0.95em
        return 0.95 if self.lang in {"ko","ja","zh"} else 0.52

    def _default_metrics(self) -> Tuple[float, float, int]:
        # density 프리셋
        if self.density == "booky":
            leading = 2.2
            para_gap_px = int(round(self.default_base_font_px * 0.8))
        else:  # readable
            leading = 1.8
            para_gap_px = int(round(self.default_base_font_px * 0.9))
        return leading, float(para_gap_px), para_gap_px

    def estimate_base_font_size(
        self,
        text_length: int,
        paragraph_count: int,
        line_height_factor: float | None = None,
        para_gap_px: int | None = None,
    ) -> float:
        if text_length <= 0:
            return float(self.default_base_font_px)

        leading, _para_gap_f, _para_gap_px = self._default_metrics()
        if line_height_factor is None:
            line_height_factor = leading
        if para_gap_px is None:
            para_gap_px = _para_gap_px

        # 1) 한 줄 평균 글자수 추정: content_w_em / char_width_em
        content_w_em = self.base_content_w_px / self.default_base_font_px
        avg_chars_per_line = max(20.0, content_w_em / self._char_width_em)

        # 2) 예상 줄 수 / 총 텍스트 높이
        estimated_lines = text_length / avg_chars_per_line
        estimated_text_h = estimated_lines * (self.default_base_font_px * line_height_factor)

        # 3) 문단 간격 높이
        total_par_gap_h = max(0, paragraph_count - 1) * para_gap_px

        # 4) 총 예상 높이 & 스케일
        total_est_h = estimated_text_h + total_par_gap_h
        if total_est_h <= 0:
            return float(self.default_base_font_px)

        scale = self.base_content_h_px / total_est_h
        base_est = self.default_base_font_px * scale

        # 5) 현실 클램프(가독성 범위)
        base_est = max(14.0, min(26.0, base_est))

        # 6) EMA로 매끈하게 (책 단위로 상태 보존 권장)
        if self.ema_alpha and self._ema_base_px is not None:
            base_est = (self.ema_alpha * base_est) + ((1 - self.ema_alpha) * self._ema_base_px)
        self._ema_base_px = base_est

        return base_est
    
    def _density_factor(self) -> float:
        # ✅ 읽기모드/책모드에 따른 “미세” 배율 (모바일은 화면이 작으므로 약간 넓게 해주자)
        return 1.0 if self.density == "booky" else 1.03  # readable ≈ 3% 더 넓게

    def build_policy(
        self,
        text_length: int,
        paragraph_count: int
    ) -> FontScalePolicy:
        
        base_px = int(round(self.estimate_base_font_size(text_length, paragraph_count)))
        leading, _, para_gap_px = self._default_metrics()

        # ✅ 최종 콘텐츠 폭 = 사용자 여백 기반 × density 미세 조정
        content_w = int(round(self.base_content_w_px * self._density_factor()))
        # # 과도 확장 방지 (예: 페이지 폭의 96% 초과 금지)
        max_w = int(round(self.page_w_px * 0.96))
        min_w = int(round(self.page_w_px * 0.70))
        content_w = max(min_w, min(max_w, content_w))

        content_h = int(round(self.base_content_h_px * self._density_factor()))
        # # 과도 확장 방지 (예: 페이지 폭의 96% 초과 금지)
        max_h = int(round(self.page_h_px * 0.96))
        min_h = int(round(self.page_h_px * 0.70))
        content_h = max(min_h, min(max_h, content_h))

        # 최상단 마진
        top_margin = int(round((self.page_h_px - content_h) / 2))
        
        pad_px = int(round(base_px * 2))

        return FontScalePolicy(
            base_px=base_px,
            leading=leading,
            para_gap_px=para_gap_px,
            content_w_px=content_w,
            content_h_px=content_h,
            top_margin=top_margin,
            pad_px=pad_px
        )

    def to_css_vars(self, policy: FontScalePolicy) -> Dict[str, str | int | float]:
        return {
            "--font-base": f"{policy.base_px}px",
            "--leading": policy.leading,
            "--para-gap": f"{policy.para_gap_px}px",
            "--content-w": f"{policy.content_w_px}px",
            "--content-h": f"{policy.content_h_px}px",
            "--top-margin": f"{policy.top_margin}px",
            "--pad": f"{policy.pad_px}px",
        }
