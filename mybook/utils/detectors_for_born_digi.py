from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import math, statistics

def _rgb_any_to_int(rgb_any) -> Optional[int]:
    """[r,g,b] (0..1) 또는 0xRRGGBB int → 0xRRGGBB int로 통일."""
    if rgb_any is None:
        return None
    if isinstance(rgb_any, int):
        return rgb_any
    if isinstance(rgb_any, (list, tuple)) and len(rgb_any) == 3:
        r = max(0, min(255, int(round(rgb_any[0] * 255))))
        g = max(0, min(255, int(round(rgb_any[1] * 255))))
        b = max(0, min(255, int(round(rgb_any[2] * 255))))
        return (r << 16) | (g << 8) | b
    return None

def _parse_dashes(dashes_str: str) -> Tuple[List[float], float]:
    """'[4 2] 0' → ([4.0,2.0], 0.0), '[] 0' → ([], 0.0)"""
    if not dashes_str:
        return [], 0.0
    s = dashes_str.strip()
    if "]" not in s:
        return [], 0.0
    head, tail = s.split("]", 1)
    arr = head.lstrip("[").strip()
    dash = [float(x) for x in arr.split() if x] if arr else []
    phase = float(tail.strip() or 0.0)
    return dash, phase

class VectorDrawingNormalizer:
    """
    1) 작은 원형 '점'을 수평 run으로 병합 (dot leader)
    2) dash 패턴 있는 stroke를 재현 가능한 속성으로 요약
    3) solid stroke / fills는 그대로(필요시 필터) 요약
    """

    def __init__(
        self,
        # dot leader 휴리스틱
        max_dot_d: float = 3.0,        # 점 지름(px) 상한
        round_ratio_max: float = 1.6,  # 원형성: w/h 허용 상한
        y_tol: float = 0.6,            # 같은 줄 묶기 Y 허용 오차
        gap_max_factor: float = 6.0,   # 다음 점까지 최대 간격 (지름의 n배)
        min_run_count: int = 5,        # run 인정 최소 개수
    ):
        self.max_dot_d = max_dot_d
        self.round_ratio_max = round_ratio_max
        self.y_tol = y_tol
        self.gap_max_factor = gap_max_factor
        self.min_run_count = min_run_count

    # ---------- public ----------

    def analyze_page(self, page) -> Dict[str, Any]:
        drawings = page.get_drawings()
        return self._normalize_drawings(drawings)

    def normalize_drawings(self, drawings: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._normalize_drawings(drawings)

    # ---------- internals ----------

    def _normalize_drawings(self, drawings: List[Dict[str, Any]]) -> Dict[str, Any]:
        dots, strokes, fills = [], [], []
        raw_idx_to_skip = set()

        # 0) 1차 분류
        for idx, d in enumerate(drawings):
            t = d.get("type")
            rect = d.get("rect")
            if t in ("f", "fs") and self._is_small_round_dot(rect):
                dot = self._dot_from_rect(rect, d)
                if dot:
                    dots.append((idx, dot))
                continue

            if t in ("s", "fs"):  # stroke 존재
                stroke = self._stroke_from_d(d)
                if stroke:
                    strokes.append((idx, stroke))
                continue

            if t == "f":  # fill-only, 점이 아닌 면
                fill = self._fill_from_d(d)
                if fill:
                    fills.append((idx, fill))
                continue

        # 1) dot leader run 병합
        dot_runs, used_dot_idx = self._group_dot_runs([d for _, d in dots])
        # 원본 dot들의 raw index 표시 (렌더에서 제외하려면 참조)
        for (raw_i, _), used in zip(dots, [True]*len(dots)):
            if used:
                raw_idx_to_skip.add(raw_i)

        # 2) stroke 요약 (dash 여부)
        dashed_strokes, solid_strokes = [], []
        for raw_i, s in strokes:
            if s["dash_array"]:
                dashed_strokes.append(s)
                raw_idx_to_skip.add(raw_i)  # 원한다면 원본 렌더 제외
            else:
                solid_strokes.append(s)
                # solid는 그대로 두고 싶다면 skip 안 해도 됨. 필요 시 raw_idx_to_skip.add(raw_i)

        # 3) fill은 그대로(또는 크기/색 기준으로 필터)
        norm_fills = [f for _, f in fills]

        return {
            "dot_runs": dot_runs,
            "dashed_strokes": dashed_strokes,
            "solid_strokes": solid_strokes,
            "fills": norm_fills,
            "raw_index_to_skip": raw_idx_to_skip,  # 원본 벡터 렌더에서 제외하고 싶을 때 활용
        }

    def _is_small_round_dot(self, rect) -> bool:
        if not rect:
            return False
        x0, y0, x1, y1 = rect
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w <= 0 or h <= 0:
            return False
        ratio = max(w, h) / max(1e-6, min(w, h))
        d_est = 0.5 * (w + h)
        return (ratio <= self.round_ratio_max) and (d_est <= self.max_dot_d)

    def _dot_from_rect(self, rect, d) -> Optional[Dict[str, Any]]:
        x0, y0, x1, y1 = rect
        cx, cy = (x0 + x1)/2.0, (y0 + y1)/2.0
        w, h = abs(x1 - x0), abs(y1 - y0)
        color = _rgb_any_to_int(d.get("fill") or d.get("fill_color") or d.get("color"))
        return {
            "cx": cx, "cy": cy, "w": w, "h": h,
            "diameter": 0.5*(w+h),
            "color": color,
        }

    def _group_dot_runs(self, dots: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
        if not dots:
            return [], []
        # y-row 그룹핑
        dots_sorted = sorted(dots, key=lambda r: (round(r["cy"]/self.y_tol), r["cx"]))
        rows: List[List[Dict[str, Any]]] = []
        for d in dots_sorted:
            if not rows or abs(rows[-1][0]["cy"] - d["cy"]) > self.y_tol:
                rows.append([d])
            else:
                rows[-1].append(d)

        runs = []
        for row in rows:
            row.sort(key=lambda r: r["cx"])
            cur = [row[0]]
            for prev, curd in zip(row, row[1:]):
                d0 = 0.5*(prev["w"] + prev["h"])
                gap = curd["cx"] - prev["cx"]
                if gap <= max(self.gap_max_factor*d0, self.max_dot_d*self.gap_max_factor):
                    cur.append(curd)
                else:
                    self._maybe_emit_run(cur, runs)
                    cur = [curd]
            self._maybe_emit_run(cur, runs)

        return runs, list(range(len(dots)))  # 여기선 단순화: 전부 사용했다고 가정

    def _maybe_emit_run(self, dots: List[Dict[str, Any]], runs: List[Dict[str, Any]]):
        if len(dots) < self.min_run_count:
            return
        xs = [d["cx"] for d in dots]
        ys = [d["cy"] for d in dots]
        diams = [d["diameter"] for d in dots]
        spacings = [dots[i+1]["cx"] - dots[i]["cx"] for i in range(len(dots)-1)] or [0.0]
        runs.append({
            "x_start": dots[0]["cx"] - dots[0]["w"]/2,
            "x_end":   dots[-1]["cx"] + dots[-1]["w"]/2,
            "y":       statistics.fmean(ys),
            "count":   len(dots),
            "diameter": statistics.fmean(diams),
            "spacing":  statistics.fmean(spacings),
            "color":    dots[0]["color"],
        })

    def _stroke_from_d(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rect = d.get("rect")
        if not rect:
            return None
        x0, y0, x1, y1 = rect
        width = float(d.get("width", 1.0))
        # dashes
        dash_arr, phase = _parse_dashes(d.get("dashes", "[] 0"))
        cap = d.get("lineCap") or [0,0,0]
        cap_style = int(cap[0]) if isinstance(cap, (list, tuple)) else int(cap)
        color = _rgb_any_to_int(d.get("color"))
        return {
            "bbox": (x0, y0, x1, y1),
            "width": width,
            "dash_array": dash_arr,
            "dash_phase": phase,
            "line_cap": cap_style,  # 0: butt, 1: round, 2: square
            "color": color,
        }

    def _fill_from_d(self, d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rect = d.get("rect")
        if not rect:
            return None
        x0, y0, x1, y1 = rect
        color = _rgb_any_to_int(d.get("fill") or d.get("fill_color"))
        return {
            "bbox": (x0, y0, x1, y1),
            "color": color,
            "even_odd": bool(d.get("even_odd")),
        }
