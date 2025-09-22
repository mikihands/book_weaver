#mybook/utils/pick_best_path.py
import re
from typing import List, Tuple

# ---- 1) d 문자열을 서브패스로 쪼개기 ----
def split_subpaths(d: str) -> List[str]:
    # 'M'으로 시작해서 'Z'로 끝나는 블록을 추출 (대문자만 가정)
    # 입력이 'M ... Z M ... Z' 형태라고 가정
    tokens = re.findall(r'M[^M]*?Z', d.replace('\n',' '))
    return [t.strip() for t in tokens]

# ---- 2) 좌표 파싱 유틸(아주 단순 버전) ----
def _numbers(s: str):
    # 공백/콤마 구분 숫자
    for num in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', s):
        yield float(num)

# ---- 3) bbox 구하기 (빠른 휴리스틱: 제어점을 포함해 범위 뽑기) ----
# 정확한 베지어 극값을 안 구해도, 대부분의 PDF 클립에서는 제어점 포함 bbox만으로 충분히 판별됨.
def bbox_of(subpath: str) -> Tuple[float,float,float,float]:
    xs, ys = [], []
    for cmd, args in re.findall(r'([MLCQZ])([^MLCQZ]*)', subpath):
        if cmd in ('M','L'):
            it = list(_numbers(args))
            # M/L는 (x,y) 페어들
            for i in range(0, len(it), 2):
                xs.append(it[i]); ys.append(it[i+1])
        elif cmd == 'C':
            it = list(_numbers(args))
            # C는 (x1,y1,x2,y2,x,y) 페어들이 이어질 수 있음
            for i in range(0, len(it), 6):
                xs += [it[i], it[i+2], it[i+4]]
                ys += [it[i+1], it[i+3], it[i+5]]
        elif cmd == 'Q':
            it = list(_numbers(args))
            for i in range(0, len(it), 4):
                xs += [it[i], it[i+2]]
                ys += [it[i+1], it[i+3]]
        elif cmd == 'Z':
            pass
    return min(xs), min(ys), max(xs), max(ys)

def bbox_contains(a, b, pad=1e-6):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 - pad <= bx1) and (ay1 - pad <= by1) and (ax2 + pad >= bx2) and (ay2 + pad >= by2)

# ---- 4) 베지어 간단 샘플링으로 면적(폴리라인 근사) 계산 ----
def _sample_cubic(p0, p1, p2, p3, n=32):
    pts = []
    for i in range(n+1):
        t = i / n
        # cubic Bezier
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        pts.append((x,y))
    return pts

def polyline_of(subpath: str, samples_per_curve=32):
    pts = []
    cur = None
    for cmd, args in re.findall(r'([MLCQZ])([^MLCQZ]*)', subpath):
        it = list(_numbers(args))
        if cmd == 'M':
            cur = (it[0], it[1])
            pts.append(cur)
        elif cmd == 'L':
            for i in range(0, len(it), 2):
                cur = (it[i], it[i+1])
                pts.append(cur)
        elif cmd == 'C':
            for i in range(0, len(it), 6):
                p0 = cur
                p1 = (it[i],   it[i+1])
                p2 = (it[i+2], it[i+3])
                p3 = (it[i+4], it[i+5])
                seg = _sample_cubic(p0,p1,p2,p3, n=samples_per_curve)
                pts += seg[1:]  # p0 중복 제거
                cur = p3
        elif cmd == 'Q':
            # Q는 C로 러프하게 확장하거나, 간단한 이차식 직접 샘플링을 추가
            # 여기선 C만 나오는 경우가 많아 생략해도 무방. 필요 시 구현.
            pass
        elif cmd == 'Z':
            if pts and pts[0] != pts[-1]:
                pts.append(pts[0])
    return pts

def polygon_area(pts):
    # shoelace
    if len(pts) < 3: return 0.0
    s = 0.0
    for i in range(len(pts)-1):
        x1,y1 = pts[i]
        x2,y2 = pts[i+1]
        s += x1*y2 - x2*y1
    return abs(s) * 0.5

def pick_final_clip(d: str) -> str:
    subs = split_subpaths(d)
    if len(subs) == 1:
        return subs[0]

    # 1) bbox 포함관계로 후보 찾기
    bboxes = [bbox_of(s) for s in subs]
    # "다른 모든 bbox에 포함되는" 경로 찾기
    for i, bb in enumerate(bboxes):
        if all(bbox_contains(other, bb) for j, other in enumerate(bboxes) if i != j):
            return subs[i]

    # 2) 실패하면 면적 최소 경로 선택 (폴리라인 근사)
    areas = []
    for s in subs:
        pts = polyline_of(s, samples_per_curve=64)
        areas.append(polygon_area(pts))
    # 면적 0 방지용 아주 작은 오차 처리
    idx = min(range(len(subs)), key=lambda k: areas[k] if areas[k] > 1e-9 else 1e-9)
    return subs[idx]