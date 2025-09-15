# mybook/utils/pdf_inspector.py
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Union
import pymupdf as fitz

# ======== Config & Heuristics ========

DEFAULT_MIN_WORDS = 40           # 텍스트 레이어로 간주할 최소 단어 수(페이지)
DEFAULT_SCORE_THRESHOLD = 0.40    # born-digital로 간주할 최소 스코어
DEFAULT_SAMPLE_PAGES = 10         # 긴 문서일 때 선행 N페이지만 검사 (퍼포먼스)
DEFAULT_MAX_PAGES_HARD = 2000     # 안전장치

@dataclass
class PageQuality:
    page_index: int
    width: float
    height: float
    rotation: int
    word_count: int
    has_text_layer: bool
    img_area_ratio: float
    ascii_ratio: float
    score: float  # 높을수록 born-digital에 가까움

@dataclass
class DocumentInspection:
    file_name: str
    file_size: int
    file_sha256: str
    page_count: int
    meta: Dict[str, Any]
    toc: List[List[Any]]
    sampled_pages: int
    pages: List[PageQuality]
    # 집계
    avg_score: float
    median_score: float
    avg_img_ratio: float
    avg_word_count: float
    text_layer_coverage: float  # (텍스트 레이어 있는 페이지 비율)
    # 라우팅
    dominant_mode: str          # 'born_digital' | 'ai_layout' | 'mixed'
    reason: str                 # 결정 사유(로그 용)

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pages"] = [asdict(p) for p in self.pages]
        return d


# ======== Core functions ========

def _sha256_of_file(fp: Union[str, bytes, io.BytesIO]) -> str:
    h = hashlib.sha256()
    if isinstance(fp, (bytes, bytearray)):
        h.update(fp)
        return h.hexdigest()
    if isinstance(fp, io.BytesIO):
        h.update(fp.getvalue())
        return h.hexdigest()
    # path-like
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _calc_ascii_ratio(words_texts: List[str]) -> float:
    total = 0
    good = 0
    for w in words_texts:
        total += len(w)
        for c in w:
            # 알파/숫자/공백 정도를 "정상 텍스트"로 가정
            if c.isascii() and (c.isalnum() or c.isspace()):
                good += 1
    return (good / total) if total else 0.0


def _page_quality(page: "fitz.Page", min_words: int = DEFAULT_MIN_WORDS) -> PageQuality:
    rect = page.rect
    width, height = rect.width, rect.height
    rotation = int(page.rotation) if hasattr(page, "rotation") else 0

    words = page.get_text("words")  # [(x0,y0,x1,y1,"w", block,line,word_idx), ...] # type: ignore
    # BUGFIX: Count words that actually contain text, not just empty bounding boxes.
    # A scanned PDF might have word-like structures with no actual character data.
    # w[4] is the text content of the word tuple.
    word_count = len([w for w in words if w[4].strip()])
    has_text_layer = word_count >= min_words

    # 이미지 면적 비율(스캔 의심)
    blocks = page.get_text("blocks") # type: ignore
    img_area = 0.0
    for b in blocks:
        # PyMuPDF 버전에 따라 block type index가 다를 수 있어 보수적으로 처리
        block_type = None
        if len(b) >= 8:
            # 일반적: (x0, y0, x1, y1, text, block_no, block_type, ...)
            bt = b[6]
            block_type = bt if isinstance(bt, int) else b[-2]
        if block_type == 1:
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            img_area += (x1 - x0) * (y1 - y0)

    page_area = (width * height) if (width and height) else 1.0
    img_area_ratio = img_area / page_area

    ascii_ratio = _calc_ascii_ratio([w[4] for w in words]) if words else 0.0

    # 스코어 계산: 텍스트 레이어 유무를 이진적으로 반영하는 대신, 단어 수에 비례한 점수를 부여하여 좀 더 유연하게 만듭니다.
    # min_words에 도달하면 1.0이 되고, 그 이하여도 0이 아닌 점수를 가집니다.
    # 이렇게 하면 단어 수가 적은 서식 파일 등이 잘못 분류되는 것을 방지할 수 있습니다.
    text_presence_score = min(1.0, word_count / min_words)

    # 최종 스코어: 텍스트 존재 점수(+), 이미지 면적 비율(-), ascii 비율(+)
    score = text_presence_score - 0.5 * img_area_ratio + 0.2 * ascii_ratio

    return PageQuality(
        page_index=int(page.number), # type: ignore
        width=float(width),
        height=float(height),
        rotation=rotation,
        word_count=word_count,
        has_text_layer=has_text_layer,
        img_area_ratio=float(img_area_ratio),
        ascii_ratio=float(ascii_ratio),
        score=float(score),
    )


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float((s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2))


def inspect_pdf(
    file_like: Union[str, bytes, io.BytesIO],
    *,
    min_words: int = DEFAULT_MIN_WORDS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    sample_pages: int = DEFAULT_SAMPLE_PAGES,
) -> DocumentInspection:
    """
    업로드된 PDF를 빠르게 검사하여 'born_digital' / 'ai_layout' / 'mixed' 라우팅 결정을 내린다.

    Parameters
    ----------
    file_like : path or bytes or BytesIO
        업로드 파일 경로 또는 바이트 객체.
    min_words : int
        페이지당 텍스트 레이어로 간주할 최소 단어 수.
    score_threshold : float
        born-digital로 판단할 스코어 임계치.
    sample_pages : int
        문서가 아주 길 경우 앞에서부터 샘플링할 페이지 수.

    Returns
    -------
    DocumentInspection
    """
    # 파일 메타
    if isinstance(file_like, (bytes, bytearray)):
        file_bytes = bytes(file_like)
        file_name = "(buffer)"
        file_size = len(file_bytes)
        file_hash = _sha256_of_file(file_bytes)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    elif isinstance(file_like, io.BytesIO):
        file_bytes = file_like.getvalue()
        file_name = "(buffer)"
        file_size = len(file_bytes)
        file_hash = _sha256_of_file(file_like)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    else:
        file_name = os.path.basename(str(file_like))
        file_size = os.path.getsize(str(file_like))
        file_hash = _sha256_of_file(str(file_like))
        doc = fitz.open(str(file_like))

    page_count = min(int(doc.page_count), DEFAULT_MAX_PAGES_HARD)

    # 문서 메타/TOC
    meta = doc.metadata or {}
    try:
        toc = doc.get_toc() # type: ignore
    except Exception:
        toc = []

    # 페이지 샘플링
    n_sample = min(page_count, sample_pages if sample_pages > 0 else page_count)

    pages: List[PageQuality] = []
    for i in range(n_sample):
        page = doc.load_page(i)
        pages.append(_page_quality(page, min_words=min_words))

    # 집계
    scores = [p.score for p in pages]
    img_ratios = [p.img_area_ratio for p in pages]
    word_counts = [p.word_count for p in pages]
    text_layer_coverage = (sum(1 for p in pages if p.has_text_layer) / n_sample) if n_sample else 0.0

    avg_score = float(sum(scores) / n_sample) if n_sample else 0.0
    median_score = _median(scores)
    avg_img_ratio = float(sum(img_ratios) / n_sample) if n_sample else 0.0
    avg_word_count = float(sum(word_counts) / n_sample) if n_sample else 0.0

    # 라우팅 결정 로직
    born_like = sum(1 for s in scores if s >= score_threshold)
    ai_like = n_sample - born_like

    if born_like == n_sample:
        dominant_mode = "born_digital"
        reason = f"all {n_sample} sampled pages scored >= {score_threshold}"
    elif ai_like == n_sample:
        dominant_mode = "ai_layout"
        reason = f"all {n_sample} sampled pages scored < {score_threshold}"
    else:
        # 혼합문서 → 기본은 born_digital, 단 스코어가 근소하면 ai_layout로 강등 가능
        # (정책에 따라 조정)
        dominant_mode = "mixed"
        reason = f"mixed: {born_like}/{n_sample} pages >= {score_threshold}"

    return DocumentInspection(
        file_name=file_name,
        file_size=file_size,
        file_sha256=file_hash,
        page_count=page_count,
        meta=meta,
        toc=toc,
        sampled_pages=n_sample,
        pages=pages,
        avg_score=avg_score,
        median_score=median_score,
        avg_img_ratio=avg_img_ratio,
        avg_word_count=avg_word_count,
        text_layer_coverage=float(text_layer_coverage),
        dominant_mode=dominant_mode,
        reason=reason,
    )


# ======== Convenience helpers for backend integration ========

def choose_processing_mode(inspect_result: DocumentInspection) -> str:
    """
    앱 라우팅용 간단 결정기.
    - born_digital → born_digital
    - ai_layout → ai_layout
    - mixed → 정책에 따라 기본 born_digital, 페이지별 분기 권장
    """
    if inspect_result.dominant_mode in ("born_digital", "ai_layout"):
        return inspect_result.dominant_mode
    # mixed일 때 기본값(정책에 따라 바꾸세요)
    return "born_digital"


def page_routing_indices(
    inspect_result: DocumentInspection,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> Dict[str, List[int]]:
    """
    혼합문서에서 페이지별로 파이프라인 분기할 때 사용.
    Returns:
      {
        "born_digital": [0,2,3,...],
        "ai_layout": [1,5,...]
      }
    """
    born_idxs, ai_idxs = [], []
    for p in inspect_result.pages:
        (born_idxs if p.score >= score_threshold else ai_idxs).append(p.page_index)
    return {"born_digital": born_idxs, "ai_layout": ai_idxs}
