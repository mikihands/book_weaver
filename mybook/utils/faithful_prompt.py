#mybook/utils/faithful_prompt.py

def build_prompt_faithful(page_ctx: dict, target_lang: str, total_pages: int) -> tuple[str, dict, str]:
    """
    page_ctx = {
      "page_no": 12,
      "size": {"w": 2480, "h": 3508, "units":"px"},
      "images": [{"ref":"img_p12_1","bbox":[x,y,w,h]}, ...],  # 이미 정규화된 좌표(0,0)~(W,H)
    }
    """
    W = int(page_ctx["size"]["w"])
    H = int(page_ctx["size"]["h"])

    # --- SYSTEM: 역할/좌표계/반환형식 계약을 강하게 명시 ---
    SYSTEM_PROMPT = f"""
You are **Weaver AI**, a document layout + translation compositor.
Your job: read the attached page, translate it to the requested language, and output a **single JSON object**
containing an HTML stage snippet plus a figures manifest.

### Coordinate Space (IMPORTANT)
- Origin: top-left (0,0)
- Page size (pixels): width={W}, height={H}
- All bounding boxes must be within this page rectangle.

### Output Contract (NO SCHEMA PROVIDED; FOLLOW EXACT KEYS)
Return **only** one JSON object with the following fields and shapes (no extra text, no code fences):

{{
  "schema_version": "weaver.page.v2",
  "page": {{
    "page_no": <int>,
    "page_w": <int>,          // must be {W}
    "page_h": <int>,          // must be {H}
    "page_units": "px"
  }},
  "html_stage": "<div>...</div>",   // stage-only HTML. NO <html>, <head>, or <body>
  "figures": [
    {{
      "ref": "<string>",      // image reference id
      "bbox_x": <int>,        // x in pixels within page
      "bbox_y": <int>,        // y in pixels within page
      "bbox_w": <int>,        // width in pixels
      "bbox_h": <int>,        // height in pixels
      "alt": "<string, optional>",
      "caption": "<string, optional>",
      "width": <int, optional>,    // rendered width hint
      "height": <int, optional>    // rendered height hint
    }},
    ...
  ]
}}

### HTML Rules
The server controls the base typography and spacing via CSS variables.
Do NOT override them. Compose content INSIDE the wrapper only.

Wrapper contract (already applied by server):
- CSS variables available: --font-base, --leading, --para-gap, --content-w, --pad
- The wrapper sets font-size, line-height, padding, and max width using those variables.

기타 HTML 관련 주의사항
1) 페이지 상,하,좌,우 여백은 서버에서 이미 구현되어 있습니다. 페이지 여백용 스타일을 추가하지 마세요.
2) font-size, margin, padding, line-height은 반드시 서버의 Wrapper에서 이미 설정한 CSS 변수 기본값에 상대적 비율을 사용해야 합니다. e.g. style="font-size: calc(var(--font-base) * 1.15)"
3) 원본과 최대한 유사한 레이아웃을 만들기 위하여 Tailwind CSS를 사용할 수 있으나, font-size, margin, padding, line-height에 대하여는 절대 Tailwind 유틸리티 클래스를 사용하지 마세요.
4) Use semantic HTML (<p>, <h1-6>, <ul/ol>, <blockquote>, <figure/figcaption>, <table>, etc.) You can add if needed.
5) For body text, do NOT set fixed px/rem sizes. Inherit wrapper base.
6) Margins/padding: use multiples of var(--font-base), e.g. style="margin: calc(var(--font-base)*1.2) 0 .4em;"
7) 문단은 실제 원본의 들여쓰기/줄간격/여백을 반영해라.
8) For each figure:
  - In HTML, insert an <img> placeholder with **data-ref="<ref>"** and width,height attributes. (do not set src).
  - The server will set the actual src using your "figures" manifest.
  - figure는 이미지 크기를 원본 비율에 맞춰 지정하고, 캡션이 있는 경우 원본에 충실히 재현, .
9) Never output <html>, <head>, or <body>. Stage-only snippet.
10) 리스트/각주/출처는 본문보다 작은 글자와 더 촘촘한 줄간격으로 한다.
11) 절대 위치 지정은 금지. 대신 여백/들여쓰기/정렬로 원본 레이아웃을 “느낌”으로 재현.

### Figures Policy
- A list of known figures (by "ref" and bbox) may be provided by the user message.
- If you see additional figures not listed, you **may** infer them and:
  1) add an <img data-ref="..." width="..." height="..." /> placeholder in HTML,
  2) add a matching entry to "figures" with an **estimated** bbox that fits inside the page.
- Do NOT place any bbox outside the page bounds. Round all numbers to integers.

### Typography & Translation
- Translate all textual content into the target language. Do not translate numbers, file names, or code syntax.
- Keep layout structure visually close to the original, but make typography readable (headings, spacing, lists).
- Avoid hallucinating content; if unsure, keep a neutral placeholder.

### Strictness
- Output **only** the JSON object. No surrounding prose. No code fences.
- All numbers must be integers where applicable (bbox and page sizes).
    """.strip()

    # --- USER: 작업 지시 + 좌표계/리소스 컨텍스트 ---
    user = {
        "task": "Translate and structure this page",
        "total_pages": total_pages,
        "target_lang": target_lang,
        "schema_version": "weaver.page.v2",
        "coordinate_space": {"width": W, "height": H, "units": "px", "origin": "top-left"},
        "target_page": page_ctx["page_no"],
        # 서버가 이미 알고 있는 이미지 자원(정규화된 bbox)을 힌트로 제공
        "figures_available": page_ctx.get("images", []),
        "instructions": [
            f"Translate all text into {target_lang}. Do not translate numbers, symbols, or code syntax.",
            f"You are processing page {page_ctx['page_no']} out of {total_pages}. Output must contain ONLY content visible on page {page_ctx['page_no']}.",
            "Maintain a visually similar layout to the original, but ensure readability with proper headings, spacing, and lists.",
            "For each figure: add <img data-ref='...' width='...' height='...' /> (no src) in HTML and a corresponding entry in 'figures'.",
            "If you infer a figure not listed in 'figures_available', add both placeholder and manifest with an estimated bbox.",
            "Ensure all bbox values are integers and strictly inside the page rectangle.",
            "Your output must be a single JSON object. DO NOT add any extra text or explanations."
        ]
    }

    # --- 예시: 좌표/키 형태를 모델에 각인시키는 샘플 (값은 임의) ---
    example_answer = f"""
{{
  "schema_version": "weaver.page.v2",
  "page": {{
    "page_no": {int(page_ctx['page_no'])},
    "page_w": {W},
    "page_h": {H},
    "page_units": "px"
  }},
  "html_stage": "<div class="prose" style="max-width: var(--content-w);">
  <h2 style="font-size: calc(var(--font-base) * 1.35); margin: calc(var(--font-base)*1.2) 0 .4em;">
    토요일 어느 시점에 경찰은 살에게 연락했다.
  </h2>

  <p>경찰이 클로이와 엠마에게 금요일 밤이나 토요일 아침에 연락했을 것이라고 추정한다…</p>

  <blockquote style="margin: var(--font-base) 0; padding-left: 1.25em; border-left: 4px solid #d1d5db; font-style: italic;">
    DI 리처드 호킨스는… <sup>6</sup>
  </blockquote>

  <figure style="margin: calc(var(--font-base)*1.2) auto;">
    <img data-ref="img_p6_1" style="display:block; width:800px; height:600px; border-radius:.375rem;">
    <figcaption style="text-align:center; color:#6b7280; font-size:.9em; margin-top:.35em;">
      그림 1. 사건 현장 스케치
    </figcaption>
  </figure>

  <ul style="margin-top: calc(var(--font-base)*1.2);">
    <li>…</li>
  </ul>
</div>",
  "figures": [
    {{
      "ref": "img_p{int(page_ctx['page_no'])}_1",
      "bbox_x": 140,
      "bbox_y": 320,
      "bbox_w": 800,
      "bbox_h": 600,
      "alt": "…",
      "caption": "…"
    }}
  ]
}}
""".strip()

    return SYSTEM_PROMPT, user, example_answer



def build_prompt_faithful_v1(page_ctx: dict, target_lang: str):
    """
    page_ctx = {
      "page_no": 12,
      "size": {"w": 2480, "h": 3508, "units":"px"},
      "images": [{"ref":"img_p12_1","bbox":[...]}],
      "ocr_text": "..."  # 선택
    }
    """
    W = page_ctx["size"]["w"]
    H = page_ctx["size"]["h"]

    SYSTEM_PROMPT = f"""
        You are Weaver AI, a document translation and structuring engine.
        Your task is to translate content into the target language and return ONLY JSON.
        The JSON must strictly follow the Weaver Page Schema (weaver.page.v1).
        You MUST use the following coordinate space for ALL bounding boxes:
        - Origin: top-left (0,0)
        - Page size (pixels): width={int(W)}, height={int(H)}
        - Return every bbox as x, y, w, h in THIS space (pixels). Do NOT use any other unit.
    """

    sys = SYSTEM_PROMPT

    user = {
        "task": "Translate and structure this page",
        "schema_version": "weaver.page.v1",
        "target_lang": target_lang,
        "mode": "faithful",
        "page": {"page_no": page_ctx["page_no"], "size": page_ctx["size"]},
        "resources": {"images": page_ctx.get("images", [])},
        "instructions": [
            "1. You MUST output valid JSON conforming to the schema.",
            "2. Each block requires: id, type, order, bbox, content, and confidence (0.0~1.0).",
            "3. Block types allowed: heading, paragraph, list, table, figure, equation, code, quote, footnote, separator.",
            f"4. Translate all text into {target_lang}. Do not translate numbers, symbols, or code syntax.",
            "5. Do NOT generate new images. Use provided image_ref values for figures.",
            "6. Tables must use <table> markup in content.html (no CSS, no style attributes).",
            "7. For footnotes, include label and text, and link back using ref_ids if provided.",
            "8. Confidence: 0.9+ if very sure, 0.6~0.8 if somewhat sure, <0.5 if unsure.",
            "9. Do NOT include any additional text or explanations outside the JSON.",
            "Return ONLY the JSON object, nothing else."
        ]
    }

    example_answer = SAMPLE_ANSWER_V1

    return sys, user, example_answer


SAMPLE_ANSWER_V1 = """
    {
        "schema_version": "weaver.page.v1",
        "target_lang": "ko",
        "mode": "faithful",
        "page": {
            "page_no": 12,
            "size": {"w": 2480, "h": 3508, "units": "px"}
        },
        "resources": {
            "images": [
            {"ref":"img_p12_1","bbox":[140,1120,1200,800]}
            ]
        },
        "blocks": [
            {
            "id": "b1",
            "type": "heading",
            "order": 1,
            "bbox": [120,220,2200,120],
            "content": {"text":"3.2 시스템 개요"},
            "confidence": 0.95,
            "attrs": {"level":2}
            },
            {
            "id": "fig2",
            "type": "figure",
            "order": 2,
            "bbox": [140,1120,1200,800],
            "content": {"image_ref":"img_p12_1","caption":"그림 3.2: 시스템 개요"},
            "confidence": 0.9
            }
        ]
    }
"""