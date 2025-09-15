#mybook/utils/faithful_prompt.py

def build_prompt_faithful(page_ctx: dict, target_lang: str, total_pages: int, book_title: str | None = None, book_genre: str | None = None, prev_page_html: str | None = None, user_feedback: str | None = None, current_translation_html: str | None = None, glossary: str | None = None) -> tuple[str, dict, str]:
    """
    page_ctx = {
      "page_no": 12,
      "size": {"w": 2480, "h": 3508, "units":"px"},
      "images": [{"ref":"img_p12_1","bbox":[x,y,w,h]}, ...],  # 이미 정규화된 좌표(0,0)~(W,H)
    }
    book_title: The title of the book.
    book_genre: The genre of the book.
    prev_page_html: The translated HTML content of the previous page for context.
    user_feedback: Specific feedback from the user for re-translation.
    current_translation_html: The current (flawed) translated HTML for this page.
    glossary: User-provided glossary for consistent translation.
    """
    W = int(page_ctx["size"]["w"])
    H = int(page_ctx["size"]["h"])

    if book_genre:
      genre_principles = get_genre_specific_principles(book_genre)

    # --- SYSTEM: 역할/좌표계/반환형식 계약을 강하게 명시 ---
    SYSTEM_PROMPT = f"""
You are **Weaver AI**, a document layout + translation compositor.
Your job: read the attached page, translate it to the requested language, and output a **single JSON object**
containing an HTML stage snippet plus a figures manifest.

{genre_principles}

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
   - **CRITICAL for editing**: Assign a unique ID to every major text block like `<p>`, `<h1>-<h6>`, `<li>`, `<blockquote>`, and `<figcaption>`. Use the format `id="el-1"`, `id="el-2"`, etc., sequentially through the document.
5) For body text, do NOT set fixed px/rem sizes. Inherit wrapper base.
6) Margins/padding: use multiples of var(--font-base), e.g. style="margin: calc(var(--font-base)*1.2) 0 .4em;"
7) 문단은 실제 원본의 들여쓰기/줄간격/여백을 반영해라. 
8) paragraph의 첫줄을 들여써야 할 경우 text-indent 사용.(Do NOT use margin)
9) For each figure:
  - In HTML, insert an <img> placeholder with **data-ref="<ref>"** and width,height attributes. (do not set src).
  - The server will set the actual src using your "figures" manifest.
  - figure는 이미지 크기를 원본 비율에 맞춰 지정하고, 캡션이 있는 경우 원본에 충실히 재현, .
10) Never output <html>, <head>, or <body>. Stage-only snippet.
11) 리스트/각주/출처는 본문보다 작은 글자와 더 촘촘한 줄간격으로 한다.
12) 절대 위치 지정은 금지. 대신 여백/들여쓰기/정렬로 원본 레이아웃을 “느낌”으로 재현.

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
        "book_context": {
            "title": book_title or "Unknown",
            "genre": book_genre or "Unknown"
        },
        "previous_page_context": {
            "note": "This is the translated HTML from the previous page (page N-1). Use it as a reference to maintain consistency in tone, style, and terminology. **Crucially, if page N-1 ended mid-sentence, your translation for the current page (page N) must seamlessly continue that sentence.**",
            "html_content": prev_page_html or "This is the first page, so no previous context is available."
        },
        "total_pages": total_pages,
        "target_lang": target_lang,
        "schema_version": "weaver.page.v2",
        "coordinate_space": {"width": W, "height": H, "units": "px", "origin": "top-left"},
        "target_page": page_ctx["page_no"],
        # 서버가 이미 알고 있는 이미지 자원(정규화된 bbox)을 힌트로 제공
        "figures_available": page_ctx.get("images", []),
        "instructions": [
            "**Sentence Continuation (Page Start):** The previous page may have ended mid-sentence. Examine `previous_page_context.html_content` and the start of the current page image. If the current page begins with a sentence fragment, your translation must seamlessly continue the sentence from the previous page.",
            "**Sentence Ending (Page End):** Similarly, examine the last sentence on the current page image. If it seems incomplete and continues to the next page (page N+1), you MUST look ahead at the next page in the document. End the translation for the current page (N) at a natural breaking point. DO NOT artificially complete the sentence; leave it open to be continued on the next page.",
            "CRITICAL: Refer to 'previous_page_context' to ensure consistent translation tone and style (e.g., formal/informal speech) with the preceding page.",
            f"The book's title is '{book_title}' and its genre is '{book_genre}'. Use this context to inform the tone and vocabulary of the translation.",
            f"Translate all text into {target_lang}. Do not translate numbers, symbols, or code syntax.",
            f"You are processing page {page_ctx['page_no']} out of {total_pages}. Output must contain ONLY content visible on page {page_ctx['page_no']}.",
            "Maintain a visually similar layout to the original, but ensure readability with proper headings, spacing, and lists.",
            "For each figure: add <img data-ref='...' width='...' height='...' /> (no src) in HTML and a corresponding entry in 'figures'.",
            "If you infer a figure not listed in 'figures_available', add both placeholder and manifest with an estimated bbox.",
            "Ensure all bbox values are integers and strictly inside the page rectangle.",
            "Your output must be a single JSON object. DO NOT add any extra text or explanations."
        ]
    }

    if user_feedback:
        user['retranslation_feedback'] = {
            "note": "This is a re-translation request. The previous attempt had issues. Pay close attention to the user's feedback below to correct the output.",
            "user_feedback": user_feedback,
            "previous_flawed_translation": current_translation_html or "No previous translation available for this page."
        }
        user['instructions'].insert(0, "**Re-translation Request:** You are correcting a previous translation. Strictly follow the user's feedback provided in `retranslation_feedback` and refer to the `previous_flawed_translation` to understand what to fix.")

    if glossary:
        user['glossary'] = {
            "note": "This is a user-provided glossary. You MUST strictly follow these rules for terminology and style.",
            "rules": glossary
        }
        user['instructions'].insert(0, "**Glossary Adherence:** A user-provided glossary is available. You MUST strictly follow the rules defined in the `glossary` section for all translations.")


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
  <h2 id="el-1" style="font-size: calc(var(--font-base) * 1.35); margin: calc(var(--font-base)*1.2) 0 .4em;">
    토요일 어느 시점에 경찰은 살에게 연락했다.
  </h2>

  <p id="el-2">경찰이 클로이와 엠마에게 금요일 밤이나 토요일 아침에 연락했을 것이라고 추정한다…</p>

  <blockquote id="el-3" style="margin: var(--font-base) 0; padding-left: 1.25em; border-left: 4px solid #d1d5db; font-style: italic;">
    DI 리처드 호킨스는… <sup>6</sup>
  </blockquote>

  <figure style="margin: calc(var(--font-base)*1.2) auto;">
    <img data-ref="img_p6_1" style="display:block; width:800px; height:600px; border-radius:.375rem;">
    <figcaption id="el-4" style="text-align:center; color:#6b7280; font-size:.9em; margin-top:.35em;">
      그림 1. 사건 현장 스케치
    </figcaption>
  </figure>

  <ul style="margin-top: calc(var(--font-base)*1.2);">
    <li id="el-5">…</li>
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

def get_genre_specific_principles(genre: str) -> str:
    LITERARY_PRINCIPLES = """
### Literary Translation Principles
Your primary goal is to produce a translation that reads as if it were **originally written in the target language**. Avoid stiff, literal translations (어색한 번역체).

1.  **Natural Flow over Literal Accuracy:** Do not translate word-for-word. Rephrase sentences to make them sound natural and fluid to a native speaker of the target language, capturing the author's prose style.
2.  **Character Voice Consistency:** Pay close attention to dialogue. Translate it to reflect the speaker's personality, age, and social context. A child's speech must sound like a child's. A formal character's speech must sound formal.
3.  **Handling Proper Nouns & Nicknames:**
    - Standard proper nouns (names, places) should be transliterated consistently.
    - **Crucially, unique nicknames or pet names (e.g., 'hippo pippo') should be transliterated (e.g., '히포 피포'), NOT translated into their literal components (e.g., '하마 피포').** Treat them as special proper nouns that define character relationships.
4.  **Cultural Nuances and Idioms:** Do not translate idioms literally. Find an equivalent expression in the target language that carries the same intent and feeling. If no direct equivalent exists, convey the meaning in a natural way.
5.  **Emotional Tone:** Preserve the emotional tone of the original text—be it suspense, romance, humor, or sorrow.
    """.strip()

    EXPOSITORY_PRINCIPLES = """
### Expository Translation Principles
Your primary goal is to convey information and arguments **clearly, accurately, and logically**. The translation must be trustworthy and easy for the reader to understand.

1.  **Clarity and Precision:** Prioritize clear and unambiguous language. Avoid creative flair or poetic language that could obscure the original meaning. The author's intent must be preserved with precision.
2.  **Logical Flow:** Maintain the logical structure of the original text. Ensure that arguments, evidence, and conclusions are connected in the same way.
3.  **Consistent Terminology:** Key concepts and specialized terms must be translated consistently throughout the entire document. If a standard translated term exists in the field, use it.
4.  **Objective Tone:** Maintain the objective and formal tone typical of non-fiction. Translate the author's voice faithfully, whether it is instructional, analytical, or persuasive.
5.  **Citations and References:** Footnotes, endnotes, and citations must be handled carefully and formatted correctly, preserving their link to the original text.
    """.strip()

    TECHNICAL_PRINCIPLES = """
### Technical Translation Principles
Your absolute highest priority is **terminological and factual accuracy**. The translation must be precise and adhere to industry-standard conventions. Natural "flow" is secondary to technical correctness.

1.  **Unyielding Terminological Accuracy:** All technical terms, scientific nomenclature, and industry-specific jargon MUST be translated to their officially accepted or most widely used equivalent in the target language. **When in doubt, prefer a direct, consistent translation over a more "natural" sounding but less precise alternative.**
2.  **Strict Consistency:** A specific technical term must be translated into the exact same word or phrase every single time it appears. There is no room for stylistic variation.
3.  **Do Not Translate Code/Formulas:** Code snippets, mathematical equations, chemical formulas, and file paths should remain in their original form. Only translate the descriptive text surrounding them.
4.  **Impersonal and Objective Tone:** The language must be direct, unambiguous, and devoid of any creative or emotional coloring. The goal is to eliminate any possibility of misinterpretation.
5.  **Preserve Hierarchical Structure:** Headings, subheadings, lists, and tables must maintain their original hierarchical structure to preserve the document's logical flow.
    """.strip()

    if genre in ["fiction", "fantasy", "science-fiction", "romance", "thriller", "children"]:
        return LITERARY_PRINCIPLES
    elif genre in ["non-fiction", "history", "self-help"]:
        return EXPOSITORY_PRINCIPLES
    elif genre in ["technical"]:
        return TECHNICAL_PRINCIPLES
    else:
        # 기본값 또는 일반적인 번역 원칙
        return """
### General Translation Principles
- Translate all textual content accurately into the target language.
- Maintain a consistent tone and style.
- Keep the layout structure visually close to the original.
        """.strip()