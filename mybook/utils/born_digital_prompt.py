# mybook/utils/born_digital_prompt.py
import json
from typing import List
from typing import Dict, Any

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

    if genre and genre.lower() in ["fiction", "fantasy", "science-fiction", "romance", "thriller", "children"]:
        return LITERARY_PRINCIPLES
    elif genre and genre.lower() in ["non-fiction", "history", "self-help"]:
        return EXPOSITORY_PRINCIPLES
    elif genre and genre.lower() in ["technical"]:
        return TECHNICAL_PRINCIPLES
    else:
        # 기본값 또는 일반적인 번역 원칙
        return """
### General Translation Principles
- Translate all textual content accurately into the target language.
- Maintain a consistent tone and style.
        """.strip()

def build_prompt_born_digital(
    spans: List[Dict[str, Any]],
    target_lang: str,
    total_pages: int,
    page_no: int,
    book_title: str | None = None,
    book_genre: str | None = None,
    glossary: str | None = None,
    previous_page_body_text: str | None = None,
    next_page_span_texts: List[str] | None = None,
) -> tuple[str, str]:
    """
    Builds a prompt for layout analysis and translation of born-digital PDFs in a single call.
    """
    genre_principles = ""
    if book_genre:
        genre_principles = get_genre_specific_principles(book_genre)

    # --- SYSTEM PROMPT ---
    system_prompt = f"""You are an expert document analyzer and translator. Your task is to analyze the layout of a PDF page from its text spans, group them into paragraphs, and translate each paragraph into {target_lang}.

{genre_principles}

### Cross-Page Context Rules
- **Tone & Style**: Refer to `previous_page_body_text` to ensure the translation tone and style (e.g., formal/informal speech) are consistent with the preceding page.
- **Sentence Continuity**: If the first span of the current page seems to continue a sentence from the previous page, use `previous_page_body_text` to create a seamless translation. Similarly, if the last span of the current page seems incomplete, use `next_page_span_texts` to decide where to naturally break the sentence.

### Core Task
1.  **Group Spans**: Analyze the provided `spans` list. Group related spans into logical paragraphs. For each paragraph, you MUST include the `span_indices` array containing the original `idx` of each span. **Crucially, treat page footers and page numbers as separate paragraphs.** For example, if a line contains "Document Title" and "Page 5", create one paragraph for "Document Title" with role `footer`, and another for "Page 5" with role `pagination`.
2. **List Items**:
   - Each list item must be its own paragraph (`role="list_item"`).
   - If the bullet/number symbol is given as a separate span, keep it as a separate paragraph. Do not merge it with the text.
   - If the bullet/number symbol is already combined with the text in a single span, keep it as-is. Do not split it.
3.  **Analyze Layout**: For each paragraph, determine its `role`, `alignment`, and `flow`. They MUST be one of the allowed `enum` values and in **lowercase**.
    - **Role**: `title`|`heading`|`body`|`list_item`|`caption`|`footer`|`pagination`|`table_data`.
    - **Alignment**: `left` | `right` | `center` | `justify`. (For `body` paragraphs, default to `justify`. Use `left` only if ragged-right alignment is clearly visible.)
    - **Flow**: "left_to_right" | "right_to_left" | "top_to_bottom". If uncertain, use "left_to_right".
    - Do NOT invent new values.
4.  **Translate with Style Markers**: internally join the text of its spans and translate this combined text into {target_lang} to create the `translated_text`.
    - **CRITICAL**: If a part of the text has special styling (bold, italic, or a different color), you MUST wrap the translated fragment with special markers.
    - For **bold** text, use: `§b§`translated bold text`§/b§`
    - For **italic** text, use: `§i§`translated italic text`§/i§`
    - For text with a **different color**, use: `§c§`translated colored text`§/c§`
    - Example: "This is a §b§very important§/b§ concept."
    - **Nesting is allowed for combined styles**: e.g., `§b§§i§bold and italic text§/i§§/b§`.
5.  **Output JSON**: Your final output MUST be a single, valid JSON **array** of paragraph objects, strictly conforming to the provided schema. Do not include the `original text` field. Do not add any extra text, explanations, or markdown formatting.

### Role Disambiguation (title vs heading)
- **title**: Reserved for a **book title page, part title page, or a unique page-level title** that represents the entire book/part. On a normal content page, it is **rare** and typically **absent**.
- **heading**: Use for **chapter/section/subsection titles** within the main content flow, including strings like **"Chapter 1. The boy who lived"**. Also use `heading` for running headers at the very top of a page (e.g., repeating book title).
- **Heuristic**:
  - If the text indicates a **chapter/section** (e.g., "Chapter 1", "1. Introduction") or appears as a recurring header, choose `heading`.
  - Only choose `title` when the text **names the whole book/part** and is likely **unique on the page** (e.g., cover/title page).
  - In case of conflict, prefer `heading` unless there is strong evidence of a unique title page.

    """.strip()

    # --- USER PROMPT (the JSON data itself) ---
    # We only need to send essential information for each span.
    span_data = [
        {
            "idx": i,
            "text": span.get("text", ""),
            "bbox": span.get("bbox", (0,0,0,0)),
            "font": span.get("font", ""),
            "size": span.get("size", 0.0),
            "color": span.get("color", 0),
        }
        for i, span in enumerate(spans)
    ]

    payload: Dict[str, Any] = {"spans": span_data}

    # 이전/다음 페이지 컨텍스트 추가
    if previous_page_body_text:
        payload["previous_page_body_text"] = {
            "note": "This is the translated text from the last few body paragraphs of the previous page. Use it for context.",
            "text": previous_page_body_text
        }
    if next_page_span_texts:
        # 다음 페이지의 첫 7개 스팬 텍스트만 컨텍스트로 제공
        payload["next_page_span_texts"] = {
            "note": "These are the first few text spans from the next page. Use them to decide how to end sentences on the current page.",
            "texts": next_page_span_texts
        }

    if glossary:
        payload["glossary"] = glossary
    if book_title or book_genre:
        payload["book_metadata"] = {
            "book_title": book_title or "Unknown",
            "book_genre": book_genre or "Unknown",
            "current_page": page_no,
            "total_pages": total_pages,
        }

    user_prompt_json = json.dumps(payload, ensure_ascii=False)

    return system_prompt, user_prompt_json

def build_prompt_retranslate_born_digital(
    paragraphs: List[Dict[str, Any]],
    target_lang: str,
    user_feedback: str,
    book_title: str | None = None,
    book_genre: str | None = None,
    glossary: str | None = None,
) -> tuple[str, str]:
    """
    Builds a prompt for re-translating paragraphs based on user feedback.
    The layout is already determined. The task is to improve the translation.
    """
    genre_principles = ""
    if book_genre:
        genre_principles = get_genre_specific_principles(book_genre)

    # --- SYSTEM PROMPT ---
    system_prompt = f"""You are an expert translator. Your task is to re-translate a list of paragraphs into {target_lang}, strictly following the user's feedback.

{genre_principles}

### Core Task
1.  **Analyze Feedback**: Carefully read the `user_feedback` and understand what needs to be corrected in the `paragraphs_to_retranslate`.
2.  **Re-translate**: Provide a new, improved translation for each paragraph in the list.
3.  **Output JSON**: Your output MUST be a JSON object with a single key "translations", which contains an array of the newly translated strings. The array must have the exact same number of elements as the input array.
    """.strip()

    # --- USER PROMPT ---
    # The user prompt contains the feedback and the original paragraphs with their flawed translations.
    payload = {
        "task_context": {
            "book_title": book_title or "Unknown",
            "book_genre": book_genre or "Unknown",
            "target_language": target_lang,
        },
        "user_feedback": user_feedback,
        "paragraphs_to_retranslate": [
            {"role": p.get("role"), "original_text": p.get("original_text"), "current_translation": p.get("translated_text")}
            for p in paragraphs
        ]
    }
    if glossary:
        payload["glossary"] = glossary

    user_prompt_json = json.dumps(payload, ensure_ascii=False)
    return system_prompt, user_prompt_json