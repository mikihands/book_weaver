# mybook/utils/born_digital_prompt.py
import json
from typing import List

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
    units: List[str],
    target_lang: str,
    book_title: str | None = None,
    book_genre: str | None = None,
    glossary: str | None = None,
    user_feedback: str | None = None,
    current_translation: List[str] | None = None
) -> tuple[str, str]:
    """
    Builds a prompt for translating a list of text units for born-digital PDFs.
    The LLM's role is purely translation.
    """
    genre_principles = ""
    if book_genre:
        genre_principles = get_genre_specific_principles(book_genre)

    # --- SYSTEM PROMPT ---
    system_prompt = f"""You are an expert translator specializing in document translation. Your task is to translate a list of text blocks (paragraphs, headings, etc.) into {target_lang}.

{genre_principles}

### Core Rules
- Your output MUST be a JSON array of strings.
- The output array MUST have the exact same number of elements as the input `units_to_translate` array.
- Each string in the output array must be the translation of the corresponding string in the input list.
- Each input string is a self-contained text block (like a paragraph or heading). Translate it as a whole, coherent unit.
- In the rare case a single sentence is split across multiple input strings (due to document formatting), your translation should still connect them smoothly while maintaining the original array structure.
- Your output must conform to the provided JSON schema, which expects a simple array of strings.
- Do not add any extra text, explanations, or code fences around the JSON output. Return ONLY the JSON array.
    """.strip()

    # --- USER PROMPT (the JSON data itself) ---
    context = {
        "task": f"Translate the following text units to {target_lang}.",
        "book_context": {
            "title": book_title or "Unknown",
            "genre": book_genre or "Unknown"
        },
    }
    if glossary:
        context["glossary"] = {
            "note": "You MUST strictly follow these rules for terminology and style.",
            "rules": glossary
        }

    if user_feedback:
        context['retranslation_request'] = {
            "note": "This is a re-translation request. The previous attempt had issues. Pay close attention to the user's feedback to correct the output. The user is providing feedback on the rendered HTML, but you should correct the underlying translated text units.",
            "user_feedback": user_feedback,
            "previous_flawed_translation_units": current_translation or "Not available."
        }

    payload = {"context": context, "units_to_translate": units}
    user_prompt_json = json.dumps(payload, ensure_ascii=False)

    return system_prompt, user_prompt_json