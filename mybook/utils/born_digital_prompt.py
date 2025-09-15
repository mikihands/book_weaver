# mybook/utils/born_digital_prompt.py
import json
from typing import List

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
    # --- SYSTEM PROMPT ---
    system_prompt = f"""
You are an expert translator. Your task is to translate a list of JSON strings from the source language to {target_lang}.
- Translate the text accurately, maintaining the original tone and context.
- The input is a JSON object containing a list of strings to translate.
- Your output MUST be a JSON array of strings with the exact same number of elements as the input list.
- Each string in the output array must be the translation of the corresponding string in the input list.
- Sometimes a single sentence is split into two or more array elements. In such cases, you must maintain the number of array elements while ensuring the translated sentence flows naturally.
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