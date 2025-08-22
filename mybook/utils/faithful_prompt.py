#mybook/utils/faithful_prompt.py
SYSTEM_PROMPT = """
    You are Weaver AI, a document translation and structuring engine.
    Your task is to translate content into the target language and return ONLY JSON.
    The JSON must strictly follow the Weaver Page Schema (weaver.page.v1).
"""


def build_prompt_faithful(page_ctx: dict, target_lang: str):
    """
    page_ctx = {
      "page_no": 12,
      "size": {"w": 2480, "h": 3508, "units":"px"},
      "images": [{"ref":"img_p12_1","bbox":[...]}],
      "ocr_text": "..."  # 선택
    }
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
            "4. Translate all text into Korean. Do not translate numbers, symbols, or code syntax.",
            "5. Do NOT generate new images. Use provided image_ref values for figures.",
            "6. Tables must use <table> markup in content.html (no CSS, no style attributes).",
            "7. For footnotes, include label and text, and link back using ref_ids if provided.",
            "8. Confidence: 0.9+ if very sure, 0.6~0.8 if somewhat sure, <0.5 if unsure.",
            "9. Do NOT include any additional text or explanations outside the JSON.",
            "Return ONLY the JSON object, nothing else."
        ]
    }

    example_answer = SAMPLE_ANSWER

    return sys, user, example_answer


SAMPLE_ANSWER = """
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