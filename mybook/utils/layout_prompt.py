# mybook/utils/layout_prompt.py
from typing import Any, Dict, List

SYS_MSG = """
You are an expert document layout analyzer. Your task is to analyze a PDF page image and its corresponding text spans to determine the paragraph structure and layout.

You will be given:
1. An image of a single PDF page.
2. A JSON object containing a list of text spans extracted from that page. Each span has an `idx`, `text`, and `bbox`.

Your goal is to group these spans into logical paragraphs and describe their properties. The visual layout in the PDF image is the most important source of truth. Use it to understand groupings, alignments, and roles that are not obvious from the text data alone.

You MUST respond with a single JSON object that strictly conforms to the provided JSON schema. Do not add any extra text, explanations, or markdown formatting around the JSON.

Key tasks:
- **Paragraph Grouping**: Identify which `span_indices` belong together to form a single paragraph. A paragraph can be a title, a block of body text, a caption, a list item, etc.
- **Role Identification**: For each paragraph, determine its semantic role (`title`, `subtitle`, `body`, `caption`, `header`, `footer`, `pagination`, etc.).
- **Layout Analysis**: Determine the `alignment` (`left`, `center`, `right`, `justify`) and text `flow` for each paragraph.
- **Font Style Detection**: Identify any text within a paragraph that has special styling like `bold` or `italic`. The `font_changes` array should contain the exact text and the styles applied.
"""

def create_layout_user_prompt(book_id: str, page_no: int, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates the user message payload for the Gemini layout analysis request.
    """
    # We only need to send essential information for each span.
    # The model will use the visual PDF for most layout cues.
    span_data = [
        {
            "idx": i,
            "text": span.get("text", ""),
            "bbox": span.get("bbox", (0,0,0,0))
        }
        for i, span in enumerate(spans)
    ]

    return {
        "task": "Analyze the layout of the provided page and its text spans. Group spans into paragraphs and identify their roles and styles based on the visual layout.",
        "book_id": book_id,
        "page_no": page_no,
        "spans": span_data
    }
