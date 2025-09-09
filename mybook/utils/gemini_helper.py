# mybook/utils/gemini_helper.py
import json
import logging
import os, time
from typing import Optional, Dict, Any, List, Tuple

from django.conf import settings
from google import genai
from google.genai import types
from jsonschema import Draft202012Validator

logger = logging.getLogger(__name__)

GEMINI_API_KEY = settings.GEMINI_API_KEY
MODEL_NAME = "gemini-2.5-flash"

def _dump_debug(payload: dict | str | list, prefix: str, attempt: int):
    debug_dir = os.path.join(settings.BASE_DIR, "media", "gemini_debug")
    os.makedirs(debug_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    is_json_serializable = isinstance(payload, (dict, list))
    path = os.path.join(debug_dir, f"{prefix}_a{attempt}_{ts}.json" if is_json_serializable
                        else f"{prefix}_a{attempt}_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        if is_json_serializable:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(payload))
    return path

class GeminiHelper:
    """
    - 파일 크기에 따라 인라인/업로드로 PDF를 전달
    - '정밀 모드 JSON' 생성을 위한 system/user 메시지 + 예시/피드백 파트 관리
    - jsonschema 검증 + 간단한 자동 재시도
    """
    def __init__(self, schema: Dict[str, Any]):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.schema = schema
        self.validator = Draft202012Validator(schema)

    def _extract_json(self, text: str) -> str:
        s, e = text.find("{"), text.rfind("}")
        return text[s:e+1] if s != -1 and e != -1 and e > s else text

    def _validate(self, data: Dict[str, Any]) -> List[str]:
        errs = sorted(self.validator.iter_errors(data), key=lambda e: e.path)
        return [f"path={list(e.path)}: {e.message}" for e in errs]

    def generate_page_json(
        self,
        file_part,                      # ✅ inline Part 또는 files.upload/get 반환 객체
        sys_msg: str,
        user_msg: Dict[str, Any],
        example_json: Optional[str] = None,
        max_retries: int = 2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
        contents = [file_part, json.dumps(user_msg, ensure_ascii=False)]
        if example_json:
            contents.append(
                "Valid example (do not copy values):\n" + example_json
            )

        last_errs = None
        for attempt in range(max_retries + 1):
            raw_text = ""
            try:
                resp = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_msg,
                        response_mime_type="application/json"
                        # response_json_schema=self.schema #아직은 시기상조
                    )
                )
                raw_text = resp.text or ""
                # 원문도 보관
                _dump_debug(raw_text, "resp_text", attempt)
                raw = self._extract_json(raw_text)
                data = json.loads(raw)

                # 파싱된 JSON도 보관
                _dump_debug(data, "parsed_json", attempt)
                
            except Exception as e:
                # parsing/호출 실패 시에도 raw_text를 남김
                _dump_debug(raw_text or str(e), "error_raw", attempt)

                last_errs = [f"parse_error: {e}"]
                contents.append("Parse error. Return ONLY JSON that matches the schema.")
                continue

            errs = self._validate(data)
            if not errs:
                # figures 보정
                W = data["page"]["page_w"]; H = data["page"]["page_h"]
                for f in data.get("figures", []):
                    f["bbox"] = clamp_bbox([f["bbox_x"], f["bbox_y"], f["bbox_w"], f["bbox_h"]], W, H)
                return data, None
            
            _dump_debug({"errors": errs}, "schema_errors", attempt)
            contents.append("Schema validation errors:\n" + "\n".join(errs[:10]))
            last_errs = errs

        return None, last_errs

    def translate_text_units(
        self,
        sys_msg: str,
        user_msg_json: str,
        max_retries: int = 2,
        expected_length: Optional[int] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Calls Gemini to translate a list of text units.
        Expects a JSON array of strings as a response, enforced by a schema.
        """
        # If an expected length is provided, make the schema stricter to enforce
        # the exact number of items in the output array. This is crucial for`spans_to_units` 함수를 개선해서, 줄바꿈으로 나뉜 문장 조각들을 미리 합치는 로직을 추가해줘.
        # preventing "unit count mismatch" errors.
        schema_to_use = self.schema
        if expected_length is not None:
            # Create a copy to avoid modifying the instance's schema object.
            schema_to_use = self.schema.copy()
            schema_to_use["minItems"] = expected_length
            schema_to_use["maxItems"] = expected_length

        contents = [user_msg_json]
        last_errs = None
        for attempt in range(max_retries + 1):
            raw_text = ""
            try:
                resp = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,# type: ignore
                    config=types.GenerateContentConfig(
                        system_instruction=sys_msg,
                        response_mime_type="application/json",
                        response_json_schema=schema_to_use
                    )
                )
                raw_text = resp.text or ""
                _dump_debug(raw_text, "born_digital_resp_text", attempt)
                data = json.loads(raw_text)
                # The schema should guarantee a list, but we can double-check.
                if not isinstance(data, list):
                    raise ValueError(f"Response was not a JSON array as expected, despite schema. Got type: {type(data)}")
                _dump_debug(data, "born_digital_parsed_json", attempt)
                return data, None
            except Exception as e:
                _dump_debug(raw_text or str(e), "born_digital_error_raw", attempt)
                last_errs = [f"Attempt {attempt+1} failed: {e}"]
                contents.append(f"Error: {e}. Please ensure the output is a valid JSON array of strings matching the schema.")
                continue
        return None, last_errs

def clamp_bbox(b, W, H):
    x,y,w,h = b
    x = max(0, min(x, W))
    y = max(0, min(y, H))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]