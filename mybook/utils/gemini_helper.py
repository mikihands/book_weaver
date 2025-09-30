# mybook/utils/gemini_helper.py
import json
import re
import logging
import os, time, inspect
from typing import Optional, Dict, Any, List, Tuple

from django.conf import settings
from google import genai
from google.genai import types
from jsonschema import Draft202012Validator
from PIL import Image

logger = logging.getLogger(__name__)

GEMINI_API_KEY = settings.GEMINI_API_KEY
MODEL_NAME = "gemini-2.5-flash-lite"
#MODEL_NAME = "gemini-2.5-flash"

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
    - '정밀 모드 JSON' 생성을 위한 system/user 메시지 + 예시/피드백 파트 관리 (faithful mode)
    - jsonschema 검증 + 간단한 자동 재시도
    """

    _THINKING_BUDGETS = {
        'standard': {'off': 0, 'medium': 7000, 'deep': 15000},
        'lite': {'off': 0, 'medium': 2500, 'deep': 5000},
    }

    def __init__(self, schema: Dict[str, Any], model_type: str = 'standard', thinking_level: str = 'medium'):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.schema = schema
        self.validator = Draft202012Validator(schema)
        self.model_type = model_type.lower()
        self.thinking_level = thinking_level.lower()

        if self.model_type not in self._THINKING_BUDGETS:
            logger.warning(f"Invalid model_type '{model_type}'. Defaulting to 'standard'.")
            self.model_type = 'standard'
        
        if self.thinking_level not in self._THINKING_BUDGETS[self.model_type]:
            logger.warning(f"Invalid thinking_level '{thinking_level}' for model '{self.model_type}'. Defaulting to 'medium'.")
            self.thinking_level = 'medium'

    @property
    def model_name(self) -> str:
        # 'lite' 모델은 'gemini-2.5-flash-lite'를 사용하고, 그 외(standard 포함)는 'gemini-2.5-flash'를 사용합니다.
        if self.model_type == 'lite':
            return 'gemini-2.5-flash-lite'
        return 'gemini-2.5-flash'

    def _extract_json(self, text: str) -> str:
        """
        텍스트에서 첫 JSON(객체/배열)을 뽑고,
        translated_text 값 내부의 미이스케이프 따옴표를 고친다.
        """
        # JSON 블록(객체/배열) 시작 위치 탐색
        start_obj = text.find('{')
        start_arr = text.find('[')
        if start_obj == -1 and start_arr == -1:
            return text

        s = min(x for x in (start_obj, start_arr) if x != -1)
        open_ch = text[s]
        close_ch = '}' if open_ch == '{' else ']'

        # 괄호 짝 맞춰서 정확한 끝 위치 찾기(스택 방식)
        depth = 0
        e = None
        for i, ch in enumerate(text[s:], s):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    e = i
                    break
        if e is None:
            return text

        json_string = text[s:e+1]

        QUOTED_FIELD = re.compile(
            r'("translated_text"\s*:\s*")'      # group 1
            r'(.*?)'                             # group 2: content (may contain ")
            r'("(?=\s*[,}]))',                   # group 3: the real closing quote
            re.DOTALL
        )

        def _fix_inner_quotes(m: re.Match) -> str:
            content = m.group(2)
            # 이미 이스케이프된 \" 는 그대로 두고, 미이스케이프 " 만 \" 로
            content = re.sub(r'(?<!\\)"', r'\\"', content)
            return m.group(1) + content + m.group(3)

        fixed = QUOTED_FIELD.sub(_fix_inner_quotes, json_string)
        return fixed

    def _validate(self, data: Dict[str, Any]) -> List[str]:
        errs = sorted(self.validator.iter_errors(data), key=lambda e: e.path)
        return [f"path={list(e.path)}: {e.message}" for e in errs]

    def _normalize_gemini_roles(self, paragraphs: list[dict]) -> list[dict]:
        """
        Normalizes the 'role' field in Gemini's response using a keyword-based approach
        to handle a wide range of variations and improve schema validation robustness.
        """
        for para in paragraphs:
            if 'role' not in para or not isinstance(para['role'], str):
                continue
            role_norm = para['role'].lower().replace(' ', '_').replace('-', '_')
            if role_norm.isdigit() or 'pag' in role_norm:
                para['role'] = 'pagination'
            elif 'foot' in role_norm:
                para['role'] = 'footer'
            elif 'list' in role_norm:
                para['role'] = 'list_item'
            elif 'capt' in role_norm:
                para['role'] = 'caption'
            elif 'sub_title' in role_norm or 'sub' in role_norm:
                para['role'] = 'subtitle'
            elif 'title' in role_norm:
                para['role'] = 'title'
            elif 'head' in role_norm:
                para['role'] = 'heading'
            elif 'table' in role_norm:
                para['role'] = 'table_data'
            elif 'body' in role_norm or 'para' in role_norm:
                para['role'] = 'body'
            else:
                para['role'] = role_norm
        return paragraphs

    def _normalize_font_changes(self, paragraphs: list[dict]) -> list[dict]:
        """
        Clears the 'font_changes' for paragraphs that consist of a single span.
        This is a safeguard, as the server can determine the style of a single span directly.
        """
        for para in paragraphs:
            if 'span_indices' in para and len(para['span_indices']) == 1:
                para['font_changes'] = []
        return paragraphs

    def _normalize_response_data(self, data: Any) -> Any:
        if isinstance(data, list):
            data = self._normalize_gemini_roles(data)
            data = self._normalize_font_changes(data)
        return data

    def generate_page_json_with_raw_response(
        self,
        file_part,
        sys_msg: str,
        user_msg: Dict[str, Any],
        example_json: Optional[str] = None,
        max_retries: int = 2,
        debug_callback: Optional[callable] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[List[str]], Optional[types.UsageMetadata]]:
        contents = [file_part, json.dumps(user_msg, ensure_ascii=False)]
        if example_json:
            contents.append("Valid example (do not copy values):\n" + example_json)

        last_errs = None
        raw_text = ""
        usage_metadata = None
        for attempt in range(max_retries + 1):
            logger.info(f"GeminiHelper: Starting attempt {attempt + 1}/{max_retries + 1}")
            try:
                # Set thinking budget based on selected model_type and thinking_level
                thinking_budget = self._THINKING_BUDGETS[self.model_type][self.thinking_level]
                config_params = {"system_instruction": sys_msg, "response_mime_type": "application/json", "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget)}

                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_params)
                )
                raw_text = resp.text or ""
                usage_metadata = resp.usage_metadata
                logger.debug(f"GeminiHelper: Attempt {attempt + 1} - Raw response received.")
                if debug_callback: debug_callback(attempt, raw_text, False)

                raw_json_str = self._extract_json(raw_text)
                data = json.loads(raw_json_str)
                if debug_callback: debug_callback(attempt, json.dumps(data, ensure_ascii=False, indent=2), True)
            except Exception as e:
                logger.warning(f"GeminiHelper: Attempt {attempt + 1} - API call or JSON parsing failed: {e}")
                last_errs = [f"Attempt {attempt+1}: API call or JSON parsing failed: {e}"]
                contents.append(f"Error: {e}. Return ONLY valid JSON that matches the schema.")
                continue

            # **IMPORTANT**: Normalize data BEFORE validation.
            data = self._normalize_response_data(data)
            logger.debug(f"GeminiHelper: Attempt {attempt + 1} - Data normalized.")

            errs = self._validate(data)
            if not errs:
                logger.info(f"GeminiHelper: Attempt {attempt + 1} - Validation successful.")
                return raw_text, data, None, usage_metadata # type: ignore

            logger.warning(f"GeminiHelper: Attempt {attempt + 1} - Schema validation failed. Errors: {errs[:3]}")
            last_errs = errs
            contents.append("Schema validation errors:\n" + "\n".join(errs[:10]))

        logger.error(f"GeminiHelper: All {max_retries + 1} attempts failed. Last errors: {last_errs}")
        return raw_text, None, last_errs, usage_metadata # type: ignore

    def generate_page_json(
        self,
        file_part,                      # ✅ inline Part 또는 files.upload/get 반환 객체
        sys_msg: str,
        user_msg: Dict[str, Any],
        example_json: Optional[str] = None,
        max_retries: int = 2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]], Optional[types.UsageMetadata]]:
        _raw_text, data, errors, usage_metadata = self.generate_page_json_with_raw_response(
            file_part, sys_msg, user_msg, example_json, max_retries
        )
        return data, errors, usage_metadata

    def translate_text_units(
        self,
        sys_msg: str,
        user_msg_json: str,
        max_retries: int = 2,
        expected_length: Optional[int] = None,
    ) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[types.UsageMetadata]]:
        """
        Calls Gemini to translate a list of text units.
        Expects a JSON array of strings as a response, enforced by a schema.
        """
        schema_to_use = self.schema
        if expected_length is not None:
            # Create a copy to avoid modifying the instance's schema object.
            schema_to_use = self.schema.copy()
            schema_to_use["minItems"] = expected_length
            schema_to_use["maxItems"] = expected_length

        contents = [user_msg_json]
        last_errs = None
        usage_metadata = None
        for attempt in range(max_retries + 1):
            raw_text = ""
            try:
                # Set thinking budget based on selected model_type and thinking_level
                thinking_budget = self._THINKING_BUDGETS[self.model_type][self.thinking_level]
                config_params = {"system_instruction": sys_msg, "response_mime_type": "application/json", "response_json_schema": schema_to_use, "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget)} # type: ignore

                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,# type: ignore
                    config=types.GenerateContentConfig(**config_params)
                )
                raw_text = resp.text or ""
                usage_metadata = resp.usage_metadata
                _dump_debug(raw_text, "born_digital_resp_text", attempt)
                data = json.loads(raw_text)
                # The schema should guarantee a list, but we can double-check.
                if not isinstance(data, list):
                    raise ValueError(f"Response was not a JSON array as expected, despite schema. Got type: {type(data)}")
                _dump_debug(data, "born_digital_parsed_json", attempt)
                return data, None, usage_metadata # type: ignore
            except Exception as e:
                _dump_debug(raw_text or str(e), "born_digital_error_raw", attempt)
                last_errs = [f"Attempt {attempt+1} failed: {e}"]
                contents.append(f"Error: {e}. Please ensure the output is a valid JSON array of strings matching the schema.")
                continue
        return None, last_errs, usage_metadata # type: ignore

    def detect_figures(self, image: Image.Image, labels: Optional[List[str]] = None) -> Tuple[Optional[List[Dict]], Optional[List[str]], Optional[types.UsageMetadata]]:
        """
        Detects prominent items in an image and returns their bounding boxes.
        This method is specialized for object detection.
        """
        prompt = """
        Detect all prominent items in the image. Do not include text.

        - **CRITICAL RULE**: If one detected item is completely inside another (e.g., an icon inside a map), you MUST only return the bounding box for the larger, containing item. Do not return nested items.
        - The response must be a JSON array where each object has a 'label' and a 'box_2d'.
        - The 'box_2d' must be `[ymin, xmin, ymax, xmax]` normalized to 0-1000.
        """
        if labels:
            prompt += f"\n- Focus on detecting these specific items if possible: {', '.join(labels)}"

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0)
            )

        try:
            response = self.client.models.generate_content( # type: ignore
                model=self.model_name,
                contents=[image, prompt],
                config=config
            )
            raw_text = response.text
            usage_metadata = response.usage_metadata
            
            # Robust JSON extraction from markdown code fences
            json_start = raw_text.find('[')
            if json_start == -1:
                json_start = raw_text.find('{')
            json_end = raw_text.rfind(']')
            if json_end == -1:
                json_end = raw_text.rfind('}')
            
            if json_start != -1 and json_end != -1:
                json_str = raw_text[json_start:json_end+1]
                bounding_boxes = json.loads(json_str)
                if isinstance(bounding_boxes, dict): # Handle single object case
                    bounding_boxes = [bounding_boxes]
                logger.debug(f"[Gemini-helper-detect_figures]bbox: {bounding_boxes}")
                return bounding_boxes, None, usage_metadata # type: ignore
            else:
                return None, [f"Could not find valid JSON in response: {raw_text}"], usage_metadata # type: ignore
        except Exception as e:
            return None, [f"Gemini figure detection API call failed: {e}"], None

def clamp_bbox(b, W, H):
    x,y,w,h = b
    x = max(0, min(x, W))
    y = max(0, min(y, H))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]