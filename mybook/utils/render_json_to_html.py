import os
from django.conf import settings

# JSON 데이터를 HTML로 변환하는 렌더러 함수
def render_json_to_html(page_json, image_info):
    html_content = ""
    for block in page_json.get("blocks", []):
        block_type = block.get("type")
        content = block.get("content")
        bbox = block.get("bbox")
        
        # 'style' 속성을 사용하여 bbox 기반 위치 지정 (정밀 모드)
        style = f"position: absolute; left: {bbox[0]}px; top: {bbox[1]}px; width: {bbox[2]}px; height: {bbox[3]}px;"

        if block_type == "heading":
            level = block.get("attrs", {}).get("level", 1)
            html_content += f'<h{level} style="{style}">{content.get("text")}</h{level}>'
        elif block_type == "paragraph":
            html_content += f'<p style="{style}">{content.get("text")}</p>'
        elif block_type == "list":
            list_type = content.get("list_type")
            items_html = ""
            for item in content.get("items", []):
                item_text = item.get("content", {}).get("text", "")
                items_html += f"<li>{item_text}</li>"
            html_content += f'<{list_type} style="{style}">{items_html}</{list_type}>'
        elif block_type == "table":
            html_content += f'<div style="{style}">{content.get("html")}</div>'
        elif block_type == "figure":
            image_ref = content.get("image_ref")
            # 이미지 정보를 통해 이미지 경로 찾기
            image_path = None
            for img in image_info:
                if img['ref'] == image_ref:
                    # Django의 MEDIA_URL을 사용하여 경로 구성
                    image_path = os.path.join(settings.MEDIA_URL, os.path.relpath(img['path'], settings.MEDIA_ROOT))
                    break
            if image_path:
                html_content += f'<img src="{image_path}" alt="{content.get("caption", "")}" style="{style}" />'
        elif block_type == "code":
            html_content += f'<pre style="{style}"><code class="language-{content.get("language")}">{content.get("text")}</code></pre>'
        
        # TODO: 다른 블록 타입(equation, quote, separator 등)에 대한 렌더링 로직 추가
    
    return html_content