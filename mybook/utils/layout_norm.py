# mybook/utils/layout_norm.py
from copy import deepcopy
from typing import List, Dict, Any

def normalize_page_json(data: dict, base_width: int = 1200) -> dict:
    """
    data (TranslatedPage.data) 를 표준 너비 기준으로 정규화한 새 dict를 반환.
    - 비율 유지
    - page.size와 blocks[].bbox, resources.images[].bbox 모두 스케일
    - meta.norm 에 scale, original_size 저장
    task 함수에서 DB에 저장할때 사용하려 했으나, 현재는 사용하지 않음. 처음 업로드할때 노멀라이즈하기로 함.
    """
    out = deepcopy(data)
    page = out.get("page", {})
    size = page.get("size", {})
    W = float(size.get("w", 0) or 0)
    H = float(size.get("h", 0) or 0)
    if W <= 0 or H <= 0:
        # 못 믿을 값이면 그냥 패스
        return out

    s = float(base_width) / W
    newW = float(base_width)
    newH = round(H * s, 2)

    # 페이지 사이즈 업데이트
    page["size"]["w"] = newW
    page["size"]["h"] = newH
    page["size"]["units"] = "px"  # 명시

    # 블록 bbox 스케일
    for b in out.get("blocks", []):
        bbox = b.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x,y,w,h = bbox
            b["bbox"] = [round(x*s,2), round(y*s,2), round(w*s,2), round(h*s,2)]

    # 이미지 리소스 bbox 스케일
    res = out.get("resources", {})
    for im in res.get("images", []) or []:
        bbox = im.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x,y,w,h = bbox
            im["bbox"] = [round(x*s,2), round(y*s,2), round(w*s,2), round(h*s,2)]

    # 메타 기록
    meta = out.setdefault("meta", {})
    meta.setdefault("norm", {})
    meta["norm"]["base_width"] = base_width
    meta["norm"]["scale"] = round(s, 6)
    meta["norm"]["original_size"] = {"w": W, "h": H, "units": size.get("units", "px")}

    return out



def normalize_pages_layout(pages: List[Dict[str, Any]], base_width: int = 1200) -> List[Dict[str, Any]]:
    """
    extract_images_and_bboxes() 결과를 정규화.
    pages: [{ "page_no": int, "size": {"w": float, "h": float}, "images":[{"ref","path","bbox":[x,y,w,h]}] }, ...]
    반환: 같은 구조이되 size/bbox가 base_width 기준으로 스케일된 값.
    각 page에 meta.norm(원본 크기, scale)도 같이 붙여줌(필요시 DB에 보관).
    """
    out: List[Dict[str, Any]] = []
    for p in pages:
        W = float(p["size"]["w"])
        H = float(p["size"]["h"])
        if W <= 0 or H <= 0:
            out.append(p); continue

        s = float(base_width) / W
        newW = float(base_width)
        newH = round(H * s, 2)

        # 페이지 사이즈
        new_page = {
            "page_no": p["page_no"],
            "size": {"w": newW, "h": newH, "units": "px"},
            "images": []
        }

        # 이미지 bbox 스케일
        for im in p.get("images", []):
            x, y, w, h = im["bbox"]
            new_page["images"].append({
                "ref": im["ref"],
                "path": im["path"],
                "bbox": [round(x*s,2), round(y*s,2), round(w*s,2), round(h*s,2)]
            })

        # 참고 메타(원본 크기/스케일)
        new_page["meta"] = {
            "norm": {
                "base_width": base_width,
                "scale": round(s, 6),
                "original_size": {"w": W, "h": H, "units": p["size"].get("units", "px")}
            }
        }
        out.append(new_page)
    return out
