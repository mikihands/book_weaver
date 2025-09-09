# mybook/utils/layout_norm.py
from copy import deepcopy
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

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
    extract_images_and_bboxes() 결과를 정규화 한다. 페이지의 가로&세로, bbox, clip_bbox, transform 를 정규화(스케일링)
    입력 pages:
      - size.w,h : px (extract에서 이미 dpi/72 적용됨)
      - images[*].bbox : [x,y,w,h] (px)
      - images[*].clip_bbox : [x,y,w,h] (px) | None
      - images[*].transform : [a,b,c,d,e,f] (px)
      - images[*].img_w, img_h : 서버에 저장된 이미지 픽셀  ← 스케일 금지!
      - images[*].origin_h : 원본의 원래 이미지 높이 px  ← 스케일 금지! (html build시 이미지 transform 보정에 사용)
      - images[*].origin_w : 원본의 원래 이미지 너비 px  ← 스케일 금지!

    반환 :
      - [{
        "page_no": int,
        "size": {"w":..., "h":...},
        "meta": {
          "norm": {
            "base_width": int,
            "scale": float,
            "original_size": {"w":..., "h":...},
            "pt_to_px": float
          }
        },
        "images": {
            ...
        }
      }, ...]
    """
    out: List[Dict[str, Any]] = []

    for p in pages:
        W = float(p["size"]["w"])   # px
        H = float(p["size"]["h"])   # px
        if W <= 0 or H <= 0:
            out.append(p)
            continue

        pt_to_px = p.get("pt_to_px", None)
        s = float(base_width) / W   # 정규화 스케일 (px→px)

        # 페이지 정규화
        newW = float(base_width)
        newH = round(H * s, 2)

        new_page = {
            "page_no": p["page_no"],
            "size": {"w": newW, "h": newH, "units": "px"},
            "images": []
        }

        for im in p.get("images", []):
            x, y, w, h = im["bbox"]               # px
            cb = im.get("clip_bbox")              # [x,y,w,h] px | None
            tf = im.get("transform")              # [a,b,c,d,e,f] px

            # bbox/clip_bbox 정규화 스케일 적용
            nb = [round(x*s, 2), round(y*s, 2), round(w*s, 2), round(h*s, 2)]
            if cb:
                cx, cy, cw, ch = cb
                ncb = [round(cx*s, 2), round(cy*s, 2), round(cw*s, 2), round(ch*s, 2)]
            else:
                ncb = None
            # transform정규화
            ntf = [round(tf[0]*s, 2), round(tf[1]*s, 2), round(tf[2]*s, 2), round(tf[3]*s, 2), round(tf[4]*s, 2), round(tf[5]*s, 2)] if tf else [1,0,0,1,0,0]

            logger.debug(f"[노멀라이즈] bbox : {nb}")
            logger.debug(f"[노멀라이즈] clip_bbox : {ncb}")
            logger.debug(f"[노멀라이즈] transform : {ntf}")

            new_page["images"].append({
                "ref": im["ref"],
                "path": im["path"],
                "xref": im.get("xref"),
                "bbox": [round(v,2) for v in nb],   # px (정규화됨)
                "clip_bbox": [round(v,2) for v in ncb] if ncb else None,   # px (정규화됨) 또는 None
                "transform": ntf,         # px (정규화됨)
                "img_w": im.get("img_w"),           # 원본 픽셀 유지
                "img_h": im.get("img_h"),
                "origin_h": im.get("origin_h"),
                "origin_w": im.get("origin_w"),
            })

        new_page["meta"] = {
            "norm": {
                "base_width": base_width,
                "scale": round(s, 6),
                "original_size": {"w": W, "h": H, "units": "px"},
                "pt_to_px": pt_to_px
            }
        }

        out.append(new_page)

    return out