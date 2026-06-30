#!/usr/bin/env python3
"""Phase 4-a: 진위 분류용 크롭 생성 (신발만 — real+fake 존재; 로고는 전부 fake라 이진분류 불가, 제외).
출력: authclf/crops/<split>/<real|fake>/<uid>_<i>.jpg (ImageFolder 구조, YOLO-cls 입력)."""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "merged"
OUT = ROOT / "authclf" / "crops"


def main() -> None:
    manifest = {m["image"]: m for m in
                (json.loads(l) for l in (MERGED / "manifest.jsonl").read_text().splitlines() if l.strip())}
    split_of = {}
    for s, imgs in json.loads((MERGED / "split_manifest.json").read_text()).items():
        for im in imgs:
            split_of[im] = s
    n = {"train": 0, "val": 0, "test": 0}
    for uid, m in manifest.items():
        split = split_of.get(uid)
        if not split:
            continue
        img_path = MERGED / split / "images" / uid
        if not img_path.exists():
            continue
        img = None
        for i, b in enumerate(m["boxes"]):
            if b["obj_type"] != "shoe" or b["authenticity"] not in ("real", "fake"):
                continue
            if img is None:
                img = Image.open(img_path).convert("RGB")
            W, H = img.size
            cx, cy, w, h = b["bbox_norm"]
            x1, y1 = max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H))
            x2, y2 = min(W, int((cx + w / 2) * W)), min(H, int((cy + h / 2) * H))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            d = OUT / split / b["authenticity"]
            d.mkdir(parents=True, exist_ok=True)
            img.crop((x1, y1, x2, y2)).save(d / f"{Path(uid).stem}_{i}.jpg")
            n[split] += 1
    print(f"[done] crops {n} → {OUT}")


if __name__ == "__main__":
    main()
