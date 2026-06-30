#!/usr/bin/env python3
"""Phase 1-d: 데이터 QA — split별 이미지/박스 수, obj_type·authenticity·source 분포, 불균형 → eval/data_qa.md."""
import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent
ROOT = DATA.parent
MERGED = DATA / "merged"


def main() -> None:
    manifest = {m["image"]: m for m in
                (json.loads(l) for l in (MERGED / "manifest.jsonl").read_text().splitlines() if l.strip())}
    lines = ["# 데이터 QA (vision-KIR)\n",
             "공개 커뮤니티 데이터(프록시). 라벨은 전문가 검증 아님. Nike AF1/AJ1 brand bias 주의.\n"]
    grand = Counter()
    for split in ("train", "val", "test"):
        img_dir = MERGED / split / "images"
        if not img_dir.exists():
            continue
        imgs = [p.name for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        obj, auth, src, nbox = Counter(), Counter(), Counter(), 0
        for im in imgs:
            m = manifest.get(im)
            if not m:
                continue
            src[m["source"]] += 1
            for b in m["boxes"]:
                obj[b["obj_type"]] += 1
                auth[b["authenticity"]] += 1
                nbox += 1
        grand.update(auth)
        lines += [f"## {split} — {len(imgs)} images, {nbox} boxes",
                  f"- source: {dict(src)}",
                  f"- obj_type: {dict(obj)}",
                  f"- authenticity: {dict(auth)}\n"]
    lines.append(f"## 전체 authenticity 분포: {dict(grand)}")
    if grand:
        tot = sum(grand.values())
        ratio = {k: round(v / tot, 3) for k, v in grand.items()}
        lines.append(f"- 비율: {ratio}  (클래스 불균형 점검)")
    out = ROOT / "eval" / "data_qa.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] → {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
