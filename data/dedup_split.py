#!/usr/bin/env python3
"""Phase 1-c: leakage-safe split.

- aHash 근접중복 제거.
- Roboflow 증강 복제(같은 source, '.rf.' 앞 stem 동일)는 한 그룹으로 묶어 같은 split에만 들어가게 → train/val leakage 차단.
- source-group 단위 70/20/10 split(원본 데이터셋엔 val/test가 사실상 없어 직접 생성).
- 통합 data.yaml(nc:2, names:[logo,shoe]) 작성.
시드 고정(재현성).
"""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

random.seed(0)
DATA = Path(__file__).resolve().parent
ALL = DATA / "merged" / "all"
MERGED = DATA / "merged"


def ahash(path: Path, size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size, size))
    px = list(img.getdata())
    avg = sum(px) / len(px)
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def source_key(uid: str) -> str:
    return uid.split(".rf.")[0] if ".rf." in uid else uid


def main() -> None:
    manifest = [json.loads(l) for l in (MERGED / "manifest.jsonl").read_text().splitlines() if l.strip()]

    # 1) 근접중복 제거 (동일 aHash)
    seen, kept = set(), []
    for m in manifest:
        h = ahash(ALL / "images" / m["image"])
        if h in seen:
            continue
        seen.add(h)
        kept.append(m)
    print(f"[dedup] {len(manifest)} → {len(kept)} (근접중복 {len(manifest) - len(kept)} 제거)")

    # 2) source-group 으로 묶기
    groups = defaultdict(list)
    for m in kept:
        groups[(m["source"], source_key(m["image"]))].append(m)

    # 3) source 기준 stratify 후 group 단위 70/20/10
    by_source = defaultdict(list)
    for (source, _), members in groups.items():
        by_source[source].append(members)
    splits = {"train": [], "val": [], "test": []}
    for source, glist in by_source.items():
        random.shuffle(glist)
        n = len(glist)
        n_tr, n_va = int(n * 0.7), int(n * 0.2)
        for i, members in enumerate(glist):
            s = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            splits[s].extend(members)

    # 4) 파일 배치
    for s, items in splits.items():
        (MERGED / s / "images").mkdir(parents=True, exist_ok=True)
        (MERGED / s / "labels").mkdir(parents=True, exist_ok=True)
        for m in items:
            stem = Path(m["image"]).stem
            shutil.copy(ALL / "images" / m["image"], MERGED / s / "images" / m["image"])
            shutil.copy(ALL / "labels" / (stem + ".txt"), MERGED / s / "labels" / (stem + ".txt"))
        print(f"  {s}: {len(items)} images")

    # 5) data.yaml
    (MERGED / "data.yaml").write_text(
        f"path: {MERGED}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: 2\nnames: [logo, shoe]\n", encoding="utf-8")
    # split manifest (재현성)
    (MERGED / "split_manifest.json").write_text(
        json.dumps({s: [m["image"] for m in items] for s, items in splits.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] data.yaml + split_manifest.json → {MERGED}")


if __name__ == "__main__":
    main()
