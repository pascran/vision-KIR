#!/usr/bin/env python3
"""Phase 1-b: 두 공개 데이터셋을 통합.

- Stage1 검출 라벨 = 2-class(logo/shoe)로 remap. ("real/fake"는 검출 클래스 아님)
- 박스별 진위(real/fake)는 manifest.jsonl에 보존 → Stage2 진위분류에서 사용.
- ★클래스 인덱스는 각 data.yaml에서 읽음 (Roboflow는 알파벳순 인덱싱이라 셋마다 다름 — 하드코딩 금지).
"""
import json
import shutil
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parent
DS = DATA / "datasets"
OUT = DATA / "merged" / "all"
OBJ_TYPES = {"logo": 0, "shoe": 1}


def authenticity_of(name: str) -> str:
    """클래스명 → real/fake. 실제 데이터 클래스:
    로고='Fake brand logo'(전부 fake), 신발='fake air force/jordan 1' + 'ori air force/jordan 1'(ori=진품)."""
    n = name.lower()
    toks = n.split()
    if "fake" in n or "counterfeit" in n:
        return "fake"
    if "ori" in toks or "original" in n or "authentic" in n or "real" in n or "genuine" in n:
        return "real"
    return "unknown"


def read_names(ds_dir: Path) -> dict:
    y = yaml.safe_load((ds_dir / "data.yaml").read_text(encoding="utf-8"))
    names = y["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}


def process(ds_name: str, obj_type: str, manifest: list) -> None:
    ds_dir = DS / ds_name
    names = read_names(ds_dir)
    obj_idx = OBJ_TYPES[obj_type]
    n_img = 0
    for split in ("train", "valid", "test"):
        img_dir = ds_dir / split / "images"
        lbl_dir = ds_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            boxes, new_lines = [], []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    ci = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:]]
                    boxes.append({"obj_type": obj_type,
                                  "authenticity": authenticity_of(names.get(ci, "")),
                                  "bbox_norm": coords})
                    new_lines.append(f"{obj_idx} " + " ".join(parts[1:]))
            if not boxes:
                continue
            uid = f"{ds_name}__{split}__{img.name}"
            shutil.copy(img, OUT / "images" / uid)
            (OUT / "labels" / (Path(uid).stem + ".txt")).write_text("\n".join(new_lines) + "\n")
            manifest.append({"image": uid, "source": ds_name, "boxes": boxes})
            n_img += 1
    print(f"  {ds_name} → {obj_type}: {n_img} images")


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    manifest: list = []
    process("logo", "logo", manifest)
    process("shoes", "shoe", manifest)
    (DATA / "merged" / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n", encoding="utf-8")
    print(f"[done] {len(manifest)} images → {OUT}")


if __name__ == "__main__":
    main()
