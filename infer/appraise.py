#!/usr/bin/env python3
"""Phase 6: 단일 이미지 → 판독근거 JSON. 검출(logo/shoe) + (신발)진위분류 + 정품 유사도 + OCR.

usage: python infer/appraise.py <image.jpg>
"""
import json
import sys
from pathlib import Path

import torch  # noqa: F401  (torchvision::nms 등록 순서 보장 — easyocr와 함께 로드 시 필요)
import torchvision  # noqa: F401
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "embed"))
DET = ROOT / "runs" / "detect_v1" / "weights" / "best.pt"
CLF = ROOT / "runs" / "authclf_v1" / "weights" / "best.pt"


def main() -> None:
    img_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "sample.jpg")
    det = YOLO(str(DET))
    clf = YOLO(str(CLF)) if CLF.exists() else None
    ref = None
    if (ROOT / "embed" / "reference.npz").exists():
        from clip_util import ReferenceIndex
        ref = ReferenceIndex(ROOT / "embed" / "reference.npz")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)  # OCR은 CPU (작은 부담, GPU cuDNN 충돌 회피)

    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    r = det.predict(img_path, verbose=False)[0]
    names = det.names
    findings = []
    for b in r.boxes:
        cls = names[int(b.cls)]
        xyxy = [int(v) for v in b.xyxy[0].tolist()]
        entry = {"object": cls, "bbox": xyxy, "det_conf": round(float(b.conf), 3)}
        if xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 2:  # 0-area crop 건너뜀
            findings.append(entry)
            continue
        crop = pil.crop(tuple(xyxy))
        txt = reader.readtext(np.array(crop), detail=0)
        if txt:
            entry["ocr"] = txt
        if cls == "shoe" and clf is not None:
            p = clf.predict(crop, verbose=False)[0].probs
            ri = next((k for k, v in clf.names.items() if v == "real"), None)
            if ri is not None:
                entry["authenticity"] = {"pred": clf.names[int(p.top1)], "p_real": round(float(p.data[ri]), 3)}
            if ref is not None:
                entry["genuine_similarity"] = round(ref.genuine_similarity(crop), 3)
        findings.append(entry)
    out = {"image": Path(img_path).name, "size": [W, H], "findings": findings,
           "note": "프록시 데모 — 커뮤니티 라벨(전문가 검증 아님), Nike 한정. verdict 아닌 1차 판독근거."}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
