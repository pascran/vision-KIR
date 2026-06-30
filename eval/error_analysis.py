#!/usr/bin/env python3
"""Phase 3-b: 검출 오류분석 — test에서 미탐(FN) box-area별, 오탐(FP), 클래스혼동 집계 → eval/error_analysis.md.
이 결과로 v2 재학습의 타겟(예: 소형 박스 약점 → 증강/해상도)을 정함."""
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "merged"
RUN = os.environ.get("RUN", "detect_v1")
DET = ROOT / "runs" / RUN / "weights" / "best.pt"


def to_xyxy(b):
    x, y, w, h = b
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou(a, b):
    ax = to_xyxy(a); bx = to_xyxy(b)
    ix1, iy1 = max(ax[0], bx[0]), max(ax[1], bx[1])
    ix2, iy2 = min(ax[2], bx[2]), min(ax[3], bx[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    model = YOLO(str(DET))
    names = model.names
    img_dir, lbl_dir = MERGED / "test" / "images", MERGED / "test" / "labels"
    fn_area = {"small(<0.02)": 0, "med": 0, "large(>0.1)": 0}
    fp, tot_gt, confus = 0, 0, {}
    for img in img_dir.glob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        gts = []
        lp = lbl_dir / (img.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    gts.append((int(float(p[0])), *map(float, p[1:])))
        r = model.predict(str(img), verbose=False, conf=0.25)[0]
        preds = [(int(b.cls), *b.xywhn[0].tolist()) for b in r.boxes]
        matched = [False] * len(preds)
        for gc, gx, gy, gw, gh in gts:
            tot_gt += 1
            best, bi = -1, -1
            for j, (pc, px, py, pw, ph) in enumerate(preds):
                v = iou((gx, gy, gw, gh), (px, py, pw, ph))
                if v > best:
                    best, bi = v, j
            if best >= 0.5 and bi >= 0:
                matched[bi] = True
                if preds[bi][0] != gc:
                    k = f"{names[gc]}->{names[preds[bi][0]]}"
                    confus[k] = confus.get(k, 0) + 1
            else:
                area = gw * gh
                key = "small(<0.02)" if area < 0.02 else ("med" if area < 0.1 else "large(>0.1)")
                fn_area[key] += 1
        fp += sum(1 for m in matched if not m)
    lines = [f"# 검출 오류분석 — {RUN} (test)\n",
             f"- GT 박스: {tot_gt}", f"- 미탐(FN) by area: {fn_area}",
             f"- 오탐(FP): {fp}", f"- 클래스 혼동: {confus or '없음'}\n",
             "→ v2 타겟: 미탐이 소형 박스에 몰리면 imgsz↑/mosaic·copy-paste 증강, 혼동 많으면 클래스 균형 보강."]
    (ROOT / "eval" / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
