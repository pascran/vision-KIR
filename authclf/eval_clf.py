#!/usr/bin/env python3
"""Phase 4-c: 진위 분류 평가 — test accuracy·ROC-AUC(real=1)·confusion → eval/results_authclf.md."""
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "authclf" / "crops" / "test"
WEIGHTS = ROOT / "runs" / "authclf_v1" / "weights" / "best.pt"


def roc_auc(y: list, s: list) -> float:
    y = np.array(y)
    order = np.argsort(s)
    y = y[order]
    P, N = int(y.sum()), int(len(y) - y.sum())
    if P == 0 or N == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def main() -> None:
    model = YOLO(str(WEIGHTS))
    names = model.names  # 폴더명 알파벳순: 0=fake, 1=real
    real_idx = next(k for k, v in names.items() if v == "real")
    ys, scores, correct, tot, cm = [], [], 0, 0, {}
    for cls_dir in TEST.iterdir():
        if not cls_dir.is_dir():
            continue
        true = cls_dir.name
        for img in cls_dir.glob("*.jpg"):
            probs = model.predict(str(img), verbose=False)[0].probs
            pred = names[int(probs.top1)]
            ys.append(1 if true == "real" else 0)
            scores.append(float(probs.data[real_idx]))
            correct += int(pred == true)
            tot += 1
            cm[f"{true}->{pred}"] = cm.get(f"{true}->{pred}", 0) + 1
    acc = correct / tot if tot else 0.0
    auc = roc_auc(ys, scores)
    lines = ["# 진위 분류 (신발 crop, test)\n",
             f"- n = {tot}", f"- **accuracy: {acc:.4f}**", f"- **ROC-AUC(real=1): {auc:.4f}**",
             f"- confusion: {cm}\n",
             "★라벨은 커뮤니티 프록시(전문가 검증 아님), Nike AF1/AJ1 한정. 로고는 전부 fake라 이진분류 제외."]
    (ROOT / "eval" / "results_authclf.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
