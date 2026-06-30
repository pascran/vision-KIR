#!/usr/bin/env python3
"""Phase 5-b: 유사도 triage 신호 측정.

가설: '정품 레퍼런스와의 유사도'가 높으면 real, 낮으면 fake. test crop(real/fake)에 대해
정품 top-k 평균 유사도를 구해 real=1로 ROC-AUC 측정. ~0.5면 신호 약함(정직 negative result).
→ verdict가 아니라 triage 사전확률. CLIP ViT-B/32 한계 → DINOv2/SigLIP 업그레이드 경로.
"""
from pathlib import Path

import numpy as np

from clip_util import ReferenceIndex

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "authclf" / "crops" / "test"


def roc_auc(y: list, s: list) -> float:
    y = np.array(y)
    y = y[np.argsort(s)]
    P, N = int(y.sum()), int(len(y) - y.sum())
    if P == 0 or N == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def main() -> None:
    ref = ReferenceIndex(ROOT / "embed" / "reference.npz")
    ys, scores = [], []
    for auth in ("real", "fake"):
        for img in (TEST / auth).glob("*.jpg"):
            scores.append(ref.genuine_similarity(str(img), k=5))
            ys.append(1 if auth == "real" else 0)
    if not ys:
        raise SystemExit("test crop 없음 — authclf/build_crops.py 먼저 실행")
    auc = roc_auc(ys, scores)
    rs = [s for s, y in zip(scores, ys) if y == 1]
    fs = [s for s, y in zip(scores, ys) if y == 0]
    real_m = np.mean(rs) if rs else float("nan")
    fake_m = np.mean(fs) if fs else float("nan")
    verdict = "유의미한 triage 신호" if auc >= 0.65 else "약한 신호(분리 거의 안 됨)"
    lines = ["# 유사도 triage 신호 (test 신발 crop)\n",
             f"- n = {len(ys)}",
             f"- 정품유사도 평균: real={real_m:.4f} vs fake={fake_m:.4f}",
             f"- **ROC-AUC(정품유사도가 real을 가름): {auc:.4f}** → {verdict}\n",
             "★이건 verdict가 아니라 triage 사전확률. CLIP ViT-B/32는 fine-grained 위조 판별엔 약함",
             "(주로 '같은 제품/포즈'를 잡음). 업그레이드 경로: DINOv2 / SigLIP / 도메인 fine-tune."]
    (ROOT / "eval" / "results_similarity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
