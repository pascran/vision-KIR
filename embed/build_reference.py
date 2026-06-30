#!/usr/bin/env python3
"""Phase 5-a: 정품(train/real 신발 crop) CLIP 임베딩 → embed/reference.npz."""
from pathlib import Path

import numpy as np

from clip_util import embed

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "authclf" / "crops" / "train" / "real"


def main() -> None:
    vecs = [embed(str(p)) for p in sorted(REAL.glob("*.jpg"))]
    if not vecs:
        raise SystemExit(f"정품 크롭 없음: {REAL} — authclf/build_crops.py 먼저 실행")
    arr = np.stack(vecs)
    np.savez(ROOT / "embed" / "reference.npz", vecs=arr)
    print(f"[done] reference {arr.shape} → embed/reference.npz")


if __name__ == "__main__":
    main()
