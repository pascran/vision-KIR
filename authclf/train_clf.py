#!/usr/bin/env python3
"""Phase 4-b: 신발 크롭 진위 분류 학습 (YOLO11s-cls, real/fake). 시드 고정."""
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "authclf" / "crops"


def main() -> None:
    model = YOLO("yolo11s-cls.pt")
    model.train(data=str(CROPS), epochs=int(os.environ.get("EPOCHS", "30")),
                imgsz=224, device=0, seed=0, deterministic=True,
                project=str(ROOT / "runs"), name="authclf_v1", exist_ok=True)


if __name__ == "__main__":
    main()
