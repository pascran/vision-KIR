#!/usr/bin/env python3
"""Phase 2: Stage1 객체 검출 학습 (YOLO11s, 2-class: logo/shoe). 시드 고정(재현성).

baseline: COCO 사전학습 YOLO11s는 logo/shoe 클래스가 없어 우리 셋 mAP≈0 → 파인튜닝 필수.
target: 자체 split test mAP@50 ≥ 0.70 (1차 목표; 실패 유형 분석 후 v2에서 개선).
env: EPOCHS(기본 80), BATCH(기본 32), RUN(기본 detect_v1).
"""
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged" / "data.yaml"


def main() -> None:
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA),
        epochs=int(os.environ.get("EPOCHS", "80")),
        imgsz=640,
        batch=int(os.environ.get("BATCH", "32")),
        device=0,
        seed=0,
        deterministic=True,  # cudnn 결정성 + 전역 시드 (ultralytics가 처리)
        project=str(ROOT / "runs"),
        name=os.environ.get("RUN", "detect_v1"),
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
