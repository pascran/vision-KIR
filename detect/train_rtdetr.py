#!/usr/bin/env python3
"""아키텍처 비교 — RT-DETR-l(Transformer 검출기)을 YOLO11s(CNN)와 동일 데이터·설정으로 학습·평가.

YOLO와 동일: data.yaml(2-class), imgsz=640, epochs(기본 80), seed=0, deterministic.
다름: RT-DETR은 Transformer라 메모리가 커 batch 기본 16. DETR 계열은 수렴이 느려
80ep에서 미수렴일 수 있음(비교 시 동일 예산 기준이라는 점·수렴 효율 차이로 해석).
"""
import os
import time
from pathlib import Path

import numpy as np
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged" / "data.yaml"
RUN = "detect_rtdetr"


def main() -> None:
    model = RTDETR("rtdetr-l.pt")
    model.train(data=str(DATA), epochs=int(os.environ.get("EPOCHS", "80")),
                imgsz=640, batch=int(os.environ.get("BATCH", "16")), device=0,
                seed=0, deterministic=True,
                project=str(ROOT / "runs"), name=RUN, exist_ok=True)

    m = model.val(data=str(DATA), split="test", project=str(ROOT / "runs"),
                  name=f"{RUN}_test", exist_ok=True, verbose=False)
    names = model.names
    lines = ["# RT-DETR-l 검출 평가 (test)\n",
             f"- mAP@50: {m.box.map50:.4f}", f"- mAP@50-95: {m.box.map:.4f}",
             f"- mean P: {m.box.mp:.4f} / R: {m.box.mr:.4f}\n", "## per-class"]
    for i, ci in enumerate(m.box.ap_class_index):
        cid = int(ci)
        cname = names[cid] if isinstance(names, (list, dict)) else str(cid)
        lines.append(f"- {cname}: P={m.box.p[i]:.3f} R={m.box.r[i]:.3f} "
                     f"AP50={m.box.ap50[i]:.3f} AP50-95={m.box.ap[i]:.3f}")

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, verbose=False, device=0)
    t = []
    for _ in range(30):
        s = time.time()
        model.predict(dummy, verbose=False, device=0)
        t.append(time.time() - s)
    lines.append(f"\n- PyTorch GPU latency: {np.median(t) * 1000:.1f} ms/img")
    (ROOT / "eval" / "results_rtdetr.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
