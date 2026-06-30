#!/usr/bin/env python3
"""Phase 7: 검출 모델 ONNX export + 지연 벤치. TRT engine은 best-effort."""
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "runs" / "detect_v1" / "weights" / "best.pt"


def bench(model, dummy, n=30, **kw) -> float:
    for _ in range(5):
        model.predict(dummy, verbose=False, **kw)
    t = []
    for _ in range(n):
        s = time.time()
        model.predict(dummy, verbose=False, **kw)
        t.append(time.time() - s)
    return float(np.median(t) * 1000)


def main() -> None:
    model = YOLO(str(DET))
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    lines = ["# 최적화 / 지연 벤치 (검출 모델)\n"]

    pt_ms = bench(model, dummy, device=0)
    lines.append(f"- PyTorch (GPU): **{pt_ms:.1f} ms** /img")

    onnx_path = model.export(format="onnx", imgsz=640)
    print("ONNX:", onnx_path)
    onnx_model = YOLO(str(onnx_path))
    onnx_ms = bench(onnx_model, dummy)
    lines.append(f"- ONNX (onnxruntime CPU): {onnx_ms:.1f} ms /img")

    try:
        eng = model.export(format="engine", half=True, imgsz=640, device=0)
        eng_ms = bench(YOLO(str(eng)), dummy, device=0)
        lines.append(f"- TensorRT FP16 (GPU): **{eng_ms:.1f} ms** /img")
    except Exception as e:
        lines.append(f"- TensorRT: export 실패(best-effort) — {str(e)[:160]}")

    (ROOT / "eval" / "results_export.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
