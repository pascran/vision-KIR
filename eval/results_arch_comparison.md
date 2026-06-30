# 검출 아키텍처 비교 — YOLO11s (CNN) vs RT-DETR-l (Transformer)

동일 조건: 같은 데이터·split, imgsz 640, 80 epochs, seed 0, deterministic. test split 평가.

| 지표 | YOLO11s (CNN) | RT-DETR-l (Transformer) |
|---|---|---|
| mAP@50 | **0.975** | 0.967 |
| mAP@50-95 | 0.832 | **0.877** |
| mean P | 0.904 | **0.938** |
| mean R | **0.970** | 0.957 |
| logo AP50-95 | 0.823 | **0.888** |
| shoe AP50-95 | 0.841 | **0.866** |
| 지연/img (PyTorch GPU) | **4.6 ms** | 10.3 ms |
| 학습 시간 (80ep) | ~0.25 h | 1.31 h |
| 파라미터(약) | ~9.4M | ~32M |

## 해석

- RT-DETR-l이 **mAP@50-95(엄격 IoU)·정밀도(P)**에서 우위 — 박스 위치가 더 정확하다. Transformer 디코더의 set-prediction이 localization을 정밀하게 한다.
- YOLO11s가 **mAP@50·recall·추론 속도(2.2배)·학습 시간(약 5배)**에서 우위 — 작고 빠르다.
- 단, RT-DETR-l(~32M)이 YOLO11s(~9.4M)보다 커서 **파라미터가 동일하지 않다**. 동급 파라미터 비교는 아니며, RT-DETR 계열의 표준 최소 변형이 rtdetr-l이라 그대로 사용했다.
- RT-DETR은 DETR 계열 특성상 수렴이 느려 80 epoch에서 미수렴 여지가 있다. 더 긴 학습 시 mAP@50도 따라잡을 가능성이 있다(동일 80ep 예산 기준 비교).

## 결론

정밀 localization·정밀도가 중요하면 RT-DETR, 속도·실시간·경량이 중요하면 YOLO. 본 과제(이미지 제출 기반, 실시간 아님)에서는 RT-DETR의 정밀도 이점이 의미 있을 수 있으나 추론 비용이 2배다. 현재 파이프라인은 속도·학습비용 우위로 YOLO11s를 1차 선택으로 유지하고, RT-DETR은 정밀도 요구 시 대안으로 둔다.
