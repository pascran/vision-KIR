# 검출 오류분석 — detect_v1 (test)

- GT 박스: 542
- 미탐(FN) by area: {'small(<0.02)': 3, 'med': 0, 'large(>0.1)': 10}
- 오탐(FP): 38
- 클래스 혼동: 없음

→ v2 타겟: 미탐이 소형 박스에 몰리면 imgsz↑/mosaic·copy-paste 증강, 혼동 많으면 클래스 균형 보강.
