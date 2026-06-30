# 데이터 QA (vision-KIR)

공개 커뮤니티 데이터(프록시). 라벨은 전문가 검증 아님. Nike AF1/AJ1 brand bias 주의.

## train — 2263 images, 3840 boxes
- source: {'logo': 324, 'shoes': 1939}
- obj_type: {'logo': 326, 'shoe': 3514}
- authenticity: {'fake': 2037, 'real': 1803}

## val — 648 images, 1095 boxes
- source: {'logo': 92, 'shoes': 556}
- obj_type: {'logo': 92, 'shoe': 1003}
- authenticity: {'fake': 633, 'real': 462}

## test — 323 images, 542 boxes
- source: {'logo': 48, 'shoes': 275}
- obj_type: {'logo': 48, 'shoe': 494}
- authenticity: {'fake': 281, 'real': 261}

## 전체 authenticity 분포: {'fake': 2951, 'real': 2526}
- 비율: {'fake': 0.539, 'real': 0.461}  (클래스 불균형 점검)
