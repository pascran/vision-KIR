# 유사도 triage 신호 (test 신발 crop)

- n = 494
- 정품유사도 평균: real=0.8687 vs fake=0.8472
- **ROC-AUC(정품유사도가 real을 가름): 0.6471** → 활용 불가(무작위 수준) — 판정 근거로 사용 금지

★이건 verdict가 아니라 triage 사전확률. CLIP ViT-B/32는 fine-grained 위조 판별엔 약함
(주로 '같은 제품/포즈'를 잡음). 업그레이드 경로: DINOv2 / SigLIP / 도메인 fine-tune.
