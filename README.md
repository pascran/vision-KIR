# vision-KIR — 위조상품 감정 비전 파이프라인

현품 이미지에서 **객체(로고/신발)를 검출 → 진위(real/fake) 판정 → 유사도 triage → 구조화 판독근거 JSON**을
생성하는 end-to-end 데모. 공개 데이터로 **데이터셋 통합·정제 → YOLO11 학습·평가 → 진위 분류 → 임베딩 유사도 → ONNX 최적화**의 풀 ML 루프를 직접 구현한다.

> ⚠️ **정직 고지**: 이건 **방법론 데모(프록시)**지 production 위조감정 시스템이 아니다.
> 데이터는 공개 커뮤니티 셋(Roboflow)으로 **라벨이 전문가 검증 아님**, 신발은 **Nike AF1/AJ1로 brand-biased**,
> "fake"는 일부 데이터셋 출처 라벨이다. 엔지니어링 관행(leakage-safe split·평가 리거·재현성)은 실무 수준으로,
> **모델·데이터는 데모 수준**임을 분명히 한다.

## 아키텍처 (2-stage + triage)

```
이미지
 ├─ Stage1  검출(YOLO11)        : 객체 위치 — logo / shoe (2-class)   ← "real/fake"는 검출 클래스 아님
 ├─ Stage2  진위 분류(crop)      : 검출 크롭에 real/fake 분류 (신발만; 로고셋은 전부 fake)
 ├─ Stage3  유사도 triage        : 크롭 임베딩(CLIP) → 정품 레퍼런스 retrieval → 의심 플래그(측정된 AUC)
 └─ 판독근거 JSON               : 검출 + 진위 + 유사도 + OCR(브랜드/시리얼)
```
"이 영역이 로고/신발이다(검출) → 진짜냐(분류+유사도)" — 검출과 판정을 분리(위조성은 박스로 학습되지 않음).

## 데이터
- **로고**: Roboflow `abner-colcq/fake-brand-logo-detetction` v2 — 단일 클래스 'Fake brand logo'(전부 fake).
- **신발**: Roboflow `ggdgdsgdg/original-or-fake-shoes` v5 — fake/ori(진품) Nike AF1·AJ1 (grayscale 강제·증강 포함).
- 통합 후 **2-class 검출**(logo/shoe) + 박스별 진위 manifest. 원본 셋엔 val/test가 사실상 없어 **직접 leakage-safe split**(aHash 중복제거 + source-group 단위 70/20/10). 분포·QA: [`eval/data_qa.md`](eval/data_qa.md).

## 실행
```bash
# 0) venv (H100, cu124)
python -m venv venv && . venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
echo "ROBOFLOW_API_KEY=..." > .env

# 1) 데이터: 다운로드 → 통합 → leakage-safe split → QA
python data/download.py && python data/merge.py && python data/dedup_split.py && python data/qa.py
# 2) 검출 학습 (YOLO11s, 2-class)
python detect/train.py
# 3) 평가 + 오류분석 → 재학습(v2) 델타
python eval/evaluate.py && python eval/error_analysis.py
# 4) 진위 분류기 (신발 crop real/fake)
python authclf/build_crops.py && python authclf/train_clf.py && python authclf/eval_clf.py
# 5) 유사도 triage 신호 측정 (AUC)
python embed/build_reference.py && python embed/measure_signal.py
# 6) 단일 이미지 판독근거 JSON
python infer/appraise.py sample.jpg
# 7) ONNX export + 지연 벤치
python export/export.py
```

## 결과 (실측, test split)
| 단계 | 지표 | 값 |
|---|---|---|
| **Stage1 검출** (YOLO11s) | mAP@50 / mAP@50-95 | **0.975 / 0.832** (logo AP50 0.968, shoe 0.982, 클래스 혼동 0) |
| **Stage2 진위 분류** (신발 crop) | accuracy / ROC-AUC | **0.792 / 0.879** (workhorse 신호) |
| **Stage3 유사도 triage** | ROC-AUC | **0.647** (real-sim 0.869 vs fake-sim 0.847 → ★약한 신호, verdict 아닌 triage) |
| **최적화** | 지연/img | PyTorch 4.6ms · **TensorRT FP16 4.4ms** · ONNX(CPU) 199.7ms |
- 오류분석([`eval/error_analysis.md`](eval/error_analysis.md)): 미탐(FN) 13(대형 10/소형 3), 오탐(FP) 38, 클래스 혼동 0. v1이 목표(mAP@50≥0.70) 초과 → v2 미실행, 약점은 문서화.
- 상세: [`eval/results_detect_v1.md`](eval/results_detect_v1.md) · [`results_authclf.md`](eval/results_authclf.md) · [`results_similarity.md`](eval/results_similarity.md) · 샘플 판독근거 [`eval/sample_appraisal.json`](eval/sample_appraisal.json)

> **읽은 것**: 진위는 **분류기(AUC 0.879)가 주 신호**, CLIP 유사도(0.647)는 약한 보조 — 측정해보니 "같은 제품/포즈"는 잡아도 진짜/가짜 미세차는 잘 못 가린다(예측된 CLIP 한계). 과장하지 않고 측정값 그대로 보고.

> ⚠️ **실전 함정**: ultralytics `export(format='engine')`는 **TensorRT를 자동 설치하며 torch를 cu124→cu130으로 업그레이드** → torchvision/cuDNN과 mismatch나 다른 스크립트가 깨진다. **TRT export는 마지막에/격리 venv에서** 하고, 끝나면 `torch==2.6.0+cu124` force-reinstall로 복원.

## 한계 (정직)
- 공개 커뮤니티 데이터 = 프록시. 라벨 전문가 검증 없음. production 감정기 아님.
- 신발 Nike AF1/AJ1 brand bias — 타 브랜드 일반화 불가.
- "fake"는 데이터셋 출처 라벨 — 박스 기반 검출이 위조성을 직접 학습하지 않음(그래서 2-stage).
- 유사도 신호는 CLIP ViT-B/32 기반 약한 triage — DINOv2/SigLIP가 업그레이드 경로.
- val/test split은 자체 생성(원본 셋엔 없음). 정확한 split은 [`data/split_manifest.json`](data/split_manifest.json)으로 고정·커밋.

## 재현성
- 시드 고정(`seed=0` + ultralytics `deterministic=True`로 cudnn 결정성), `requirements.txt` 버전 핀, split manifest 커밋.
- 단 **완전 비트단위 재현은 보장 안 됨**: Roboflow 데이터 버전·CUDA 12.4·GPU 커널 비결정성으로 소수점 변동 가능. mAP/AUC는 동일 환경에서 ±0.01 내 재현 목표.

## 기술 스택
`PyTorch` · `Ultralytics YOLO11` · `CLIP(open_clip)` · `Qdrant` · `easyOCR` · `ONNX` · H100(cu124)
