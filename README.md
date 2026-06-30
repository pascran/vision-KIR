# vision-KIR — 위조상품 감정 비전 파이프라인

현품 이미지에서 객체(로고/신발) 검출 → 진위 분류 → 유사도 triage → 판독근거 JSON을 생성한다.
공개 데이터로 데이터셋 통합·정제, YOLO11 학습·평가, 진위 분류, 임베딩 유사도, ONNX 최적화까지 구현한 ML 파이프라인.

방법론 데모(프록시)이며 production 감정 시스템이 아니다. 데이터는 공개 커뮤니티 셋(Roboflow), 라벨은 전문가 검증을 거치지 않았고, 신발은 Nike AF1/AJ1로 편향돼 있다. 상세 한계는 아래 참조.

## 아키텍처 (2-stage + triage)

```
이미지
 ├─ Stage1  검출(YOLO11)     : 객체 위치 — logo / shoe (2-class)
 ├─ Stage2  진위 분류(crop)   : 검출 크롭에 real/fake 분류 (신발만; 로고셋은 전부 fake)
 ├─ Stage3  유사도 triage     : 크롭 임베딩(CLIP) → 정품 레퍼런스 유사도 (측정 AUC 0.647, 판정 제외)
 └─ 판독근거 JSON            : 검출 + 진위 + 유사도(비신뢰 표시) + OCR
```

검출과 판정을 분리한다. 위조성은 박스로 학습되지 않으므로, 검출은 객체 위치(logo/shoe)만 맡고 진위는 크롭 분류·유사도로 다룬다.

## 데이터

- 로고: Roboflow `abner-colcq/fake-brand-logo-detetction` v2. 단일 클래스 'Fake brand logo'(전부 fake).
- 신발: Roboflow `ggdgdsgdg/original-or-fake-shoes` v5. fake / ori(진품) Nike AF1·AJ1. grayscale 강제·증강 포함.
- 통합 후 2-class 검출(logo/shoe) + 박스별 진위 manifest. 원본 셋엔 val/test가 없어 직접 split한다: aHash 중복 제거 후 source-group 단위 70/20/10. 분포는 [`eval/data_qa.md`](eval/data_qa.md).

## 실행

```bash
# 0) venv (H100, cu124)
python -m venv venv && . venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
echo "ROBOFLOW_API_KEY=..." > .env

# 1) 데이터: 다운로드 → 통합 → split → QA
python data/download.py && python data/merge.py && python data/dedup_split.py && python data/qa.py
# 2) 검출 학습 (YOLO11s, 2-class)
python detect/train.py
# 3) 평가 + 오류분석
python eval/evaluate.py && python eval/error_analysis.py
# 4) 진위 분류기 (신발 crop)
python authclf/build_crops.py && python authclf/train_clf.py && python authclf/eval_clf.py
# 5) 유사도 신호 측정 (AUC)
python embed/build_reference.py && python embed/measure_signal.py
# 6) 판독근거 JSON
python infer/appraise.py sample.jpg
# 7) ONNX export + 지연 벤치
python export/export.py
```

## 결과 (test split)

| 단계 | 지표 | 값 |
|---|---|---|
| 검출 (YOLO11s) | mAP@50 / mAP@50-95 | 0.975 / 0.832 (logo AP50 0.968, shoe 0.982, 클래스 혼동 0) |
| 진위 분류 (신발 crop) | accuracy / ROC-AUC | 0.792 / 0.879 |
| 유사도 triage | ROC-AUC | 0.647 (real 0.869 vs fake 0.847) — 무작위 수준, 판정에서 제외 |
| 최적화 | 지연/img | PyTorch 4.6ms / TensorRT FP16 4.4ms / ONNX(CPU) 199.7ms |

- 진위 판정은 분류기(AUC 0.879)만 신호로 쓴다. 유사도 0.647은 정품/위조를 거의 못 가르므로(real-fake 평균차 0.022) 판독근거 JSON에 `reliable: false`로 표시만 하고 판정에 반영하지 않는다.
- 오류분석([`eval/error_analysis.md`](eval/error_analysis.md)): 미탐(FN) 13, 오탐(FP) 38, 클래스 혼동 0. v1이 목표(mAP@50 ≥ 0.70)를 넘어 v2는 실행하지 않았다.
- 상세: [`results_detect_v1.md`](eval/results_detect_v1.md) · [`results_authclf.md`](eval/results_authclf.md) · [`results_similarity.md`](eval/results_similarity.md) · [`sample_appraisal.json`](eval/sample_appraisal.json)

## 한계

- 공개 커뮤니티 데이터(프록시). 라벨 전문가 검증 없음. production 감정기가 아니다.
- 신발 Nike AF1/AJ1 편향. 타 브랜드 일반화 불가. 실제 적용 시 브랜드별 재학습 필요.
- "fake"는 데이터셋 출처 라벨이라 박스 검출이 위조성을 직접 학습하지 않는다(그래서 2-stage).
- 유사도는 CLIP ViT-B/32 기반 약한 신호. DINOv2/SigLIP 또는 도메인 fine-tune이 개선 경로.
- val/test split은 자체 생성(원본 셋엔 없음). split은 [`data/split_manifest.json`](data/split_manifest.json)으로 고정한다.

## 재현성

- 시드 고정(`seed=0`, ultralytics `deterministic=True`), `requirements.txt` 버전 핀, split manifest 커밋.
- 비트 단위 재현은 보장하지 않는다. Roboflow 데이터 버전·CUDA 12.4·GPU 커널 비결정성으로 소수점 변동이 있을 수 있다(동일 환경 ±0.01 목표).
- 순수 함수 단위 테스트는 `tests/`, CI는 `.github/workflows/ci.yml`(compile + lint + pytest).

## 노트

ultralytics `export(format='engine')`는 TensorRT 설치 과정에서 torch를 cu124→cu130으로 올려 torchvision/cuDNN을 깨뜨린다. TRT export는 격리 venv에서 마지막에 실행하고, 끝나면 `torch==2.6.0+cu124`로 복원한다.

## 기술 스택

PyTorch · Ultralytics YOLO11 · CLIP(open_clip) · Qdrant · easyOCR · ONNX · H100(cu124)
