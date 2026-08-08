<div align="center">

# 👶 AtoCatch

**AI 기반 영유아 아토피 위험도 예측 · 맞춤 홈케어 서비스**

> 스마트폰 피부 사진 한 장과 간단한 설문만으로, 집에서도 아토피 위험 신호를 확인하고 관리한다

[![Streamlit App](https://img.shields.io/badge/🔗_라이브_데모-atocatch.streamlit.app-1b6554?style=for-the-badge)](https://atocatch.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

---

## 📌 프로젝트 소개

AtoCatch는 피부 이미지 분석 모델과 공공데이터 기반 설문 위험도 예측 모델을 결합해, 영유아 아토피 피부염의 위험 신호를 조기에 확인하고 관리하도록 돕는 AI 헬스케어 서비스입니다. 국내 아토피 환자 중 0~4세가 13.5만 명으로 가장 많고, 방치 시 1인당 의료비가 20대 기준 7배까지 늘어난다는 점에서 영유아기 조기 발견의 필요성에서 출발했습니다.

| 구분 | 내용 |
|---|---|
| 개발 기간 | 2026.05 · 약 1개월 |
| 팀 구성 | 3인 |
| 타깃 | 영유아(0~5세) 보호자 |
| 이미지 데이터 | AI Hub 합성 10,800장 + DermNet NZ 265장 |
| 설문 데이터 | 한국아동패널 1~10차, N=1,967명 |
| 이미지 모델 | EfficientNetV2-S (아토피 유무 + IGA 중증도) |
| 설문 모델 | Logistic Regression |
| 서비스 | Streamlit 멀티페이지 웹앱 + RAG 챗봇 |

**핵심 기능**

- **피부 이미지 분석** — 스마트폰 사진으로 아토피 유무·IGA 중증도 판별, Grad-CAM으로 근거 부위 시각화
- **설문 위험도 예측** — 부모 병력·환경 노출 등 11개 변수로 아토피 발병 위험도를 저/중/고위험 3단계로 안내
- **AI 홈케어 챗봇** — 임상 가이드라인 기반 RAG 챗봇 정보 제공 + 분석 이력 자동 기록

---

## 👩‍💻 담당 역할

3인 팀에서 **이미지 분류 모델 전체**와 **설문 데이터 선정·분석**을 담당했습니다.

| 구분 | 담당 내용 |
|---|---|
| 이미지 모델링 | 데이터 전처리, 아키텍처 비교, 학습·평가, Grad-CAM, 배포 후 leakage 감사 및 재학습까지 전체 |
| 설문 데이터 | 한국아동패널 데이터 선정, 전처리·파생변수 설계·통계분석(단변량/VIF/로지스틱) |

> 설문 위험도 모델의 학습·threshold 선정(Logistic Regression)은 팀원이 진행했습니다. 웹 서비스(Streamlit 앱 구현)는 다른 팀원이 담당했습니다.

---

## 📊 주요 결과

### 이미지 분류 모델

| 태스크 | Accuracy | F1 | AUC | Sensitivity |
|---|---:|---:|---:|---:|
| 아토피 유무 (DermNet holdout) | 79.6% | 80.9% | 0.839 | 88.5% |
| IGA 중증도 (내부 검증) | 83.9% | 83.7% | 0.876 | 90.6% |

> 🔑 합성 데이터 단독 모델은 DermNet에서 62.6%로 하락했고(도메인 갭), 실제 이미지 mixing으로 76.9%까지 개선했습니다. 이후 base-id leakage를 수정해 재학습한 현재 배포 모델은 79.6%, AUC 0.839를 기록했습니다. 위 수치는 완전히 독립적인 external test가 아니라 **DermNet holdout 기준**입니다 — 자세한 내용은 아래 "모델 검증과 의사결정" 참고.

### 설문 위험도 모델

| 지표 | 결과 |
|---|---:|
| Test AUC | 0.629 |
| Test Recall | 0.7667 (threshold 0.12) |
| 핵심 위험 인자 (p<0.05) | 항생제 3회↑(OR 1.96) · 부모 아토피(1.72) · 부모 비염(1.39) · 실내 곰팡이(1.37) |

> 통계적으로 유의한 위험 인자가 확인됐지만, 개인 단위 발병 예측력(AUC 0.629)은 제한적이었습니다. 배포 모델은 그대로 유지하되, 사후 검증에서 outcome 정의에 방법론적 한계가 있는 것을 발견해 별도로 투명하게 공개하고 있습니다 — 아래 참고.

---

## 🌐 서비스 흐름

```
로그인 → 홈
   ├─ 설문조사 ──────→ Logistic Regression → 저 / 중 / 고위험 3단계
   └─ 피부 스캔 ──────→ EfficientNetV2-S (아토피 유무) → Grad-CAM
                              │ 아토피 의심 시
                              ▼
                        EfficientNetV2-S (IGA 중증도) → Grad-CAM
   ↓
종합 분석 결과 리포트 → AI 챗봇 정보 제공(RAG) / 분석 이력 관리
```

설문 모델과 이미지 모델은 서로 다른 정보(과거 병력·환경 vs. 현재 피부 상태)를 측정하기 때문에 하나의 점수로 합산하지 않고 독립적으로 제시합니다.

---

## 🔍 모델 검증과 의사결정

### 1. 합성 → 실사 도메인 갭과 DermNet mixing

AI Hub 합성 이미지로만 학습하면 내부 성능은 95.8%였지만, 실제 DermNet 이미지 265장 평가에서는 62.6%로 급락했습니다. DermNet 실제 이미지 157장을 학습에 소량 믹싱하는 것만으로 나머지 108장 홀드아웃 기준 Acc가 76.9%까지(+14.2%p) 개선돼, 이 방식을 최종 파이프라인에 반영했습니다. 이후 아래 2번의 base-id leakage를 수정해 재학습한 현재 배포 모델은 이 108장 홀드아웃 기준 79.6%, AUC 0.839를 기록합니다.

### 2. AI Hub base-id 그룹 leakage 발견 → 재학습

배포 후 재검토에서 AI Hub 라벨 데이터의 `정면` 폴더 내부에 **같은 base-id가 신체부위·병변 코드만 다르게 여러 장 존재하는 중복 패턴**을 발견했습니다(같은 합성 케이스로 추정 — 실제 동일 "환자"라는 원본 메타데이터로 증명된 것은 아니라 patient-level이 아닌 **base-id group-preserving split**이라고 표기합니다). 기존 `data_split.py`의 순수 랜덤 분할은 이 그룹을 쪼개 train/test에 걸치게 했고(전체 풀의 5.33%, 16개 그룹은 train·test 동시 노출로 직접적 leakage), `make_grouped_split.py`(GroupShuffleSplit)로 그룹을 보존해 재학습했습니다.

| | 기존(랜덤 split, leakage 있음) | 재학습(그룹 보존 split) |
|---|---:|---:|
| DermNet holdout Acc | 80.6% | 79.6% |
| DermNet holdout AUC | 0.828 | **0.839** |
| DermNet holdout Sensitivity | 69.2% | **88.5%** |

leakage 수정 후에도 동일 DermNet holdout에서는 성능 저하가 관찰되지 않았고, AUC와 Sensitivity는 오히려 개선되었습니다. 검증 후 재학습된 모델로 배포를 교체했습니다.

<details>
<summary>재현 스크립트와 남은 한계 보기</summary>

- 점검: `check_aihub_subject_leakage.py` / 재분할 유틸: `make_grouped_split.py` / 재학습: `train_binary_grouped_final.py`(`eval_comparison.py`의 학습 루프를 그대로 재사용하고 데이터 split만 교체)
- DermNet holdout(108장)이 아키텍처 선택(`eval_comparison.py`)과 threshold 설정(`utils_threshold.py`)에도 재사용돼, 위 수치는 완전히 독립적인 external test는 아닙니다.
- 결과는 seed=42 1회 실행 기준이라 다른 seed에서도 안정적인지는 아직 확인하지 못했습니다.

</details>

### 3. 설문 outcome(Y) 정의 감사

배포된 설문 모델의 원본 학습 스크립트(`train_final_service_model.py`)를 찾아 재실행한 결과, 배포본과 계수가 최대 오차 2.9×10⁻¹⁶(부동소수점 수준)로 완전히 일치함을 확인했습니다. 이 과정에서 Y(7~10차 신규 아토피 발생) 정의를 한국아동패널 공식 코드북과 대조하다가 결측 처리 문제를 발견했습니다: 8·9차는 코드북상 `1=예/2=아니오/99999999=무응답`으로 명확한데 기존 전처리가 무응답까지 전부 "미진단"으로 처리하고 있었고, 추적 상태를 엄격히 재구성하면 기존 Y=0(1,666명) 중 **430명(25.8%)은 실제로는 follow-up 정보가 없는 상태**였습니다.

| 분석 | N | AUC |
|---|---:|---:|
| 배포 서비스 모델 (Original) | 1,967 | 0.629 |
| 7~10차 엄격 추적 검토 (Sensitivity) | 1,537 | 0.555 |
| 9차 명시적 응답만 사용 (Primary) | 1,306 | 0.575 |

더 엄격한 정의에서는 AUC가 낮아져, 기존 성능이 outcome 결측 처리 방식에 영향을 받았을 가능성을 확인했습니다. 다만 N·발생률·대상자 구성이 함께 바뀌기 때문에 "기존 모델이 잘못된 라벨 덕분에 성능이 높았다"고 단정하지는 않습니다. **이 재검증은 방법론 감사 목적이며 배포 모델(threshold 0.12)에는 반영하지 않았습니다** — 서비스 코드 자체(전처리 파이프라인, threshold 흐름)는 정상 동작을 실제 실행으로 확인했고, 문제를 찾아 투명하게 공개해 관리하는 상태입니다.

<details>
<summary>재현 스크립트와 세부 근거 보기</summary>

- 재학습: `training/survey_model/train_corrected_outcome_model.py` (threshold는 버전마다 validation에서 새로 탐색, 기존 0.12 재사용 안 함)
- 7·10차는 문항 형식이 달라 참여 여부 변수(`DCh14hlt021`, `JCh17int001`)로 조사 참여만 확인했고, "무응답=미진단" skip-logic 자체는 코드북에서 직접 확인하지 못해 `observed_negative`(운영적 정의)로 표기 — 8·9차만 쓰는 9차 단독 분석이 방법론적으로 가장 신뢰도가 높습니다.
- outcome-unknown 430명은 6차 시점 예측변수 결측률도 ~92%로 전반적 추적 탈락 패턴과 일치하지만, 무작위 탈락임을 통계적으로 입증하지는 못해 attrition bias 가능성은 남아 있습니다.

</details>

### 4. 생존분석 탐색 후 미채택

같은 데이터로 발병 시점까지 고려하는 Kaplan-Meier/Cox 비례위험모델도 시도했지만, 변수별로 검정 방법(Log-rank ↔ Cox 단변량 ↔ Cox 다변량)에 따라 유의성이 흔들려 신뢰도 있는 위험 인자를 확정하기 어려웠습니다. 로지스틱 회귀는 동일 데이터에서 4개 인자 모두 p<0.05로 안정적으로 유의해 서비스용 모델로 확정했습니다. 분석 코드는 `training/survival_analysis/`에 참고용으로 보존합니다.

---

## 🖼️ 스크린샷

| 로그인 | 홈 | 설문 | 피부 스캔 |
|:------:|:---:|:----:|:--------:|
| ![로그인](app/screenshots/login.png) | ![홈](app/screenshots/home.png) | ![설문](app/screenshots/survey.png) | ![스캔](app/screenshots/scan.png) |

| 분석 결과 | AI 챗봇 | 기록 보기 |
|:---------:|:-------:|:--------:|
| ![결과](app/screenshots/results.png) | ![챗봇](app/screenshots/chatbot.png) | ![기록](app/screenshots/history.png) |

---

## 🛠️ 기술 스택

Python 3.10+ · PyTorch · timm (EfficientNetV2-S) · scikit-learn · Streamlit · OpenAI API + RAG · Grad-CAM · Plotly

## 📁 데이터

| 데이터 | 출처 | 규모 | 용도 |
|--------|------|------|------|
| 아토피·피부 이미지 | AI Hub (한국지능정보사회진흥원) | 10,800장 (6 클래스 × 1,800) | 이미지 분류 모델 학습 |
| 실제 피부 이미지 | DermNet NZ (직접 크롤링) | 265장 | Holdout 평가·도메인 갭 보완 |
| 영유아 패널 데이터 | 한국아동패널 (1~10차) | N=1,967명 | 설문 위험도 모델 학습 |

> ⚠️ AI Hub 데이터는 라이선스 제한으로 레포지토리에 포함되지 않습니다. DermNet NZ 크롤러는 `training/image_classification/data_crawl_dermnet.py`에서 확인할 수 있습니다(DermNet NZ 저작권 하에 있으므로 재사용 시 출처를 명시하세요).

---

<details>
<summary><b>🗂️ 프로젝트 구조</b></summary>

```
AtoCatch/
├── app/                              # Streamlit 웹 앱 (streamlit run app/app_main.py 로만 실행)
│   ├── app_main.py                   # 메인 앱 (로그인·회원가입·홈·설문·이미지 분석·챗봇·기록 전부 포함)
│   ├── gradcam_module.py             # Grad-CAM 시각화 모듈
│   ├── rag_engine.py                 # RAG 챗봇 엔진
│   ├── model_config.json             # 현재 배포 아토피 유무 모델의 threshold·성능 메타데이터
│   ├── model_config2.json            # 중증도 모델 메타데이터
│   ├── atopy_service_model.joblib    # 설문 위험도 모델 (학습된 결과물)
│   ├── best_model.pth / best_iga_model.pth  # 이미지 모델 가중치 (레포에 직접 커밋)
│   ├── requirements.txt
│   ├── design/                       # UI 이미지 에셋
│   ├── screenshots/                  # 앱 스크린샷
│   └── survey_model/                 # 설문 모델 계수·학습 데이터
│
└── training/                         # 모델 학습 코드
    ├── image_classification/
    │   ├── train_binary.py               # 아토피 유무 이진 분류 학습 (v1, 합성데이터만)
    │   ├── train_binary_dermnet_mix_v2.py # 이진 분류 학습 v2 (DermNet 믹싱, efficientnet_b0)
    │   ├── eval_comparison.py            # 5개 아키텍처 비교 실험 (아키텍처 선정용)
    │   ├── eval_binary_legacy_holdout.py # (구) DermNet 60/40 holdout 재평가 코드 — 참고용
    │   ├── make_grouped_split.py         # base-id 그룹 보존 split 유틸 (leakage 수정용)
    │   ├── train_binary_grouped_final.py # 현재 배포 모델(best_model.pth) 실제 학습 스크립트
    │   ├── check_aihub_subject_leakage.py # AI Hub base-id 중복(leakage) 점검
    │   ├── predict.py                    # 단일 이미지 추론 (현재 배포 모델 사용)
    │   ├── data_setup.py / data_split.py / data_split_raw.py / data_prepare.py / data_matching.py
    │   ├── utils_gradcam.py              # Grad-CAM 모듈
    │   ├── utils_threshold.py            # 임계값 최적화 (Youden's J)
    │   └── data_crawl_dermnet.py         # DermNet NZ 이미지 크롤러
    ├── survey_model/
    │   ├── train_features.py             # 설문 피처 가공 (모델 학습은 하지 않음)
    │   ├── train_xgboost.py              # XGBoost 비교 실험
    │   ├── eval_logistic_v2.py           # 로지스틱 회귀 v1/v2 변수셋 비교
    │   ├── train_final_service_model.py  # atopy_service_model.joblib 원본 학습 스크립트 (bit-exact 검증됨)
    │   ├── train_corrected_outcome_model.py # 사후 outcome 감사용 재학습 (서비스 미적용)
    │   ├── data_merge.py / data_build_ad_history.py
    │   └── eval_univariate.py
    └── survival_analysis/                # 생존분석 (탐색 후 미채택, 참고용)
        ├── eval_survival_v1.py / eval_survival_v2.py / eval_survival_v3_early_predictors.py
        └── results/
```

> ⚠️ IGA 중증도 모델(`best_iga_model.pth`)을 실제로 학습한 스크립트는 어디에도 없습니다. 확인된 것은 `tf_efficientnetv2_s(num_classes=2)`에 strict load로 정상 로드된다는 아키텍처 일치뿐이고, optimizer·split·seed 등 학습 설정은 불명이라 추정해서 만든 코드는 추가하지 않았습니다(아키텍처 추정 재현 ≠ 실제 재현이기 때문).

</details>

<details>
<summary><b>⚙️ 실행 방법</b></summary>

`best_model.pth`, `best_iga_model.pth` 모두 `app/`에 레포와 함께 커밋되어 있어 별도 다운로드가 필요 없습니다(각 77.8MB, GitHub 100MB 리밋 이내).

```bash
cd app
pip install -r requirements.txt
```

`app/` 폴더 안에 `.env` 파일 생성:
```env
OPENAI_API_KEY=your_openai_api_key
```

```bash
streamlit run app/app_main.py
```

</details>

<details>
<summary><b>🧹 프로젝트 종료 후 리팩터링</b></summary>

포트폴리오 정리 과정에서 저장소와 서비스 코드를 다시 검증해 다음을 수정했습니다.

- **설문 위험도 예측 인코딩 반전 버그 수정**: 8개 설문 항목의 예/아니오가 학습 데이터와 반대로 모델에 들어가던 버그, 항생제 응답 카테고리 off-by-one — 실제 서비스 예측 결과에 영향 있던 버그라 실행 테스트로 수정 확인
- 파일명과 실제 내용이 다른 스크립트 다수 정리·개명, 죽은 코드(안 쓰는 로그인/회원가입 모듈, 구버전 앱 스냅샷 등) 삭제
- 원본 학습 스크립트를 찾아 배포된 설문 모델과 계수 bit-exact 일치 검증, 사업계획서의 Isotonic Calibration 오기재 정정
- 모델 가중치를 Google Drive 런타임 다운로드 방식에서 레포 직접 커밋으로 전환 (외부 의존성 제거)
- AI Hub base-id leakage 발견 → grouped split 재학습 → 배포 모델 교체 (위 "모델 검증과 의사결정" 참고)
- 설문 모델 outcome(Y) 정의를 공식 코드북과 대조해 방법론적 한계 발견·투명하게 공개 (위 참고)
- 재학습 스크립트의 출력 경로 등 프로젝트 내부 경로를 상대경로로 정리 (AI Hub/DermNet 원본처럼 레포에 없는 외부 대용량 데이터 루트는 다른 학습 스크립트들과 동일하게 사용자가 직접 지정하는 절대경로로 유지)

</details>

---

## ⚠️ 한계

- **설문 위험도 모델**: 배포 모델의 outcome(7~10차 신규 아토피 발생) 정의를 코드북과 대조한 결과 기존 Y=0의 25.8%가 추적 정보 불충분 상태였고, 엄격한 정의로 재분석하면 AUC가 낮아집니다. 이 재검증은 방법론 감사 목적이며 배포 모델에는 반영하지 않았습니다.
- 위와 같은 이유로 outcome-unknown 제외 표본의 attrition bias 가능성이 남아 있습니다(무작위 탈락임을 통계적으로 입증하지는 못함).
- **이미지 모델**: DermNet holdout(108장)이 아키텍처·threshold 선택과 최종 성능 보고에 반복 사용돼, 보고된 수치가 완전히 독립적인 external test 성능은 아닙니다.
- base-id 그룹 보존 재분할 결과가 seed=42 1회 실행 기준이라, 다른 seed에서도 안정적인지는 추가 검증이 필요합니다.

---

## ⚠️ 사용 안내

본 서비스의 결과는 AI 예측 수치이며, **의사의 진단이나 치료를 대체할 수 없습니다.** 정확한 진단과 치료는 소아청소년과 또는 피부과 전문의와 상담하시기 바랍니다.
