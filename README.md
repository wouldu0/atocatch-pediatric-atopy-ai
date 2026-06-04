<div align="center">

# 👶 AtoCatch

### AI 기반 영유아 아토피 위험도 예측 · 맞춤 홈케어 서비스

*"아이의 피부 사진 한 장과 간단한 설문으로, 집에서도 아토피를 조기에 잡는다"*

[![Streamlit App](https://img.shields.io/badge/🔗_라이브_데모-atocatch.streamlit.app-1b6554?style=for-the-badge)](https://atocatch.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

---

## 🍼 왜 만들었나요?

> **"영유아 아토피, 부모가 미리 알아차리기 어렵다"**

건조함·붉어짐·가려움은 흔한 증상이라 아토피 위험 신호인지 단순 피부 트러블인지 구분하기 어렵습니다.  
말 못하는 아이는 불편함을 표현할 수 없고, 맞벌이 부모는 반복 통원 자체가 큰 부담입니다.

| 📊 국내 아토피 현황 | |
|---|---|
| 👥 연간 진료 환자 | **97.3만 명** (전 연령대 만성 질환) |
| 🏥 연간 외래 청구 | **241만 건** (99.9% 소액 반복 구조) |
| 💸 요양급여 비용 | **1,972억 원** (2022년 대비 12% 증가) |
| 👶 최다 환자 연령 | **0~4세** (13.5만 명) |
| 📈 방치 시 비용 폭증 | 영유아기 1인당 5.6만 원 → **20대 39.9만 원 (7배)** |

> 💡 **영유아기 조기 발견·관리가 성인기 만성화와 의료비 부담을 줄이는 핵심입니다.**

---

## 🎯 AtoCatch가 해결하는 문제

| 문제 | 해결 방법 | 서비스 |
|------|-----------|--------|
| 잠재 위험 요인을 종합적으로 판단하기 어려움 | 로지스틱 회귀 기반 **아토피 발병 위험도 사전 예측** | 🔮 우리 아이 아토피 위험 미리보기 |
| 비전문가가 피부 트러블과 아토피를 육안으로 구분 불가 | EfficientNetV2-S 기반 **스마트폰 사진만으로 아토피 여부·중증도 확인** | 📸 우리 아이 아토피 상태 바로보기 |
| 잦은 통원과 반복 확인에 따른 시간·비용 부담 | LLM·RAG 기반 **AI 홈케어 + 증상 이력 자동 기록** | 🤖 24시간 AI 아토피 챗봇 & 기록보기 |

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🔍 **피부 이미지 분석** | EfficientNetV2-S — 아토피 유무 이진 분류 |
| 📊 **IGA 중증도 분류** | mild vs. moderate~severe 2단계 분류 |
| 🌡️ **Grad-CAM 히트맵** | 모델이 주목한 피부 부위 시각화 |
| 📋 **설문 위험도 예측** | 9가지 유전·환경 인자 기반 아토피 발병 위험도 |
| 🤖 **AI 챗봇 상담** | 임상 가이드라인 기반 RAG 아토피 전문 챗봇 |
| 📄 **리포트 출력** | HTML 형식 분석 결과 보고서 다운로드 |
| 📅 **기록 관리** | 회원별 진단 이력 조회 (최근 50건) |

---

## 🤖 AI 모델 상세

### 📸 이미지 분류 모델 (EfficientNetV2-S)

| | Model 2-A : 아토피 유무 | Model 2-B : IGA 중증도 |
|---|---|---|
| **Accuracy** | **80.6%** | **83.9%** |
| **F1-Score** | 81.1% | 83.7% |
| **AUC** | 0.828 | 0.876 |
| **Sensitivity** | 69.2% | 90.6% |
| **Threshold** | 0.29 (Youden's J) | 0.38 (F1 최적) |
| 학습 데이터 | AI Hub 합성 3,600장 + DermNet 160장 | AI Hub IGA 라벨 1,800장 |
| 외부 검증 | DermNet 실제 이미지 265장 | 합성 데이터 내부 검증 |

> 🔑 **핵심 인사이트**: 합성 데이터 단독 내부 성능 ~95% → 실제 이미지 외부 검증 55%로 급락.  
> DermNet 실제 이미지 160장 소량 믹싱만으로 외부 Acc **+14.3%p** 향상 (도메인 갭 공략)

### 📋 설문 위험도 모델 (Logistic Regression)

한국아동패널 데이터 N=1,967명 기반으로, 6차 시점(영유아기) 유전·환경 요인으로 7~10차 아토피 신규 발생을 예측합니다.

**통계적으로 유의한 핵심 위험 인자 (p < 0.05)**

| 위험 인자 | Odds Ratio | p-value | 해석 |
|-----------|-----------|---------|------|
| 💊 항생제 3회 이상 복용 | **1.963** | p<0.001*** | 아토피 위험 약 2배 |
| 👨‍👩‍👧 부모 아토피 진단 | **1.720** | p=0.011* | 유전적 소인 |
| 🌿 부모 알레르기 비염 | **1.389** | p=0.025* | 알레르기 체질 연관 |
| 🍄 실내 곰팡이 노출 | **1.367** | p=0.028* | 환경 위험 요인 |

| 모델 설정 | |
|---|---|
| 알고리즘 | Logistic Regression (L2, C=0.01) |
| 보정 | Isotonic Calibration |
| Threshold | 0.12 (고위험군 민감도 우선) |
| **Recall** | **0.7667** |
| 입력 변수 | 11개 이진 변수 |

---

## 📊 데이터셋

| 데이터 | 출처 | 규모 | 용도 |
|--------|------|------|------|
| 아토피·피부 이미지 | AI Hub (한국지능정보사회진흥원) | 10,800장 (6 클래스 × 1,800) | 이미지 분류 모델 학습 |
| 실제 피부 이미지 | DermNet NZ (직접 크롤링) | 265장 | 외부 검증·도메인 갭 보완 |
| 영유아 패널 데이터 | 한국아동패널 (1~10차) | N=1,967명 | 설문 위험도 모델 학습 |

> ⚠️ AI Hub 데이터는 라이선스 제한으로 레포지토리에 포함되지 않습니다.

---

## 🏗️ 프로젝트 구조

```
AtoCatch/
├── app/                              # Streamlit 웹 앱
│   ├── app_main.py                   # 메인 앱 (홈·설문·이미지 분석·챗봇·기록)
│   ├── gradcam_module.py             # Grad-CAM 시각화 모듈
│   ├── rag_engine.py                 # RAG 챗봇 엔진
│   ├── login_page.py                 # 로그인 페이지
│   ├── signup_page.py                # 회원가입 페이지
│   ├── model_config.json             # 아토피 유무 모델 설정
│   ├── model_config2.json            # 중증도 모델 설정
│   ├── atopy_service_model.joblib    # 설문 위험도 모델
│   ├── requirements.txt
│   ├── design/                       # UI 이미지 에셋
│   ├── screenshots/                  # 앱 스크린샷
│   └── survey_model/                 # 설문 모델 학습 데이터
│
└── training/                         # 모델 학습 코드
    ├── image_classification/
    │   ├── train_binary.py           # 아토피 유무 이진 분류 학습
    │   ├── train_binary_final.py     # 최종 이진 분류 모델
    │   ├── train_severity.py         # IGA 중증도 분류 학습
    │   ├── predict.py                # 단일 이미지 추론
    │   ├── data_setup.py             # 데이터 초기 설정
    │   ├── data_split.py             # Train/Val/Test 분할
    │   ├── data_split_raw.py         # 원시 데이터 분할 유틸
    │   ├── data_prepare.py           # 데이터셋 준비
    │   ├── data_processing.py        # 이미지 전처리
    │   ├── data_matching.py          # 이미지 매칭
    │   ├── utils_gradcam.py          # Grad-CAM 모듈
    │   ├── utils_grade.py            # 중증도 등급 모듈
    │   ├── utils_threshold.py        # 임계값 최적화 (Youden's J)
    │   ├── utils_early_stopping.py   # Early stopping 구현
    │   ├── utils_log.py              # 학습 로그
    │   └── eval_comparison.py        # 모델 비교 실험
    └── survey_model/
        ├── train_features.py         # 설문 피처 학습 (로지스틱 회귀)
        ├── train_xgboost.py          # XGBoost 실험
        ├── data_merge.py             # 원시 데이터 병합
        ├── data_merge_v2.py          # 데이터 병합 v2
        └── eval_univariate.py        # 단변량 분석
```

---

## 🚀 빠른 시작

### 1. 모델 가중치 다운로드

`.pth` 파일은 용량 문제로 GitHub에 포함되지 않습니다. `app/` 디렉토리에 위치시켜 주세요.

| 파일 | 용도 | 링크 |
|------|------|------|
| `best_model.pth` | 아토피 유무 분류 | [Google Drive](https://drive.google.com/file/d/1khrt-QelCpdcf8PbDCvMb5kVnRc2evP5/view) |
| `best_iga_model.pth` | IGA 중증도 분류 | [Google Drive](https://drive.google.com/file/d/1VydTQalT3hol_WwrA03NtrmlVURuUbnV/view) |

> Streamlit Cloud 배포 시에는 앱 실행 중 자동으로 다운로드됩니다.

### 2. 환경 설정

```bash
cd app
pip install -r requirements.txt
```

`app/` 폴더 안에 `.env` 파일 생성:
```env
OPENAI_API_KEY=your_openai_api_key
```

### 3. 앱 실행

```bash
streamlit run app/app_main.py
```

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

| 분류 | 기술 |
|------|------|
| Frontend | Streamlit |
| 딥러닝 | PyTorch, timm (EfficientNetV2-S) |
| 머신러닝 | scikit-learn (Logistic Regression + Isotonic Calibration) |
| AI 챗봇 | OpenAI API + RAG |
| 시각화 | Grad-CAM, Plotly |
| 배포 | Streamlit Community Cloud |

---

## ⚠️ 면책 조항

본 서비스의 결과는 AI 예측 수치이며, **의사의 진단이나 치료를 대체할 수 없습니다.**  
정확한 진단과 치료는 소아청소년과 또는 피부과 전문의와 상담하시기 바랍니다.
