# AtoCatch 🩺

> **AI 기반 아토피 피부염 조기 예측 및 홈케어 플랫폼**

아이의 피부 사진 한 장과 간단한 설문만으로 아토피 여부 및 중증도를 예측하고, 맞춤형 케어 가이드와 AI 챗봇 상담을 제공하는 Streamlit 웹 서비스입니다.

🔗 **라이브 데모**: https://atocatch.streamlit.app/

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 🔍 피부 이미지 분석 | EfficientNetV2-S 기반 아토피 유무 이진 분류 (Acc 80.6%, AUC 0.828) |
| 📊 중증도 분류 | IgA 기반 mild / moderate~severe 분류 (Acc 84.4%, AUC 0.876) |
| 🌡️ 히트맵 시각화 | Grad-CAM으로 모델이 주목한 피부 부위 시각화 |
| 📋 설문 위험도 예측 | 9가지 환경·유전 인자 기반 로지스틱 회귀 모델 |
| 🤖 AI 챗봇 상담 | OpenAI API 기반 아토피 전문 RAG 챗봇 |
| 📄 리포트 출력 | HTML 형식 분석 결과 보고서 다운로드 |
| 📅 기록 관리 | 회원별 진단 이력 조회 (최근 50건) |

---

## 모델 성능

| 모델 | Accuracy | F1-Score | AUC |
|------|----------|----------|-----|
| 아토피 유무 (EfficientNetV2-S) | 80.6% | 0.811 | 0.828 |
| 중증도 IgA (EfficientNetV2-S) | 84.4% | 0.843 | 0.876 |
| 위험도 설문 (Logistic Regression) | - | - | - |

---

## 프로젝트 구조

```
AtoCatch/
├── app/                              # Streamlit 웹 앱
│   ├── app_main.py                   # 메인 앱 (홈, 설문, 이미지 분석, 챗봇, 기록)
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
    │   ├── train_severity.py         # IgA 중증도 분류 학습
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

## 빠른 시작

### 1. 모델 가중치 다운로드

`.pth` 파일은 용량 문제로 GitHub에 포함되지 않습니다. 아래 Google Drive에서 받아 `app/` 디렉토리에 위치시켜 주세요.

| 파일 | 경로 | 링크 |
|------|------|------|
| `best_model.pth` | `app/best_model.pth` | [Google Drive](https://drive.google.com/file/d/1khrt-QelCpdcf8PbDCvMb5kVnRc2evP5/view) |
| `best_iga_model.pth` | `app/best_iga_model.pth` | [Google Drive](https://drive.google.com/file/d/1VydTQalT3hol_WwrA03NtrmlVURuUbnV/view) |

> Streamlit Cloud 배포 시에는 앱 실행 중 자동으로 다운로드됩니다.

### 2. 환경 설정

```bash
cd app
pip install -r requirements.txt
```

루트에 `.env` 파일 생성:
```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. 앱 실행

```bash
streamlit run app/app_main.py
```

---

## 데이터셋

| 데이터 | 출처 | 용도 |
|--------|------|------|
| 아토피 피부 이미지 | AI Hub (한국지능정보사회진흥원) | 이미지 분류 모델 학습 |
| 비아토피 이미지 | DermNet NZ | 이진 분류 비교군 |
| 설문 데이터 | KNHANES (국민건강영양조사) | 위험도 예측 모델 학습 |

> AI Hub 데이터는 라이선스 제한으로 본 레포지토리에 포함되지 않습니다.

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Frontend | Streamlit |
| 딥러닝 | PyTorch, timm (EfficientNetV2-S) |
| 머신러닝 | scikit-learn (Logistic Regression) |
| AI 챗봇 | OpenAI API |
| 시각화 | Grad-CAM, Plotly |
| 배포 | Streamlit Community Cloud |

---

## 스크린샷

| 로그인 | 홈 | 설문 | 피부 스캔 |
|--------|-----|------|----------|
| ![로그인](app/screenshots/login.png) | ![홈](app/screenshots/home.png) | ![설문](app/screenshots/survey.png) | ![스캔](app/screenshots/scan.png) |

| 분석 결과 | AI 챗봇 | 기록 보기 |
|-----------|---------|----------|
| ![결과](app/screenshots/results.png) | ![챗봇](app/screenshots/chatbot.png) | ![기록](app/screenshots/history.png) |

---

## 면책 조항

본 서비스의 결과는 AI 예측 수치이며, 의사의 진단이나 치료를 대체할 수 없습니다.  
정확한 진단은 소아청소년과 또는 피부과 전문의와 상담하시기 바랍니다.
