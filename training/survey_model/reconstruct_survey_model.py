"""
설문 위험도 모델(app/atopy_service_model.joblib) 재구성 스크립트
==============================================================
역할: 저장된 배포 모델(joblib)에서 확인한 preprocessing/model architecture를
      기반으로 학습 파이프라인을 재구성한다. 원 학습 split/seed 기록이 남아
      있지 않아 기존 모델과 동일한 coefficients의 완전 재현(exact reproduction)은
      보장하지 않는다.

      이 스크립트는 "원 학습 코드를 복원한 것"이 아니라 "현재 시점에서 배포
      artifact로부터 역추적한 구조를 새로 재구성한 것"이다 — 둘은 다른 것이다.
      과거 학습 로그·노트북 등 원본이 발견되면 이 스크립트를 대체해야 한다.

[joblib.load()로 확인한 배포 파이프라인 구조 — 추측 아님, 직접 읽은 사실]
    Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat',  Pipeline([SimpleImputer(strategy='most_frequent'),
                                OneHotEncoder(handle_unknown='ignore', sparse_output=False)]),
                     [antibiotic, parent_AD, parent_AR, mold_ever, parent_asthma,
                      sibling_allergy, pet_ever, passive_smoke_ever, child_passive_smoke]),
            ('cont', Pipeline([SimpleImputer(strategy='median'), StandardScaler()]),
                     [rural_years, outdoor_avg]),
        ])),
        ('model', LogisticRegression(C=0.01, max_iter=3000, random_state=42)),
    ])
    ※ README/사업계획서 PPT는 "Isotonic Calibration 적용"이라고 설명하지만,
      실제 joblib 안에는 CalibratedClassifierCV도 IsotonicRegression도 없다.
      즉 배포된 모델은 보정되지 않은 순수 LogisticRegression이다 (이 사실은 확실함).

[재현 결과 — 불확실한 부분]
    위 구조 그대로 pskc_final.csv(N=1,967) 전체로 재학습하면 계수의 부호·상대적
    크기는 배포본과 일치하지만 절대값은 정확히 일치하지 않는다 (예: mold_ever=1.0
    계수가 배포본 0.1438 vs 재현 0.1314). test_size 0.15/0.2/0.25/0.3 ×
    stratify 유무 조합 8가지를 추가로 시도했으나 정확히 일치하는 조합을 찾지 못했다.
    11개 입력 변수 모두 27~34%의 결측률을 가지고 있어(예: sibling_allergy 673/1967
    결측), imputation 전략·train/test split·행 필터링 중 어느 하나라도 원본과
    다르면 이 정도 오차가 발생할 수 있다. 즉 파이프라인 구조는 확인됐지만
    "정확히 어떤 데이터 서브셋으로 학습했는지"는 이 레포의 자료만으로는 특정할 수 없다.
    → 아래 REPRODUCE()는 전체 데이터로 학습하는 가장 근접한 시도이며,
      원본 joblib을 덮어쓰지 않는다.

실행: python reconstruct_survey_model.py
      (backup/data/pskc_final.csv 경로를 DATA_PATH에 맞게 수정할 것 — 대용량 원본
       데이터라 이 레포에는 포함되어 있지 않음)
"""
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = "pskc_final.csv"  # ← 본인 경로로 수정 (백업의 backup/data/pskc_final.csv)
DEPLOYED_JOBLIB_PATH = "../../app/atopy_service_model.joblib"  # 비교 대상, 절대 덮어쓰지 않음
OUTPUT_PATH = "reconstructed_model.joblib"  # 이 스크립트의 산출물 (배포본과 별도 파일, exact reproduction 아님)

CAT_COLS = ["antibiotic", "parent_AD", "parent_AR", "mold_ever", "parent_asthma",
            "sibling_allergy", "pet_ever", "passive_smoke_ever", "child_passive_smoke"]
CONT_COLS = ["rural_years", "outdoor_avg"]


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CAT_COLS),
        ("cont", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), CONT_COLS),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(C=0.01, max_iter=3000, random_state=42)),
    ])


def coef_table(pipe):
    lr = pipe.named_steps["model"]
    ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    names = list(ohe.get_feature_names_out()) + CONT_COLS
    return pd.Series(lr.coef_[0], index=names), lr.intercept_[0]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[CAT_COLS + CONT_COLS]
    y = df["Y"]
    print(f"데이터: {len(df)}행, 양성(Y=1) {int(y.sum())}명 ({y.mean()*100:.1f}%)")
    print("결측률:")
    print((X.isna().mean() * 100).round(1).to_string())

    pipe = build_pipeline()
    pipe.fit(X, y)  # 원본 split/seed를 알 수 없어 전체 데이터로 재구성 학습 (가장 근접한 시도)
    my_coef, my_intercept = coef_table(pipe)

    print(f"\n[재구성 결과] intercept={my_intercept:.4f}")
    print(my_coef.round(4).to_string())

    try:
        deployed = joblib.load(DEPLOYED_JOBLIB_PATH)
        dep_coef, dep_intercept = coef_table(deployed)
        compare = pd.DataFrame({"deployed": dep_coef, "reconstructed": my_coef})
        compare["diff"] = (compare["reconstructed"] - compare["deployed"]).round(4)
        print(f"\n[배포본과 비교] intercept 차이: {my_intercept - dep_intercept:.4f}")
        print(compare.round(4).to_string())
        print(
            "\n⚠️ 부호와 상대적 크기는 배포본과 일치하지만 절대값은 정확히 일치하지 않습니다 "
            "(exact reproduction 아님). 원본 train/test split·random seed 기록이 남아있지 않아 "
            "완전히 동일한 coefficients 재현은 보장하지 않습니다. 자세한 내용은 이 파일 상단 docstring 참고."
        )
    except FileNotFoundError:
        print(f"\n배포본 joblib을 찾지 못해 비교를 건너뜁니다: {DEPLOYED_JOBLIB_PATH}")

    joblib.dump(pipe, OUTPUT_PATH)
    print(f"\n재구성 결과를 저장했습니다 (배포본과 별도 파일, 배포본을 대체하지 않음): {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
