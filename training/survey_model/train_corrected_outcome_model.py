"""
설문 위험도 모델 — 수정된 Outcome(Y) 정의로 재학습 (기존 서비스 모델과 별도)
==============================================================
배경: train_final_service_model.py가 만드는 배포 모델(atopy_service_model.joblib)의
Y(7~10차 아토피 신규 발생) 정의를 한국아동패널 공식 코드북과 대조한 결과, 다음
문제가 확인됐다.

  - 8차(KCh15adx004)·9차(KCh16adx004): 코드북상 1=예/2=아니오/99999999=무응답인데,
    기존 코드는 1이 아니면(빈칸·2·99999999 전부) 0으로 처리 — 8차 549명, 9차 661명이
    "미진단"으로 잘못 들어감.
  - 10차(DCh17hlt031k)는 "진단 연도" 필드라 값이 있으면 진단은 맞지만, 빈칸이
    미진단인지 미참여인지 구분이 안 됨. 참여 게이트(JCh17int001=1)로 확인한 결과
    636명이 실제로는 10차 자체에 참여하지 않은 미응답자였음.
  - 7차(DCh14hlt031f)도 체크리스트형이라 같은 문제. 참여 게이트(DCh14hlt021 응답 여부)로
    확인 결과 508명이 7차 건강모듈 미참여자.
  - 이 결과, 기존 Y=0(1,666명) 중 430명(25.8%)은 7~10차 어디에서도 명시적 "아니오"
    확인이 안 되는 outcome_unknown(추적 근거 없음)이었다. 이 430명은 6차 시점
    predictor 결측률도 ~92%로 전반적인 추적 탈락 패턴과 일치했다 — 다만 추적 탈락이
    무작위라는 것을 입증할 수는 없어 attrition bias 가능성은 남아 있다.

이 스크립트는 위 문제를 반영해 Y를 다시 정의하고, 기존과 동일한 파이프라인
구조(11개 변수, LogisticRegression C=0.01, 60/20/20 split)로 두 가지 버전을
새로 학습한다. **threshold 0.12는 기존 Y 정의에서 선택된 값이라 재사용하지
않고, 각 버전마다 validation에서 F2 기준으로 새로 탐색한다.**

  - Primary analysis  : 9차(KCh16adx004, "출생~현재 진단 여부") 단독을 outcome으로
                         사용. 8·9차는 코드북상 명시적 1/2/무응답 코드가 있어 가장
                         신뢰도 높은 endpoint.
  - Sensitivity analysis: 7~10차 전체 정보를 쓰되, 각 차수 missing/skip 구조를
                         엄격히 적용.

⚠️ **"음성" 라벨의 근거 수준이 차수마다 다르다.** 8·9차는 코드북에 1=예/2=아니오가
명시된 진짜 응답이라 신뢰도가 높다. 반면 7·10차의 "음성"은 참여 게이트 변수
(DCh14hlt021 응답 여부, JCh17int001=1)로 "이 회차 조사에 참여했다"만 확인한 것이지,
"아토피 항목에 대한 무응답=미진단"이라는 skip-logic 자체를 코드북에서 직접 확인한
것은 아니다. 그래서 이 라벨을 `confirmed_negative`가 아니라 `observed_negative`
(운영적 정의상 음성 — 참여는 확인됐지만 항목별 skip 구조까지 검증되지는 않음)로
부른다. 8·9차만 쓰는 primary analysis가 방법론적으로 가장 신뢰할 수 있는 버전이다.

⚠️ 이 스크립트는 기존 atopy_service_model.joblib과 README 핵심 수치를 변경하지
않는다. 산출물은 별도 폴더(outputs_corrected_outcome/)에 저장하고, 기존 서비스
모델과의 비교 결과만 출력한다.

실행 전: MERGED_PATH(backup/data/merged.csv), PSKC_FINAL_PATH(pskc_final.csv),
W7_RAW_PATH/W10_RAW_PATH(원본 wave 7/10 csv, 참여 게이트 변수용) 경로를 맞출 것.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score, fbeta_score,
    confusion_matrix,
)

DATA_DIR = "data/"  # ← 본인 경로로 수정
MERGED_PATH = os.path.join(DATA_DIR, "merged.csv")
PSKC_FINAL_PATH = os.path.join(DATA_DIR, "pskc_final.csv")
W7_RAW_PATH = os.path.join(DATA_DIR, "w7_2014_data_230411.csv")
W10_RAW_PATH = os.path.join(DATA_DIR, "w10_2017_data_250211.csv")
DEPLOYED_JOBLIB_PATH = "../../app/atopy_service_model.joblib"

OUT_DIR = "outputs_corrected_outcome"
os.makedirs(OUT_DIR, exist_ok=True)

SERVICE_CAT_COLS = [
    "antibiotic", "parent_AD", "parent_AR", "mold_ever", "parent_asthma",
    "sibling_allergy", "pet_ever", "passive_smoke_ever", "child_passive_smoke",
]
SERVICE_CONT_COLS = ["rural_years", "outdoor_avg"]
SERVICE_FEATURES = SERVICE_CAT_COLS + SERVICE_CONT_COLS
RANDOM_STATE = 42


def build_corrected_outcomes():
    """Y_old(기존 정의)와 Y_primary_w9 / Y_sensitivity_7to10(수정 정의)를 만든다."""
    df = pd.read_csv(MERGED_PATH, low_memory=False)
    raw7 = pd.read_csv(W7_RAW_PATH, low_memory=False, encoding="cp949")[["N_ID", "DCh14hlt021"]]
    raw10 = pd.read_csv(W10_RAW_PATH, low_memory=False, encoding="cp949")[["N_ID", "JCh17int001"]]
    df = df.merge(raw7, on="N_ID", how="left").merge(raw10, on="N_ID", how="left")

    def raw_str(col):
        return df[col].astype(str).str.strip() if col in df.columns else pd.Series("", index=df.index)

    # baseline: 6차까지 미진단 (기존과 동일)
    for w, col in [(3, "ECh10hlt035_w3"), (4, "DCh11hlt035_w4"), (5, "DCh12hlt035_w5"), (6, "DCh13hlt035_w6")]:
        df[f"ad_w{w}"] = (raw_str(col) == "6").astype(float)
    ad_3to6 = [f"ad_w{w}" for w in [3, 4, 5, 6]]
    baseline_mask = df[ad_3to6].apply(lambda row: not any(row == 1), axis=1)
    base = df[baseline_mask].copy()

    # 기존(OLD) 방식
    base["ad_w7_old"] = (base["DCh14hlt031f_w7"].astype(str).str.strip() == "6").astype(float)
    base["ad_w8_old"] = (base["KCh15adx004_w8"].astype(str).str.strip() == "1").astype(float)
    base["ad_w9_old"] = (base["KCh16adx004_w9"].astype(str).str.strip() == "1").astype(float)
    base["ad_w10_old"] = base["DCh17hlt031k_w10"].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip() in ["", "nan", "0"] else 1
    ).astype(float)
    base["Y_old"] = base[["ad_w7_old", "ad_w8_old", "ad_w9_old", "ad_w10_old"]].apply(
        lambda row: 1 if any(row == 1) else 0, axis=1
    )

    # 수정(NEW) 방식: 참여 게이트 + 명시적 코드
    g7 = base["DCh14hlt021"].astype(str).str.strip() != ""
    d7 = base["DCh14hlt031f_w7"].astype(str).str.strip() == "6"
    base["ad_w7_new"] = np.where(d7, 1, np.where(g7, 0, np.nan))

    def code12(col):
        s = base[col].astype(str).str.strip()
        out = pd.Series(np.nan, index=base.index)
        out[s == "1"] = 1
        out[s == "2"] = 0
        return out

    base["ad_w8_new"] = code12("KCh15adx004_w8")
    base["ad_w9_new"] = code12("KCh16adx004_w9")

    g10 = base["JCh17int001"] == 1
    d10 = base["DCh17hlt031k_w10"].astype(str).str.strip() != ""
    base["ad_w10_new"] = np.where(d10, 1, np.where(g10, 0, np.nan))

    def classify(row):
        vals = [v for v in row if pd.notna(v)]
        if any(v == 1 for v in vals):
            return "positive"
        if any(v == 0 for v in vals):
            return "observed_negative"
        return "outcome_unknown"

    cols_new = ["ad_w7_new", "ad_w8_new", "ad_w9_new", "ad_w10_new"]
    base["outcome_class"] = base[cols_new].apply(classify, axis=1)
    base["Y_sensitivity_7to10"] = base["outcome_class"].map(
        {"positive": 1, "observed_negative": 0, "outcome_unknown": np.nan}
    )
    base["Y_primary_w9"] = base["ad_w9_new"]

    return base[["N_ID", "Y_old", "Y_primary_w9", "Y_sensitivity_7to10", "outcome_class"]]


def make_pipeline():
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), SERVICE_CAT_COLS),
        ("cont", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), SERVICE_CONT_COLS),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=3000, solver="lbfgs", penalty="l2", C=0.01, random_state=RANDOM_STATE)),
    ])


def find_best_threshold_by_f2(y_true, proba):
    rows = []
    for th in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= th).astype(int)
        rows.append({
            "threshold": th,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        })
    th_df = pd.DataFrame(rows)
    best = th_df.sort_values(["f2", "recall"], ascending=False).iloc[0]
    return float(best["threshold"]), th_df


def train_and_eval(X, y, label):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    best_th, _ = find_best_threshold_by_f2(y_valid, valid_proba)

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_th).astype(int)

    result = {
        "label": label,
        "N_total": len(X) + 0,
        "N_train": len(X_train), "N_valid": len(X_valid), "N_test": len(X_test),
        "positive_rate": float(y.mean()),
        "threshold": best_th,
        "AUC": roc_auc_score(y_test, test_proba),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "F1": f1_score(y_test, test_pred, zero_division=0),
    }

    joblib.dump(pipe, os.path.join(OUT_DIR, f"model_{label}.joblib"))
    return result, pipe


def main():
    y_variants = build_corrected_outcomes()
    feat = pd.read_csv(PSKC_FINAL_PATH)
    merged = y_variants.merge(feat[["N_ID"] + SERVICE_FEATURES], on="N_ID", how="left")

    print("[Y 정의 수정 전/후 비교]")
    print(f"{'항목':<18}{'기존(Y_old)':>14}{'sensitivity(7~10차)':>22}{'primary(9차만)':>16}")
    n_old_valid = merged["Y_old"].notna().sum()
    n_sens_valid = merged["Y_sensitivity_7to10"].notna().sum()
    n_prim_valid = merged["Y_primary_w9"].notna().sum()
    print(f"{'분석대상 N':<18}{n_old_valid:>14}{n_sens_valid:>22}{n_prim_valid:>16}")
    print(f"{'Y=1':<18}{int((merged['Y_old']==1).sum()):>14}{int((merged['Y_sensitivity_7to10']==1).sum()):>22}{int((merged['Y_primary_w9']==1).sum()):>16}")
    print(f"{'Y=0':<18}{int((merged['Y_old']==0).sum()):>14}{int((merged['Y_sensitivity_7to10']==0).sum()):>22}{int((merged['Y_primary_w9']==0).sum()):>16}")
    print(f"{'발생률':<18}{merged['Y_old'].mean()*100:>13.2f}%{merged.loc[merged['Y_sensitivity_7to10'].notna(),'Y_sensitivity_7to10'].mean()*100:>21.2f}%{merged.loc[merged['Y_primary_w9'].notna(),'Y_primary_w9'].mean()*100:>15.2f}%")

    results = []

    # 기존 정의 (비교 기준선 — 배포본과 동일한 데이터로 재현)
    X_old = merged[SERVICE_FEATURES]
    y_old = merged["Y_old"].astype(int)
    r_old, _ = train_and_eval(X_old, y_old, "original_Y_old")
    results.append(r_old)

    # sensitivity analysis
    sub = merged[merged["Y_sensitivity_7to10"].notna()]
    r_sens, _ = train_and_eval(sub[SERVICE_FEATURES], sub["Y_sensitivity_7to10"].astype(int), "sensitivity_7to10")
    results.append(r_sens)

    # primary analysis (9차)
    sub9 = merged[merged["Y_primary_w9"].notna()]
    r_prim, _ = train_and_eval(sub9[SERVICE_FEATURES], sub9["Y_primary_w9"].astype(int), "primary_w9")
    results.append(r_prim)

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(OUT_DIR, "comparison_original_vs_corrected.csv"), index=False)
    print("\n[모델 비교]")
    print(result_df.round(4).to_string(index=False))
    print(f"\n저장 폴더: {OUT_DIR}/ (기존 app/atopy_service_model.joblib은 변경하지 않음)")


if __name__ == "__main__":
    main()
