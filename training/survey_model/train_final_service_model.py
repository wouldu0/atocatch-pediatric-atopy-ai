"""
설문 위험도 모델(app/atopy_service_model.joblib) 원본 학습 스크립트
==============================================================
로컬 백업에서 나중에 찾은 원본 파일. 배포된 joblib과 계수를 직접 대조해
bit-exact 일치를 확인했다 (최대 절대 오차 2.9e-16, 부동소수점 수준).
즉 이 파일이 `atopy_service_model.joblib`을 실제로 만든 스크립트다 —
`reconstruct_survey_model.py`처럼 구조를 추정해 재구성한 게 아니다.

파이프라인 전체 흐름:
  [1]~[6] 파생변수 생성 (merged.csv → pskc_final.csv, train_features.py와 동일)
  [7]     결측률/단변량 검정/VIF 기반 유의 변수 선별 (통계적 해석용)
  [8]     아파트/모유수유 변수 진단 (탐색적 점검)
  [9]     핵심 4개 변수(antibiotic, parent_AD, parent_AR, mold_ever)만으로 1차 모델 비교
  [10]    core4 + 환경/출생/식습관 변수 세트(A~F) × 4개 샘플링 전략 × 4개 모델
          RandomizedSearchCV 튜닝 비교 → C_core4_environment(11개 변수) +
          NoSampling + LogisticRegression(C=0.01)이 최종 채택됨
  [11]    서비스용 최종 모델 학습·저장 (atopy_service_model.joblib, threshold=0.12)

⚠️ 주의: [1]~[6]의 아토피 진단(ad_w3~ad_w10) 인코딩·Y 정의는 공식 한국아동패널
코드북 대조 결과 8~9차(KCh15/16adx004)에서 문제가 있는 것으로 확인됨
(1=예/2=아니오 명시적 구조인데 빈칸을 전부 0=미진단으로 처리 중, 99999999
정리도 Y 생성 이후에 실행됨). 이 스크립트는 원본 그대로 보존한 것이고,
Y 정의 수정은 별도로 검증 후 진행한다.

원본 파일명: "step2. features (5).py"
입력: merged.csv (data_merge.py 출력) / pskc_ad_history.csv
출력: pskc_final.csv, atopy_service_model.joblib, atopy_service_model_coefficients.csv

실행 전 DATA_DIR 경로만 수정할 것
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# ============================================================
# [설정]
# ============================================================
DATA_DIR = r"data/"  # ← 본인 경로로 수정
INPUT_FILE  = os.path.join(DATA_DIR, "merged.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "pskc_final.csv")

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"입력: {len(df)}행, {len(df.columns)}컬럼")


# ============================================================
# 유틸: 숫자 변환 헬퍼
# ============================================================
def to_num(series):
    """문자열 포함 컬럼을 숫자로 강제 변환, 변환 불가 → NaN"""
    return pd.to_numeric(series, errors='coerce')

def get_col(col):
    """컬럼이 있으면 숫자 변환해서 반환, 없으면 NaN 시리즈"""
    if col in df.columns:
        return to_num(df[col])
    return pd.Series(np.nan, index=df.index)


# ============================================================
# [1] 아토피 진단 → 0/1 변환
# ============================================================
print("\n[1] 아토피 진단 변환")

AD_COLS = {
    3:  ("ECh10hlt035_w3",  "str6"),
    4:  ("DCh11hlt035_w4",  "str6"),
    5:  ("DCh12hlt035_w5",  "str6"),
    6:  ("DCh13hlt035_w6",  "str6"),
    7:  ("DCh14hlt031f_w7", "str6"),
    8:  ("KCh15adx004_w8",  "str1"),
    9:  ("KCh16adx004_w9",  "str1"),
    10: ("DCh17hlt031k_w10","notnull"),
}

for w, (col, method) in AD_COLS.items():
    if col not in df.columns:
        print(f"  ⚠ {col} 없음")
        df[f"ad_w{w}"] = np.nan
        continue

    if method == "str6":
        df[f"ad_w{w}"] = (df[col].astype(str).str.strip() == '6').astype(float)
    elif method == "str1":
        df[f"ad_w{w}"] = (df[col].astype(str).str.strip() == '1').astype(float)
    elif method == "notnull":
        df[f"ad_w{w}"] = df[col].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() in ['', 'nan', '0'] else 1
        ).astype(float)

    n = int(df[f"ad_w{w}"].sum())
    print(f"  {w}차 AD 진단: {n}명")


# ============================================================
# [2] 6차까지 미진단 필터링
# ============================================================
print("\n[2] 6차까지 미진단 필터링")

ad_3to6 = [f"ad_w{w}" for w in [3,4,5,6] if f"ad_w{w}" in df.columns]
mask = df[ad_3to6].apply(lambda row: not any(row == 1), axis=1)
df = df[mask].copy()
print(f"  6차까지 미진단: {len(df)}명")


# ============================================================
# [3] 종속변수 Y 생성
# ============================================================
print("\n[3] 종속변수 Y 생성")

ad_7to10 = [f"ad_w{w}" for w in [7,8,9,10] if f"ad_w{w}" in df.columns]
df["Y"] = df[ad_7to10].apply(
    lambda row: 1 if any(row == 1) else 0, axis=1
)
print(f"  양성(Y=1): {int(df['Y'].sum())}명")
print(f"  음성(Y=0): {int((df['Y']==0).sum())}명")

# ============================================================
# [3.5] 원본 컬럼 무응답(99999999) 사전 처리
# ============================================================
print("\n[3.5] 원본 컬럼 무응답 사전 처리")
replaced_total = 0  # ← 여기서 초기화

for col in df.columns:
    mask_num = (df[col] == 99999999)
    mask_str = (df[col].astype(str).str.strip() == '99999999')
    combined = mask_num | mask_str
    n = int(combined.sum())
    if n > 0:
        df[col] = df[col].where(~combined, np.nan)
        replaced_total += n
        print(f"  {col}: {n}개 → NaN 처리")

print(f"  총 {replaced_total}개 → NaN 처리")

# ============================================================
# [4] 파생변수 생성
# ============================================================
print("\n[4] 파생변수 생성")

feat = pd.DataFrame()
feat["N_ID"] = df["N_ID"]
feat["Y"]    = df["Y"]


# --- 성별 ---
feat["sex"] = to_num(df.get("DCh09dmg001_w2", np.nan)).map({1: 0, 2: 1})
print("  ✓ 성별")

# --- 재태기간 ---
feat["gestation_weeks"] = get_col("BMt08pnb001_w1")

# --- 체중 ---
feat["birth_weight"] = get_col("BCh08hlt001_w1")

# --- 출산 방식 ---
feat["c_section"] = to_num(df.get("BMt08pnb003_w1", np.nan)).map({1: 0, 2: 1, 3: 1})
print("  ✓ 성별")

# --- 모유수유 ---
bf_yes_cols   = ["DCh11fed001_w4", "DCh10fed001_w3", "DCh09fed001_w2","DCh08fed001_w1"]
bf_month_cols = ["DCh11fed002_w4", "DCh10fed002_w3", "DCh09fed002_w2","DCh08fed001_w1"]

avail_yes   = [c for c in bf_yes_cols   if c in df.columns]
avail_month = [c for c in bf_month_cols if c in df.columns]

if avail_yes:
    # 4차 우선 → 3차 → 2차
    bf_raw = df[avail_yes].apply(
        lambda col: to_num(col)
    ).bfill(axis=1).iloc[:, 0]
    
    # 1=수유중, 2=중단 → 둘 다 breastfed=1
    # NaN → 처음부터 안 함 → breastfed=0
    feat["breastfed"] = bf_raw.apply(
        lambda x: 1 if x in [1, 2] else 0
    )
else:
    feat["breastfed"] = np.nan

if avail_month:
    # 중단 시기는 fed001=2인 경우만 값 있음
    bf_months_raw = df[avail_month].apply(
        lambda col: to_num(col)
    ).bfill(axis=1).iloc[:, 0]
    feat["breastfed_months"] = to_num(bf_months_raw)
else:
    feat["breastfed_months"] = np.nan

# 수유 안 한 경우(breastfed=0) → 기간 0
feat.loc[feat["breastfed"] == 0, "breastfed_months"] = 0
# 수유 중(fed001=1)인 경우 → 기간 NaN 유지
print("  ✓ 모유수유 여부 & 기간")


# --- 흡연 ---
# 임신 중 모 흡연 (1=흡연, 2=아니오)
col = "KMt13smk011_w6"
if col in df.columns:
    feat["prenatal_smoke"] = to_num(df[col]).apply(
        lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
    )
else:
    feat["prenatal_smoke"] = np.nan

# 2~6차 부/모 흡연 누적 (1=핌, 2=평소피우나임신수유로못핌 → 둘다 흡연자)
smoke_cols = [
    "EMt08smk001_w1", "FFt08smk001_w1",
    "EMt09smk001_w2", "FFt09smk001_w2",
    "EMt10smk001_w3", "FFt10smk001_w3",
    "EMt11smk001_w4", "FFt11smk001_w4",
    "EMt12smk001_w5", "FFt12smk001_w5",
    "EMt13smk001_w6", "FFt13smk001_w6",
]
smoke_avail = [c for c in smoke_cols if c in df.columns]
if smoke_avail:
    smoke_df = df[smoke_avail].apply(lambda col: to_num(col))
    feat["passive_smoke_ever"] = smoke_df.apply(
        lambda row: 1 if any(row.isin([1, 2])) else (0 if row.notna().any() else np.nan),
        axis=1
    )
else:
    feat["passive_smoke_ever"] = np.nan

# 아동 현재 간접흡연 (1=예, 2=아니오)
col = "KCh13smk008_w6"
if col in df.columns:
    feat["child_passive_smoke"] = to_num(df[col]).apply(
        lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
    )
else:
    feat["child_passive_smoke"] = np.nan
print("  ✓ 흡연 3개")


# --- 거주환경 ---
RURAL_CODE = 2  # 읍면 코드

cmm002_cols = [
    "DHu08cmm002_w1","DHu09cmm002_w2", "DHu10cmm002_w3",
    "DHu11cmm002_w4", "EHu12cmm002_w5", "DHu13cmm002_w6",
]
cmm002_avail = [c for c in cmm002_cols if c in df.columns]
if cmm002_avail:
    cmm_df = df[cmm002_avail].apply(lambda col: to_num(col))
    feat["rural_years"] = cmm_df.apply(
        lambda row: sum(row == RURAL_CODE), axis=1
    )
else:
    feat["rural_years"] = np.nan

feat["area_type_w6"] = to_num(get_col("EHu13cmm003_w6"))
print("  ✓ 거주환경 2개")


# --- 가족 알레르기 ---
def any_one(cols, yes_val=1):
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return pd.Series(np.nan, index=df.index)
    num_df = df[avail].apply(lambda col: to_num(col))
    return num_df.apply(
        lambda row: 1 if any(row == yes_val) else (0 if row.notna().any() else np.nan),
        axis=1
    )

feat["parent_AD"]       = any_one(["KFt13adx004_w6", "KMt13adx004_w6"])
feat["parent_AR"]       = any_one(["KFt13arx009_w6", "KMt13arx009_w6"])
feat["parent_asthma"]   = any_one(["KFt13asx011_w6", "KMt13asx011_w6"])
feat["sibling_allergy"] = any_one(["KHu13adx004_w6", "KHu13arx009_w6", "KHu13asx011_w6"])
print("  ✓ 가족 알레르기 4개")


# --- 애완동물 ---
feat["pet_ever"] = any_one(["KCh13pet001_w6", "KCh13pet003_w6"])
print("  ✓ 애완동물")


# --- 항생제 ---
feat["antibiotic"] = to_num(get_col("KCh13drg004_w6"))
print("  ✓ 항생제")


# --- 실내환경 (곰팡이) ---
feat["mold_ever"] = any_one(["KCh13mld001_w6", "KCh13mld006_w6", "KCh13mld008_w6"])
print("  ✓ 실내환경(곰팡이)")


# --- 실외활동 ---
# 3~4차: 분 단위 → 시간 변환 후 합산
# 5~6차: 이미 시간 단위

def outdoor_sum_to_hour(row, col_a, col_b):
    a = row.get(col_a, np.nan)
    b = row.get(col_b, np.nan)
    vals = [v for v in [a, b] if pd.notna(v)]
    if not vals:
        return np.nan
    return sum(vals) / 60

# 3~4차 분→시간 변환
for col in ["ECh10dlc003_w3", "ECh10dlc010_w3",
            "DCh11dlc004_w4", "DCh11dlc010_w4"]:
    if col in df.columns:
        df[col] = to_num(df[col])

df["outdoor_w3"] = df.apply(
    lambda row: outdoor_sum_to_hour(row, "ECh10dlc003_w3", "ECh10dlc010_w3"), axis=1
)
df["outdoor_w4"] = df.apply(
    lambda row: outdoor_sum_to_hour(row, "DCh11dlc004_w4", "DCh11dlc010_w4"), axis=1
)
df["outdoor_w5"] = to_num(get_col("DCh12dlc010_w5"))
df["outdoor_w6"] = to_num(get_col("DCh13dlc010_w6"))

feat["outdoor_avg"] = df[["outdoor_w3","outdoor_w4",
                           "outdoor_w5","outdoor_w6"]].mean(axis=1)
print("  ✓ 실외활동 평균 (3~6차, 단위: 시간)")


# --- 식습관 ---
def wave_avg(col5, col6):
    c5 = f"{col5}_w5"
    c6 = f"{col6}_w6"
    avail = [c for c in [c5, c6] if c in df.columns]
    if not avail:
        return pd.Series(np.nan, index=df.index)
    return df[avail].apply(lambda col: to_num(col)).mean(axis=1)

def fillback(col6, col5):
    c6 = f"{col6}_w6"
    c5 = f"{col5}_w5"
    s6 = to_num(get_col(c6))
    s5 = to_num(get_col(c5))
    if c6 in df.columns and c5 in df.columns:
        return s6.fillna(s5)
    elif c6 in df.columns:
        return s6
    elif c5 in df.columns:
        return s5
    return pd.Series(np.nan, index=df.index)

# 그룹 A: 5~6차 평균 (누적 노출)
feat["eat_snack_avg"]     = wave_avg("ECh12eat016", "ECh13eat016")
feat["eat_breakfast_avg"] = wave_avg("ECh12eat022", "ECh13eat022")
feat["eat_delivery_avg"]  = wave_avg("ECh12eat024", "ECh13eat024")

# 그룹 B: 6차 기준, 결측시 5차 보완
feat["eat_picky"]      = fillback("ECh13eat019", "ECh12eat019")
feat["eat_regularity"] = fillback("ECh13eat017", "ECh12eat017")
feat["eat_amount"]     = fillback("ECh13eat018", "ECh12eat018")
feat["eat_speed"]      = fillback("ECh13eat020", "ECh12eat020")
print("  ✓ 식습관 7개")


# ============================================================
# [5] 이상값 전체 확인
# ============================================================
print(f"\n{'='*60}")
print("이상값 확인")
print("="*60)

# 무응답(99999999) 확인
print("\n[무응답 99999999]")
found = False
for col in feat.columns:
    if col in ['N_ID', 'Y']:
        continue
    n = (feat[col] == 99999999).sum()
    if n > 0:
        print(f"  {col:<25}: {n}개")
        found = True
if not found:
    print("  없음 ✅")

# 결측치 확인
print("\n[결측치 NaN]")
found = False
missing = feat.isnull().sum()
for col in feat.columns:
    if missing[col] > 0:
        pct = missing[col] / len(feat) * 100
        print(f"  {col:<25}: {missing[col]}개 ({pct:.1f}%)")
        found = True
if not found:
    print("  없음 ✅")

# 이상값 확인 (999 이상 비정상값)
print("\n[이상값 (999 이상, 99999999 제외)]")
found = False
for col in feat.columns:
    if col in ['N_ID', 'Y']:
        continue
    try:
        n = ((feat[col] >= 999) & (feat[col] != 99999999)).sum()
        if n > 0:
            print(f"  {col:<25}: {n}개, 최대값={feat[col].max()}")
            found = True
    except:
        pass
if not found:
    print("  없음 ✅")

# 각 변수 기술통계 (범위 확인용)
print("\n[변수별 기술통계]")
print(feat.drop(columns=['N_ID']).describe().round(2).to_string())

# ============================================================
# [6] 최종 확인 및 저장
# ============================================================
print(f"\n{'='*60}")
print("최종 테이블 확인")
print("="*60)

print(f"\n행 수: {len(feat)}")
print(f"컬럼 수: {len(feat.columns)} (N_ID, Y 포함)")
print(f"\nY 분포:")
print(f"  양성(Y=1): {int(feat['Y'].sum())}명 ({feat['Y'].mean()*100:.1f}%)")
print(f"  음성(Y=0): {int((feat['Y']==0).sum())}명")

feat.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {OUTPUT_FILE}")

#%%
# ============================================================
# [7] 유의한 변수 선별
# ============================================================
print(f"\n{'='*60}")
print("유의한 변수 선별")
print("="*60)

# 결과 저장 폴더
SELECT_DIR = os.path.join(DATA_DIR, "feature_selection_outputs")
os.makedirs(SELECT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 설정값
# ------------------------------------------------------------
TARGET_COL = "Y"
ID_COL = "N_ID"

# 결측률 기준
MISSING_THRESHOLD = 0.50

# 단변량 후보 기준
UNIVARIATE_P_THRESHOLD = 0.2

# 최종 유의수준
FINAL_P_THRESHOLD = 0.05

# VIF 기준
VIF_THRESHOLD = 10.0

# 수동 제외 변수
MANUAL_DROP_COLS = [
    "prenatal_smoke",   # 결측률 58.5%라 제외
]

# ============================================================
# [7-1] 분석 대상 변수 정리
# ============================================================
print("\n[7-1] 분석 대상 변수 정리")

analysis_df = feat.copy()

feature_cols = [
    c for c in analysis_df.columns
    if c not in [ID_COL, TARGET_COL]
]

# 수동 제외
feature_cols = [
    c for c in feature_cols
    if c not in MANUAL_DROP_COLS
]

print("\n[수동 제외 변수]")
for c in MANUAL_DROP_COLS:
    if c in analysis_df.columns:
        print(f"  제외: {c}")

# 결측률 요약
missing_summary = pd.DataFrame({
    "variable": feature_cols,
    "missing_count": [analysis_df[c].isna().sum() for c in feature_cols],
    "missing_rate": [analysis_df[c].isna().mean() for c in feature_cols],
    "n_unique": [analysis_df[c].nunique(dropna=True) for c in feature_cols],
})

missing_summary = missing_summary.sort_values("missing_rate", ascending=False)

missing_summary.to_csv(
    os.path.join(SELECT_DIR, "01_missing_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

# 결측률 높은 변수 제거
high_missing_cols = missing_summary.loc[
    missing_summary["missing_rate"] > MISSING_THRESHOLD,
    "variable"
].tolist()

if high_missing_cols:
    print(f"\n[결측률 {MISSING_THRESHOLD * 100:.0f}% 초과 변수 제거]")
    for c in high_missing_cols:
        print(f"  제거: {c} / 결측률={analysis_df[c].isna().mean() * 100:.1f}%")

feature_cols = [
    c for c in feature_cols
    if c not in high_missing_cols
]

# 값이 하나뿐인 변수 제거
single_value_cols = [
    c for c in feature_cols
    if analysis_df[c].nunique(dropna=True) <= 1
]

if single_value_cols:
    print("\n[값이 하나뿐인 변수 제거]")
    for c in single_value_cols:
        print(f"  제거: {c}")

feature_cols = [
    c for c in feature_cols
    if c not in single_value_cols
]

print(f"\n최종 분석 후보 변수 수: {len(feature_cols)}")
print(feature_cols)
#%%
# ============================================================
# [7-2] 변수 타입 분류
# ============================================================
print("\n[7-2] 변수 타입 분류")

# ------------------------------------------------------------
# 연속형 변수
# - 실제 측정값: 재태기간, 출생체중, 모유수유 기간, 실외활동 시간
# - 누적/평균값: rural_years, 식습관 평균 변수
# ------------------------------------------------------------
continuous_cols = [
    "gestation_weeks",      # 실제 값은 211~293 정도 → 주(weeks)가 아니라 일(days)에 가까움
    "birth_weight",         # 출생체중 kg
    "breastfed_months",     # 모유수유 기간
    "rural_years",          # 읍면 거주 누적 연수 0~6
    "outdoor_avg",          # 실외활동 평균 시간
    "eat_snack_avg",        # 5~6차 평균값
    "eat_breakfast_avg",    # 5~6차 평균값
    "eat_delivery_avg",     # 5~6차 평균값
]

continuous_cols = [
    c for c in continuous_cols
    if c in feature_cols
]

# ------------------------------------------------------------
# 범주형 변수
# - 0/1 이진 변수
# - 코드값 변수
# - 단일 시점 1~5 척도 변수
# ------------------------------------------------------------
categorical_cols = [
    c for c in feature_cols
    if c not in continuous_cols
]

print("\n[범주형 변수]")
print(categorical_cols)

print("\n[연속형 변수]")
print(continuous_cols)

#%%
# ============================================================
# [7-3] 단변량 검정
# ============================================================
print("\n[7-3] 단변량 검정")

univariate_results = []

# ------------------------------------------------------------
# 범주형 변수: 카이제곱 검정
# ------------------------------------------------------------
for col in categorical_cols:
    temp = analysis_df[[TARGET_COL, col]].dropna()

    if len(temp) == 0:
        continue

    if temp[col].nunique() <= 1:
        continue

    try:
        table = pd.crosstab(temp[col], temp[TARGET_COL])

        # Y=0, Y=1 둘 다 있어야 검정 가능
        if table.shape[1] < 2:
            continue

        chi2, p, dof, expected = chi2_contingency(table)

        univariate_results.append({
            "variable": col,
            "type": "categorical",
            "test": "chi-square",
            "statistic": chi2,
            "p_value": p,
            "n_used": len(temp),
            "n_unique": temp[col].nunique(),
            "mean_y0": np.nan,
            "mean_y1": np.nan,
        })

    except Exception as e:
        print(f"  ⚠ {col} 카이제곱 실패: {e}")

# ------------------------------------------------------------
# 연속형 변수: Mann-Whitney U test
# ------------------------------------------------------------
for col in continuous_cols:
    temp = analysis_df[[TARGET_COL, col]].dropna()

    if len(temp) == 0:
        continue

    group0 = temp.loc[temp[TARGET_COL] == 0, col]
    group1 = temp.loc[temp[TARGET_COL] == 1, col]

    if len(group0) < 3 or len(group1) < 3:
        continue

    try:
        stat, p = mannwhitneyu(
            group0,
            group1,
            alternative="two-sided"
        )

        univariate_results.append({
            "variable": col,
            "type": "continuous",
            "test": "Mann-Whitney U",
            "statistic": stat,
            "p_value": p,
            "n_used": len(temp),
            "n_unique": temp[col].nunique(),
            "mean_y0": group0.mean(),
            "mean_y1": group1.mean(),
        })

    except Exception as e:
        print(f"  ⚠ {col} Mann-Whitney 실패: {e}")


univ_df = pd.DataFrame(univariate_results)

if len(univ_df) == 0:
    raise ValueError("단변량 검정 결과가 없습니다. 변수 타입이나 결측 상태를 확인하세요.")

univ_df = univ_df.sort_values("p_value")

univ_df.to_csv(
    os.path.join(SELECT_DIR, "02_univariate_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[단변량 검정 결과]")
print(univ_df.to_string(index=False))

#%%
# ============================================================
# [7-4] 단변량 기준 후보 변수 선택
# ============================================================
print("\n[7-4] 단변량 기준 후보 변수 선택")

selected_vars = univ_df.loc[
    univ_df["p_value"] < UNIVARIATE_P_THRESHOLD,
    "variable"
].tolist()

print(f"\n단변량 p < {UNIVARIATE_P_THRESHOLD} 변수 수: {len(selected_vars)}")
print(selected_vars)

selected_univ_df = univ_df[
    univ_df["variable"].isin(selected_vars)
].copy()

selected_univ_df.to_csv(
    os.path.join(SELECT_DIR, "03_selected_by_univariate.csv"),
    index=False,
    encoding="utf-8-sig"
)

if len(selected_vars) == 0:
    print("\n⚠ 단변량 기준을 만족하는 변수가 없습니다.")
    print("UNIVARIATE_P_THRESHOLD 값을 0.20에서 0.30으로 높여보세요.")
    raise ValueError("선택된 변수가 없어 분석을 중단합니다.")


#%%
# ============================================================
# [7-5] 로지스틱 회귀용 데이터 생성
# ============================================================
print("\n[7-5] 로지스틱 회귀용 데이터 생성")

selected_categorical = [
    c for c in selected_vars
    if c in categorical_cols
]

selected_continuous = [
    c for c in selected_vars
    if c in continuous_cols
]

y = analysis_df[TARGET_COL].astype(float)

# ------------------------------------------------------------
# 범주형 변수 결측 처리 + 더미 인코딩
# ------------------------------------------------------------
if len(selected_categorical) > 0:
    X_cat = analysis_df[selected_categorical].copy()

    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_cat_imputed = pd.DataFrame(
        cat_imputer.fit_transform(X_cat),
        columns=selected_categorical,
        index=analysis_df.index
    )

    # 더미화를 위해 문자열 처리
    for c in selected_categorical:
        X_cat_imputed[c] = X_cat_imputed[c].astype(str)

    X_cat_dummy = pd.get_dummies(
        X_cat_imputed,
        drop_first=True,
        dtype=float
    )
else:
    X_cat_dummy = pd.DataFrame(index=analysis_df.index)


# ------------------------------------------------------------
# 연속형 변수 결측 처리 + 표준화
# ------------------------------------------------------------
if len(selected_continuous) > 0:
    X_cont = analysis_df[selected_continuous].copy()

    cont_imputer = SimpleImputer(strategy="median")

    X_cont_imputed = pd.DataFrame(
        cont_imputer.fit_transform(X_cont),
        columns=selected_continuous,
        index=analysis_df.index
    )

    scaler = StandardScaler()

    X_cont_scaled = pd.DataFrame(
        scaler.fit_transform(X_cont_imputed),
        columns=selected_continuous,
        index=analysis_df.index
    )
else:
    X_cont_scaled = pd.DataFrame(index=analysis_df.index)


X = pd.concat([X_cat_dummy, X_cont_scaled], axis=1)
X = X.astype(float)

print(f"로지스틱 회귀 입력 변수 수: {X.shape[1]}")
print(X.columns.tolist())
#%%
## ============================================================
# [7-6] VIF 계산 및 다중공선성 제거
# ============================================================
print("\n[7-6] VIF 계산 및 다중공선성 제거")

def calculate_vif(X_data):
    vif_list = []

    for i, col in enumerate(X_data.columns):
        try:
            vif = variance_inflation_factor(X_data.values, i)
        except Exception:
            vif = np.inf

        vif_list.append({
            "variable": col,
            "VIF": vif
        })

    return pd.DataFrame(vif_list).sort_values("VIF", ascending=False)


# ------------------------------------------------------------
# 1) VIF 제거 전 결과 저장
# ------------------------------------------------------------
vif_before = calculate_vif(X)

vif_before.to_csv(
    os.path.join(SELECT_DIR, "04_1_vif_before_filter.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[VIF 제거 전 결과]")
print(vif_before.to_string(index=False))


# ------------------------------------------------------------
# 2) VIF 기준 초과 변수 반복 제거
# ------------------------------------------------------------
X_vif = X.copy()
removed_vif = []

while X_vif.shape[1] > 1:
    vif_df = calculate_vif(X_vif)

    max_vif = vif_df["VIF"].max()
    max_var = vif_df.iloc[0]["variable"]

    if max_vif <= VIF_THRESHOLD:
        break

    print(f"  VIF 제거: {max_var} / VIF={max_vif:.2f}")

    X_vif = X_vif.drop(columns=[max_var])
    removed_vif.append(max_var)


# ------------------------------------------------------------
# 3) VIF 제거 후 결과 저장
# ------------------------------------------------------------
vif_final = calculate_vif(X_vif)

vif_final.to_csv(
    os.path.join(SELECT_DIR, "04_2_vif_after_filter.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[VIF 제거 후 결과]")
print(vif_final.to_string(index=False))

print("\n[VIF 제거 변수 목록]")
if len(removed_vif) > 0:
    for v in removed_vif:
        print(f"  제거: {v}")
else:
    print("  제거된 변수 없음")
    
#============================================================
#%%

# ============================================================
# [7-7] 다변량 로지스틱 회귀
# ============================================================
print("\n[7-7] 다변량 로지스틱 회귀")

X_model = sm.add_constant(X_vif, has_constant="add")

try:
    logit_model = sm.Logit(y, X_model)
    result = logit_model.fit(disp=False, maxiter=1000)

except Exception as e:
    print("\n⚠ 일반 Logit 실패")
    print(f"원인: {e}")
    print("GLM Binomial로 재시도합니다.")

    glm_model = sm.GLM(
        y,
        X_model,
        family=sm.families.Binomial()
    )
    result = glm_model.fit()


# ============================================================
# [7-8] OR, CI, p-value 정리
# ============================================================
print("\n[7-8] 로지스틱 회귀 결과 정리")

params = result.params
conf = result.conf_int()
pvalues = result.pvalues

logit_result_df = pd.DataFrame({
    "variable": params.index,
    "coef": params.values,
    "OR": np.exp(params.values),
    "CI_lower": np.exp(conf[0].values),
    "CI_upper": np.exp(conf[1].values),
    "p_value": pvalues.values
})

logit_result_df = logit_result_df[
    logit_result_df["variable"] != "const"
].copy()

logit_result_df = logit_result_df.sort_values("p_value")

logit_result_df.to_csv(
    os.path.join(SELECT_DIR, "05_logistic_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[다변량 로지스틱 회귀 결과]")
print(logit_result_df.to_string(index=False))


# ============================================================
# [7-9] 최종 유의 변수 선별
# ============================================================
print("\n[7-9] 최종 유의 변수 선별")

significant_df = logit_result_df[
    logit_result_df["p_value"] < FINAL_P_THRESHOLD
].copy()

significant_df.to_csv(
    os.path.join(SELECT_DIR, "06_significant_features_final.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\n최종 유의 변수 수: {len(significant_df)}")
print(f"기준: p < {FINAL_P_THRESHOLD}")

if len(significant_df) > 0:
    print("\n[최종 유의 변수]")
    print(significant_df.to_string(index=False))
else:
    print("\n최종 유의 변수가 없습니다.")
    print("이 경우 단변량 기준을 p < 0.20에서 p < 0.30으로 완화하거나,")
    print("변수 수와 양성 표본 수를 다시 확인하세요.")
#%%

# ============================================================
# [7-10] 전체 저장 위치 출력
# ============================================================
print(f"\n{'='*60}")
print("유의 변수 선별 완료")
print("="*60)

print(f"\n결과 저장 폴더: {SELECT_DIR}")
print("저장 파일:")
print("  01_missing_summary.csv")
print("  02_univariate_results.csv")
print("  03_selected_by_univariate.csv")
print("  04_vif_final.csv")
print("  05_logistic_results.csv")
print("  06_significant_features_final.csv")

#%%
# ============================================================
# [8] 모유수유 / 아파트 변수 결과 진단 코드
# ============================================================
print(f"\n{'='*80}")
print("[8] 모유수유 / 아파트 변수 결과 진단")
print("="*80)

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DIAG_DIR = os.path.join(DATA_DIR, "diagnostic_breastfed_apartment")
os.makedirs(DIAG_DIR, exist_ok=True)

TARGET_COL = "Y"

diag_df = feat.copy()

# ------------------------------------------------------------
# 공통 함수 1: 범주형 변수별 Y 분포 확인
# ------------------------------------------------------------
def categorical_y_summary(data, col, target="Y", save_name=None):
    print(f"\n{'-'*80}")
    print(f"[범주형 진단] {col}")
    print("-"*80)

    tmp = data[[target, col]].copy()

    # NaN도 하나의 그룹으로 보기 위해 문자열 처리
    tmp[col] = tmp[col].astype("object")
    tmp[col] = tmp[col].where(tmp[col].notna(), "MISSING")

    tab = pd.crosstab(tmp[col], tmp[target])

    # Y=0, Y=1 컬럼이 없을 수도 있으므로 보정
    for yval in [0, 1]:
        if yval not in tab.columns:
            tab[yval] = 0

    tab = tab[[0, 1]].copy()
    tab.columns = ["Y0_count", "Y1_count"]
    tab["total"] = tab["Y0_count"] + tab["Y1_count"]
    tab["Y1_rate_pct"] = tab["Y1_count"] / tab["total"] * 100
    tab["group_pct_total"] = tab["total"] / len(tmp) * 100

    print("\n[그룹별 전체 수 / 아토피 수 / 아토피율]")
    print(tab.round(2).to_string())

    # "아토피 환자 중 어떤 그룹이 많은가" 확인
    col_dist_in_y = pd.crosstab(tmp[target], tmp[col], normalize="index") * 100
    print("\n[Y별 그룹 구성비 (%): 아토피 환자 중 아파트가 많은지 확인]")
    print(col_dist_in_y.round(2).to_string())

    # 카이제곱 검정
    try:
        chi_tab = pd.crosstab(tmp[col], tmp[target])
        if chi_tab.shape[0] >= 2 and chi_tab.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(chi_tab)
            print(f"\n[카이제곱 검정] chi2={chi2:.4f}, p={p:.6f}")
        else:
            print("\n[카이제곱 검정] 그룹 수 부족으로 생략")
    except Exception as e:
        print(f"\n[카이제곱 검정 실패] {e}")

    if save_name:
        tab.to_csv(
            os.path.join(DIAG_DIR, f"{save_name}_summary.csv"),
            encoding="utf-8-sig"
        )
        col_dist_in_y.to_csv(
            os.path.join(DIAG_DIR, f"{save_name}_dist_within_Y.csv"),
            encoding="utf-8-sig"
        )

    return tab


# ------------------------------------------------------------
# 공통 함수 2: 연속형 변수별 Y 분포 확인
# ------------------------------------------------------------
def continuous_y_summary(data, col, target="Y", save_name=None):
    print(f"\n{'-'*80}")
    print(f"[연속형 진단] {col}")
    print("-"*80)

    tmp = data[[target, col]].copy()

    print("\n[결측 여부별 Y 분포]")
    tmp[f"{col}_missing"] = tmp[col].isna().astype(int)
    missing_tab = pd.crosstab(tmp[f"{col}_missing"], tmp[target])

    for yval in [0, 1]:
        if yval not in missing_tab.columns:
            missing_tab[yval] = 0

    missing_tab = missing_tab[[0, 1]]
    missing_tab.columns = ["Y0_count", "Y1_count"]
    missing_tab["total"] = missing_tab["Y0_count"] + missing_tab["Y1_count"]
    missing_tab["Y1_rate_pct"] = missing_tab["Y1_count"] / missing_tab["total"] * 100
    print(missing_tab.round(2).to_string())

    nonmiss = tmp.dropna(subset=[col]).copy()

    print("\n[Y별 기술통계]")
    desc = nonmiss.groupby(target)[col].agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        max="max"
    )
    print(desc.round(3).to_string())

    # Mann-Whitney U
    try:
        g0 = nonmiss.loc[nonmiss[target] == 0, col]
        g1 = nonmiss.loc[nonmiss[target] == 1, col]

        if len(g0) >= 3 and len(g1) >= 3:
            stat, p = mannwhitneyu(g0, g1, alternative="two-sided")
            print(f"\n[Mann-Whitney U] stat={stat:.4f}, p={p:.6f}")
        else:
            print("\n[Mann-Whitney U] 표본 수 부족으로 생략")
    except Exception as e:
        print(f"\n[Mann-Whitney 실패] {e}")

    if save_name:
        desc.to_csv(
            os.path.join(DIAG_DIR, f"{save_name}_describe_by_Y.csv"),
            encoding="utf-8-sig"
        )
        missing_tab.to_csv(
            os.path.join(DIAG_DIR, f"{save_name}_missing_by_Y.csv"),
            encoding="utf-8-sig"
        )

    return desc


# ============================================================
# [8-1] 아파트 변수 진단
# ============================================================
print(f"\n{'='*80}")
print("[8-1] 아파트 변수 area_type_w6 진단")
print("="*80)

if "area_type_w6" in diag_df.columns:
    print("\n[area_type_w6 원자료 분포]")
    print(diag_df["area_type_w6"].value_counts(dropna=False).sort_index())

    # 네 말 기준: 2.0 = 아파트라고 가정
    diag_df["apt_w6"] = np.where(
        diag_df["area_type_w6"].isna(),
        np.nan,
        np.where(diag_df["area_type_w6"] == 2, 1, 0)
    )

    print("\n[apt_w6 정의]")
    print("  apt_w6 = 1: area_type_w6 == 2, 즉 아파트")
    print("  apt_w6 = 0: area_type_w6 != 2, 즉 비아파트")

    area_tab = categorical_y_summary(
        diag_df,
        "area_type_w6",
        target=TARGET_COL,
        save_name="area_type_w6"
    )

    apt_tab = categorical_y_summary(
        diag_df,
        "apt_w6",
        target=TARGET_COL,
        save_name="apt_w6"
    )

else:
    print("⚠ area_type_w6 컬럼이 없습니다.")


# ============================================================
# [8-2] 모유수유 변수 진단
# ============================================================
print(f"\n{'='*80}")
print("[8-2] 모유수유 변수 진단")
print("="*80)

if "breastfed" in diag_df.columns:
    categorical_y_summary(
        diag_df,
        "breastfed",
        target=TARGET_COL,
        save_name="breastfed"
    )
else:
    print("⚠ breastfed 컬럼이 없습니다.")

if "breastfed_months" in diag_df.columns:
    print("\n[breastfed_months 원자료 값 분포: 상위 30개]")
    print(diag_df["breastfed_months"].value_counts(dropna=False).sort_index().head(30))

    continuous_y_summary(
        diag_df,
        "breastfed_months",
        target=TARGET_COL,
        save_name="breastfed_months"
    )

    # 구간화해서 보기
    diag_df["breastfed_months_bin"] = pd.cut(
        diag_df["breastfed_months"],
        bins=[-0.1, 0, 3, 6, 12, 24, 999],
        labels=[
            "0_미수유",
            "1_1~3개월",
            "2_4~6개월",
            "3_7~12개월",
            "4_13~24개월",
            "5_25개월이상"
        ]
    )

    categorical_y_summary(
        diag_df,
        "breastfed_months_bin",
        target=TARGET_COL,
        save_name="breastfed_months_bin"
    )

else:
    print("⚠ breastfed_months 컬럼이 없습니다.")


# ============================================================
# [8-3] 네 코드에서 선택된 원본 모유수유 값 확인
# ============================================================
print(f"\n{'='*80}")
print("[8-3] 원본 모유수유 컬럼 확인")
print("="*80)

bf_yes_cols = ["DCh11fed001_w4", "DCh10fed001_w3", "DCh09fed001_w2", "DCh08fed001_w1"]
bf_month_cols = ["DCh11fed002_w4", "DCh10fed002_w3", "DCh09fed002_w2", "DCh08fed001_w1"]

avail_yes = [c for c in bf_yes_cols if c in df.columns]
avail_month = [c for c in bf_month_cols if c in df.columns]

print("\n[모유수유 여부 후보 컬럼]")
print(avail_yes)

print("\n[모유수유 기간 후보 컬럼]")
print(avail_month)

print("\n⚠ 확인 포인트")
print("  bf_month_cols 마지막 컬럼이 DCh08fed001_w1 입니다.")
print("  이 컬럼이 진짜 '기간' 컬럼인지, 아니면 '수유 여부' 컬럼인지 반드시 확인하세요.")
print("  기간 컬럼이어야 한다면 fed002 계열이어야 할 가능성이 있습니다.")

if avail_yes:
    bf_raw_selected = df[avail_yes].apply(lambda col: pd.to_numeric(col, errors="coerce")).bfill(axis=1).iloc[:, 0]

    bf_check = pd.DataFrame({
        "Y": diag_df[TARGET_COL],
        "bf_raw_selected": bf_raw_selected,
        "breastfed": diag_df.get("breastfed", np.nan),
        "breastfed_months": diag_df.get("breastfed_months", np.nan)
    })

    bf_check["bf_raw_label"] = bf_check["bf_raw_selected"].map({
        1: "1_수유중",
        2: "2_중단"
    })
    bf_check["bf_raw_label"] = bf_check["bf_raw_label"].fillna("MISSING_OR_NO")

    print("\n[선택된 모유수유 원본값 분포]")
    print(bf_check["bf_raw_label"].value_counts(dropna=False))

    categorical_y_summary(
        bf_check,
        "bf_raw_label",
        target="Y",
        save_name="bf_raw_selected"
    )

    print("\n[bf_raw_selected가 MISSING인데 breastfed=0으로 처리된 사람 수 확인]")
    missing_but_zero = (
        bf_check["bf_raw_selected"].isna()
        & (bf_check["breastfed"] == 0)
    )
    print(f"  원본 모유수유 여부가 NaN인데 breastfed=0 처리된 수: {int(missing_but_zero.sum())}명")

    bf_check.to_csv(
        os.path.join(DIAG_DIR, "bf_raw_selected_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )

for col in avail_yes:
    print(f"\n[원본 여부 컬럼 분포] {col}")
    print(pd.to_numeric(df[col], errors="coerce").value_counts(dropna=False).sort_index())

for col in avail_month:
    print(f"\n[원본 기간 컬럼 분포] {col}")
    s = pd.to_numeric(df[col], errors="coerce")
    print(s.value_counts(dropna=False).sort_index().head(50))
    print(s.describe())


# ============================================================
# [8-4] 아파트와 다른 위험요인의 관계 확인
# ============================================================
print(f"\n{'='*80}")
print("[8-4] 아파트 거주군과 다른 위험요인의 관계 확인")
print("="*80)

check_vars = [
    "parent_AD",
    "parent_AR",
    "mold_ever",
    "antibiotic",
    "breastfed",
    "breastfed_months",
    "rural_years",
    "passive_smoke_ever",
    "child_passive_smoke",
    "outdoor_avg"
]

check_vars = [c for c in check_vars if c in diag_df.columns]

if "apt_w6" in diag_df.columns:
    profile_rows = []

    for v in check_vars:
        tmp = diag_df[["apt_w6", v, TARGET_COL]].copy()

        for apt_val, g in tmp.groupby("apt_w6", dropna=False):
            profile_rows.append({
                "group_var": "apt_w6",
                "group_value": apt_val,
                "variable": v,
                "n": g[v].notna().sum(),
                "missing": g[v].isna().sum(),
                "mean_or_rate": g[v].mean(),
                "median": g[v].median(),
                "Y_rate": g[TARGET_COL].mean()
            })

    profile_df = pd.DataFrame(profile_rows)
    print("\n[아파트/비아파트별 주요 변수 평균 또는 비율]")
    print(profile_df.round(3).to_string(index=False))

    profile_df.to_csv(
        os.path.join(DIAG_DIR, "apt_group_profile.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 아파트 x 곰팡이 조합별 아토피율
    if "mold_ever" in diag_df.columns:
        print("\n[아파트 여부 x 곰팡이 노출 조합별 아토피율]")
        combo = diag_df.copy()
        combo["apt_mold_group"] = (
            "apt=" + combo["apt_w6"].astype("object").astype(str)
            + "_mold=" + combo["mold_ever"].astype("object").astype(str)
        )
        categorical_y_summary(
            combo,
            "apt_mold_group",
            target=TARGET_COL,
            save_name="apt_mold_group"
        )


# ============================================================
# [8-5] 단순 모델 vs 보정 모델 비교
# ============================================================
print(f"\n{'='*80}")
print("[8-5] 단순 모델 vs 보정 모델 비교")
print("="*80)

def fit_logit_model(data, cat_cols=None, cont_cols=None, model_name="model",
                    target="Y", impute=True, scale_cont=True):
    cat_cols = cat_cols or []
    cont_cols = cont_cols or []

    use_cols = [target] + cat_cols + cont_cols
    use_cols = [c for c in use_cols if c in data.columns]

    tmp = data[use_cols].copy()

    # impute=False면 완전사례분석
    if not impute:
        tmp = tmp.dropna()

    y = tmp[target].astype(float)

    X_parts = []

    # 범주형 처리
    real_cat_cols = [c for c in cat_cols if c in tmp.columns]
    if real_cat_cols:
        X_cat = tmp[real_cat_cols].copy()

        if impute:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_cat = pd.DataFrame(
                cat_imputer.fit_transform(X_cat),
                columns=real_cat_cols,
                index=tmp.index
            )

        for c in real_cat_cols:
            X_cat[c] = X_cat[c].astype(str)

        X_cat_dummy = pd.get_dummies(X_cat, drop_first=True, dtype=float)
        X_parts.append(X_cat_dummy)

    # 연속형 처리
    real_cont_cols = [c for c in cont_cols if c in tmp.columns]
    if real_cont_cols:
        X_cont = tmp[real_cont_cols].copy()

        if impute:
            cont_imputer = SimpleImputer(strategy="median")
            X_cont = pd.DataFrame(
                cont_imputer.fit_transform(X_cont),
                columns=real_cont_cols,
                index=tmp.index
            )

        if scale_cont:
            scaler = StandardScaler()
            X_cont = pd.DataFrame(
                scaler.fit_transform(X_cont),
                columns=real_cont_cols,
                index=tmp.index
            )

        X_parts.append(X_cont.astype(float))

    if not X_parts:
        print(f"\n[{model_name}] 입력 변수가 없어 생략")
        return None

    X = pd.concat(X_parts, axis=1).astype(float)

    # 상수항 추가
    X_model = sm.add_constant(X, has_constant="add")

    try:
        result = sm.Logit(y, X_model).fit(disp=False, maxiter=1000)
    except Exception as e:
        print(f"\n[{model_name}] Logit 실패 → GLM Binomial 재시도: {e}")
        result = sm.GLM(y, X_model, family=sm.families.Binomial()).fit()

    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    out = pd.DataFrame({
        "model": model_name,
        "n_used": len(tmp),
        "variable": params.index,
        "coef": params.values,
        "OR": np.exp(params.values),
        "CI_lower": np.exp(conf[0].values),
        "CI_upper": np.exp(conf[1].values),
        "p_value": pvals.values,
        "impute": impute,
        "scale_cont": scale_cont
    })

    out = out[out["variable"] != "const"].copy()
    out = out.sort_values("p_value")

    print(f"\n[{model_name}] n={len(tmp)}, impute={impute}, scale_cont={scale_cont}")
    print(out.round(4).to_string(index=False))

    out.to_csv(
        os.path.join(DIAG_DIR, f"{model_name}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return out


model_outputs = []

# 모델 1: 아파트만
if "area_type_w6" in diag_df.columns:
    out = fit_logit_model(
        diag_df,
        cat_cols=["area_type_w6"],
        cont_cols=[],
        model_name="M1_area_only",
        impute=True,
        scale_cont=True
    )
    if out is not None:
        model_outputs.append(out)

# 모델 2: 모유수유 기간만
if "breastfed_months" in diag_df.columns:
    out = fit_logit_model(
        diag_df,
        cat_cols=[],
        cont_cols=["breastfed_months"],
        model_name="M2_breastfed_months_only_scaled",
        impute=True,
        scale_cont=True
    )
    if out is not None:
        model_outputs.append(out)

# 모델 3: 아파트 + 모유수유 기간
if {"area_type_w6", "breastfed_months"}.issubset(diag_df.columns):
    out = fit_logit_model(
        diag_df,
        cat_cols=["area_type_w6"],
        cont_cols=["breastfed_months"],
        model_name="M3_area_plus_breastfed",
        impute=True,
        scale_cont=True
    )
    if out is not None:
        model_outputs.append(out)

# 모델 4: 최종 유의 변수 중심 보정 모델
final_cat_vars = [
    "antibiotic",
    "area_type_w6",
    "parent_AD",
    "parent_AR",
    "mold_ever"
]
final_cont_vars = [
    "breastfed_months"
]

final_cat_vars = [c for c in final_cat_vars if c in diag_df.columns]
final_cont_vars = [c for c in final_cont_vars if c in diag_df.columns]

out = fit_logit_model(
    diag_df,
    cat_cols=final_cat_vars,
    cont_cols=final_cont_vars,
    model_name="M4_final_like_model_imputed_scaled",
    impute=True,
    scale_cont=True
)
if out is not None:
    model_outputs.append(out)

# 모델 5: 같은 모델을 완전사례분석으로 비교
out = fit_logit_model(
    diag_df,
    cat_cols=final_cat_vars,
    cont_cols=final_cont_vars,
    model_name="M5_final_like_model_complete_case_scaled",
    impute=False,
    scale_cont=True
)
if out is not None:
    model_outputs.append(out)

# 모델 6: breastfed_months를 표준화하지 않고 넣기
out = fit_logit_model(
    diag_df,
    cat_cols=final_cat_vars,
    cont_cols=final_cont_vars,
    model_name="M6_final_like_model_imputed_unscaled",
    impute=True,
    scale_cont=False
)
if out is not None:
    model_outputs.append(out)


if model_outputs:
    compare_df = pd.concat(model_outputs, axis=0)
    compare_df.to_csv(
        os.path.join(DIAG_DIR, "model_comparison_all.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[핵심 변수만 비교]")
    key_mask = compare_df["variable"].str.contains(
        "area_type_w6|apt|breastfed_months|mold|parent_AD|parent_AR|antibiotic",
        regex=True
    )
    print(compare_df.loc[key_mask].round(4).to_string(index=False))


# ============================================================
# [8-6] breastfed_months OR 해석 단위 확인
# ============================================================
print(f"\n{'='*80}")
print("[8-6] breastfed_months OR 해석 단위 확인")
print("="*80)

if "breastfed_months" in diag_df.columns:
    raw_bf = diag_df["breastfed_months"].copy()

    median_bf = raw_bf.median()
    imputed_bf = raw_bf.fillna(median_bf)

    # StandardScaler는 ddof=0 기준에 가까움
    sd_bf = imputed_bf.std(ddof=0)

    print(f"\n[breastfed_months]")
    print(f"  원자료 중앙값: {median_bf:.4f}")
    print(f"  중앙값 대체 후 표준편차: {sd_bf:.4f}")
    print("  네 기존 모델에서 breastfed_months는 StandardScaler 적용 후 들어감")
    print("  따라서 기존 OR은 '1개월 증가'가 아니라 '1 표준편차 증가' 기준임")

    # 기존 logit_result_df가 있으면 실제 OR을 1개월 기준으로 환산
    if "logit_result_df" in globals():
        bf_row = logit_result_df.loc[
            logit_result_df["variable"] == "breastfed_months"
        ].copy()

        if len(bf_row) > 0 and sd_bf > 0:
            coef_scaled = bf_row.iloc[0]["coef"]
            or_scaled = bf_row.iloc[0]["OR"]

            coef_per_month = coef_scaled / sd_bf
            or_per_month = np.exp(coef_per_month)

            print("\n[기존 최종 모델 기준 환산]")
            print(f"  표준화 기준 coef: {coef_scaled:.6f}")
            print(f"  표준화 기준 OR: {or_scaled:.6f}")
            print(f"  1개월 증가 기준 환산 OR: {or_per_month:.6f}")
            print("  즉, 보고서에는 기존 OR을 '1개월당 OR'이라고 쓰면 안 됨")
        else:
            print("\n기존 logit_result_df에서 breastfed_months 결과를 찾지 못했습니다.")
    else:
        print("\nlogit_result_df 객체가 없어 기존 최종 OR 환산은 생략합니다.")


# ============================================================
# [8-7] 진단 결과 저장 위치
# ============================================================
print(f"\n{'='*80}")
print("[8] 진단 완료")
print("="*80)
print(f"저장 폴더: {DIAG_DIR}")
print("확인할 주요 파일:")
print("  - area_type_w6_summary.csv")
print("  - apt_w6_summary.csv")
print("  - breastfed_summary.csv")
print("  - breastfed_months_describe_by_Y.csv")
print("  - breastfed_months_bin_summary.csv")
print("  - bf_raw_selected_check.csv")
print("  - apt_group_profile.csv")
print("  - model_comparison_all.csv")

#%%
# ============================================================
# [9] 핵심 4개 변수만 사용한 모델링
# ============================================================
print(f"\n{'='*80}")
print("[9] 핵심 4개 변수 모델링")
print("="*80)

CORE_DIR = os.path.join(DATA_DIR, "core_4vars_model_outputs")
os.makedirs(CORE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1) 핵심 4개 변수
# ------------------------------------------------------------
core_cat_vars = [
    "antibiotic",
    "parent_AD",
    "parent_AR",
    "mold_ever"
]

core_cat_vars = [c for c in core_cat_vars if c in feat.columns]

print("\n[사용할 핵심 변수]")
print(core_cat_vars)

missing_core = [c for c in ["antibiotic", "parent_AD", "parent_AR", "mold_ever"] if c not in feat.columns]
if missing_core:
    raise ValueError(f"feat에 없는 핵심 변수: {missing_core}")

# ------------------------------------------------------------
# 2) 변수별 결측률 / 분포 확인
# ------------------------------------------------------------
core_df = feat[["Y"] + core_cat_vars].copy()

print("\n[Y 분포]")
print(core_df["Y"].value_counts())
print(f"양성 비율: {core_df['Y'].mean() * 100:.2f}%")

print("\n[핵심 변수 결측률]")
core_missing = pd.DataFrame({
    "variable": core_cat_vars,
    "missing_count": [core_df[c].isna().sum() for c in core_cat_vars],
    "missing_rate": [core_df[c].isna().mean() for c in core_cat_vars],
    "n_unique": [core_df[c].nunique(dropna=True) for c in core_cat_vars],
})
print(core_missing.to_string(index=False))

core_missing.to_csv(
    os.path.join(CORE_DIR, "01_core_4vars_missing_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[핵심 변수 원자료 분포]")
for c in core_cat_vars:
    print(f"\n--- {c} ---")
    print(core_df[c].value_counts(dropna=False).sort_index())

# ------------------------------------------------------------
# 3) 핵심 4개 변수 로지스틱 회귀
#    네가 위에서 만든 fit_logit_model 함수 재사용
# ------------------------------------------------------------
core_result_df = fit_logit_model(
    data=feat,
    cat_cols=core_cat_vars,
    cont_cols=[],
    model_name="M7_core_4vars_only",
    target="Y",
    impute=True,
    scale_cont=True
)

# ------------------------------------------------------------
# 4) 결과 저장
# ------------------------------------------------------------
if core_result_df is not None:
    core_result_df.to_csv(
        os.path.join(CORE_DIR, "02_core_4vars_logistic_OR_CI_p.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    core_sig_df = core_result_df[
        core_result_df["p_value"] < 0.05
    ].copy()

    print("\n[핵심 4개 모델 내 p < 0.05 변수]")
    if len(core_sig_df) > 0:
        print(core_sig_df.to_string(index=False))
    else:
        print("p < 0.05인 변수가 없습니다.")

    core_sig_df.to_csv(
        os.path.join(CORE_DIR, "03_core_4vars_significant.csv"),
        index=False,
        encoding="utf-8-sig"
    )

print(f"\n{'='*80}")
print("[9] 핵심 4개 변수 모델링 완료")
print("="*80)
print(f"저장 폴더: {CORE_DIR}")
print("저장 파일:")
print("  01_core_4vars_missing_summary.csv")
print("  02_core_4vars_logistic_OR_CI_p.csv")
print("  03_core_4vars_significant.csv")

#%%
# ============================================================
# [9] 핵심 4개 변수 기반 모델 비교
#     - No Sampling
#     - Class Weight
#     - RandomOverSampler
#     - SMOTEN
# ============================================================
print(f"\n{'='*80}")
print("[9] 핵심 4개 변수 기반 모델 비교")
print("="*80)

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTEN
except ImportError:
    raise ImportError(
        "imbalanced-learn이 설치되어 있지 않습니다.\n"
        "아래 명령어 실행 후 다시 돌리세요:\n"
        "pip install imbalanced-learn"
    )

CORE_MODEL_DIR = os.path.join(DATA_DIR, "core_4vars_smote_model_outputs")
os.makedirs(CORE_MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1) 핵심 4개 변수 설정
# ------------------------------------------------------------
TARGET_COL = "Y"

core_vars = [
    "antibiotic",
    "parent_AD",
    "parent_AR",
    "mold_ever"
]

missing_cols = [c for c in core_vars + [TARGET_COL] if c not in feat.columns]
if missing_cols:
    raise ValueError(f"feat에 없는 컬럼이 있습니다: {missing_cols}")

data = feat[[TARGET_COL] + core_vars].copy()

print("\n[사용 변수]")
print(core_vars)

print("\n[Y 분포]")
print(data[TARGET_COL].value_counts())
print(f"양성 비율: {data[TARGET_COL].mean() * 100:.2f}%")

print("\n[변수별 결측률]")
missing_df = pd.DataFrame({
    "variable": core_vars,
    "missing_count": [data[c].isna().sum() for c in core_vars],
    "missing_rate": [data[c].isna().mean() for c in core_vars],
    "n_unique": [data[c].nunique(dropna=True) for c in core_vars]
})
print(missing_df.to_string(index=False))

missing_df.to_csv(
    os.path.join(CORE_MODEL_DIR, "01_core_4vars_missing_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 2) Train / Validation / Test 분리
#    Train: 60%, Validation: 20%, Test: 20%
# ------------------------------------------------------------
X_raw = data[core_vars].copy()
y = data[TARGET_COL].astype(int)

X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X_raw,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

X_valid_raw, X_test_raw, y_valid, y_test = train_test_split(
    X_temp_raw,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\n[데이터 분리]")
print(f"Train      : {X_train_raw.shape}, 양성비율={y_train.mean() * 100:.2f}%")
print(f"Validation : {X_valid_raw.shape}, 양성비율={y_valid.mean() * 100:.2f}%")
print(f"Test       : {X_test_raw.shape}, 양성비율={y_test.mean() * 100:.2f}%")

# ------------------------------------------------------------
# 3) OneHotEncoder 호환 함수
# ------------------------------------------------------------
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ------------------------------------------------------------
# 4) Threshold 선택 함수
#    Validation에서 F2가 가장 좋은 threshold 선택
#    F2는 recall을 precision보다 더 중요하게 봄
# ------------------------------------------------------------
def find_best_threshold_by_f2(y_true, proba):
    thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []
    for th in thresholds:
        pred = (proba >= th).astype(int)

        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)

        rows.append({
            "threshold": th,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2
        })

    th_df = pd.DataFrame(rows)
    best = th_df.sort_values(["f2", "recall"], ascending=False).iloc[0]
    return float(best["threshold"]), th_df

# ------------------------------------------------------------
# 5) 평가 함수
# ------------------------------------------------------------
def evaluate_model(y_true, proba, threshold, model_name, sampler_name, split_name):
    pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)

    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)

    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "sampler": sampler_name,
        "model": model_name,
        "split": split_name,
        "threshold": threshold,
        "AUC": auc,
        "PR_AUC": pr_auc,
        "Brier": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

# ------------------------------------------------------------
# 6) 샘플링 + 인코딩 함수
# ------------------------------------------------------------
def prepare_data_for_sampler(
    X_train_raw,
    y_train,
    X_valid_raw,
    X_test_raw,
    sampler_name
):
    """
    sampler_name:
      - none
      - random_over
      - smoten
    """

    # 결측은 train 기준 최빈값으로 대체
    imputer = SimpleImputer(strategy="most_frequent")

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=core_vars,
        index=X_train_raw.index
    )

    X_valid_imp = pd.DataFrame(
        imputer.transform(X_valid_raw),
        columns=core_vars,
        index=X_valid_raw.index
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=core_vars,
        index=X_test_raw.index
    )

    # 전부 범주형 문자열 처리
    for c in core_vars:
        X_train_imp[c] = X_train_imp[c].astype(str)
        X_valid_imp[c] = X_valid_imp[c].astype(str)
        X_test_imp[c] = X_test_imp[c].astype(str)

    # -----------------------------
    # No Sampling
    # -----------------------------
    if sampler_name == "none":
        encoder = make_onehot_encoder()

        X_train_enc = encoder.fit_transform(X_train_imp)
        X_valid_enc = encoder.transform(X_valid_imp)
        X_test_enc = encoder.transform(X_test_imp)

        y_train_res = y_train.copy()

        return X_train_enc, y_train_res, X_valid_enc, X_test_enc, encoder

    # -----------------------------
    # RandomOverSampler
    # 원핫 인코딩 후 단순 복제 오버샘플링
    # -----------------------------
    elif sampler_name == "random_over":
        encoder = make_onehot_encoder()

        X_train_enc = encoder.fit_transform(X_train_imp)
        X_valid_enc = encoder.transform(X_valid_imp)
        X_test_enc = encoder.transform(X_test_imp)

        ros = RandomOverSampler(random_state=42)
        X_train_res, y_train_res = ros.fit_resample(X_train_enc, y_train)

        return X_train_res, y_train_res, X_valid_enc, X_test_enc, encoder

    # -----------------------------
    # SMOTEN
    # 범주형 변수 전용 SMOTE
    # -----------------------------
    elif sampler_name == "smoten":
        smoten = SMOTEN(random_state=42, k_neighbors=5)

        X_train_res_raw, y_train_res = smoten.fit_resample(X_train_imp, y_train)

        X_train_res_raw = pd.DataFrame(
            X_train_res_raw,
            columns=core_vars
        )

        for c in core_vars:
            X_train_res_raw[c] = X_train_res_raw[c].astype(str)

        encoder = make_onehot_encoder()

        X_train_enc = encoder.fit_transform(X_train_res_raw)
        X_valid_enc = encoder.transform(X_valid_imp)
        X_test_enc = encoder.transform(X_test_imp)

        return X_train_enc, y_train_res, X_valid_enc, X_test_enc, encoder

    else:
        raise ValueError(f"알 수 없는 sampler_name: {sampler_name}")

# ------------------------------------------------------------
# 7) 모델 목록
# ------------------------------------------------------------
def get_models(class_weight_mode=False):
    if class_weight_mode:
        lr_class_weight = "balanced"
        rf_class_weight = "balanced"
    else:
        lr_class_weight = None
        rf_class_weight = None

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight=lr_class_weight
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight=rf_class_weight,
            min_samples_leaf=5
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
    }

    return models

# ------------------------------------------------------------
# 8) 실험 설정
# ------------------------------------------------------------
experiment_settings = [
    {
        "sampler_name": "none",
        "label": "NoSampling",
        "class_weight": False
    },
    {
        "sampler_name": "none",
        "label": "ClassWeight",
        "class_weight": True
    },
    {
        "sampler_name": "random_over",
        "label": "RandomOverSampler",
        "class_weight": False
    },
    {
        "sampler_name": "smoten",
        "label": "SMOTEN",
        "class_weight": False
    }
]

all_results = []
all_thresholds = []

# ------------------------------------------------------------
# 9) 모델 학습 / 평가 반복
# ------------------------------------------------------------
for exp in experiment_settings:
    sampler_name = exp["sampler_name"]
    label = exp["label"]
    class_weight = exp["class_weight"]

    print(f"\n{'='*80}")
    print(f"[실험] {label}")
    print("="*80)

    X_train_enc, y_train_res, X_valid_enc, X_test_enc, encoder = prepare_data_for_sampler(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_valid_raw=X_valid_raw,
        X_test_raw=X_test_raw,
        sampler_name=sampler_name
    )

    print("[샘플링 후 Train 분포]")
    print(pd.Series(y_train_res).value_counts().sort_index())

    models = get_models(class_weight_mode=class_weight)

    for model_name, model in models.items():
        print(f"\n--- {label} + {model_name} ---")

        model.fit(X_train_enc, y_train_res)

        valid_proba = model.predict_proba(X_valid_enc)[:, 1]
        test_proba = model.predict_proba(X_test_enc)[:, 1]

        best_th, th_df = find_best_threshold_by_f2(y_valid, valid_proba)

        th_df["sampler"] = label
        th_df["model"] = model_name
        all_thresholds.append(th_df)

        valid_result = evaluate_model(
            y_true=y_valid,
            proba=valid_proba,
            threshold=best_th,
            model_name=model_name,
            sampler_name=label,
            split_name="validation"
        )

        test_result = evaluate_model(
            y_true=y_test,
            proba=test_proba,
            threshold=best_th,
            model_name=model_name,
            sampler_name=label,
            split_name="test"
        )

        all_results.append(valid_result)
        all_results.append(test_result)

        print(f"Best threshold by Validation F2: {best_th:.2f}")

        print("[Validation]")
        print(
            f"AUC={valid_result['AUC']:.4f}, "
            f"PR-AUC={valid_result['PR_AUC']:.4f}, "
            f"Recall={valid_result['recall']:.4f}, "
            f"Precision={valid_result['precision']:.4f}, "
            f"F2={valid_result['f2']:.4f}, "
            f"Brier={valid_result['Brier']:.4f}"
        )

        print("[Test]")
        print(
            f"AUC={test_result['AUC']:.4f}, "
            f"PR-AUC={test_result['PR_AUC']:.4f}, "
            f"Recall={test_result['recall']:.4f}, "
            f"Precision={test_result['precision']:.4f}, "
            f"F2={test_result['f2']:.4f}, "
            f"Brier={test_result['Brier']:.4f}"
        )

        print("[Test Confusion Matrix]")
        print(np.array([
            [test_result["TN"], test_result["FP"]],
            [test_result["FN"], test_result["TP"]]
        ]))

# ------------------------------------------------------------
# 10) 결과 정리 및 저장
# ------------------------------------------------------------
result_df = pd.DataFrame(all_results)

result_df = result_df.sort_values(
    by=["split", "AUC", "PR_AUC", "f2"],
    ascending=[True, False, False, False]
)

result_df.to_csv(
    os.path.join(CORE_MODEL_DIR, "02_core_4vars_model_comparison.csv"),
    index=False,
    encoding="utf-8-sig"
)

threshold_df = pd.concat(all_thresholds, axis=0)

threshold_df.to_csv(
    os.path.join(CORE_MODEL_DIR, "03_threshold_search_validation.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(f"\n{'='*80}")
print("[전체 모델 비교 결과]")
print("="*80)

print("\n[Test 기준 결과]")
test_result_df = result_df[result_df["split"] == "test"].copy()
test_result_df = test_result_df.sort_values(
    by=["AUC", "PR_AUC", "f2"],
    ascending=False
)

print(test_result_df.round(4).to_string(index=False))

test_result_df.to_csv(
    os.path.join(CORE_MODEL_DIR, "04_test_result_sorted.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 11) 추천 모델 자동 출력
# ------------------------------------------------------------
best_auc_model = test_result_df.sort_values(
    by=["AUC", "PR_AUC"],
    ascending=False
).iloc[0]

best_f2_model = test_result_df.sort_values(
    by=["f2", "recall", "PR_AUC"],
    ascending=False
).iloc[0]

print("\n[추천 모델 - AUC 기준]")
print(best_auc_model.to_string())

print("\n[추천 모델 - F2/Recall 기준]")
print(best_f2_model.to_string())

print(f"\n{'='*80}")
print("[9] 핵심 4개 변수 모델 비교 완료")
print("="*80)
print(f"저장 폴더: {CORE_MODEL_DIR}")
print("저장 파일:")
print("  01_core_4vars_missing_summary.csv")
print("  02_core_4vars_model_comparison.csv")
print("  03_threshold_search_validation.csv")
print("  04_test_result_sorted.csv")
#%%
# ============================================================
# [10] 핵심 4개 + 추가 변수 세트별 예측 모델 비교
#      - NoSampling
#      - ClassWeight
#      - RandomOverSampler
#      - ProperSMOTE
#        * 범주형만 있으면 SMOTEN
#        * 범주형 + 연속형이면 SMOTENC
#      - 하이퍼파라미터 튜닝 포함
# ============================================================

print(f"\n{'='*80}")
print("[10] 핵심 4개 + 추가 변수 세트별 예측 모델 비교")
print("     + 하이퍼파라미터 튜닝")
print("     + SMOTE after OneHot 제거")
print("     + SMOTEN / SMOTENC 적용")
print("="*80)

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    raise ImportError(
        "imbalanced-learn이 설치되어 있지 않습니다.\n"
        "터미널에서 아래 명령어 실행 후 다시 돌리세요:\n"
        "pip install imbalanced-learn"
    )

EXPAND_DIR = os.path.join(DATA_DIR, "expanded_prediction_model_outputs_tuned")
os.makedirs(EXPAND_DIR, exist_ok=True)

TARGET_COL = "Y"
RANDOM_STATE = 42

# 튜닝 기준
# 아토피 양성 비율이 낮으므로 accuracy보다 PR-AUC가 더 적절함
TUNING_SCORING = "average_precision"

# 너무 오래 걸리면 10으로 낮추고,
# 시간이 괜찮으면 20~30으로 올려도 됨
N_ITER_SEARCH = 15

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ------------------------------------------------------------
# 1) 변수 세트 정의
# ------------------------------------------------------------
core_cat = [
    "antibiotic",
    "parent_AD",
    "parent_AR",
    "mold_ever"
]

birth_cat = [
    "sex",
    "c_section"
]

birth_cont = [
    "gestation_weeks",
    "birth_weight"
]

env_cat = [
    "parent_asthma",
    "sibling_allergy",
    "pet_ever",
    "passive_smoke_ever",
    "child_passive_smoke"
]

env_cont = [
    "rural_years",
    "outdoor_avg"
]

diet_cat = [
    "eat_picky",
    "eat_regularity",
    "eat_amount",
    "eat_speed"
]

diet_cont = [
    "eat_snack_avg",
    "eat_breakfast_avg",
    "eat_delivery_avg"
]

# 해석이 민감했던 변수는 별도 민감도 세트로만 확인
sensitive_cat = [
    "area_type_w6"
]

sensitive_cont = [
    "breastfed_months"
]

def exists(cols):
    return [c for c in cols if c in feat.columns]

variable_sets = {
    "A_core4": {
        "cat": exists(core_cat),
        "cont": []
    },
    "B_core4_birth": {
        "cat": exists(core_cat + birth_cat),
        "cont": exists(birth_cont)
    },
    "C_core4_environment": {
        "cat": exists(core_cat + env_cat),
        "cont": exists(env_cont)
    },
    "D_core4_diet": {
        "cat": exists(core_cat + diet_cat),
        "cont": exists(diet_cont)
    },
    "E_core4_birth_env_diet": {
        "cat": exists(core_cat + birth_cat + env_cat + diet_cat),
        "cont": exists(birth_cont + env_cont + diet_cont)
    },
    "F_sensitivity_plus_area_breastfed": {
        "cat": exists(core_cat + birth_cat + env_cat + diet_cat + sensitive_cat),
        "cont": exists(birth_cont + env_cont + diet_cont + sensitive_cont)
    }
}

print("\n[변수 세트 확인]")
for set_name, v in variable_sets.items():
    print(f"\n{set_name}")
    print("  categorical:", v["cat"])
    print("  continuous :", v["cont"])

# ------------------------------------------------------------
# 2) 인코더 호환 함수
# ------------------------------------------------------------
def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def make_ordinal_encoder():
    return OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

# ------------------------------------------------------------
# 3) Threshold 선택: Validation F2 기준
# ------------------------------------------------------------
def find_best_threshold_by_f2(y_true, proba):
    rows = []

    for th in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= th).astype(int)

        rows.append({
            "threshold": th,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "f2": fbeta_score(y_true, pred, beta=2, zero_division=0)
        })

    th_df = pd.DataFrame(rows)
    best = th_df.sort_values(["f2", "recall"], ascending=False).iloc[0]

    return float(best["threshold"]), th_df

# ------------------------------------------------------------
# 4) 평가 함수
# ------------------------------------------------------------
def evaluate_model(y_true, proba, threshold, set_name, sampler_name, model_name, split_name, best_params):
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    return {
        "variable_set": set_name,
        "sampler": sampler_name,
        "model": model_name,
        "split": split_name,
        "threshold": threshold,
        "AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "best_params": str(best_params)
    }

# ------------------------------------------------------------
# 5) 일반 전처리기
#    - NoSampling / ClassWeight / RandomOverSampler에서 사용
#    - 범주형: 최빈값 대체 후 OneHot
#    - 연속형: 중앙값 대체 후 StandardScaler
# ------------------------------------------------------------
def make_onehot_preprocessor(cat_cols, cont_cols):
    transformers = []

    if len(cat_cols) > 0:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_onehot_encoder())
                ]),
                cat_cols
            )
        )

    if len(cont_cols) > 0:
        transformers.append(
            (
                "cont",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                cont_cols
            )
        )

    return ColumnTransformer(transformers)

# ------------------------------------------------------------
# 6) Proper SMOTE용 전처리기
#    핵심:
#      - OneHot 전에 SMOTEN/SMOTENC 적용
#      - 범주형은 OrdinalEncoder로 임시 숫자화
#      - SMOTE 후 다시 OneHot
# ------------------------------------------------------------
def make_presmote_preprocessor(cat_cols, cont_cols):
    transformers = []

    if len(cat_cols) > 0:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", make_ordinal_encoder())
                ]),
                cat_cols
            )
        )

    if len(cont_cols) > 0:
        transformers.append(
            (
                "cont",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                cont_cols
            )
        )

    return ColumnTransformer(transformers)

def make_postsmote_onehot(cat_cols, cont_cols):
    """
    presmote_preprocessor 결과 배열 순서:
      앞쪽: cat_cols 개수만큼 범주형 ordinal
      뒤쪽: cont_cols 개수만큼 연속형 scaled

    SMOTEN/SMOTENC 후 범주형 부분을 다시 OneHot으로 바꿔 모델에 넣는다.
    """
    transformers = []

    n_cat = len(cat_cols)
    n_cont = len(cont_cols)

    if n_cat > 0:
        cat_idx = list(range(n_cat))
        transformers.append(
            (
                "cat_after_smote",
                make_onehot_encoder(),
                cat_idx
            )
        )

    if n_cont > 0:
        cont_idx = list(range(n_cat, n_cat + n_cont))
        transformers.append(
            (
                "cont_after_smote",
                "passthrough",
                cont_idx
            )
        )

    return ColumnTransformer(transformers)

# ------------------------------------------------------------
# 7) 모델 기본값 및 하이퍼파라미터 후보
# ------------------------------------------------------------
def get_model_and_param_dist(model_name, class_weight=False):
    if model_name == "LogisticRegression":
        lr_weight = "balanced" if class_weight else None

        model = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight=lr_weight,
            random_state=RANDOM_STATE
        )

        param_dist = {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            "model__penalty": ["l2"]
        }

    elif model_name == "RandomForest":
        rf_weight = "balanced" if class_weight else None

        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight=rf_weight,
            n_jobs=-1
        )

        param_dist = {
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 3, 5, 8, 12],
            "model__min_samples_leaf": [1, 3, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", None]
        }

    elif model_name == "GradientBoosting":
        model = GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )

        param_dist = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_leaf": [1, 3, 5, 10],
            "model__subsample": [0.7, 0.85, 1.0]
        }

    elif model_name == "MLP":
        model = MLPClassifier(
            max_iter=1500,
            early_stopping=True,
            random_state=RANDOM_STATE
        )

        param_dist = {
            "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.0005, 0.001, 0.003]
        }

    else:
        raise ValueError(f"지원하지 않는 모델명: {model_name}")

    return model, param_dist

# ------------------------------------------------------------
# 8) 파이프라인 생성
# ------------------------------------------------------------
def make_training_pipeline(cat_cols, cont_cols, sampler_name, model, class_weight=False):
    """
    sampler_name:
      - NoSampling
      - ClassWeight
      - RandomOverSampler
      - ProperSMOTE
    """

    # -----------------------------
    # NoSampling / ClassWeight
    # -----------------------------
    if sampler_name in ["NoSampling", "ClassWeight"]:
        preprocessor = make_onehot_preprocessor(cat_cols, cont_cols)

        pipe = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        return pipe

    # -----------------------------
    # RandomOverSampler
    # 단순 복제라서 OneHot 이후 적용해도 문제 없음
    # -----------------------------
    if sampler_name == "RandomOverSampler":
        preprocessor = make_onehot_preprocessor(cat_cols, cont_cols)

        pipe = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
                ("model", model)
            ]
        )

        return pipe

    # -----------------------------
    # ProperSMOTE
    # OneHot 전에 SMOTEN / SMOTENC 적용
    # -----------------------------
    if sampler_name == "ProperSMOTE":
        n_cat = len(cat_cols)
        n_cont = len(cont_cols)

        pre_smote = make_presmote_preprocessor(cat_cols, cont_cols)

        # 범주형 + 연속형 혼합
        if n_cat > 0 and n_cont > 0:
            sampler = SMOTENC(
                categorical_features=list(range(n_cat)),
                random_state=RANDOM_STATE,
                k_neighbors=5
            )

            post_smote = make_postsmote_onehot(cat_cols, cont_cols)

            pipe = ImbPipeline(
                steps=[
                    ("pre_smote", pre_smote),
                    ("sampler", sampler),
                    ("post_smote", post_smote),
                    ("model", model)
                ]
            )

        # 범주형만 있는 경우
        elif n_cat > 0 and n_cont == 0:
            sampler = SMOTEN(
                random_state=RANDOM_STATE,
                k_neighbors=5
            )

            post_smote = make_postsmote_onehot(cat_cols, cont_cols)

            pipe = ImbPipeline(
                steps=[
                    ("pre_smote", pre_smote),
                    ("sampler", sampler),
                    ("post_smote", post_smote),
                    ("model", model)
                ]
            )

        # 연속형만 있는 경우
        elif n_cat == 0 and n_cont > 0:
            sampler = SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=5
            )

            pipe = ImbPipeline(
                steps=[
                    ("pre_smote", pre_smote),
                    ("sampler", sampler),
                    ("model", model)
                ]
            )

        else:
            raise ValueError("범주형/연속형 변수가 모두 없습니다.")

        return pipe

    raise ValueError(f"지원하지 않는 sampler_name: {sampler_name}")

# ------------------------------------------------------------
# 9) 실험 설정
# ------------------------------------------------------------
sampling_settings = [
    {
        "name": "NoSampling",
        "class_weight": False
    },
    {
        "name": "ClassWeight",
        "class_weight": True
    },
    {
        "name": "RandomOverSampler",
        "class_weight": False
    },
    {
        "name": "ProperSMOTE",
        "class_weight": False
    }
]

model_names = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting",
    "MLP"
]

all_results = []
all_thresholds = []
all_best_params = []

# ------------------------------------------------------------
# 10) 변수 세트별 모델 학습 + 튜닝
# ------------------------------------------------------------
for set_name, vars_info in variable_sets.items():
    cat_cols = vars_info["cat"]
    cont_cols = vars_info["cont"]

    use_cols = cat_cols + cont_cols

    if len(use_cols) == 0:
        print(f"\n⚠ {set_name}: 사용할 변수가 없어 생략")
        continue

    print(f"\n{'='*80}")
    print(f"[변수 세트] {set_name}")
    print("="*80)

    model_df = feat[[TARGET_COL] + use_cols].copy()

    X = model_df[use_cols]
    y = model_df[TARGET_COL].astype(int)

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_valid_raw, X_test_raw, y_valid, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Train      : {X_train_raw.shape}, 양성비율={y_train.mean()*100:.2f}%")
    print(f"Validation : {X_valid_raw.shape}, 양성비율={y_valid.mean()*100:.2f}%")
    print(f"Test       : {X_test_raw.shape}, 양성비율={y_test.mean()*100:.2f}%")

    for sampling in sampling_settings:
        sampler_name = sampling["name"]
        class_weight = sampling["class_weight"]

        print(f"\n--- Sampling: {sampler_name} ---")

        for model_name in model_names:
            print(f"\n[튜닝 시작] {set_name} | {sampler_name} | {model_name}")

            model, param_dist = get_model_and_param_dist(
                model_name=model_name,
                class_weight=class_weight
            )

            pipe = make_training_pipeline(
                cat_cols=cat_cols,
                cont_cols=cont_cols,
                sampler_name=sampler_name,
                model=model,
                class_weight=class_weight
            )

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_dist,
                n_iter=N_ITER_SEARCH,
                scoring=TUNING_SCORING,
                cv=CV,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
                refit=True
            )

            try:
                search.fit(X_train_raw, y_train)
            except Exception as e:
                print(f"  ⚠ 실패: {set_name} | {sampler_name} | {model_name}")
                print(f"  원인: {e}")
                continue

            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_

            all_best_params.append({
                "variable_set": set_name,
                "sampler": sampler_name,
                "model": model_name,
                "best_cv_score": best_cv_score,
                "best_params": str(best_params)
            })

            valid_proba = best_model.predict_proba(X_valid_raw)[:, 1]
            test_proba = best_model.predict_proba(X_test_raw)[:, 1]

            # Validation에서 F2 기준 threshold 선택
            best_th, th_df = find_best_threshold_by_f2(y_valid, valid_proba)

            th_df["variable_set"] = set_name
            th_df["sampler"] = sampler_name
            th_df["model"] = model_name
            all_thresholds.append(th_df)

            valid_result = evaluate_model(
                y_true=y_valid,
                proba=valid_proba,
                threshold=best_th,
                set_name=set_name,
                sampler_name=sampler_name,
                model_name=model_name,
                split_name="validation",
                best_params=best_params
            )

            test_result = evaluate_model(
                y_true=y_test,
                proba=test_proba,
                threshold=best_th,
                set_name=set_name,
                sampler_name=sampler_name,
                model_name=model_name,
                split_name="test",
                best_params=best_params
            )

            all_results.append(valid_result)
            all_results.append(test_result)

            print(
                f"  BestCV({TUNING_SCORING})={best_cv_score:.4f} | "
                f"th={best_th:.2f} | "
                f"Test AUC={test_result['AUC']:.4f}, "
                f"PR-AUC={test_result['PR_AUC']:.4f}, "
                f"Recall={test_result['recall']:.4f}, "
                f"Precision={test_result['precision']:.4f}, "
                f"F2={test_result['f2']:.4f}, "
                f"Brier={test_result['Brier']:.4f}"
            )

# ------------------------------------------------------------
# 11) 전체 결과 저장
# ------------------------------------------------------------
result_df = pd.DataFrame(all_results)

result_path = os.path.join(EXPAND_DIR, "01_all_expanded_model_results_tuned.csv")
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

if len(all_thresholds) > 0:
    threshold_df = pd.concat(all_thresholds, axis=0)
else:
    threshold_df = pd.DataFrame()

threshold_path = os.path.join(EXPAND_DIR, "02_all_threshold_search_results_tuned.csv")
threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")

best_params_df = pd.DataFrame(all_best_params)
best_params_path = os.path.join(EXPAND_DIR, "03_best_params_by_model.csv")
best_params_df.to_csv(best_params_path, index=False, encoding="utf-8-sig")

test_df = result_df[result_df["split"] == "test"].copy()

# 최종 순위:
# 1순위 AUC
# 2순위 PR-AUC
# 3순위 F2
# 4순위 Brier는 낮을수록 좋음
test_df = test_df.sort_values(
    by=["AUC", "PR_AUC", "f2", "Brier"],
    ascending=[False, False, False, True]
)

test_sorted_path = os.path.join(EXPAND_DIR, "04_test_results_sorted_tuned.csv")
test_df.to_csv(test_sorted_path, index=False, encoding="utf-8-sig")

print(f"\n{'='*80}")
print("[Test 기준 전체 모델 순위]")
print("="*80)
print(test_df.round(4).to_string(index=False))

# ------------------------------------------------------------
# 12) Core4 대비 성능 향상 확인
# ------------------------------------------------------------
core_best = test_df[test_df["variable_set"] == "A_core4"].sort_values(
    by=["AUC", "PR_AUC", "f2", "Brier"],
    ascending=[False, False, False, True]
).head(1)

overall_best = test_df.sort_values(
    by=["AUC", "PR_AUC", "f2", "Brier"],
    ascending=[False, False, False, True]
).head(1)

print(f"\n{'='*80}")
print("[Core4 모델 vs 전체 최우수 모델]")
print("="*80)

print("\n[Core4 최우수]")
print(core_best.to_string(index=False))

print("\n[전체 최우수]")
print(overall_best.to_string(index=False))

print(f"\n저장 폴더: {EXPAND_DIR}")
print("저장 파일:")
print("  01_all_expanded_model_results_tuned.csv")
print("  02_all_threshold_search_results_tuned.csv")
print("  03_best_params_by_model.csv")
print("  04_test_results_sorted_tuned.csv")

#%%
# ============================================================
# [11] 서비스용 최종 모델 저장
#      최종 선택:
#      C_core4_environment + NoSampling + LogisticRegression
#      best_params = {'model__penalty': 'l2', 'model__C': 0.01}
#      threshold = 0.12
# ============================================================

print(f"\n{'='*80}")
print("[11] 서비스용 최종 모델 저장")
print("     C_core4_environment + NoSampling + LogisticRegression")
print("="*80)

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 0) 저장 경로
# ------------------------------------------------------------
SERVICE_DIR = os.path.join(DATA_DIR, "atopy_service_model_final")
os.makedirs(SERVICE_DIR, exist_ok=True)

TARGET_COL = "Y"
RANDOM_STATE = 42

# ------------------------------------------------------------
# 1) 최종 서비스 모델 변수
#    C_core4_environment = core4 + environment
# ------------------------------------------------------------
SERVICE_CAT_COLS = [
    # core4
    "antibiotic",
    "parent_AD",
    "parent_AR",
    "mold_ever",

    # environment
    "parent_asthma",
    "sibling_allergy",
    "pet_ever",
    "passive_smoke_ever",
    "child_passive_smoke",
]

SERVICE_CONT_COLS = [
    "rural_years",
    "outdoor_avg",
]

SERVICE_FEATURES = SERVICE_CAT_COLS + SERVICE_CONT_COLS

missing_cols = [
    c for c in SERVICE_FEATURES + [TARGET_COL]
    if c not in feat.columns
]

if missing_cols:
    raise ValueError(f"feat에 없는 컬럼: {missing_cols}")

service_df = feat[[TARGET_COL] + SERVICE_FEATURES].copy()

X = service_df[SERVICE_FEATURES]
y = service_df[TARGET_COL].astype(int)

print("\n[서비스 모델 입력 변수]")
print(SERVICE_FEATURES)

print("\n[Y 분포]")
print(y.value_counts().sort_index())
print(f"양성 비율: {y.mean() * 100:.2f}%")

# ------------------------------------------------------------
# 2) Train / Validation / Test 분리
#    [10]과 동일하게 60 / 20 / 20 구조 유지
# ------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=y
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\n[데이터 분리]")
print(f"Train      : {X_train.shape}, 양성비율={y_train.mean() * 100:.2f}%")
print(f"Validation : {X_valid.shape}, 양성비율={y_valid.mean() * 100:.2f}%")
print(f"Test       : {X_test.shape}, 양성비율={y_test.mean() * 100:.2f}%")

# ------------------------------------------------------------
# 3) OneHotEncoder 버전 호환
# ------------------------------------------------------------
try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

# ------------------------------------------------------------
# 4) 전처리기
#    범주형: 최빈값 대체 + OneHot
#    연속형: 중앙값 대체 + StandardScaler
# ------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot)
            ]),
            SERVICE_CAT_COLS
        ),
        (
            "cont",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            SERVICE_CONT_COLS
        )
    ]
)

# ------------------------------------------------------------
# 5) 최종 서비스 모델
#    [10] 튜닝 결과 기준:
#    C_core4_environment + NoSampling + LogisticRegression
#    best_params = C=0.01, penalty=l2
# ------------------------------------------------------------
service_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            penalty="l2",
            C=0.01,
            random_state=RANDOM_STATE
        ))
    ]
)

service_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 6) 서비스 Threshold
#    [10] 결과 기준:
#    C_core4_environment + NoSampling + LogisticRegression
# ------------------------------------------------------------
SERVICE_THRESHOLD = 0.12

# ------------------------------------------------------------
# 7) Test 성능 확인
# ------------------------------------------------------------
test_proba = service_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= SERVICE_THRESHOLD).astype(int)

auc = roc_auc_score(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)
brier = brier_score_loss(y_test, test_proba)
recall = recall_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)
f2 = fbeta_score(y_test, test_pred, beta=2, zero_division=0)
cm = confusion_matrix(y_test, test_pred)

print("\n[Test 성능 확인]")
print(f"AUC       : {auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"Brier     : {brier:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1        : {f1:.4f}")
print(f"F2        : {f2:.4f}")

print("\n[Confusion Matrix]")
print(cm)

print("\n[Classification Report]")
print(classification_report(y_test, test_pred, zero_division=0))

# ------------------------------------------------------------
# 8) 전처리 후 feature 이름 확인
# ------------------------------------------------------------
try:
    feature_names = service_model.named_steps["preprocessor"].get_feature_names_out()
except Exception:
    feature_names = None

# ------------------------------------------------------------
# 9) Logistic Regression 계수 저장
# ------------------------------------------------------------
coef_df = None

if feature_names is not None:
    coef = service_model.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "OR": np.exp(coef)
    }).sort_values("coef", ascending=False)

    coef_path = os.path.join(SERVICE_DIR, "atopy_service_model_coefficients.csv")

    coef_df.to_csv(
        coef_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[계수 저장 완료]")
    print(coef_path)

# ------------------------------------------------------------
# 10) 모델 저장
# ------------------------------------------------------------
model_path = os.path.join(SERVICE_DIR, "atopy_service_model.joblib")
meta_joblib_path = os.path.join(SERVICE_DIR, "atopy_service_meta.joblib")
meta_json_path = os.path.join(SERVICE_DIR, "atopy_service_meta.json")

joblib.dump(service_model, model_path)

meta = {
    "features": SERVICE_FEATURES,
    "cat_cols": SERVICE_CAT_COLS,
    "cont_cols": SERVICE_CONT_COLS,
    "threshold": SERVICE_THRESHOLD,
    "variable_set": "C_core4_environment",
    "sampler": "NoSampling",
    "model_name": "LogisticRegression",
    "best_params": {
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 3000
    },
    "test_metrics": {
        "AUC": float(auc),
        "PR_AUC": float(pr_auc),
        "Brier": float(brier),
        "Recall": float(recall),
        "Precision": float(precision),
        "F1": float(f1),
        "F2": float(f2),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1])
    },
    "note": "진단 확정 모델이 아니라 문진 기반 아토피 조기 위험군 선별 모델"
}

joblib.dump(meta, meta_joblib_path)

with open(meta_json_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n[저장 완료]")
print(model_path)
print(meta_joblib_path)
print(meta_json_path)

print("\n[최종 서비스 모델 요약]")
print("모델     : C_core4_environment + NoSampling + LogisticRegression")
print("C        : 0.01")
print("threshold:", SERVICE_THRESHOLD)