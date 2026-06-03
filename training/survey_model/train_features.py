"""
Step 2: 파생변수 생성
======================
입력: merged.csv (step1_merge.py 출력)
출력: pskc_final.csv (분석용 최종 테이블)

실행 전 DATA_DIR 경로만 수정할 것
"""

import pandas as pd
import numpy as np
import os

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
def get_breastfed(row):
    for col in ["DCh11fed001_w4", "DCh10fed001_w3",
                "DCh09fed001_w2", "DCh08fed001_w1"]:
        if col in df.columns:
            val = pd.to_numeric(row.get(col), errors='coerce')
            if pd.notna(val):
                return val
    return np.nan

feat["breastfed_raw"] = df.apply(get_breastfed, axis=1)
feat["breastfed"] = feat["breastfed_raw"].apply(
    lambda x: 1 if x in [1, 2] else 0
)

def get_breastfed_months(row):
    for col in ["DCh11fed002_w4", "DCh10fed002_w3", "DCh09fed002_w2"]:
        if col in df.columns:
            val = pd.to_numeric(row.get(col), errors='coerce')
            if pd.notna(val):
                return val
    return np.nan

feat["breastfed_months"] = df.apply(get_breastfed_months, axis=1)
feat["breastfed_months"] = pd.to_numeric(feat["breastfed_months"], errors='coerce')
feat.loc[feat["breastfed"] == 0, "breastfed_months"] = 0
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