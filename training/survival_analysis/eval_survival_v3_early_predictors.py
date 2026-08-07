"""
Step 5: 1~3차 + 시점불변 변수 → 4~10차 아토피 예측
=====================================================
독립변수: 1~3차 직접 측정 + 6차 시점불변(가족력/곰팡이/항생제 등)
종속변수: 4~10차 중 한 번이라도 AD 발병 여부
분석: 로지스틱 회귀 + 생존분석 (KM + Cox)

입력: pskc_ad_history.csv + merged.csv
출력: 로지스틱 결과, KM 곡선, Cox 결과

사전 설치: pip install lifelines
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ============================================================
# [설정]
# ============================================================
DATA_DIR   = r"data/"   # ← 본인 경로
OUTPUT_DIR = os.path.join(DATA_DIR, "model_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# [1] 데이터 로드
# ============================================================
print("=" * 60)
print("Step 1: 데이터 로드")
print("=" * 60)

history = pd.read_csv(os.path.join(DATA_DIR, "pskc_ad_history.csv"))
merged  = pd.read_csv(os.path.join(DATA_DIR, "merged.csv"), low_memory=False)

print(f"  AD 이력: {len(history)}명")
print(f"  merged: {len(merged)}행, {len(merged.columns)}컬럼")


# ============================================================
# [2] 무응답 처리
# ============================================================
for col in merged.columns:
    mask = (merged[col].astype(str).str.strip() == '99999999')
    if mask.sum() > 0:
        merged[col] = merged[col].where(~mask, np.nan)

def to_num(s):
    return pd.to_numeric(s, errors='coerce')

def get_col(col):
    if col in merged.columns:
        return to_num(merged[col])
    return pd.Series(np.nan, index=merged.index)


# ============================================================
# [3] 파생변수 생성 (1~3차 + 시점불변)
# ============================================================
print(f"\n{'=' * 60}")
print("Step 2: 파생변수 생성")
print("=" * 60)

feat = pd.DataFrame()
feat["N_ID"] = merged["N_ID"]

# --- 출생 정보 (1차) ---
feat["gestation_weeks"] = get_col("BMt08pnb001_w1")
feat["birth_weight"]    = get_col("BCh08hlt001_w1")
feat["c_section"]       = to_num(merged.get("BMt08pnb003_w1",
                          np.nan)).map({1: 0, 2: 1, 3: 1})
feat["sex"]             = to_num(merged.get("DCh09dmg001_w2",
                          np.nan)).map({1: 0, 2: 1})
print("  ✓ 출생 정보 (재태기간, 체중, 분만형태, 성별)")

# --- 모유수유 기간 (2~4차, 4차 우선) ---
def get_bf_months(row):
    for col in ["DCh11fed002_w4", "DCh10fed002_w3", "DCh09fed002_w2"]:
        if col in merged.columns:
            val = pd.to_numeric(row.get(col), errors='coerce')
            if pd.notna(val):
                return val
    return np.nan

feat["breastfed_months"] = merged.apply(get_bf_months, axis=1)
feat["breastfed_months"] = to_num(feat["breastfed_months"])
# 1차에서 NaN = 처음부터 수유 안 함 → 0
if "DCh08fed001_w1" in merged.columns:
    never_bf = to_num(merged["DCh08fed001_w1"]).isna()
    feat.loc[never_bf, "breastfed_months"] = 0
print("  ✓ 모유수유 기간")

# --- 부모 흡연 누적 (1~3차) ---
smoke_cols_early = [
    "EMt08smk001_w1", "FFt08smk001_w1",
    "EMt09smk001_w2", "FFt09smk001_w2",
    "EMt10smk001_w3", "FFt10smk001_w3",
]
smoke_avail = [c for c in smoke_cols_early if c in merged.columns]
if smoke_avail:
    smoke_df = merged[smoke_avail].apply(lambda col: to_num(col))
    feat["passive_smoke_early"] = smoke_df.apply(
        lambda row: 1 if any(row.isin([1, 2]))
        else (0 if row.notna().any() else np.nan), axis=1
    )
print("  ✓ 부모 흡연 누적 (1~3차)")

# --- 거주환경 누적 (1~3차) ---
cmm_early = ["DHu08cmm002_w1", "DHu09cmm002_w2", "DHu10cmm002_w3"]
cmm_avail = [c for c in cmm_early if c in merged.columns]
if cmm_avail:
    cmm_df = merged[cmm_avail].apply(lambda col: to_num(col))
    feat["rural_years_early"] = cmm_df.apply(
        lambda row: sum(row == 2), axis=1
    )
print("  ✓ 농촌 거주 누적 (1~3차)")

# --- 실외활동 (3차) ---
cols_3 = ["ECh10dlc003_w3", "ECh10dlc010_w3"]
for c in cols_3:
    if c in merged.columns:
        merged[c] = to_num(merged[c])

def outdoor_w3(row):
    vals = [pd.to_numeric(row.get(c), errors='coerce')
            for c in cols_3 if c in merged.columns]
    vals = [v for v in vals if pd.notna(v)]
    return sum(vals) / 60 if vals else np.nan

feat["outdoor_w3"] = merged.apply(outdoor_w3, axis=1)
print("  ✓ 실외활동 (3차, 시간)")

# --- 시점불변 변수 (6차 회고, 출생~첫돌 사건) ---
def any_one(cols, yes_val=1):
    avail = [c for c in cols if c in merged.columns]
    if not avail:
        return pd.Series(np.nan, index=merged.index)
    num_df = merged[avail].apply(lambda col: to_num(col))
    return num_df.apply(
        lambda row: 1 if any(row == yes_val)
        else (0 if row.notna().any() else np.nan), axis=1
    )

feat["parent_AD"]      = any_one(["KFt13adx004_w6", "KMt13adx004_w6"])
feat["parent_AR"]      = any_one(["KFt13arx009_w6", "KMt13arx009_w6"])
feat["parent_asthma"]  = any_one(["KFt13asx011_w6", "KMt13asx011_w6"])
feat["mold_ever"]      = any_one(["KCh13mld001_w6",
                                   "KCh13mld006_w6", "KCh13mld008_w6"])
feat["antibiotic"]     = to_num(get_col("KCh13drg004_w6"))
feat["pet_ever"]       = any_one(["KCh13pet001_w6", "KCh13pet003_w6"])
feat["prenatal_smoke"] = to_num(merged.get("KMt13smk011_w6",
                         np.nan)).apply(
    lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
)
print("  ✓ 시점불변 변수 (가족력, 곰팡이, 항생제 등)")

print(f"\n  총 변수: {len(feat.columns)-1}개")


# ============================================================
# [4] 종속변수 & 생존분석용 time/event 생성
# ============================================================
print(f"\n{'=' * 60}")
print("Step 3: 종속변수 생성")
print("=" * 60)

# AD 진단 이력에서 3차까지 미진단 필터
for w in [3, 4, 5, 6, 7, 8, 9, 10]:
    col_map = {
        3: ("ECh10hlt035_w3", "str6"),
        4: ("DCh11hlt035_w4", "str6"),
        5: ("DCh12hlt035_w5", "str6"),
        6: ("DCh13hlt035_w6", "str6"),
        7: ("DCh14hlt031f_w7","str6"),
        8: ("KCh15adx004_w8", "str1"),
        9: ("KCh16adx004_w9", "str1"),
        10:("DCh17hlt031k_w10","notnull"),
    }
    col, method = col_map[w]
    if col not in merged.columns:
        merged[f"ad_w{w}"] = np.nan
        continue
    if method == "str6":
        merged[f"ad_w{w}"] = (merged[col].astype(str).str.strip()=='6').astype(float)
    elif method == "str1":
        merged[f"ad_w{w}"] = (merged[col].astype(str).str.strip()=='1').astype(float)
    elif method == "notnull":
        merged[f"ad_w{w}"] = merged[col].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() in ['','nan','0'] else 1
        ).astype(float)

# 3차까지 미진단 필터
ad_w3 = merged["ad_w3"] if "ad_w3" in merged.columns else pd.Series(0, index=merged.index)
no_ad_w3 = merged[ad_w3 == 0].copy()
feat = feat[feat["N_ID"].isin(no_ad_w3["N_ID"])].copy()
feat = feat.merge(no_ad_w3[["N_ID"] +
    [f"ad_w{w}" for w in range(4,11) if f"ad_w{w}" in no_ad_w3.columns]],
    on="N_ID", how="left")

print(f"  3차까지 미진단: {len(feat)}명")

# 이진 종속변수 Y
ad_4to10 = [f"ad_w{w}" for w in range(4,11) if f"ad_w{w}" in feat.columns]
feat["Y"] = feat[ad_4to10].apply(
    lambda row: 1 if any(row == 1) else 0, axis=1
)
print(f"  양성(Y=1): {int(feat['Y'].sum())}명 ({feat['Y'].mean()*100:.1f}%)")
print(f"  음성(Y=0): {int((feat['Y']==0).sum())}명")

# 생존분석용 time (최초 발병 차수)
def get_onset(row):
    for w in range(4, 11):
        col = f"ad_w{w}"
        if col in row.index and row[col] == 1:
            return w
    return 10

feat["time"]  = feat.apply(get_onset, axis=1)
feat["event"] = feat["Y"]


# ============================================================
# [5] 로지스틱 회귀
# ============================================================
print(f"\n{'=' * 60}")
print("Step 4: 로지스틱 회귀")
print("=" * 60)

indep_vars = [
    "gestation_weeks", "birth_weight", "c_section", "sex",
    "breastfed_months", "passive_smoke_early", "rural_years_early",
    "outdoor_w3",
    "parent_AD", "parent_AR", "parent_asthma",
    "mold_ever", "antibiotic", "pet_ever", "prenatal_smoke",
]
indep_vars = [v for v in indep_vars if v in feat.columns]

# 결측치 처리
cat_vars  = ["c_section", "sex", "passive_smoke_early", "rural_years_early",
             "parent_AD", "parent_AR", "parent_asthma",
             "mold_ever", "pet_ever", "prenatal_smoke"]
cont_vars = ["gestation_weeks", "birth_weight", "breastfed_months",
             "outdoor_w3", "antibiotic"]
cat_vars  = [v for v in cat_vars  if v in feat.columns]
cont_vars = [v for v in cont_vars if v in feat.columns]

X = feat[indep_vars].copy()
if cat_vars:
    X[cat_vars] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_vars])
if cont_vars:
    X[cont_vars] = SimpleImputer(strategy='median').fit_transform(X[cont_vars])

Y = feat["Y"].values

# 5-Fold CV AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logit = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
auc_scores = cross_val_score(logit, X_scaled, Y, cv=cv, scoring='roc_auc')
print(f"\n  AUC (5-Fold CV): {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# statsmodels OR + p값
X_sm = sm.add_constant(X.astype(float))
logit_sm = sm.Logit(Y, X_sm).fit(disp=0)

coef    = logit_sm.params
pvalues = logit_sm.pvalues
ci      = logit_sm.conf_int()

result_df = pd.DataFrame({
    "variable": coef.index,
    "coef":     coef.values,
    "OR":       np.exp(coef.values),
    "CI_lower": np.exp(ci[0].values),
    "CI_upper": np.exp(ci[1].values),
    "p_value":  pvalues.values
}).query("variable != 'const'").sort_values("p_value")

print(f"\n  {'변수':<22} {'OR':>7} {'95% CI':>16} {'p값':>8} {'유의'}")
print("  " + "-" * 60)
for _, row in result_df.iterrows():
    sig = "✅" if row["p_value"] < 0.05 else ("△" if row["p_value"] < 0.10 else "❌")
    ci_str = f"({row['CI_lower']:.2f}~{row['CI_upper']:.2f})"
    print(f"  {row['variable']:<22} {row['OR']:>7.3f} {ci_str:>16} {row['p_value']:>8.4f} {sig}")

result_df.to_csv(
    os.path.join(OUTPUT_DIR, "logit_early_results.csv"),
    index=False, encoding='utf-8-sig'
)

sig_df = result_df[result_df["p_value"] < 0.05]
print(f"\n  유의 변수 (p<0.05): {len(sig_df)}개")


# ============================================================
# [6] 단변량 Cox → 다변량 Cox
# ============================================================
print(f"\n{'=' * 60}")
print("Step 5: Cox 비례위험모델")
print("=" * 60)

cox_data = feat[["time", "event"] + indep_vars].copy()
cox_data = cox_data.apply(lambda col: pd.to_numeric(col, errors='coerce'))
if cat_vars:
    cox_data[cat_vars] = SimpleImputer(
        strategy='most_frequent').fit_transform(cox_data[cat_vars])
if cont_vars:
    cox_data[cont_vars] = SimpleImputer(
        strategy='median').fit_transform(cox_data[cont_vars])
cox_data = cox_data.dropna()

print(f"  분석 대상: {len(cox_data)}명")

# 단변량 Cox
print(f"\n  [단변량 Cox]")
uni_results = []
for var in indep_vars:
    sub = cox_data[["time", "event", var]].dropna()
    if len(sub) < 50 or sub[var].nunique() < 2:
        continue
    try:
        cph = CoxPHFitter()
        cph.fit(sub.astype(float), duration_col="time", event_col="event")
        s = cph.summary
        hr  = s.loc[var, "exp(coef)"]
        p   = s.loc[var, "p"]
        ci_lo = s.loc[var, "exp(coef) lower 95%"]
        ci_hi = s.loc[var, "exp(coef) upper 95%"]
        sel = "✅" if p < 0.20 else "❌"
        print(f"  {var:<22} HR={hr:.3f} ({ci_lo:.2f}~{ci_hi:.2f}) p={p:.4f} {sel}")
        uni_results.append({"variable": var, "HR": hr, "p": p, "selected": p < 0.20})
    except:
        pass

selected = [r["variable"] for r in uni_results if r["selected"]]
print(f"\n  p<0.20 후보: {selected}")

# 다변량 Cox
if len(selected) > 0:
    print(f"\n  [다변량 Cox]")
    multi_data = cox_data[["time", "event"] + selected].dropna().astype(float)
    print(f"  분석 대상: {len(multi_data)}명, 발병: {int(multi_data['event'].sum())}명")

    cph_multi = CoxPHFitter()
    cph_multi.fit(multi_data, duration_col="time", event_col="event")
    print(f"  Concordance Index: {cph_multi.concordance_index_:.3f}")

    multi_result = cph_multi.summary[[
        "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"
    ]].copy()
    multi_result.columns = ["HR", "CI_lower", "CI_upper", "p_value"]
    multi_result = multi_result.sort_values("p_value")

    print(f"\n  {'변수':<22} {'HR':>7} {'95% CI':>16} {'p값':>8} {'유의'}")
    print("  " + "-" * 60)
    for var, row in multi_result.iterrows():
        sig = "✅" if row["p_value"] < 0.05 else ("△" if row["p_value"] < 0.10 else "❌")
        ci_str = f"({row['CI_lower']:.2f}~{row['CI_upper']:.2f})"
        print(f"  {var:<22} {row['HR']:>7.3f} {ci_str:>16} {row['p_value']:>8.4f} {sig}")

    multi_result.to_csv(
        os.path.join(OUTPUT_DIR, "cox_early_multivariate.csv"),
        index=True, encoding='utf-8-sig'
    )


# ============================================================
# [7] KM 곡선 시각화
# ============================================================
print(f"\n{'=' * 60}")
print("Step 6: KM 곡선 시각화")
print("=" * 60)

km_vars = ["parent_AD", "mold_ever", "passive_smoke_early"]
km_vars = [v for v in km_vars if v in cox_data.columns]

n_plots = 1 + len(km_vars)
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
if n_plots == 1:
    axes = [axes]

# 전체 KM
kmf = KaplanMeierFitter()
kmf.fit(cox_data["time"], cox_data["event"], label="전체")
kmf.plot_survival_function(ax=axes[0], ci_show=True, linewidth=2)
axes[0].set_title("전체 KM 곡선\n(3차 이후 아토피 발병)", fontsize=11)
axes[0].set_xlabel("차수", fontsize=9)
axes[0].set_ylabel("미발병 확률", fontsize=9)
axes[0].set_ylim(0.6, 1.01)
axes[0].grid(alpha=0.3)

# 누적 발병률 표시
for t in range(4, 11):
    ci = (1 - kmf.survival_function_at_times(t).values[0]) * 100
    axes[0].annotate(f"{ci:.1f}%", xy=(t, 1-ci/100),
                     fontsize=7, color='navy', ha='center')

# 위험인자별 KM
var_labels = {
    "parent_AD":           "부모 아토피",
    "mold_ever":           "실내 곰팡이",
    "passive_smoke_early": "부모 흡연 (1~3차)",
}

for i, var in enumerate(km_vars):
    ax = axes[i + 1]
    sub = cox_data[["time", "event", var]].dropna()
    sub[var] = pd.to_numeric(sub[var], errors='coerce')
    sub = sub.dropna()

    g0 = sub[sub[var] == 0]
    g1 = sub[sub[var] == 1]

    if len(g0) < 10 or len(g1) < 10:
        ax.set_visible(False)
        continue

    kmf0 = KaplanMeierFitter()
    kmf1 = KaplanMeierFitter()
    label = var_labels.get(var, var)
    kmf0.fit(g0["time"].astype(float), g0["event"].astype(float),
             label=f"{label}=없음")
    kmf1.fit(g1["time"].astype(float), g1["event"].astype(float),
             label=f"{label}=있음")

    kmf0.plot_survival_function(ax=ax, ci_show=False,
                                linewidth=2, color='steelblue')
    kmf1.plot_survival_function(ax=ax, ci_show=False,
                                linewidth=2, color='crimson', linestyle='--')

    lr = logrank_test(
        g0["time"].astype(float), g1["time"].astype(float),
        event_observed_A=g0["event"].astype(float),
        event_observed_B=g1["event"].astype(float)
    )
    p_str = '<0.001' if lr.p_value < 0.001 else f'{lr.p_value:.3f}'
    ax.set_title(f"{label}\n(Log-rank p={p_str})", fontsize=11)
    ax.set_xlabel("차수", fontsize=9)
    ax.set_ylabel("미발병 확률", fontsize=9)
    ax.set_ylim(0.6, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("1~3차 위험인자 기반 생존분석", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "km_early_model.png"), dpi=150)
plt.close()
print(f"  저장: km_early_model.png")


# ============================================================
# 최종 요약
# ============================================================
print(f"""
{'=' * 60}
최종 요약
{'=' * 60}

  분석 대상: {len(feat)}명
  양성(Y=1): {int(feat['Y'].sum())}명 ({feat['Y'].mean()*100:.1f}%)

  로지스틱 AUC (5-Fold): {auc_scores.mean():.3f} ± {auc_scores.std():.3f}
  로지스틱 유의변수 (p<0.05): {len(sig_df)}개
""")
for _, row in sig_df.iterrows():
    print(f"    {row['variable']:<22} OR={row['OR']:.3f}  p={row['p_value']:.4f}")

print(f"""
  저장 파일:
  {OUTPUT_DIR}/
  ├ logit_early_results.csv
  ├ cox_early_multivariate.csv
  └ km_early_model.png
""")