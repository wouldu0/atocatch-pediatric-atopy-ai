"""
Step 6: 단변량 검정 (1~3차 기반 변수)
=======================================
AD군(Y=1) vs 비AD군(Y=0) 변수별 비교
- 연속형: Mann-Whitney U 검정
- 범주형: 카이제곱 검정

입력: step5에서 생성된 feat DataFrame (또는 재생성)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
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
# [1] 데이터 로드 (step5와 동일하게 생성)
# ============================================================
print("=" * 60)
print("Step 1: 데이터 로드")
print("=" * 60)

merged  = pd.read_csv(os.path.join(DATA_DIR, "merged.csv"), low_memory=False)

# 무응답 처리
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
# [2] 파생변수 생성 (step5와 동일)
# ============================================================
feat = pd.DataFrame()
feat["N_ID"] = merged["N_ID"]

feat["gestation_weeks"]   = get_col("BMt08pnb001_w1")
feat["birth_weight"]      = get_col("BCh08hlt001_w1")
feat["c_section"]         = to_num(merged.get("BMt08pnb003_w1",
                            np.nan)).map({1: 0, 2: 1, 3: 1})
feat["sex"]               = to_num(merged.get("DCh09dmg001_w2",
                            np.nan)).map({1: 0, 2: 1})

def get_bf_months(row):
    for col in ["DCh11fed002_w4", "DCh10fed002_w3", "DCh09fed002_w2"]:
        if col in merged.columns:
            val = pd.to_numeric(row.get(col), errors='coerce')
            if pd.notna(val):
                return val
    return np.nan

feat["breastfed_months"] = merged.apply(get_bf_months, axis=1)
feat["breastfed_months"] = to_num(feat["breastfed_months"])
if "DCh08fed001_w1" in merged.columns:
    never_bf = to_num(merged["DCh08fed001_w1"]).isna()
    feat.loc[never_bf, "breastfed_months"] = 0

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

cmm_early = ["DHu08cmm002_w1", "DHu09cmm002_w2", "DHu10cmm002_w3"]
cmm_avail = [c for c in cmm_early if c in merged.columns]
if cmm_avail:
    cmm_df = merged[cmm_avail].apply(lambda col: to_num(col))
    feat["rural_years_early"] = cmm_df.apply(
        lambda row: sum(row == 2), axis=1
    )

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

# ============================================================
# [3] 종속변수 생성 (3차 미진단 → 4~10차 발병)
# ============================================================
for w in range(3, 11):
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

ad_w3 = merged["ad_w3"] if "ad_w3" in merged.columns else pd.Series(0, index=merged.index)
no_ad_w3 = merged[ad_w3 == 0].copy()
feat = feat[feat["N_ID"].isin(no_ad_w3["N_ID"])].copy()
feat = feat.merge(
    no_ad_w3[["N_ID"] + [f"ad_w{w}" for w in range(4,11)
              if f"ad_w{w}" in no_ad_w3.columns]],
    on="N_ID", how="left"
)

ad_4to10 = [f"ad_w{w}" for w in range(4,11) if f"ad_w{w}" in feat.columns]
feat["Y"] = feat[ad_4to10].apply(
    lambda row: 1 if any(row == 1) else 0, axis=1
)

print(f"  3차 미진단: {len(feat)}명")
print(f"  Y=1: {int(feat['Y'].sum())}명, Y=0: {int((feat['Y']==0).sum())}명")

# ============================================================
# [4] 단변량 검정
# ============================================================
print(f"\n{'=' * 60}")
print("Step 2: 단변량 검정")
print("=" * 60)

# 변수 분류
cat_vars = {
    "sex":                "성별 (여아=1)",
    "c_section":          "제왕절개 (예=1)",
    "passive_smoke_early":"부모 흡연 누적 (1~3차)",
    "parent_AD":          "부모 아토피",
    "parent_AR":          "부모 알레르기 비염",
    "parent_asthma":      "부모 천식",
    "mold_ever":          "실내 곰팡이",
    "pet_ever":           "애완동물",
    "prenatal_smoke":     "임신 중 흡연",
}

cont_vars = {
    "gestation_weeks":   "재태기간 (주)",
    "birth_weight":      "출생체중 (kg)",
    "breastfed_months":  "모유수유 기간 (개월)",
    "rural_years_early": "농촌 거주 차수 (1~3차)",
    "outdoor_w3":        "실외활동 시간 (3차, 시간)",
    "antibiotic":        "항생제 횟수",
}

ad0 = feat[feat["Y"] == 0]
ad1 = feat[feat["Y"] == 1]

results = []

# 범주형: 카이제곱
print(f"\n  [범주형 변수 — 카이제곱 검정]")
print(f"  {'변수':<25} {'AD없음(%)':>12} {'AD있음(%)':>12} {'p값':>8} {'유의'}")
print("  " + "-" * 65)

for var, label in cat_vars.items():
    if var not in feat.columns:
        continue
    v0 = ad0[var].dropna()
    v1 = ad1[var].dropna()
    if len(v0) < 10 or len(v1) < 10:
        continue

    ct = pd.crosstab(feat[var].dropna(), feat.loc[feat[var].notna(), "Y"])
    if ct.shape != (2, 2):
        continue

    chi2, p, _, _ = stats.chi2_contingency(ct)
    pct0 = v0.mean() * 100
    pct1 = v1.mean() * 100
    sig = "✅" if p < 0.05 else ("△" if p < 0.10 else "❌")

    print(f"  {label:<25} {pct0:>11.1f}% {pct1:>11.1f}% {p:>8.4f} {sig}")
    results.append({
        "변수": label, "유형": "범주형",
        "AD없음": f"{pct0:.1f}%", "AD있음": f"{pct1:.1f}%",
        "검정": "카이제곱", "p값": round(p, 4),
        "유의": "✅" if p < 0.05 else ("△" if p < 0.10 else "❌")
    })

# 연속형: Mann-Whitney U
print(f"\n  [연속형 변수 — Mann-Whitney U 검정]")
print(f"  {'변수':<25} {'AD없음(중앙값)':>14} {'AD있음(중앙값)':>14} {'p값':>8} {'유의'}")
print("  " + "-" * 70)

for var, label in cont_vars.items():
    if var not in feat.columns:
        continue
    v0 = ad0[var].dropna()
    v1 = ad1[var].dropna()
    if len(v0) < 10 or len(v1) < 10:
        continue

    stat, p = stats.mannwhitneyu(v0, v1, alternative='two-sided')
    med0 = v0.median()
    med1 = v1.median()
    sig = "✅" if p < 0.05 else ("△" if p < 0.10 else "❌")

    print(f"  {label:<25} {med0:>14.2f} {med1:>14.2f} {p:>8.4f} {sig}")
    results.append({
        "변수": label, "유형": "연속형",
        "AD없음": f"{med0:.2f}", "AD있음": f"{med1:.2f}",
        "검정": "Mann-Whitney", "p값": round(p, 4),
        "유의": "✅" if p < 0.05 else ("△" if p < 0.10 else "❌")
    })

# 저장
result_df = pd.DataFrame(results).sort_values("p값")
result_df.to_csv(
    os.path.join(OUTPUT_DIR, "univariate_early.csv"),
    index=False, encoding='utf-8-sig'
)

# ============================================================
# [5] 시각화
# ============================================================
print(f"\n{'=' * 60}")
print("Step 3: 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

plot_vars = [
    ("parent_AD",         "부모 아토피", "cat"),
    ("parent_AR",         "부모 비염",   "cat"),
    ("mold_ever",         "실내 곰팡이", "cat"),
    ("gestation_weeks",   "재태기간 (주)", "cont"),
    ("breastfed_months",  "수유 기간 (개월)", "cont"),
    ("antibiotic",        "항생제 횟수", "cont"),
]

for i, (var, label, vtype) in enumerate(plot_vars):
    ax = axes[i]
    if var not in feat.columns:
        ax.set_visible(False)
        continue

    if vtype == "cat":
        v0 = ad0[var].dropna()
        v1 = ad1[var].dropna()
        pct = [v0.mean()*100, v1.mean()*100]
        bars = ax.bar(["AD 없음", "AD 있음"], pct,
                      color=["steelblue", "crimson"], width=0.5)
        ax.set_ylabel("비율 (%)", fontsize=10)
        for bar, val in zip(bars, pct):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha='center', fontsize=10)

        ct = pd.crosstab(feat[var].dropna(),
                         feat.loc[feat[var].notna(), "Y"])
        if ct.shape == (2, 2):
            _, p, _, _ = stats.chi2_contingency(ct)
            ax.set_title(f"{label}\np={'<0.001' if p<0.001 else f'{p:.3f}'}",
                         fontsize=11)
    else:
        v0 = ad0[var].dropna()
        v1 = ad1[var].dropna()
        ax.boxplot([v0, v1], labels=["AD 없음", "AD 있음"],
                   patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.6),
                   medianprops=dict(color='red', linewidth=2))
        ax.set_ylabel(label, fontsize=10)
        _, p = stats.mannwhitneyu(v0, v1, alternative='two-sided')
        ax.set_title(
            f"{label}\n중앙값: {v0.median():.1f} vs {v1.median():.1f}  "
            f"p={'<0.001' if p<0.001 else f'{p:.3f}'}",
            fontsize=10
        )
    ax.grid(axis='y', alpha=0.3)

plt.suptitle("AD군 vs 비AD군 단변량 비교 (1~3차 기반)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "univariate_early.png"), dpi=150)
plt.close()
print(f"  저장: univariate_early.png")

# ============================================================
# 최종 요약
# ============================================================
print(f"\n{'=' * 60}")
print("최종 요약")
print("=" * 60)

sig_results = result_df[result_df["유의"] == "✅"]
print(f"\n  유의 변수 (p<0.05): {len(sig_results)}개")
print(f"  {'변수':<25} {'AD없음':>10} {'AD있음':>10} {'p값':>8}")
print("  " + "-" * 55)
for _, row in sig_results.iterrows():
    print(f"  {row['변수']:<25} {row['AD없음']:>10} {row['AD있음']:>10} {row['p값']:>8}")

print(f"""
  저장 파일:
  {OUTPUT_DIR}/
  ├ univariate_early.csv   전체 단변량 결과
  └ univariate_early.png   시각화
""")