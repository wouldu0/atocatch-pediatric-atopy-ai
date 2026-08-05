"""
Step 4: 생존분석 (Survival Analysis)
======================================
입력: pskc_ad_history.csv + pskc_final.csv
출력: 생존곡선, Cox HR 결과

사전 설치: pip install lifelines
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# ============================================================
# [설정]
# ============================================================
DATA_DIR   = r"data/"   # ← 본인 경로
OUTPUT_DIR = os.path.join(DATA_DIR, "model_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


# ============================================================
# [1] 데이터 준비 — 생존분석용 time, event 변수 생성
# ============================================================
print("=" * 60)
print("Step 1: 생존분석 데이터 준비")
print("=" * 60)

# AD 진단 이력
history = pd.read_csv(os.path.join(DATA_DIR, "pskc_ad_history.csv"))

# 파생변수 (위험인자)
final = pd.read_csv(os.path.join(DATA_DIR, "pskc_final.csv"))

# time: 최초 발병 차수 (없으면 마지막 관찰 차수 10)
# event: 발병 여부 (1=발병, 0=중도절단)
history["time"]  = history["onset_wave"].fillna(10).astype(int)
history["event"] = history["ever_ad"].astype(int)

print(f"  전체 아동: {len(history)}명")
print(f"  발병(event=1): {history['event'].sum()}명")
print(f"  중도절단(event=0): {(history['event']==0).sum()}명")

print(f"\n  발병 차수 분포:")
onset_dist = history[history['event']==1]['time'].value_counts().sort_index()
for t, n in onset_dist.items():
    print(f"    {t}차 (만{t-1}세): {n}명")


# ============================================================
# [2] Kaplan-Meier 전체 생존곡선
# ============================================================
print(f"\n{'=' * 60}")
print("Step 2: Kaplan-Meier 전체 생존곡선")
print("=" * 60)

kmf = KaplanMeierFitter()
kmf.fit(
    durations=history["time"],
    event_observed=history["event"],
    label="전체 아동"
)

# 각 차수별 누적 발병률
print(f"\n  차수별 누적 발병률 (1 - 생존율):")
for t in range(3, 11):
    surv = kmf.survival_function_at_times(t).values[0]
    cum_incidence = (1 - surv) * 100
    print(f"    {t}차 (만{t-1}세): {cum_incidence:.1f}%")

# 시각화
fig, ax = plt.subplots(figsize=(9, 5))
kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2)
ax.set_xlabel("차수 (만 나이 = 차수 - 1세)", fontsize=12)
ax.set_ylabel("아토피 미발병 확률", fontsize=12)
ax.set_title("Kaplan-Meier 생존곡선\n(소아 아토피 피부염 발병 패턴)", fontsize=13)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"{t}차\n(만{t-1}세)" for t in range(1, 11)], fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.7, 1.01)

# 중앙 발병 시점 표시
median = kmf.median_survival_time_
if not np.isinf(median):
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label=f'중앙 발병시점: {median}차')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "km_overall.png"), dpi=150)
plt.close()
print(f"  저장: km_overall.png")


# ============================================================
# [3] 위험인자별 Kaplan-Meier 비교 + Log-rank test
# ============================================================
print(f"\n{'=' * 60}")
print("Step 3: 위험인자별 KM 곡선 비교")
print("=" * 60)

# pskc_final과 history 병합
merged = history.merge(
    final[['N_ID', 'parent_AD', 'antibiotic', 'child_passive_smoke',
           'mold_ever', 'area_type_w6', 'parent_AR']],
    on='N_ID', how='left'
)

# 비교할 변수들
compare_vars = [
    ("parent_AD",          {0: "부모 아토피 없음", 1: "부모 아토피 있음"}),
    ("antibiotic_high",    {0: "항생제 0-2회", 1: "항생제 3회 이상"}),
    ("child_passive_smoke",{0: "간접흡연 없음", 1: "간접흡연 있음"}),
    ("mold_ever",          {0: "곰팡이 없음", 1: "곰팡이 있음"}),
    ("area_apt",           {0: "비아파트 거주", 1: "아파트 거주"}),
]

# 파생 이진변수 생성
merged["antibiotic_high"] = (merged["antibiotic"] >= 3).astype(float)
merged["area_apt"] = (merged["area_type_w6"] == 2).astype(float)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

logrank_results = []

for idx, (var, labels) in enumerate(compare_vars):
    ax = axes[idx]
    col = merged[var].dropna()
    valid = merged.dropna(subset=[var])

    g0 = valid[valid[var] == 0]
    g1 = valid[valid[var] == 1]

    if len(g0) < 10 or len(g1) < 10:
        ax.set_visible(False)
        continue

    kmf0 = KaplanMeierFitter()
    kmf1 = KaplanMeierFitter()

    kmf0.fit(g0["time"], g0["event"], label=labels[0])
    kmf1.fit(g1["time"], g1["event"], label=labels[1])

    kmf0.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='steelblue')
    kmf1.plot_survival_function(ax=ax, ci_show=False, linewidth=2,
                                color='crimson', linestyle='--')

    # Log-rank test
    results = logrank_test(
        g0["time"], g1["time"],
        event_observed_A=g0["event"],
        event_observed_B=g1["event"]
    )
    p = results.p_value

    ax.set_title(f"{labels[1].split(' ')[0]}\np={'<0.001' if p < 0.001 else f'{p:.3f}'}",
                 fontsize=11)
    ax.set_xlabel("차수", fontsize=9)
    ax.set_ylabel("미발병 확률", fontsize=9)
    ax.set_ylim(0.6, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    logrank_results.append({
        "변수": var,
        "그룹0_N": len(g0),
        "그룹1_N": len(g1),
        "그룹0_발병": g0["event"].sum(),
        "그룹1_발병": g1["event"].sum(),
        "p_value": round(p, 4),
        "유의": "✅" if p < 0.05 else "❌"
    })

    print(f"  {var}: p={p:.4f} {'✅ 유의' if p < 0.05 else '❌ 비유의'}")
    print(f"    {labels[0]}: {len(g0)}명 중 {int(g0['event'].sum())}명 발병")
    print(f"    {labels[1]}: {len(g1)}명 중 {int(g1['event'].sum())}명 발병")

# 마지막 subplot 정리
axes[-1].set_visible(False)

plt.suptitle("위험인자별 아토피 발병 생존곡선 비교\n(Log-rank test)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "km_by_risk_factors.png"), dpi=150)
plt.close()
print(f"\n  저장: km_by_risk_factors.png")

# Log-rank 결과 저장
lr_df = pd.DataFrame(logrank_results)
lr_df.to_csv(os.path.join(OUTPUT_DIR, "logrank_results.csv"),
             index=False, encoding='utf-8-sig')


# ============================================================
# [4] Cox 비례위험모델
# ============================================================
print(f"\n{'=' * 60}")
print("Step 4: Cox 비례위험모델")
print("=" * 60)

cox_vars = [
    'parent_AD', 'parent_AR', 'antibiotic_high',
    'child_passive_smoke', 'mold_ever', 'area_apt'
]

cox_data = merged[['time', 'event'] + cox_vars].dropna()
print(f"  분석 대상: {len(cox_data)}명 (결측 제외)")

# 더미 인코딩 (이미 이진형이라 그대로)
cox_data = cox_data.astype(float)

cph = CoxPHFitter()
cph.fit(cox_data, duration_col='time', event_col='event')

print(f"\n  [Cox 모델 결과]")
print(f"  Concordance Index: {cph.concordance_index_:.3f}")
print()
cph.print_summary(decimals=3)

# HR 정리
cox_result = cph.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                           'exp(coef) upper 95%', 'p']].copy()
cox_result.columns = ['coef', 'HR', 'CI_lower', 'CI_upper', 'p_value']
cox_result['유의'] = cox_result['p_value'].apply(lambda x: '✅' if x < 0.05 else '❌')
cox_result = cox_result.sort_values('p_value')

print(f"\n  [HR 요약 (발병 위험비)]")
print(f"  {'변수':<25} {'HR':>6} {'95% CI':>18} {'p값':>8} {'유의'}")
print("  " + "-" * 65)
for var, row in cox_result.iterrows():
    ci = f"({row['CI_lower']:.2f}~{row['CI_upper']:.2f})"
    print(f"  {var:<25} {row['HR']:>6.3f} {ci:>18} {row['p_value']:>8.4f} {row['유의']}")

cox_result.to_csv(os.path.join(OUTPUT_DIR, "cox_results.csv"),
                  index=True, encoding='utf-8-sig')


# ============================================================
# [5] Cox HR Forest Plot
# ============================================================
print(f"\n{'=' * 60}")
print("Step 5: Forest Plot (HR)")
print("=" * 60)

var_labels = {
    'parent_AD':           '부모 아토피 진단',
    'parent_AR':           '부모 알레르기 비염',
    'antibiotic_high':     '항생제 3회 이상',
    'child_passive_smoke': '아동 간접흡연',
    'mold_ever':           '실내 곰팡이 노출',
    'area_apt':            '아파트 거주',
}

plot_data = cox_result.copy()
plot_data.index = [var_labels.get(v, v) for v in plot_data.index]
plot_data = plot_data.sort_values('HR')

colors = ['crimson' if p < 0.05 else 'steelblue'
          for p in plot_data['p_value']]

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = range(len(plot_data))

ax.scatter(plot_data['HR'], y_pos, color=colors, zorder=3, s=80)
ax.hlines(y_pos, plot_data['CI_lower'], plot_data['CI_upper'],
          color=colors, linewidth=2)
ax.axvline(x=1, color='black', linestyle='--', linewidth=1)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(plot_data.index, fontsize=10)
ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
ax.set_title('Cox 비례위험모델 — Forest Plot\n(빨강=p<0.05, 파랑=p≥0.05)', fontsize=13)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cox_forest_plot.png"), dpi=150)
plt.close()
print(f"  저장: cox_forest_plot.png")


# ============================================================
# [6] 로지스틱 OR vs Cox HR 비교
# ============================================================
print(f"\n{'=' * 60}")
print("Step 6: 로지스틱 OR vs Cox HR 비교")
print("=" * 60)

# 기존 로지스틱 결과 (v2)
logit_file = os.path.join(OUTPUT_DIR, "logit_v2_results.csv")
if os.path.exists(logit_file):
    logit = pd.read_csv(logit_file)

    var_map = {
        'parent_AD_1.0':           'parent_AD',
        'parent_AR_1.0':           'parent_AR',
        'antibiotic_3.0':          'antibiotic_high',
        'child_passive_smoke_1.0': 'child_passive_smoke',
        'mold_ever_1.0':           'mold_ever',
        'area_type_w6_2.0':        'area_apt',
    }

    print(f"\n  {'변수':<25} {'로지스틱 OR':>12} {'Cox HR':>10} {'해석'}")
    print("  " + "-" * 65)

    for logit_var, cox_var in var_map.items():
        logit_row = logit[logit['variable'] == logit_var]
        if len(logit_row) == 0:
            continue
        or_val = logit_row['OR'].values[0]
        or_p   = logit_row['p_value'].values[0]

        if cox_var in cox_result.index:
            hr_val = cox_result.loc[cox_var, 'HR']
            hr_p   = cox_result.loc[cox_var, 'p_value']
        else:
            hr_val = np.nan
            hr_p   = np.nan

        direction = "일치 ✅" if (or_val > 1) == (hr_val > 1) else "방향 다름 ⚠"
        print(f"  {logit_var:<25} OR={or_val:.2f}(p={or_p:.3f})  "
              f"HR={hr_val:.2f}(p={hr_p:.3f})  {direction}")
else:
    print("  logit_v2_results.csv 없음 — 로지스틱 결과 비교 스킵")


# ============================================================
# 최종 요약
# ============================================================
print(f"""
{'=' * 60}
최종 요약
{'=' * 60}

  Cox 모델 Concordance Index: {cph.concordance_index_:.3f}
  (AUC와 유사한 지표, 0.5=랜덤, 1.0=완벽)

  저장 파일:
  {OUTPUT_DIR}/
  ├ km_overall.png          전체 KM 생존곡선
  ├ km_by_risk_factors.png  위험인자별 KM 비교
  ├ cox_forest_plot.png     Cox HR Forest Plot
  ├ cox_results.csv         Cox 결과
  └ logrank_results.csv     Log-rank test 결과
""")