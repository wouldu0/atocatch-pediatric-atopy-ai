"""
Step 4-2: 생존분석 (처음부터 제대로)
======================================
절차:
  1. 생존분석용 데이터 구성 (전체 2,150명)
  2. 전체 변수 단변량 Cox (각 변수 하나씩)
  3. p < 0.20 후보 선별
  4. 다변량 Cox (후보 변수 동시 투입)
  5. KM 곡선 시각화

입력: pskc_ad_history.csv + pskc_final.csv (merged.csv도 있으면 활용)
출력: 단변량/다변량 Cox 결과, KM 곡선

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
OUTPUT_DIR = os.path.join(DATA_DIR, "model_results(KM)")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


# ============================================================
# [1] 생존분석용 데이터 구성
#     전체 2,150명 기준
#     time = 최초 발병 차수 (없으면 10)
#     event = 발병 여부
# ============================================================
print("=" * 60)
print("Step 1: 생존분석용 데이터 구성")
print("=" * 60)

history = pd.read_csv(os.path.join(DATA_DIR, "pskc_ad_history.csv"))
final   = pd.read_csv(os.path.join(DATA_DIR, "pskc_final.csv"))

# time, event 생성
history["time"]  = history["onset_wave"].fillna(10).astype(int)
history["event"] = history["ever_ad"].astype(int)

# pskc_final과 병합 (파생변수)
df = history[["N_ID", "time", "event"]].merge(
    final.drop(columns=["Y"], errors='ignore'),
    on="N_ID", how="left"
)

print(f"  전체: {len(df)}명")
print(f"  발병: {df['event'].sum()}명 ({df['event'].mean()*100:.1f}%)")
print(f"  중도절단: {(df['event']==0).sum()}명")
print(f"  독립변수 후보: {len(df.columns)-3}개")
print(f"  컬럼: {[c for c in df.columns if c not in ['N_ID','time','event']]}")

# 이진 파생변수 추가
df["antibiotic_high"] = (df["antibiotic"] >= 3).astype(float)
df["area_apt"]        = (df["area_type_w6"] == 2).astype(float)
df["area_rural"]      = (df["area_type_w6"] == 5).astype(float)


# ============================================================
# [2] 단변량 Cox (각 변수 하나씩)
# ============================================================
print(f"\n{'=' * 60}")
print("Step 2: 단변량 Cox 비례위험모델")
print("=" * 60)

# 분석할 변수 목록 (범주형은 더미 처리된 것 사용)
candidate_vars = [
    # 성별
    "sex",
    # 수유
    "breastfed", "breastfed_months",
    # 흡연
    "passive_smoke_ever", "child_passive_smoke",
    # 거주환경
    "rural_years", "area_apt", "area_rural", "area_type_w6",
    # 가족 알레르기
    "parent_AD", "parent_AR", "parent_asthma", "sibling_allergy",
    # 애완동물
    "pet_ever",
    # 항생제
    "antibiotic", "antibiotic_high",
    # 실내환경
    "mold_ever",
    # 실외활동
    "outdoor_avg",
    # 식습관
    "eat_snack_avg", "eat_breakfast_avg", "eat_delivery_avg",
    "eat_picky", "eat_regularity", "eat_amount", "eat_speed",
]

# 실제 존재하는 컬럼만
candidate_vars = [v for v in candidate_vars if v in df.columns]

univariate_results = []

print(f"\n  {'변수':<25} {'HR':>7} {'95% CI':>18} {'p값':>8} {'선택'}")
print("  " + "-" * 65)

for var in candidate_vars:
    sub = df[["time", "event", var]].dropna()
    if len(sub) < 50 or sub[var].nunique() < 2:
        continue

    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col="time", event_col="event")
        s = cph.summary

        hr  = s.loc[var, "exp(coef)"]
        ci_lo = s.loc[var, "exp(coef) lower 95%"]
        ci_hi = s.loc[var, "exp(coef) upper 95%"]
        p   = s.loc[var, "p"]

        selected = "✅" if p < 0.20 else "❌"
        ci_str = f"({ci_lo:.2f}~{ci_hi:.2f})"
        print(f"  {var:<25} {hr:>7.3f} {ci_str:>18} {p:>8.4f} {selected}")

        univariate_results.append({
            "variable": var,
            "HR": round(hr, 3),
            "CI_lower": round(ci_lo, 3),
            "CI_upper": round(ci_hi, 3),
            "p_value": round(p, 4),
            "selected": p < 0.20
        })
    except Exception as e:
        print(f"  {var:<25} 오류: {e}")

uni_df = pd.DataFrame(univariate_results).sort_values("p_value")
uni_df.to_csv(
    os.path.join(OUTPUT_DIR, "cox_univariate.csv"),
    index=False, encoding='utf-8-sig'
)

selected_vars = uni_df[uni_df["selected"]]["variable"].tolist()
print(f"\n  p<0.20 후보 변수 ({len(selected_vars)}개): {selected_vars}")


# ============================================================
# [3] 다변량 Cox (후보 변수 동시 투입)
# ============================================================
print(f"\n{'=' * 60}")
print("Step 3: 다변량 Cox 비례위험모델")
print("=" * 60)

if len(selected_vars) == 0:
    print("  ⚠ 후보 변수 없음. p<0.30으로 기준 완화")
    selected_vars = uni_df.head(5)["variable"].tolist()

# antibiotic과 antibiotic_high 중복 방지
if "antibiotic" in selected_vars and "antibiotic_high" in selected_vars:
    selected_vars.remove("antibiotic")
if "area_type_w6" in selected_vars and "area_apt" in selected_vars:
    selected_vars.remove("area_type_w6")
if "area_type_w6" in selected_vars and "area_rural" in selected_vars:
    selected_vars.remove("area_type_w6")

print(f"  다변량 투입 변수: {selected_vars}")

multi_data = df[["time", "event"] + selected_vars].dropna().astype(float)
print(f"  분석 대상: {len(multi_data)}명 (결측 제외)")
print(f"  발병: {int(multi_data['event'].sum())}명")

cph_multi = CoxPHFitter()
cph_multi.fit(multi_data, duration_col="time", event_col="event")

print(f"\n  Concordance Index: {cph_multi.concordance_index_:.3f}")
cph_multi.print_summary(decimals=3)

# 결과 정리
multi_result = cph_multi.summary[[
    "coef", "exp(coef)",
    "exp(coef) lower 95%", "exp(coef) upper 95%", "p"
]].copy()
multi_result.columns = ["coef", "HR", "CI_lower", "CI_upper", "p_value"]
multi_result["유의"] = multi_result["p_value"].apply(
    lambda x: "✅" if x < 0.05 else ("△" if x < 0.10 else "❌")
)
multi_result = multi_result.sort_values("p_value")

print(f"\n  [다변량 Cox 결과 요약]")
print(f"  {'변수':<25} {'HR':>7} {'95% CI':>18} {'p값':>8} {'유의'}")
print("  " + "-" * 65)
for var, row in multi_result.iterrows():
    ci_str = f"({row['CI_lower']:.2f}~{row['CI_upper']:.2f})"
    print(f"  {var:<25} {row['HR']:>7.3f} {ci_str:>18} {row['p_value']:>8.4f} {row['유의']}")

multi_result.to_csv(
    os.path.join(OUTPUT_DIR, "cox_multivariate.csv"),
    index=True, encoding='utf-8-sig'
)

sig_vars = multi_result[multi_result["p_value"] < 0.05]
print(f"\n  최종 유의 변수 (p<0.05): {len(sig_vars)}개")
for var, row in sig_vars.iterrows():
    direction = "위험 증가" if row["HR"] > 1 else "보호적"
    print(f"    {var}: HR={row['HR']:.3f} ({direction})")


# ============================================================
# [4] 전체 KM 곡선
# ============================================================
print(f"\n{'=' * 60}")
print("Step 4: KM 곡선 시각화")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 전체 KM
kmf = KaplanMeierFitter()
kmf.fit(df["time"], df["event"], label="전체 아동")
kmf.plot_survival_function(ax=axes[0], ci_show=True, linewidth=2)
axes[0].set_title("전체 KM 생존곡선\n소아 아토피 발병 패턴", fontsize=12)
axes[0].set_xlabel("차수 (만 나이 = 차수-1세)", fontsize=10)
axes[0].set_ylabel("아토피 미발병 확률", fontsize=10)
axes[0].set_xticks(range(1, 11))
axes[0].set_xticklabels([f"{t}차" for t in range(1, 11)], fontsize=8)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0.7, 1.01)

# 누적 발병률
cum_inc = [(t, (1-kmf.survival_function_at_times(t).values[0])*100)
           for t in range(3, 11)]
for t, ci in cum_inc:
    axes[0].annotate(
        f"{ci:.1f}%",
        xy=(t, 1-ci/100),
        fontsize=7, color='navy', ha='center', va='bottom'
    )

# 유의한 변수 중 첫 번째로 KM 비교
if len(sig_vars) > 0:
    best_var = sig_vars.index[0]
elif len(selected_vars) > 0:
    best_var = selected_vars[0]
else:
    best_var = "parent_AD"

if best_var in df.columns:
    sub_km = df[["time", "event", best_var]].dropna()
    
    # 값을 숫자로 강제 변환
    sub_km[best_var] = pd.to_numeric(sub_km[best_var], errors='coerce')
    sub_km = sub_km.dropna()
    
    # 중앙값 기준으로 이분 (연속형 변수 대응)
    median_val = sub_km[best_var].median()
    g0 = sub_km[sub_km[best_var] <= median_val]
    g1 = sub_km[sub_km[best_var] > median_val]
    
    if len(g0) < 10 or len(g1) < 10:
        axes[1].set_visible(False)
    else:
        kmf0 = KaplanMeierFitter()
        kmf1 = KaplanMeierFitter()
        kmf0.fit(
            g0["time"].astype(float),
            g0["event"].astype(float),
            label=f"{best_var} ≤{median_val:.1f}"
        )
        kmf1.fit(
            g1["time"].astype(float),
            g1["event"].astype(float),
            label=f"{best_var} >{median_val:.1f}"
        )

        kmf0.plot_survival_function(ax=axes[1], ci_show=False,
                                    linewidth=2, color='steelblue')
        kmf1.plot_survival_function(ax=axes[1], ci_show=False,
                                    linewidth=2, color='crimson', linestyle='--')

        lr = logrank_test(
            g0["time"].astype(float), g1["time"].astype(float),
            event_observed_A=g0["event"].astype(float),
            event_observed_B=g1["event"].astype(float)
        )

        axes[1].set_title(
            f"{best_var} 비교\n(Log-rank p={'<0.001' if lr.p_value < 0.001 else f'{lr.p_value:.3f}'})",
            fontsize=12
        )
        axes[1].set_xlabel("차수", fontsize=10)
        axes[1].set_ylabel("아토피 미발병 확률", fontsize=10)
        axes[1].set_xticks(range(1, 11))
        axes[1].set_xticklabels([f"{t}차" for t in range(1, 11)], fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0.7, 1.01)
        axes[1].legend(fontsize=9)

plt.suptitle("생존분석 결과", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "km_survival_v2.png"), dpi=150)
plt.close()
print(f"  저장: km_survival_v2.png")


# ============================================================
# [5] Cox HR Forest Plot (다변량)
# ============================================================
var_labels = {
    "sex":                 "성별 (여아)",
    "breastfed":           "모유수유 여부",
    "breastfed_months":    "모유수유 기간",
    "passive_smoke_ever":  "부모 흡연 누적",
    "child_passive_smoke": "아동 간접흡연",
    "rural_years":         "농촌 거주 차수",
    "area_apt":            "아파트 거주",
    "area_rural":          "농산어촌 거주",
    "parent_AD":           "부모 아토피",
    "parent_AR":           "부모 알레르기 비염",
    "parent_asthma":       "부모 천식",
    "sibling_allergy":     "형제 알레르기",
    "pet_ever":            "애완동물",
    "antibiotic":          "항생제 횟수",
    "antibiotic_high":     "항생제 3회 이상",
    "mold_ever":           "실내 곰팡이",
    "outdoor_avg":         "실외활동 평균",
    "eat_snack_avg":       "간식 빈도",
    "eat_breakfast_avg":   "아침식사 빈도",
    "eat_delivery_avg":    "외식·배달 빈도",
    "eat_picky":           "편식",
    "eat_regularity":      "식사 규칙성",
    "eat_amount":          "식사량",
    "eat_speed":           "식사 속도",
}

plot_df = multi_result.copy()
plot_df.index = [var_labels.get(v, v) for v in plot_df.index]
plot_df = plot_df.sort_values("HR")

colors = ["crimson" if p < 0.05 else ("orange" if p < 0.10 else "steelblue")
          for p in plot_df["p_value"]]

fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.5)))
y_pos = range(len(plot_df))

ax.scatter(plot_df["HR"], y_pos, color=colors, zorder=3, s=80)
ax.hlines(y_pos, plot_df["CI_lower"], plot_df["CI_upper"],
          color=colors, linewidth=2)
ax.axvline(x=1, color="black", linestyle="--", linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(plot_df.index, fontsize=10)
ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12)
ax.set_title(
    "다변량 Cox — Forest Plot\n(빨강=p<0.05, 주황=p<0.10, 파랑=p≥0.10)",
    fontsize=13
)
ax.grid(axis="x", alpha=0.3)

# p값 텍스트 추가
for i, (_, row) in enumerate(plot_df.iterrows()):
    ax.text(
        plot_df["CI_upper"].max() * 1.05, i,
        f"p={row['p_value']:.3f}",
        va="center", fontsize=8, color="gray"
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cox_forest_v2.png"), dpi=150)
plt.close()
print(f"  저장: cox_forest_v2.png")


# ============================================================
# 최종 요약
# ============================================================
print(f"""
{'=' * 60}
최종 요약
{'=' * 60}

  단변량 후보 변수 (p<0.20): {len(selected_vars)}개
    {selected_vars}

  다변량 Cox Concordance Index: {cph_multi.concordance_index_:.3f}

  최종 유의 변수 (p<0.05): {len(sig_vars)}개
""")
for var, row in sig_vars.iterrows():
    print(f"    {var:<25} HR={row['HR']:.3f}  p={row['p_value']:.4f}")

print(f"""
  저장 파일:
  {OUTPUT_DIR}/
  ├ cox_univariate.csv      단변량 결과 전체
  ├ cox_multivariate.csv    다변량 결과
  ├ km_survival_v2.png      KM 곡선
  └ cox_forest_v2.png       Forest Plot
""")