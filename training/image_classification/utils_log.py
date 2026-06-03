"""
Step 3-2: 변수 추가 후 로지스틱 재분석
=========================================
단변량 p<0.20 변수 전부 포함
breastfed 제외 (교란변수)
child_passive_smoke, sibling_allergy, sex, eat_breakfast_avg 추가

입력: pskc_final.csv
출력: model_results/logit_v2_results.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# [설정]
# ============================================================
DATA_DIR   = r"data/"   # ← 본인 경로
OUTPUT_DIR = os.path.join(DATA_DIR, "model_results(log)")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ============================================================
# [1] 데이터 로드
# ============================================================
print("=" * 60)
print("Step 1: 데이터 로드")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "pskc_final.csv"))
print(f"  전체: {len(df)}행")
print(f"  Y=1: {int(df['Y'].sum())}명, Y=0: {int((df['Y']==0).sum())}명")


# ============================================================
# [2] 변수 정의
#     v1: 로지스틱 유의 변수만 (breastfed 포함)
#     v2: 단변량 p<0.20 전부 + breastfed 제외
# ============================================================

# 모델 v1 (기존)
VARS_V1_CAT  = ['parent_AD', 'parent_AR', 'antibiotic',
                'mold_ever', 'area_type_w6']
VARS_V1_CONT = []

# 모델 v2 (확장 — breastfed 제외, 나머지 단변량 p<0.20 전부)
VARS_V2_CAT  = [
    'sex',                  # p=0.091
    'parent_AD',            # p=0.007
    'parent_AR',            # p=0.060
    'parent_asthma',        # p=0.212 (경계)
    'sibling_allergy',      # p=0.093
    'mold_ever',            # p=0.041
    'area_type_w6',         # p=0.058
    'antibiotic',           # p=0.135
    'child_passive_smoke',  # p=0.231 (XGBoost 2위라 추가)
]
VARS_V2_CONT = [
    'breastfed_months',     # p=0.000 (수유 기간은 교란 덜함)
    'eat_breakfast_avg',    # p=0.099
]

# 실제 존재하는 컬럼만
VARS_V2_CAT  = [c for c in VARS_V2_CAT  if c in df.columns]
VARS_V2_CONT = [c for c in VARS_V2_CONT if c in df.columns]

print(f"\n  v2 범주형: {VARS_V2_CAT}")
print(f"  v2 연속형: {VARS_V2_CONT}")


# ============================================================
# [3] 전처리 함수
# ============================================================
def preprocess(df, cat_vars, cont_vars):
    Y = df['Y'].values

    if cat_vars:
        cat_imp = SimpleImputer(strategy='most_frequent')
        X_cat = pd.DataFrame(
            cat_imp.fit_transform(df[cat_vars]),
            columns=cat_vars
        )
        for c in cat_vars:
            X_cat[c] = X_cat[c].astype(str)
        X_cat_dummy = pd.get_dummies(X_cat, drop_first=True)
    else:
        X_cat_dummy = pd.DataFrame(index=df.index)

    if cont_vars:
        cont_imp = SimpleImputer(strategy='median')
        X_cont_raw = pd.DataFrame(
            cont_imp.fit_transform(df[cont_vars]),
            columns=cont_vars
        )
        scaler = StandardScaler()
        X_cont = pd.DataFrame(
            scaler.fit_transform(X_cont_raw),
            columns=cont_vars
        )
    else:
        X_cont = pd.DataFrame(index=df.index)

    X = pd.concat([X_cat_dummy, X_cont], axis=1)
    return X, Y


# ============================================================
# [4] 5-Fold CV AUC 비교
# ============================================================
print(f"\n{'=' * 60}")
print("Step 2: 5-Fold CV AUC 비교")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
logit = LogisticRegression(max_iter=1000, class_weight='balanced',
                           random_state=42)

# v1 (기존 — breastfed 포함)
X_v1, Y = preprocess(df, VARS_V1_CAT, VARS_V1_CONT)
auc_v1 = cross_val_score(logit, X_v1, Y, cv=cv, scoring='roc_auc')

# v2 (확장 — breastfed 제외)
X_v2, Y = preprocess(df, VARS_V2_CAT, VARS_V2_CONT)
auc_v2 = cross_val_score(logit, X_v2, Y, cv=cv, scoring='roc_auc')

print(f"\n  [v1: 기존 유의변수 (breastfed 포함)]")
print(f"  AUC: {[round(x,3) for x in auc_v1]}")
print(f"  평균: {auc_v1.mean():.3f} ± {auc_v1.std():.3f}")

print(f"\n  [v2: 확장 변수 (breastfed 제외)]")
print(f"  AUC: {[round(x,3) for x in auc_v2]}")
print(f"  평균: {auc_v2.mean():.3f} ± {auc_v2.std():.3f}")

diff = auc_v2.mean() - auc_v1.mean()
print(f"\n  AUC 변화: {diff:+.3f}")
if diff > 0:
    print(f"  → v2가 {diff:.3f} 높음 ✅")
else:
    print(f"  → v1이 더 좋거나 유사 ⚠")


# ============================================================
# [5] v2 statsmodels 로지스틱 — OR + CI + p값
# ============================================================
print(f"\n{'=' * 60}")
print("Step 3: v2 로지스틱 회귀 상세 결과 (OR, CI, p)")
print("=" * 60)

X_v2_full, Y = preprocess(df, VARS_V2_CAT, VARS_V2_CONT)
X_v2_sm = sm.add_constant(X_v2_full.astype(float))

logit_sm = sm.Logit(Y, X_v2_sm)
result = logit_sm.fit(disp=0)

coef    = result.params
pvalues = result.pvalues
ci      = result.conf_int()

result_df = pd.DataFrame({
    'variable': coef.index,
    'coef':     coef.values,
    'OR':       np.exp(coef.values),
    'CI_lower': np.exp(ci[0].values),
    'CI_upper': np.exp(ci[1].values),
    'p_value':  pvalues.values
}).query("variable != 'const'").sort_values('p_value')

print(f"\n  전체 결과:")
print(result_df.round(3).to_string(index=False))

sig_df = result_df[result_df['p_value'] < 0.05]
print(f"\n  유의 변수 (p<0.05): {len(sig_df)}개")
print(sig_df.round(3).to_string(index=False))

# 저장
result_df.to_csv(
    os.path.join(OUTPUT_DIR, "logit_v2_results.csv"),
    index=False, encoding='utf-8-sig'
)


# ============================================================
# [6] ROC 커브 v1 vs v2 비교
# ============================================================
print(f"\n{'=' * 60}")
print("Step 4: ROC 커브 v1 vs v2")
print("=" * 60)

logit.fit(X_v1, Y)
prob_v1 = logit.predict_proba(X_v1)[:, 1]

logit.fit(X_v2, Y)
prob_v2 = logit.predict_proba(X_v2)[:, 1]

auc_v1_full = roc_auc_score(Y, prob_v1)
auc_v2_full = roc_auc_score(Y, prob_v2)

fpr1, tpr1, _ = roc_curve(Y, prob_v1)
fpr2, tpr2, _ = roc_curve(Y, prob_v2)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr1, tpr1,
        label=f'v1: 유의변수만 (AUC={auc_v1_full:.3f})',
        linewidth=2, linestyle='--')
ax.plot(fpr2, tpr2,
        label=f'v2: 확장변수 (AUC={auc_v2_full:.3f})',
        linewidth=2)
ax.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve: v1 vs v2', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_v1_vs_v2.png"), dpi=150)
plt.close()
print(f"  저장: roc_v1_vs_v2.png")


# ============================================================
# [7] Forest Plot (OR 시각화)
# ============================================================
print(f"\n{'=' * 60}")
print("Step 5: Forest Plot")
print("=" * 60)

plot_df = result_df[result_df['variable'] != 'const'].copy()
plot_df = plot_df.sort_values('OR')

colors = ['red' if p < 0.05 else 'steelblue'
          for p in plot_df['p_value']]

fig, ax = plt.subplots(figsize=(8, 8))
y_pos = range(len(plot_df))

ax.scatter(plot_df['OR'], y_pos, color=colors, zorder=3, s=60)
ax.hlines(
    y_pos,
    plot_df['CI_lower'],
    plot_df['CI_upper'],
    color=colors, linewidth=1.5
)
ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(plot_df['variable'], fontsize=9)
ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12)
ax.set_title('Forest Plot — v2 모델\n(빨강=p<0.05)', fontsize=13)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "forest_plot_v2.png"), dpi=150)
plt.close()
print(f"  저장: forest_plot_v2.png")


# ============================================================
# 최종 요약
# ============================================================
print(f"""
{'=' * 60}
최종 요약
{'=' * 60}

  v1 AUC (기존): {auc_v1.mean():.3f}
  v2 AUC (확장): {auc_v2.mean():.3f}
  변화:          {auc_v2.mean()-auc_v1.mean():+.3f}

  v2 유의변수 ({len(sig_df)}개):
""")
for _, row in sig_df.iterrows():
    print(f"    {row['variable']:<30} OR={row['OR']:.2f}  p={row['p_value']:.3f}")

print(f"""
  저장 파일:
  {OUTPUT_DIR}/
  ├ logit_v2_results.csv
  ├ roc_v1_vs_v2.png
  └ forest_plot_v2.png
""")