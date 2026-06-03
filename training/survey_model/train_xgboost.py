"""
Step 3: XGBoost 모델링
=======================
입력: pskc_final.csv (step2 출력)
출력: 모델 성능, feature importance, 로지스틱 비교

실행 전 DATA_DIR 경로만 수정할 것
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================
# [설정]
# ============================================================
DATA_DIR   = r"data/"   # ← 본인 경로
OUTPUT_DIR = os.path.join(DATA_DIR, "model_results(xg)")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한글 폰트 설정 (Windows)
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
print(f"  전체: {len(df)}행, {len(df.columns)}컬럼")
print(f"  Y=1: {int(df['Y'].sum())}명, Y=0: {int((df['Y']==0).sum())}명")

# ============================================================
# [2] 변수 정의
# ============================================================
# 전체 독립변수 (N_ID, Y 제외)
ALL_FEATURES = [c for c in df.columns if c not in ['N_ID', 'Y']]

# 범주형 변수
CAT_VARS = [
    'sex', 'breastfed', 'passive_smoke_ever', 'child_passive_smoke',
    'parent_AD', 'parent_AR', 'parent_asthma', 'sibling_allergy',
    'pet_ever', 'mold_ever', 'area_type_w6', 'antibiotic',
    'eat_picky', 'eat_regularity', 'eat_amount', 'eat_speed'
]

# 연속형 변수
CONT_VARS = [
    'breastfed_months', 'rural_years', 'outdoor_avg',
    'eat_snack_avg', 'eat_breakfast_avg', 'eat_delivery_avg'
]

# 실제 존재하는 컬럼만
CAT_VARS  = [c for c in CAT_VARS  if c in df.columns]
CONT_VARS = [c for c in CONT_VARS if c in df.columns]

print(f"\n  범주형 변수: {len(CAT_VARS)}개")
print(f"  연속형 변수: {len(CONT_VARS)}개")

# ============================================================
# [3] 전처리
# ============================================================
print(f"\n{'=' * 60}")
print("Step 2: 전처리")
print("=" * 60)

Y = df['Y'].values

# 범주형: 최빈값 대체 + 더미 인코딩
if CAT_VARS:
    cat_imp = SimpleImputer(strategy='most_frequent')
    X_cat = pd.DataFrame(
        cat_imp.fit_transform(df[CAT_VARS]),
        columns=CAT_VARS
    )
    for c in CAT_VARS:
        X_cat[c] = X_cat[c].astype(str)
    X_cat_dummy = pd.get_dummies(X_cat, drop_first=True)
else:
    X_cat_dummy = pd.DataFrame(index=df.index)

# 연속형: 중앙값 대체 (스케일링은 로지스틱용)
if CONT_VARS:
    cont_imp = SimpleImputer(strategy='median')
    X_cont = pd.DataFrame(
        cont_imp.fit_transform(df[CONT_VARS]),
        columns=CONT_VARS
    )
else:
    X_cont = pd.DataFrame(index=df.index)

# XGBoost용: 스케일링 없이 그대로
X_xgb = pd.concat([X_cat_dummy, X_cont], axis=1)

# 로지스틱용: 연속형 표준화
scaler = StandardScaler()
X_cont_scaled = pd.DataFrame(
    scaler.fit_transform(X_cont),
    columns=CONT_VARS
) if len(CONT_VARS) > 0 else pd.DataFrame()
X_logit = pd.concat([X_cat_dummy, X_cont_scaled], axis=1)

print(f"  최종 변수 수: {X_xgb.shape[1]}개")
print(f"  샘플 수: {X_xgb.shape[0]}명")

# ============================================================
# [4] 모델 학습 (5-Fold CV)
# ============================================================
print(f"\n{'=' * 60}")
print("Step 3: 5-Fold Cross Validation")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(Y[Y==0]) / len(Y[Y==1]),  # 클래스 불균형 보정
    eval_metric='auc',
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)

xgb_auc = cross_val_score(
    xgb_model, X_xgb, Y,
    cv=cv, scoring='roc_auc'
)

print(f"\n  [XGBoost]")
print(f"  AUC per fold: {[round(x,3) for x in xgb_auc]}")
print(f"  평균 AUC: {xgb_auc.mean():.3f} ± {xgb_auc.std():.3f}")

# --- 로지스틱 회귀 (비교용) ---
logit_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

logit_auc = cross_val_score(
    logit_model, X_logit, Y,
    cv=cv, scoring='roc_auc'
)

print(f"\n  [로지스틱 회귀]")
print(f"  AUC per fold: {[round(x,3) for x in logit_auc]}")
print(f"  평균 AUC: {logit_auc.mean():.3f} ± {logit_auc.std():.3f}")

# ============================================================
# [5] 전체 데이터로 최종 학습 (Feature Importance용)
# ============================================================
print(f"\n{'=' * 60}")
print("Step 4: Feature Importance")
print("=" * 60)

xgb_model.fit(X_xgb, Y)
logit_model.fit(X_logit, Y)

# XGBoost Feature Importance
importance_df = pd.DataFrame({
    'feature': X_xgb.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  [XGBoost Feature Importance 상위 15개]")
print(importance_df.head(15).to_string(index=False))

# 저장
importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv"),
    index=False, encoding='utf-8-sig'
)

# ============================================================
# [6] ROC 커브 비교
# ============================================================
print(f"\n{'=' * 60}")
print("Step 5: ROC 커브 시각화")
print("=" * 60)

# 전체 데이터로 예측 확률
xgb_prob  = xgb_model.predict_proba(X_xgb)[:, 1]
logit_prob = logit_model.predict_proba(X_logit)[:, 1]

xgb_auc_full   = roc_auc_score(Y, xgb_prob)
logit_auc_full = roc_auc_score(Y, logit_prob)

fpr_xgb,   tpr_xgb,   _ = roc_curve(Y, xgb_prob)
fpr_logit, tpr_logit, _ = roc_curve(Y, logit_prob)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr_xgb,   tpr_xgb,
        label=f'XGBoost (AUC={xgb_auc_full:.3f})', linewidth=2)
ax.plot(fpr_logit, tpr_logit,
        label=f'Logistic (AUC={logit_auc_full:.3f})',
        linewidth=2, linestyle='--')
ax.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve: XGBoost vs Logistic Regression', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()
print(f"  ROC 커브 저장 완료")

# ============================================================
# [7] Feature Importance 시각화
# ============================================================
top15 = importance_df.head(15)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(
    top15['feature'][::-1],
    top15['importance'][::-1],
    color='steelblue', edgecolor='white'
)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('XGBoost Feature Importance (Top 15)', fontsize=13)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()
print(f"  Feature Importance 시각화 저장 완료")

# ============================================================
# [8] 최적 임계값 탐색
# ============================================================
print(f"\n{'=' * 60}")
print("Step 6: 최적 임계값 (Threshold) 탐색")
print("=" * 60)

thresholds = np.arange(0.1, 0.6, 0.05)
print(f"\n  {'임계값':>8} {'정확도':>8} {'민감도':>8} {'특이도':>8} {'F1':>8}")
print("  " + "-" * 48)

best_f1 = 0
best_threshold = 0.5

from sklearn.metrics import f1_score

for t in thresholds:
    pred = (xgb_prob >= t).astype(int)
    acc  = (pred == Y).mean()
    tn, fp, fn, tp = confusion_matrix(Y, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1   = f1_score(Y, pred, zero_division=0)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

    print(f"  {t:>8.2f} {acc:>8.3f} {sens:>8.3f} {spec:>8.3f} {f1:>8.3f}")

print(f"\n  최적 임계값: {best_threshold:.2f} (F1={best_f1:.3f})")
print(f"  → 서비스에서 이 임계값으로 위험군 분류")

# ============================================================
# [9] 최종 결과 요약
# ============================================================
print(f"\n{'=' * 60}")
print("최종 결과 요약")
print("=" * 60)

print(f"""
  [모델 성능 비교] (5-Fold CV)
  XGBoost  AUC: {xgb_auc.mean():.3f} ± {xgb_auc.std():.3f}
  Logistic AUC: {logit_auc.mean():.3f} ± {logit_auc.std():.3f}

  [XGBoost Feature Importance 상위 5개]
""")
for _, row in importance_df.head(5).iterrows():
    print(f"    {row['feature']:<30} {row['importance']:.4f}")

print(f"""
  [저장 파일]
  {OUTPUT_DIR}/
  ├ roc_curve.png
  ├ feature_importance.png
  └ xgb_feature_importance.csv
""")

# 결과 요약 CSV 저장
summary = pd.DataFrame({
    'model': ['XGBoost', 'Logistic'],
    'auc_mean': [xgb_auc.mean(), logit_auc.mean()],
    'auc_std':  [xgb_auc.std(),  logit_auc.std()],
    'best_threshold': [best_threshold, 0.5]
})
summary.to_csv(
    os.path.join(OUTPUT_DIR, "model_summary.csv"),
    index=False, encoding='utf-8-sig'
)
print("  model_summary.csv 저장 완료")