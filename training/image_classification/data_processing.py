"""
PSKC 아토피 피부염 — 차수별 진단 이력 구축 & 양성N 확인
==========================================================
실행 전 DATA_DIR과 FILE_MAP만 수정하세요.
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# [설정] 여기만 수정
# ============================================================

DATA_DIR = r"data/"  # ← 본인 경로

# 차수별 CSV 파일명 (본인 파일명으로 수정)
FILE_MAP = {
    3:  "w3_2010_data_220603.csv",
    4:  "w4_2011_data_230411.csv",
    5:  "w5_2012_data_230411.csv",
    6:  "w6_2013_data(일반)_230411.csv",
    7:  "w7_2014_data_230411.csv",
    8:  "w8_2015_data_250211.csv",
    9:  "w9_2016_data_250211.csv",
    10: "w10_2017_data_250211.csv",
}

# 차수별 AD 진단 변수명 & "진단=예" 판정 로직
WAVE_CONFIG = {
    3:  {"var": "ECh10hlt035",   "yes_value": '6'},      # 문자열 '6'
    4:  {"var": "DCh11hlt035",   "yes_value": '6'},
    5:  {"var": "DCh12hlt035",   "yes_value": '6'},
    6:  {"var": "DCh13hlt035",   "yes_value": '6'},
    7:  {"var": "DCh14hlt031f",  "yes_value": '6'},
    8:  {"var": "KCh15adx004",   "yes_value": '1'},       # 이것도 문자열일 수 있음
    9:  {"var": "KCh16adx004",   "yes_value": '1'},
    10: {"var": "DCh17hlt031k",  "yes_value": "notnull"},
}


# ============================================================
# Step 1-2: 각 차수에서 [N_ID, AD진단여부] 추출
# ============================================================

print("=" * 60)
print("Step 1-2: 차수별 AD 진단 변수 추출")
print("=" * 60)

wave_dfs = {}

for w, fname in FILE_MAP.items():
    filepath = os.path.join(DATA_DIR, fname)
    
    if not os.path.exists(filepath):
        print(f"  ⚠ {w}차 파일 없음: {filepath}")
        continue
    
    try:
        df = pd.read_csv(filepath, encoding='cp949')
    except:
        try:
            df = pd.read_csv(filepath, encoding='euc-kr')
        except:
            df = pd.read_csv(filepath, encoding='utf-8')
    
    cfg = WAVE_CONFIG[w]
    var_name = cfg["var"]
    
    if "N_ID" not in df.columns:
        print(f"  ⚠ {w}차: N_ID 없음! 첫 컬럼: {df.columns[0]}")
        continue
    
    if var_name not in df.columns:
        print(f"  ⚠ {w}차: {var_name} 없음!")
        similar = [c for c in df.columns if 'hlt03' in c or 'adx' in c]
        print(f"     비슷한 컬럼: {similar[:10]}")
        continue
    
    # AD 진단 여부를 0/1로 변환
    if cfg["yes_value"] == "notnull":
        # 빈 문자열, 공백, NaN 모두 "진단 아님"으로 처리
        ad_binary = df[var_name].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() == '' or str(x).strip() == '0' else 1
        )
    else:
        # 문자열 비교로 통일
        ad_binary = (df[var_name].astype(str).str.strip() == str(cfg["yes_value"])).astype(int)
    
    result = pd.DataFrame({
        "N_ID": df["N_ID"],
        f"ad_w{w}": ad_binary
        
    })
    
    wave_dfs[w] = result
    n_yes = int(ad_binary.sum())
    n_total = len(df)
    print(f"  {w}차 (만{w-1}세): N={n_total}, AD진단={n_yes}명 ({n_yes/n_total*100:.1f}%)")
\

# ============================================================
# Step 3: N_ID 기준 wide-format 조인
# ============================================================

print(f"\n{'=' * 60}")
print("Step 3: 아이별 차수별 진단 이력 테이블 생성")
print("=" * 60)

waves_sorted = sorted(wave_dfs.keys())
merged = wave_dfs[waves_sorted[0]].copy()

for w in waves_sorted[1:]:
    merged = merged.merge(wave_dfs[w], on="N_ID", how="outer")

print(f"  전체 아이 수: {len(merged)}")

print(f"\n  차수별 진단 수:")
for w in waves_sorted:
    col = f"ad_w{w}"
    if col in merged.columns:
        print(f"    {w}차 (만{w-1}세): {int(merged[col].sum())}명")


# ============================================================
# Step 4: 최초 보고 차수(onset_wave) 산출
# ============================================================

print(f"\n{'=' * 60}")
print("Step 4: 최초 보고 차수(onset_wave) 산출")
print("=" * 60)

def find_onset(row):
    for w in waves_sorted:
        col = f"ad_w{w}"
        if col in row.index and row[col] == 1:
            return w
    return np.nan

merged["onset_wave"] = merged.apply(find_onset, axis=1)
merged["ever_ad"] = merged["onset_wave"].notna()

print(f"  AD 진단 경험: {int(merged['ever_ad'].sum())}명 ({merged['ever_ad'].mean()*100:.1f}%)")
print(f"  진단 없음: {int((~merged['ever_ad']).sum())}명")

print(f"\n  최초 보고 차수별 분포:")
for w, n in merged["onset_wave"].value_counts().sort_index().items():
    print(f"    {int(w)}차 (만{int(w)-1}세): {n}명")


# ============================================================
# Step 5: 기준 시점별 양성 N 계산
# ============================================================

print(f"\n{'=' * 60}")
print("Step 5: 기준 시점별 양성 N 확인")
print("=" * 60)

print(f"\n  {'기준':<6} {'나이':<7} {'대상N':>6} {'양성':>6} {'음성':>6} {'비율':>7} {'판정'}")
print("  " + "-" * 52)

for cutoff in waves_sorted[:-1]:
    eligible = merged[
        (merged["onset_wave"].isna()) |
        (merged["onset_wave"] > cutoff)
    ]
    
    n_pos = int((eligible["onset_wave"] > cutoff).sum())
    n_neg = int(eligible["onset_wave"].isna().sum())
    n_total = len(eligible)
    ratio = n_pos / n_total * 100 if n_total > 0 else 0
    
    if n_pos >= 80:    v = "✅ 충분"
    elif n_pos >= 50:  v = "⚠️ 가능"
    elif n_pos >= 30:  v = "⚠️ 빠듯"
    else:              v = "❌ 부족"
    
    print(f"  {cutoff}차    만{cutoff-1}세  {n_total:>6} {n_pos:>6} {n_neg:>6} {ratio:>6.1f}% {v}")


# ============================================================
# Step 6: 들쭉날쭉 응답 확인
# ============================================================

print(f"\n{'=' * 60}")
print("Step 6: 응답 일관성 확인")
print("=" * 60)

diagnosed = merged[merged["ever_ad"]].copy()
flip_count = 0

for _, row in diagnosed.iterrows():
    found_yes = False
    for w in waves_sorted:
        col = f"ad_w{w}"
        if col not in row.index:
            continue
        if row[col] == 1:
            found_yes = True
        elif found_yes and row[col] == 0:
            flip_count += 1
            break

print(f"  AD 경험자: {len(diagnosed)}명")
print(f"  예→아니오 전환: {flip_count}명 ({flip_count/max(len(diagnosed),1)*100:.1f}%)")
print(f"  → 최초 '예' 기준 onset_wave 확정, 이후 '아니오'는 무시")


# ============================================================
# 저장
# ============================================================

output_path = os.path.join(DATA_DIR, "pskc_ad_history.csv")
merged.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n{'=' * 60}")
print(f"저장 완료: {output_path}")
print(f"""
다음 단계:
  1. Step 5 결과를 나한테 공유
  2. 양성 N 충분한 기준 시점 확정
  3. 독립변수 조인 코드 작성
""")