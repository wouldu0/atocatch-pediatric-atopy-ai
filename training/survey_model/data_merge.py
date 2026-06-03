"""
Step 1: 차수별 CSV 조인
========================
차수별 CSV에서 필요한 컬럼만 추출하고
N_ID 기준으로 조인하여 merged.csv 저장

실행 전 DATA_DIR 경로만 수정할 것
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# [설정]
# ============================================================
DATA_DIR = r"data/"  # ← 본인 경로로 수정
OUTPUT_FILE = os.path.join(DATA_DIR, "merged.csv")

FILE_MAP = {
    1: "w1_2008_data_220324.csv",
    2:  "w2_2009_data1_220324.csv",
    3:  "w3_2010_data_220603.csv",
    4:  "w4_2011_data_230411.csv",
    5:  "w5_2012_data_230411.csv",
    6:  "w6_2013_data(일반)_230411.csv",
    7:  "w7_2014_data_230411.csv",
    8:  "w8_2015_data_250211.csv",
    9:  "w9_2016_data_250211.csv",
    10: "w10_2017_data_250211.csv",
}

USE_COLS = {
    1: ["N_ID",
        "BMt08pnb001","BMt08pnb003", "BCh08hlt001",
        "DCh08fed001","DCh08fed002",
        "EMt08smk001", "FFt08smk001",
        "DHu08cmm002","DHu08ses004",
    ],
    2: [
        "N_ID",
        "DCh09dmg001",
        "DCh09fed001", "DCh09fed002",
        "EMt09smk001", "FFt09smk001",
        "DHu09cmm002", "EHu09cmm003",
    ],
    3: [
        "N_ID",
        "ECh10hlt035",
        "DCh10fed001", "DCh10fed002",
        "EMt10smk001", "FFt10smk001",
        "DHu10cmm002", "EHu10cmm003",
        "ECh10dlc003", "ECh10dlc010",
    ],
    4: [
        "N_ID",
        "DCh11hlt035",
        "DCh11fed001", "DCh11fed002",
        "EMt11smk001", "FFt11smk001",
        "DHu11cmm002", "EHu11cmm003",
        "DCh11dlc004", "DCh11dlc010",
    ],
    5: [
        "N_ID",
        "DCh12hlt035",
        "EHu12cmm002",
        "EMt12smk001", "FFt12smk001",
        "ECh12eat016", "ECh12eat017",
        "ECh12eat018", "ECh12eat019",
        "ECh12eat020", "ECh12eat022", "ECh12eat024",
        "DCh12dlc010",
    ],
    6: [
        "N_ID",
        "DCh13hlt035",
        "KFt13adx004", "KMt13adx004",
        "KFt13arx009", "KMt13arx009",
        "KFt13asx011", "KMt13asx011",
        "KHu13adx004", "KHu13arx009", "KHu13asx011",
        "KCh13pet001", "KCh13pet003",
        "KCh13drg004",
        "KCh13smk008", "KMt13smk010",
        "KMt13smk011", "KCh13smk012",
        "EMt13smk001", "FFt13smk001",
        "KCh13mld001", "KCh13mld006", "KCh13mld008",
        "DHu13cmm002", "EHu13cmm003",
        "ECh13eat016", "ECh13eat017", "ECh13eat018",
        "ECh13eat019", "ECh13eat020", "ECh13eat022", "ECh13eat024",
        "DCh13dlc010",
    ],
    7:  ["N_ID", "DCh14hlt031f"],
    8:  ["N_ID", "KCh15adx004"],
    9:  ["N_ID", "KCh16adx004"],
    10: ["N_ID", "DCh17hlt031k"],
}


# ============================================================
# 로드 및 조인
# ============================================================
print("=" * 60)
print("차수별 CSV 로드 및 조인")
print("=" * 60)

def load_wave(wave_num):
    filepath = os.path.join(DATA_DIR, FILE_MAP[wave_num])

    if not os.path.exists(filepath):
        print(f"  ⚠ {wave_num}차 파일 없음")
        return None

    try:
        df = pd.read_csv(filepath, encoding='cp949', low_memory=False)
    except:
        try:
            df = pd.read_csv(filepath, encoding='euc-kr', low_memory=False)
        except:
            df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)

    want = USE_COLS[wave_num]
    missing = [c for c in want if c not in df.columns]
    if missing:
        print(f"  ⚠ {wave_num}차 없는 컬럼: {missing}")
    available = [c for c in want if c in df.columns]

    result = df[available].copy()
    rename = {c: f"{c}_w{wave_num}" for c in available if c != "N_ID"}
    result = result.rename(columns=rename)

    print(f"  {wave_num}차: {len(result)}행, {len(available)-1}개 변수 추출")
    return result


waves = sorted(FILE_MAP.keys())
merged = load_wave(waves[0])

for w in waves[1:]:
    df_w = load_wave(w)
    if df_w is not None:
        merged = merged.merge(df_w, on="N_ID", how="outer")

print(f"\n조인 완료: {len(merged)}행, {len(merged.columns)}컬럼")
print(f"\n컬럼 목록:")
for c in merged.columns:
    print(f"  {c}")

merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {OUTPUT_FILE}")
print("다음: step2_features.py 실행")