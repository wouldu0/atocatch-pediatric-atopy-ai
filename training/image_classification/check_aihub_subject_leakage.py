"""
AI Hub 정면/측면 이미지의 subject-level 중복(leakage) 점검
==============================================================
배경: data_matching.py가 AI Hub 라벨을 정면(TL/VL_아토피_정면)과 측면
(TL/VL_아토피_측면) 폴더에서 각각 수집해 하나의 학습 샘플 목록으로 합치고,
data_split.py는 그 목록을 이미지 단위로 완전 랜덤 7:2:1 분할한다(subject 단위
그룹 분할 아님). 같은 아이를 정면·측면 두 장 찍은 것이라면, 한 아이의 두 이미지가
train과 test에 각각 걸칠 수 있다 — 이게 실제로 발생하는지 원본 라벨 JSON을
직접 열어봐야 확인 가능한데, 로컬에는 AI Hub 원본(10GB, 라이선스상 재배포 불가)이
없어서 여태 확인을 못 했다. 이 스크립트는 그 확인용이다.

⚠️ 실행 전 필수: STEP 0 스키마 확인 결과를 먼저 눈으로 확인할 것.
    라벨 JSON 구조를 100% 확신할 수 없어서(로컬에 샘플이 없었음),
    STEP 0가 실제 JSON 안의 모든 키를 그대로 출력한다. 그 출력을 보고
    "정면·측면 두 이미지가 같은 사람이라는 걸 알려주는 필드"가
    `identifier`인지, 아니면 다른 필드(예: patient_id, subject_id,
    person_id 등)인지 먼저 사람이 확인해야 한다 — 필요하면 아래
    CANDIDATE_ID_FIELDS에 그 필드명을 추가해서 재실행할 것.

실행:
    python check_aihub_subject_leakage.py
    (AIHUB_ROOT 경로만 본인 환경에 맞게 수정)
"""
import os
import re
import json
import random
from collections import defaultdict, Counter

# ============================================================
# [설정] 본인 경로로 수정
# ============================================================
AIHUB_ROOT = r"E:\atopic\AIHub"  # data_split.py / data_matching.py와 동일 루트

LABEL_ROOTS = {
    "train": os.path.join(AIHUB_ROOT, "Training", "02.라벨링데이터"),
    "val": os.path.join(AIHUB_ROOT, "Validation", "02.라벨링데이터"),
}
LABEL_PREFIX = {"train": "TL", "val": "VL"}
VIEWS = ["정면", "측면"]

# data_split.py의 이진분류 split과 동일 시드/설정 (leakage 재현용)
SEED = 42
TEST_SIZE_1 = 0.3   # train_test_split 1차 (7 vs 2+1)
TEST_SIZE_2 = 1 / 3  # 2차 분할 (val vs test)


# ============================================================
# STEP 0: 라벨 JSON 스키마 직접 확인 (사람이 눈으로 봐야 함)
# ============================================================
def inspect_schema(n_samples=3):
    print("=" * 70)
    print("STEP 0: 라벨 JSON 스키마 확인 (정면/측면 각각 몇 개씩 원본 그대로 출력)")
    print("=" * 70)
    for split in ["train", "val"]:
        for view in VIEWS:
            folder = os.path.join(LABEL_ROOTS[split], f"{LABEL_PREFIX[split]}_아토피_{view}")
            if not os.path.isdir(folder):
                print(f"[없음] {folder}")
                continue
            files = [f for f in os.listdir(folder) if f.endswith(".json")][:n_samples]
            for fname in files:
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"\n--- {split}/{view}/{fname} ---")
                print("최상위 키:", list(data.keys()))
                if "annotations" in data and len(data["annotations"]) > 0:
                    print("annotations[0] 키:", list(data["annotations"][0].keys()))
                print(json.dumps(data, ensure_ascii=False, indent=2)[:1500])
    print("\n" + "=" * 70)
    print("⚠️ 위 출력에서 '정면 파일과 측면 파일이 같은 아이라는 걸 나타내는 필드'가")
    print("   무엇인지 확인하고, 필요하면 CANDIDATE_ID_FIELDS를 수정해서 다시 실행할 것.")
    print("=" * 70)


# ============================================================
# STEP 1: 전체 라벨 수집 (identifier + 후보 subject id 필드들)
# ============================================================
CANDIDATE_ID_FIELDS = ["identifier", "patient_id", "subject_id", "person_id", "case_id"]


def collect_all_labels():
    records = []
    for split in ["train", "val"]:
        for view in VIEWS:
            folder = os.path.join(LABEL_ROOTS[split], f"{LABEL_PREFIX[split]}_아토피_{view}")
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                ann = data.get("annotations", [{}])[0]
                rec = {"split": split, "view": view, "filename": fname}
                for field in CANDIDATE_ID_FIELDS:
                    rec[field] = ann.get(field) or data.get(field)
                records.append(rec)
    return records


# ============================================================
# STEP 2: identifier에서 "정면/측면 공통 접두어(추정 subject id)" 추출 시도
# 예) identifier가 "P0001_front"/"P0001_side" 같은 패턴이면 "_front"/"_side" 등
# 접미어를 떼면 같은 사람끼리 묶일 수 있음. 정확한 규칙은 STEP 0 확인 후 조정할 것.
# ============================================================
def guess_subject_id(identifier: str) -> str:
    if not identifier:
        return identifier
    # 흔한 패턴 후보: 끝의 숫자/구분자, "_정면"/"_측면", "-1"/"-2" 등 제거
    s = re.sub(r'(_정면|_측면|_front|_side|_F|_S)$', '', str(identifier), flags=re.IGNORECASE)
    s = re.sub(r'[-_]?(\d{1,2})$', '', s)  # 끝자리 짧은 순번 제거 (과도할 수 있으니 결과 검증 필요)
    return s


def analyze_subject_overlap(records):
    print("\n" + "=" * 70)
    print("STEP 2: identifier 기준 정면/측면 subject 중복 분석")
    print("=" * 70)

    by_identifier = Counter(r["identifier"] for r in records if r.get("identifier"))
    dup_identifier = {k: v for k, v in by_identifier.items() if v > 1}
    print(f"identifier 자체가 정면/측면 등에서 그대로 중복되는 경우: {len(dup_identifier)}건")
    if dup_identifier:
        print("예시 5개:", list(dup_identifier.items())[:5])

    for r in records:
        r["guessed_subject"] = guess_subject_id(r.get("identifier"))

    subj_to_views = defaultdict(set)
    for r in records:
        subj_to_views[r["guessed_subject"]].add(r["view"])

    multi_view_subjects = {k: v for k, v in subj_to_views.items() if len(v) > 1}
    print(f"\n[추정 규칙 기준] 정면+측면 둘 다 가진 것으로 추정되는 subject 수: {len(multi_view_subjects)}")
    print("이 숫자가 0이거나 이상하게 크면 guess_subject_id()의 정규식이 실제 명명 규칙과")
    print("안 맞는 것 — STEP 0 출력을 보고 직접 규칙을 다시 짤 것.")

    return records


# ============================================================
# STEP 3: 실제 data_split.py 로직을 재현해서, 같은 subject의 이미지가
# train/val/test에 걸쳐 있는지 확인 (진짜 leakage 여부)
# ============================================================
def simulate_split_and_check_leakage(records):
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 70)
    print("STEP 3: data_split.py와 동일한 랜덤 분할 재현 후 subject 중복 확인")
    print("=" * 70)

    paths = [r["filename"] for r in records]
    subjects = [r["guessed_subject"] for r in records]

    random.seed(SEED)
    train_p, temp_p, train_s, temp_s = train_test_split(
        paths, subjects, test_size=TEST_SIZE_1, random_state=SEED
    )
    val_p, test_p, val_s, test_s = train_test_split(
        temp_p, temp_s, test_size=TEST_SIZE_2, random_state=SEED
    )

    train_subj, val_subj, test_subj = set(train_s), set(val_s), set(test_s)

    leak_train_val = train_subj & val_subj
    leak_train_test = train_subj & test_subj
    leak_val_test = val_subj & test_subj

    print(f"Train subject 수: {len(train_subj)} / Val: {len(val_subj)} / Test: {len(test_subj)}")
    print(f"Train∩Val 중복 subject: {len(leak_train_val)}")
    print(f"Train∩Test 중복 subject: {len(leak_train_test)}  ← 이게 제일 중요 (모델이 학습 중 본 얼굴을 테스트에서 다시 봄)")
    print(f"Val∩Test 중복 subject: {len(leak_val_test)}")

    total_leaked_subjects = len(leak_train_val | leak_train_test | leak_val_test)
    total_subjects = len(set(subjects))
    print(f"\n[핵심 숫자] 전체 subject {total_subjects}명 중 "
          f"split 간 중복된 subject: {total_leaked_subjects}명 "
          f"({total_leaked_subjects/total_subjects*100:.1f}%)")

    return {
        "train_test_leak": len(leak_train_test),
        "total_subjects": total_subjects,
        "leaked_subjects": total_leaked_subjects,
    }


if __name__ == "__main__":
    inspect_schema(n_samples=3)
    print("\n\n위 STEP 0 출력을 확인했다면 Enter, 코드를 먼저 고쳐야 하면 Ctrl+C.")
    input()

    records = collect_all_labels()
    print(f"\n전체 라벨 수집 완료: {len(records)}건")

    records = analyze_subject_overlap(records)
    result = simulate_split_and_check_leakage(records)

    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)
    if result["train_test_leak"] > 0:
        print(f"⚠️ Train/Test 간 subject 중복 {result['train_test_leak']}명 확인 — "
              f"실제 leakage입니다. GroupShuffleSplit으로 base-id 그룹 보존 split 재구성 필요.")
    else:
        print("✅ 이 실행에서는 Train/Test subject 중복이 발견되지 않았습니다. "
              "단, guess_subject_id()의 정규식이 실제 명명 규칙과 맞는지 STEP 0 출력으로 "
              "반드시 재검증할 것 — 규칙이 틀리면 '중복 없음'이 거짓양성일 수 있습니다.")
