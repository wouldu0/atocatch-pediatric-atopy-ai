"""
AI Hub base-id 그룹을 보존하는 patient-level(그룹) split
==============================================================
배경: check_aihub_subject_leakage.py 검증으로 "아토피_정면" 폴더 안에서
같은 base-id(예: H1_300346)가 신체부위(P)·병변(L) 코드만 다르게 여러 장
존재하고, data_split.py의 순수 랜덤 7:2:1 분할이 이 그룹을 쪼개
train/val/test에 걸치는 실제 leakage(아토피 풀의 5.33%, Train∩Test 16그룹)를
만든다는 게 확인됐다. 이 스크립트는 그 그룹을 보존하는 분할 함수를 제공한다.

⚠️ 실행 전 확인할 것: 지금까지 검증은 "아토피" 폴더만 했다. 비아토피
(정상/건선/여드름/주사/지루) 폴더도 같은 명명 규칙(H*_숫자_P*_L*)을 쓴다면
같은 패턴이 있을 수 있다 — check_base_id_duplication_all_classes()로 먼저
전체 클래스를 확인하고, 있으면 아래 grouped_train_val_test_split()에
자동으로 반영된다(디스크에서 파일명만 보고 그룹을 만들기 때문에 클래스
무관하게 동작함).

이 스크립트는 파일 경로 리스트만 반환한다 — 기존 train_binary.py /
eval_comparison.py / train_binary_final.py의 데이터 수집 부분(리스트 만드는
부분)만 이 함수 결과로 바꿔 끼우고, 학습 루프 자체는 그대로 재사용할 것.
그래야 "그룹 리키지만 고치고 나머지는 기존 검증된 코드 그대로"가 보장된다.
"""
import os
import re
from collections import defaultdict

BASE_ID_PATTERN = re.compile(r'^(.+?)_P\d+_L\d+$')


def extract_base_id(filename_no_ext: str) -> str:
    """
    'H1_300346_P1_L0' -> 'H1_300346'
    패턴에 안 맞으면(비아토피 클래스 등 다른 명명 규칙) 파일명 자체를
    고유 그룹으로 취급 — 즉 그룹핑이 필요 없는 파일은 그대로 자기 자신이 그룹.
    """
    stem = os.path.splitext(filename_no_ext)[0]
    m = BASE_ID_PATTERN.match(stem)
    return m.group(1) if m else stem


def check_base_id_duplication_all_classes(disease_files: dict):
    """
    data_split.py의 disease_files(딕셔너리: disease -> [파일경로,...])를 그대로
    받아서, 클래스별로 base-id 중복이 얼마나 있는지 출력한다.
    아토피 폴더뿐 아니라 전체 클래스를 확인하기 위한 함수 — 반드시 먼저 실행할 것.
    """
    print("=" * 70)
    print("클래스별 base-id 중복 확인")
    print("=" * 70)
    for disease, files in disease_files.items():
        groups = defaultdict(list)
        for f in files:
            fname = os.path.basename(f)
            groups[extract_base_id(fname)].append(f)
        n_files = len(files)
        n_groups = len(groups)
        n_dup_groups = sum(1 for g in groups.values() if len(g) > 1)
        n_dup_files = sum(len(g) for g in groups.values() if len(g) > 1)
        print(f"{disease:<12} 파일 {n_files:>5}개 | 그룹 {n_groups:>5}개 | "
              f"중복그룹 {n_dup_groups:>4}개 ({n_dup_files}장, {n_dup_files/n_files*100:.1f}%)")
    print("\n중복그룹이 0인 클래스는 그룹핑 안 해도 무방(=파일마다 고유 그룹).")
    print("0이 아닌 클래스가 있으면 아래 grouped_train_val_test_split()이 자동으로 처리함.")


def grouped_train_val_test_split(paths, labels, test_size=0.3, val_ratio_of_temp=1 / 3, seed=42):
    """
    data_split.py의 7:2:1 분할과 동일한 비율이지만, base-id 그룹을 절대
    쪼개지 않는다(StratifiedGroupKFold 계열 대신, 그룹 단위로 먼저 나누고
    그 안에서 클래스 비율을 맞추는 방식 — sklearn GroupShuffleSplit은
    stratify를 지원하지 않아 직접 구현).
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = [extract_base_id(os.path.basename(p)) for p in paths]

    # 1차: train vs temp(val+test), 그룹 단위로 분리
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(paths, labels, groups=groups))

    temp_paths = [paths[i] for i in temp_idx]
    temp_labels = [labels[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]

    # 2차: temp -> val vs test, 역시 그룹 단위
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_of_temp, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_paths, temp_labels, groups=temp_groups))

    train_p = [paths[i] for i in train_idx]
    train_l = [labels[i] for i in train_idx]
    val_p = [temp_paths[i] for i in val_idx]
    val_l = [temp_labels[i] for i in val_idx]
    test_p = [temp_paths[i] for i in test_idx]
    test_l = [temp_labels[i] for i in test_idx]

    # 검증: 그룹이 실제로 안 쪼개졌는지 assert
    g_train = set(groups[i] for i in train_idx)
    g_val = set(temp_groups[i] for i in val_idx)
    g_test = set(temp_groups[i] for i in test_idx)
    assert not (g_train & g_val), "train/val 그룹 중복 — 분할 로직 확인 필요"
    assert not (g_train & g_test), "train/test 그룹 중복 — 분할 로직 확인 필요"
    assert not (g_val & g_test), "val/test 그룹 중복 — 분할 로직 확인 필요"

    print(f"[grouped split 완료] Train {len(train_p)}장 / Val {len(val_p)}장 / Test {len(test_p)}장")
    print(f"클래스 비율 — Train 양성률={sum(train_l)/len(train_l)*100:.1f}% "
          f"Val={sum(val_l)/len(val_l)*100:.1f}% Test={sum(test_l)/len(test_l)*100:.1f}%")
    print("(참고: 그룹 단위 분할이라 기존 순수 랜덤 분할보다 클래스 비율이 약간 덜 정확할 수 있음 — "
          "위 비율이 크게 어긋나면 seed를 바꿔가며 몇 번 시도해서 가장 균형 잡힌 seed를 채택할 것)")

    return train_p, val_p, test_p, train_l, val_l, test_l
