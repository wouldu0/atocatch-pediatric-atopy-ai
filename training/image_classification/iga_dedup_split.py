"""
iga_dedup_split.py
==================
IGA 1,800장 exact content duplicate 제거 + leakage-free clean split 생성.

입력: manifests/iga_grouped_split_seed42_sha256.csv  (sha256 컬럼 포함)

수행 순서:
  Step 1. SHA-256 full audit (cross-split + within-split 포함 전체)
  Step 2. Label / IGA grade conflict 검사 → 충돌 시 즉시 중단
  Step 3. SHA별 대표 이미지 1개 선택 (identifier 오름차순 first)
  Step 4. Connected components (same base_id OR same sha256) 생성
  Step 5. Grade-stratified component-level 7:2:1 split (seed=42)
  Step 6. 6종 overlap 검증 (base_id + sha256 각 3쌍 = 0)
  Step 7. 결과 저장

출력 (manifests/ 기준):
  iga_content_dedup_manifest.csv                  — 전체 1800행 + dedup 상태
  iga_content_dedup_grouped_split_seed42.csv      — clean split (대표 이미지만)
  iga_content_dedup_split_verification.json       — 검증 결과

주의:
  - 원본 raw 파일은 절대 삭제/수정하지 않음
  - 기존 iga_grouped_split_seed42.csv / sha256 버전은 건드리지 않음
"""
import os, sys, csv, json, random
from collections import defaultdict
from datetime import datetime

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_DIR = os.path.join(REPO_ROOT, "training", "image_classification", "manifests")
IGA_SHA_CSV  = os.path.join(MANIFEST_DIR, "iga_grouped_split_seed42_sha256.csv")

SEED        = 42
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.2


# ── Union-Find (path compression) ──
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.p[py] = px


def clean_key(k):
    return k.strip().lstrip('﻿').strip('"').strip("'")


# ══════════════════════════════════════════════
# Step 1. CSV 읽기
# ══════════════════════════════════════════════
print("=" * 65)
print("Step 1. IGA manifest 읽기")
print("=" * 65)

rows = []
with open(IGA_SHA_CSV, encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    raw_headers = next(reader)
    headers = [clean_key(h) for h in raw_headers]
    for line in reader:
        rows.append(dict(zip(headers, line)))

N = len(rows)
print(f"  총 행: {N}")
print(f"  컬럼: {headers}")
if N == 0:
    print("  ⚠ 빈 파일")
    sys.exit(1)


# ══════════════════════════════════════════════
# Step 2. SHA-256 전체 audit
# ══════════════════════════════════════════════
print("\nStep 2. SHA-256 전체 audit")

sha_groups: dict = defaultdict(list)
for i, r in enumerate(rows):
    sha_groups[r['sha256']].append(i)

unique_sha  = len(sha_groups)
dup_groups  = {sha: idxs for sha, idxs in sha_groups.items() if len(idxs) > 1}
n_extras    = sum(len(v) - 1 for v in dup_groups.values())
clean_count = unique_sha

print(f"  unique SHA:          {unique_sha}")
print(f"  duplicate 그룹 수:   {len(dup_groups)}")
print(f"  제거 대상 이미지 수: {n_extras}")
print(f"  clean 이미지 수:     {clean_count}")

# 현재 split별 분포 (원본, leakage 포함)
split_idx = defaultdict(list)
for i, r in enumerate(rows):
    split_idx[r['split']].append(i)
print("\n  현재 split 분포 (원본, leakage 포함):")
for sp in ['train', 'val', 'test']:
    grp = split_idx[sp]
    n1  = sum(1 for i in grp if int(rows[i]['label']) == 1)
    print(f"    {sp:<6}: {len(grp):>5}장  mod-sev {n1} ({n1/len(grp)*100:.1f}%)")

# duplicate 그룹 상세
print("\n  [전체 duplicate 그룹 목록]")
cross_split_count = 0
within_split_count = 0
for sha, idxs in sorted(dup_groups.items()):
    splits_in = set(rows[i]['split'] for i in idxs)
    cross = len(splits_in) > 1
    if cross: cross_split_count += 1
    else:     within_split_count += 1
    cross_mark = "cross-split" if cross else "within-split"
    print(f"\n  [{cross_mark}] SHA {sha[:16]}...  ({len(idxs)}장)")
    for i in sorted(idxs, key=lambda j: rows[j]['identifier']):
        r = rows[i]
        print(f"    {r['identifier']:<30} label={r['label']} grade={r['iga_grade']}"
              f" view={r['view']} split={r['split']}")

print(f"\n  cross-split: {cross_split_count}그룹  within-split: {within_split_count}그룹")


# ══════════════════════════════════════════════
# Step 3. Label/IGA grade conflict 검사
# ══════════════════════════════════════════════
print("\nStep 3. Label/IGA grade conflict 검사")

conflicts = []
for sha, idxs in dup_groups.items():
    labels = set(rows[i]['label']     for i in idxs)
    grades = set(rows[i]['iga_grade'] for i in idxs)
    if len(labels) > 1 or len(grades) > 1:
        conflicts.append({
            'sha256':      sha,
            'labels':      sorted(labels),
            'grades':      sorted(grades),
            'identifiers': sorted(rows[i]['identifier'] for i in idxs),
        })

if conflicts:
    print(f"  ⚠ CONFLICT {len(conflicts)}건 발견 — 즉시 중단")
    for c in conflicts:
        print(f"  SHA {c['sha256'][:16]}: labels={c['labels']}  grades={c['grades']}")
        for iid in c['identifiers']:
            print(f"    {iid}")
    sys.exit(1)
else:
    print("  OK: label/grade conflict 없음 ✓")


# ══════════════════════════════════════════════
# Step 4. 대표 이미지 선택 (identifier 오름차순 first)
# ══════════════════════════════════════════════
print("\nStep 4. 대표 이미지 선택")

is_rep: list = [False] * N
sha_rep: dict = {}  # sha → representative index

for sha, idxs in sha_groups.items():
    rep_idx = min(idxs, key=lambda i: rows[i]['identifier'])
    is_rep[rep_idx] = True
    sha_rep[sha] = rep_idx

n_reps = sum(is_rep)
print(f"  대표 이미지: {n_reps}장  (= unique SHA {unique_sha}와 일치: {n_reps == unique_sha})")


# ══════════════════════════════════════════════
# Step 5. Connected components (base_id OR sha256)
# ══════════════════════════════════════════════
print("\nStep 5. Connected components (base_id OR sha256)")

uf = UF(N)

# union by base_id
bid_groups: dict = defaultdict(list)
for i, r in enumerate(rows):
    bid_groups[r['base_id']].append(i)
for bid, idxs in bid_groups.items():
    for j in range(1, len(idxs)):
        uf.union(idxs[0], idxs[j])

n_comps_before_sha = len(set(uf.find(i) for i in range(N)))

# union by sha256 (sha256가 다른 base_id를 연결할 수 있음)
for sha, idxs in sha_groups.items():
    if len(idxs) > 1:
        for j in range(1, len(idxs)):
            uf.union(idxs[0], idxs[j])

comp_map: dict = defaultdict(list)
for i in range(N):
    comp_map[uf.find(i)].append(i)

n_comps = len(comp_map)
print(f"  base_id 기준 그룹:    {len(bid_groups)}")
print(f"  sha256 연결 전 comp:  {n_comps_before_sha}")
print(f"  sha256 연결 후 comp:  {n_comps}")
print(f"  sha256로 병합된 그룹: {n_comps_before_sha - n_comps}")


# ══════════════════════════════════════════════
# Step 6. Grade-stratified component split
# ══════════════════════════════════════════════
print("\nStep 6. Grade-stratified component split (seed=42, 7:2:1)")

comp_roots = sorted(comp_map.keys())

def dominant_label(root):
    clean_in = [i for i in comp_map[root] if is_rep[i]]
    if not clean_in:
        return 0
    lbls = [int(rows[i]['label']) for i in clean_in]
    return int(sum(lbls) / len(lbls) >= 0.5)

comp_label = {root: dominant_label(root) for root in comp_roots}

mild_comps  = sorted(r for r in comp_roots if comp_label[r] == 0)
msev_comps  = sorted(r for r in comp_roots if comp_label[r] == 1)

def proportional_split(comps: list, seed: int):
    n = len(comps)
    rng = random.Random(seed)
    shuf = comps[:]
    rng.shuffle(shuf)
    n_train = int(round(n * TRAIN_RATIO))
    n_val   = int(round(n * VAL_RATIO))
    return (set(shuf[:n_train]),
            set(shuf[n_train:n_train + n_val]),
            set(shuf[n_train + n_val:]))

tr0, va0, te0 = proportional_split(mild_comps, SEED)
tr1, va1, te1 = proportional_split(msev_comps, SEED)

train_comps = tr0 | tr1
val_comps   = va0 | va1
test_comps  = te0 | te1

sample_split = {}
for i in range(N):
    root = uf.find(i)
    if   root in train_comps: sample_split[i] = 'train'
    elif root in val_comps:   sample_split[i] = 'val'
    else:                     sample_split[i] = 'test'


# ══════════════════════════════════════════════
# Step 7. 6종 overlap 검증
# ══════════════════════════════════════════════
print("\nStep 7. 6종 overlap 검증")

train_all   = [i for i in range(N) if sample_split[i] == 'train']
val_all     = [i for i in range(N) if sample_split[i] == 'val']
test_all    = [i for i in range(N) if sample_split[i] == 'test']

train_clean = [i for i in train_all if is_rep[i]]
val_clean   = [i for i in val_all   if is_rep[i]]
test_clean  = [i for i in test_all  if is_rep[i]]


def bid_set(idx_list): return set(rows[i]['base_id'] for i in idx_list)
def sha_set(idx_list): return set(rows[i]['sha256']  for i in idx_list)

tv_bid = bid_set(train_all)   & bid_set(val_all)
tt_bid = bid_set(train_all)   & bid_set(test_all)
vt_bid = bid_set(val_all)     & bid_set(test_all)
tv_sha = sha_set(train_clean) & sha_set(val_clean)
tt_sha = sha_set(train_clean) & sha_set(test_clean)
vt_sha = sha_set(val_clean)   & sha_set(test_clean)

overlap_checks = [
    ("base_id train↔val",  tv_bid),
    ("base_id train↔test", tt_bid),
    ("base_id val↔test",   vt_bid),
    ("sha256  train↔val",  tv_sha),
    ("sha256  train↔test", tt_sha),
    ("sha256  val↔test",   vt_sha),
]
all_zero = all(len(s) == 0 for _, s in overlap_checks)
for name, s in overlap_checks:
    mark = "✓" if len(s) == 0 else "✗"
    print(f"  {mark} {name}: {len(s)}")

if not all_zero:
    print("\n  ⚠ Overlap 검증 실패 — 중단")
    sys.exit(1)
print("  → ALL OVERLAPS = 0 ✓")

# split 분포 출력
print("\n  Clean split 분포:")
for sp_name, clean_idxs in [("train", train_clean), ("val", val_clean), ("test", test_clean)]:
    n  = len(clean_idxs)
    n1 = sum(1 for i in clean_idxs if int(rows[i]['label']) == 1)
    n0 = n - n1
    bids = set(rows[i]['base_id'] for i in clean_idxs)
    print(f"  {sp_name:<6}: {n:>5}장  base_id {len(bids):>4}  "
          f"mild {n0} ({n0/n*100:.1f}%)  mod-sev {n1} ({n1/n*100:.1f}%)")


# ══════════════════════════════════════════════
# Step 8. 저장
# ══════════════════════════════════════════════
print("\nStep 8. 저장")

os.makedirs(MANIFEST_DIR, exist_ok=True)

# 8a. 전체 dedup manifest (1800행)
full_path = os.path.join(MANIFEST_DIR, "iga_content_dedup_manifest.csv")
with open(full_path, "w", newline="", encoding="utf-8") as f:
    fields = ['relative_path', 'identifier', 'base_id', 'label', 'iga_grade',
              'view', 'sha256', 'component_id', 'duplicate_status',
              'representative_path', 'assigned_split']
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for i, r in enumerate(rows):
        sha       = r['sha256']
        rep_idx   = sha_rep[sha]
        is_me_rep = is_rep[i]
        writer.writerow({
            'relative_path':      r['relative_path'],
            'identifier':         r['identifier'],
            'base_id':            r['base_id'],
            'label':              r['label'],
            'iga_grade':          r['iga_grade'],
            'view':               r['view'],
            'sha256':             sha,
            'component_id':       str(uf.find(i)),
            'duplicate_status':   'representative' if is_me_rep else 'duplicate',
            'representative_path': '' if is_me_rep else rows[rep_idx]['relative_path'],
            'assigned_split':     sample_split[i],
        })
print(f"  저장: {full_path}  ({N}행)")

# 8b. Clean split CSV (대표 이미지만, split 순서로 정렬)
clean_path = os.path.join(MANIFEST_DIR, "iga_content_dedup_grouped_split_seed42.csv")
with open(clean_path, "w", newline="", encoding="utf-8") as f:
    fields = ['relative_path', 'identifier', 'base_id', 'label', 'iga_grade',
              'view', 'sha256', 'component_id', 'split']
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for sp_name, clean_idxs in [("train", train_clean), ("val", val_clean), ("test", test_clean)]:
        for i in sorted(clean_idxs, key=lambda j: rows[j]['identifier']):
            r = rows[i]
            writer.writerow({
                'relative_path': r['relative_path'],
                'identifier':    r['identifier'],
                'base_id':       r['base_id'],
                'label':         r['label'],
                'iga_grade':     r['iga_grade'],
                'view':          r['view'],
                'sha256':        r['sha256'],
                'component_id':  str(uf.find(i)),
                'split':         sp_name,
            })
clean_total = len(train_clean) + len(val_clean) + len(test_clean)
print(f"  저장: {clean_path}  ({clean_total}행)")


def grade_dist(clean_idxs):
    n = len(clean_idxs)
    if n == 0:
        return {}
    n1 = sum(1 for i in clean_idxs if int(rows[i]['label']) == 1)
    return {
        "mild_or_below":   n - n1,
        "moderate_severe": n1,
        "mild_pct":        round((n - n1) / n * 100, 1),
        "msev_pct":        round(n1 / n * 100, 1),
    }

# 8c. Verification JSON
verif = {
    "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_total_original":   N,
    "n_unique_sha":       unique_sha,
    "n_duplicate_groups": len(dup_groups),
    "n_cross_split_dup_groups": cross_split_count,
    "n_within_split_dup_groups": within_split_count,
    "n_extras_removed":   n_extras,
    "n_clean":            clean_count,
    "n_components":       n_comps,
    "label_grade_conflict": False,
    "duplicate_groups": [
        {
            "sha256":         sha,
            "n_images":       len(idxs),
            "label":          rows[sha_rep[sha]]['label'],
            "iga_grade":      rows[sha_rep[sha]]['iga_grade'],
            "identifiers":    sorted(rows[i]['identifier'] for i in idxs),
            "representative": rows[sha_rep[sha]]['identifier'],
            "cross_split":    len(set(rows[i]['split'] for i in idxs)) > 1,
        }
        for sha, idxs in sorted(dup_groups.items())
    ],
    "split": {
        "train": len(train_clean),
        "val":   len(val_clean),
        "test":  len(test_clean),
    },
    "grade_distribution": {
        "train": grade_dist(train_clean),
        "val":   grade_dist(val_clean),
        "test":  grade_dist(test_clean),
    },
    "overlap_verification": {
        "base_id": {
            "train_val":  len(tv_bid),
            "train_test": len(tt_bid),
            "val_test":   len(vt_bid),
        },
        "sha256": {
            "train_val":  len(tv_sha),
            "train_test": len(tt_sha),
            "val_test":   len(vt_sha),
        },
        "all_zero": all_zero,
    },
    "historical_note": (
        "기존 iga_grouped_split_seed42.csv (1800장 base_id only split) 및 "
        "iga_grouped_split_seed42_sha256.csv는 historical result로 보존됨."
    ),
}

verif_path = os.path.join(MANIFEST_DIR, "iga_content_dedup_split_verification.json")
with open(verif_path, "w", encoding="utf-8") as f:
    json.dump(verif, f, indent=2, ensure_ascii=False)
print(f"  저장: {verif_path}")

print("\n완료.")
