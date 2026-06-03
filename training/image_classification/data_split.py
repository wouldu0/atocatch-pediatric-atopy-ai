import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# ── 설정 ──
TRAIN_ROOT = r"E:\atopic\AIHub\Training\01.원천데이터"
VAL_ROOT = r"E:\atopic\AIHub\Validation\01.원천데이터"
SEED = 42
random.seed(SEED)

# ── 1. 전체 데이터 수집 ──
disease_files = {}
for root in [TRAIN_ROOT, VAL_ROOT]:
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        parts = folder.split("_")
        if len(parts) < 3:
            continue
        disease = parts[1]
        if disease not in disease_files:
            disease_files[disease] = []
        for f in os.listdir(folder_path):
            if f.lower().endswith((".png", ".jpg")):
                disease_files[disease].append(os.path.join(folder_path, f))

print("=== 전체 데이터 ===")
for d, files in sorted(disease_files.items()):
    print(f"  {d}: {len(files)}장")

# ── 2. 이진 데이터셋 구성 ──
NON_ATOPY = {"정상": 1200, "건선": 150, "여드름": 150, "주사": 150, "지루": 150}
samples = []

for f in disease_files["아토피"]:
    samples.append((f, 1))
for disease, n in NON_ATOPY.items():
    selected = random.sample(disease_files[disease], n)
    for f in selected:
        samples.append((f, 0))

labels = [s[1] for s in samples]
print(f"\n=== 이진 데이터셋 ===")
print(f"  아토피 (1): {labels.count(1)}장")
print(f"  비아토피 (0): {labels.count(0)}장")

# ── 3. 7:2:1 Split ──
paths = [s[0] for s in samples]
labels = [s[1] for s in samples]

train_p, temp_p, train_l, temp_l = train_test_split(
    paths, labels, test_size=0.3, stratify=labels, random_state=SEED
)
val_p, test_p, val_l, test_l = train_test_split(
    temp_p, temp_l, test_size=1/3, stratify=temp_l, random_state=SEED
)

print(f"\n=== 7:2:1 Split ===")
print(f"  Train: {len(train_p)}장 (아토피:{train_l.count(1)} / 비아토피:{train_l.count(0)})")
print(f"  Val:   {len(val_p)}장 (아토피:{val_l.count(1)} / 비아토피:{val_l.count(0)})")
print(f"  Test:  {len(test_p)}장 (아토피:{test_l.count(1)} / 비아토피:{test_l.count(0)})")