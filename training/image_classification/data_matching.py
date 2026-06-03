import os
import json
from collections import Counter

label_roots = {
    "train": r"E:\atopic\AIHub\Training\02.라벨링데이터",
    "val":   r"E:\atopic\AIHub\Validation\02.라벨링데이터",
}
img_roots = {
    "train": r"E:\atopic\AIHub\Training\01.원천데이터",
    "val":   r"E:\atopic\AIHub\Validation\01.원천데이터",
}

IGA_MAP = {
    "Clear": 0, "Almost Clear": 0, "Mild": 0,
    "Moderate": 1, "Severe": 1,
}

samples = []
iga_counts = Counter()
not_found = 0

for split in ["train", "val"]:
    label_root = label_roots[split]
    img_root = img_roots[split]
    prefix = "TS" if split == "train" else "VS"
    label_prefix = "TL" if split == "train" else "VL"

    for view in ["정면", "측면"]:
        label_folder = os.path.join(label_root, f"{label_prefix}_아토피_{view}")
        img_folder = os.path.join(img_root, f"{prefix}_아토피_{view}")

        print(f"\n[{split}-{view}]")
        print(f"  라벨 폴더: {label_folder} → 존재: {os.path.exists(label_folder)}")
        print(f"  이미지 폴더: {img_folder} → 존재: {os.path.exists(img_folder)}")

        if not os.path.exists(label_folder) or not os.path.exists(img_folder):
            continue

        jsons = [f for f in os.listdir(label_folder) if f.endswith('.json')]
        print(f"  JSON 수: {len(jsons)}")

        found_here = 0
        for fname in jsons:
            fpath = os.path.join(label_folder, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ann = data['annotations'][0]
            identifier = ann['identifier']
            iga = ann['diagnosis_info']['easi_score']['iga_grade']
            label = IGA_MAP.get(iga, -1)
            if label == -1:
                continue
            img_path = os.path.join(img_folder, f"{identifier}.png")
            if os.path.exists(img_path):
                samples.append((img_path, label))
                iga_counts[iga] += 1
                found_here += 1
            else:
                not_found += 1

        print(f"  찾음: {found_here}장")

print(f"\n총 샘플: {len(samples)}장")
print(f"이진 라벨:")
label_list = [s[1] for s in samples]
print(f"  mild 이하 (0):       {label_list.count(0)}장")
print(f"  moderate-severe (1): {label_list.count(1)}장")