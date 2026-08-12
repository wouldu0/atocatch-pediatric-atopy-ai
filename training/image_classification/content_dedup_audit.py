"""
content_dedup_audit.py
=======================
binary + IGA manifest에 대한 content-level (SHA-256) duplicate 검사.

1. binary manifest: cross-split exact duplicate 전수 검사
2. 367장 deduplicated internal test 재평가 (재학습 없음)
3. IGA manifest: SHA-256 컬럼 추가 + cross-split duplicate 검사
"""
import os, sys, csv, json, hashlib, random
import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix,
)

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_DIR = os.path.join(REPO_ROOT, "training", "image_classification", "manifests")
DATA_ROOT    = r"E:\atopic\AIHub"   # 절대경로는 스크립트 내부 참조용, 출력·저장에는 미사용

BINARY_CSV   = os.path.join(MANIFEST_DIR, "binary_split_candidate_seed42.csv")
IGA_CSV_IN   = os.path.join(MANIFEST_DIR, "iga_grouped_split_seed42.csv")
IGA_CSV_OUT  = os.path.join(MANIFEST_DIR, "iga_grouped_split_seed42_sha256.csv")

MODEL_PATH   = r"E:\atopic\models\binary_grouped_split\tf_efficientnetv2_s\best_model.pth"
IMG_SIZE     = 224
BATCH_SIZE   = 32
LABEL_NAMES  = ["non_atopy", "atopy"]


# ── 헬퍼 ──
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def abs_path(rel):
    return os.path.join(DATA_ROOT, rel.replace("/", os.sep))


eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(abs_path(r["relative_path"])).convert("RGB")
        return self.transform(img), int(r["label"])

def run_inference(model, rows, device):
    loader = DataLoader(SkinDataset(rows, eval_transform),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_l, all_p = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            out = model(imgs.to(device))
            prbs = torch.softmax(out, dim=1)
            all_l.extend(lbls.numpy())
            all_p.extend(prbs[:, 1].cpu().numpy())
    return np.array(all_l), np.array(all_p)

def mets(labels, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "n":           len(labels),
        "threshold":   thr,
        "accuracy":    round(float(accuracy_score(labels, preds)), 4),
        "f1_weighted": round(float(f1_score(labels, preds, average="weighted")), 4),
        "roc_auc":     round(float(roc_auc_score(labels, probs)), 4),
        "sensitivity": round(float(tp/(tp+fn)) if (tp+fn)>0 else 0, 4),
        "specificity": round(float(tn/(tn+fp)) if (tn+fp)>0 else 0, 4),
        "confusion_matrix": cm.tolist(),
    }


# ══════════════════════════════════════════════
# 1. Binary manifest — SHA-256 cross-split 검사
# ══════════════════════════════════════════════
print("=" * 65)
print("1. Binary manifest SHA-256 cross-split duplicate 검사")
print("=" * 65)

with open(BINARY_CSV, encoding="utf-8") as f:
    binary_rows = list(csv.DictReader(f))

# sha256 → [(split, relative_path), ...]
sha_map = {}
for r in binary_rows:
    sha = r["sha256"]
    sha_map.setdefault(sha, []).append((r["split"], r["relative_path"]))

# cross-split groups (같은 sha가 2개 이상의 서로 다른 split에 존재)
cross_split_groups = {}
for sha, entries in sha_map.items():
    splits_seen = set(e[0] for e in entries)
    if len(splits_seen) > 1:
        cross_split_groups[sha] = entries

print(f"  총 행: {len(binary_rows)}")
print(f"  unique SHA: {len(sha_map)}")
print(f"  cross-split SHA duplicate 그룹: {len(cross_split_groups)}")

KNOWN_SHA = "403993759f0654d29a08b63ef19b0aa2347c5c82140f249649e3a96a1fb38dd9"
if cross_split_groups:
    print("\n  [Cross-split duplicate 목록]")
    for sha, entries in cross_split_groups.items():
        known_mark = "  ← 기보고된 쌍" if sha == KNOWN_SHA else ""
        print(f"\n  SHA: {sha[:16]}...{known_mark}")
        for split, rel in sorted(entries):
            print(f"    {split:<20}  {rel}")
else:
    print("  → cross-split duplicate 없음")

# 기보고된 쌍 byte-identical 재확인
print(f"\n  [기보고된 쌍 byte-identical 재확인]")
known_entries = cross_split_groups.get(KNOWN_SHA, [])
if len(known_entries) >= 2:
    paths = [abs_path(e[1]) for e in known_entries]
    hashes = []
    for p in paths:
        h = sha256_file(p)
        hashes.append(h)
        print(f"    {p}")
        print(f"    SHA256: {h}")
    print(f"  byte-identical: {'✓' if len(set(hashes))==1 else '✗'}")
else:
    print("  → KNOWN_SHA가 cross-split_groups에 없음 (duplicate 아님)")


# ══════════════════════════════════════════════
# 2. 367장 dedup internal test 재평가
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("2. 367장 deduplicated internal test 재평가")
print("=" * 65)

# test split에서 cross-split duplicate 제거
dup_rels_in_test = set()
for sha, entries in cross_split_groups.items():
    for split, rel in entries:
        if split == "test":
            dup_rels_in_test.add(rel)

test_368 = [r for r in binary_rows if r["split"] == "test"]
test_367 = [r for r in test_368 if r["relative_path"] not in dup_rels_in_test]

print(f"  원본 test: {len(test_368)}장")
print(f"  제거된 test duplicate: {len(dup_rels_in_test)}장")
print(f"  deduplicated test: {len(test_367)}장")
for rel in dup_rels_in_test:
    print(f"    제거: {rel}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  모델 로드: {MODEL_PATH}")
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))

labels_367, probs_367 = run_inference(model, test_367, device)
m367 = mets(labels_367, probs_367, thr=0.5)

print(f"\n  367장 결과 (threshold=0.5):")
print(f"    Accuracy:    {m367['accuracy']:.4f}")
print(f"    F1 weighted: {m367['f1_weighted']:.4f}")
print(f"    ROC-AUC:     {m367['roc_auc']:.4f}")
print(f"    Sensitivity: {m367['sensitivity']:.4f}")
print(f"    Specificity: {m367['specificity']:.4f}")
print(f"    CM: {m367['confusion_matrix']}")

dedup_result = {
    "note": "best_model.pth (grouped split, no retraining) evaluated on 367-image deduplicated test set",
    "removed_from_test": list(dup_rels_in_test),
    "n_original_test":    len(test_368),
    "n_deduplicated_test": len(test_367),
    "n_positive":         int(labels_367.sum()),
    "n_negative":         int((1 - labels_367).sum()),
    "metrics_thr_0.5":    m367,
    "reference_368_metrics": {
        "accuracy": 0.9511, "f1_weighted": 0.9511,
        "roc_auc": 0.9874, "sensitivity": 0.9286, "specificity": 0.9731,
    },
}
dedup_json = os.path.join(MANIFEST_DIR, "binary_test_367_dedup_eval.json")
with open(dedup_json, "w", encoding="utf-8") as f:
    json.dump(dedup_result, f, indent=2, ensure_ascii=False)
print(f"\n  저장: {dedup_json}")


# ══════════════════════════════════════════════
# 3. IGA manifest SHA-256 생성 + cross-split 검사
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("3. IGA manifest SHA-256 생성 및 cross-split duplicate 검사")
print("=" * 65)

with open(IGA_CSV_IN, encoding="utf-8-sig") as f:
    iga_rows = list(csv.DictReader(f))
# DictReader가 따옴표 포함된 헤더를 그대로 쓸 수 있으므로 정규화
for r in iga_rows:
    keys = list(r.keys())
    for k in keys:
        clean = k.strip().strip('"').strip("'")
        if clean != k:
            r[clean] = r.pop(k)

print(f"  IGA 총 행: {len(iga_rows)}")

IGA_DATA_ROOT = r"E:\atopic"
iga_sha_errors = 0
for r in iga_rows:
    abs_p = os.path.join(IGA_DATA_ROOT, r["relative_path"].replace("/", os.sep).replace('"', ''))
    try:
        r["sha256"] = sha256_file(abs_p)
    except Exception as e:
        r["sha256"] = "ERROR"
        iga_sha_errors += 1

print(f"  SHA256 생성 완료 (오류: {iga_sha_errors})")

# cross-split duplicate 검사
iga_sha_map = {}
for r in iga_rows:
    iga_sha_map.setdefault(r["sha256"], []).append((r["split"], r["relative_path"]))

iga_cross = {}
for sha, entries in iga_sha_map.items():
    splits = set(e[0] for e in entries)
    if len(splits) > 1:
        iga_cross[sha] = entries

print(f"  unique SHA: {len(iga_sha_map)}")
print(f"  cross-split SHA duplicate 그룹: {len(iga_cross)}")

if iga_cross:
    print("\n  [IGA cross-split duplicate 목록]")
    for sha, entries in iga_cross.items():
        print(f"\n  SHA: {sha[:16]}...")
        for split, rel in sorted(entries):
            print(f"    {split:<12}  {rel}")
else:
    print("  → IGA cross-split duplicate 없음 ✓")

# IGA CSV 저장 (sha256 컬럼 추가)
fieldnames = list(iga_rows[0].keys())
with open(IGA_CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(iga_rows)
print(f"\n  저장: {IGA_CSV_OUT}")


# ══════════════════════════════════════════════
# 6. binary_split_validation_result.json manifest_path 수정
# ══════════════════════════════════════════════
val_json_path = os.path.join(MANIFEST_DIR, "binary_split_validation_result.json")
with open(val_json_path, encoding="utf-8") as f:
    val_result = json.load(f)

old_mp = val_result.get("manifest_path", "")
new_mp = "training/image_classification/manifests/binary_split_candidate_seed42.csv"
val_result["manifest_path"] = new_mp

with open(val_json_path, "w", encoding="utf-8") as f:
    json.dump(val_result, f, indent=2, ensure_ascii=False)
print(f"\n[6] manifest_path 수정")
print(f"  이전: {old_mp}")
print(f"  이후: {new_mp}")


# ══════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════
print("\n" + "=" * 65)
print("최종 요약")
print("=" * 65)
print(f"  binary cross-split SHA duplicate 총 개수: {len(cross_split_groups)}")
print(f"  367장 dedup internal test:")
print(f"    Acc={m367['accuracy']:.4f}  F1={m367['f1_weighted']:.4f}  AUC={m367['roc_auc']:.4f}")
print(f"    Sens={m367['sensitivity']:.4f}  Spec={m367['specificity']:.4f}")
print(f"  IGA SHA 생성 성공: {iga_sha_errors==0} (오류 {iga_sha_errors}건)")
print(f"  IGA cross-split SHA duplicate 개수: {len(iga_cross)}")
print(f"  변경 파일:")
print(f"    manifests/binary_test_367_dedup_eval.json  (신규)")
print(f"    manifests/iga_grouped_split_seed42_sha256.csv  (신규)")
print(f"    manifests/binary_split_validation_result.json  (manifest_path 수정)")
raw_safe = len(cross_split_groups) == 1 and len(iga_cross) == 0
print(f"  raw 데이터 삭제 가능 여부: {'조건 충족 (binary 1쌍, IGA 0쌍)' if raw_safe else '추가 검토 필요'}")
