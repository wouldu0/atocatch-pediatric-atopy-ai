"""
eval_binary_skindisnet_external.py
====================================
현재 production Binary(아토피 유무) 모델을 SkinDisNet(Preprocessed) 외부 데이터셋에
그대로(threshold/전처리/모델 전부 동결) 1회 평가하는 독립 external evaluation 스크립트.

MODEL FREEZE:
  - app/best_model.pth, app/model_config.json을 source of truth로 그대로 로드한다.
  - threshold/전처리/architecture 전부 production 그대로 사용, 이 스크립트 안에서
    재학습·fine-tuning·threshold 재탐색은 절대 하지 않는다.

Primary class mapping (고정, 결과를 보고 바꾸지 않음):
  positive: AD
  negative: CD, SC, SD, TC
  excluded: EC (atopic dermatitis와 의미가 겹칠 수 있어 ground-truth ambiguity 회피)

산출물: training/image_classification/outputs/external_skindisnet/
  metrics.json, predictions.csv, confusion_matrix.png, dataset_audit.json,
  evaluation_summary.md

기존 training 결과(app/, 다른 outputs/ 하위 디렉터리)는 건드리지 않는다.

실행:
    python eval_binary_skindisnet_external.py --data-root "/path/to/Preprocessed"
    (선택) --dermnet-root "/path/to/dermnet_images"  # 지정하면 cross-dataset dedup audit도 수행
    경로는 SKINDISNET_PATH / DERMNET_PATH 환경변수로도 지정 가능. 개인 PC 절대경로를
    이 스크립트에 하드코딩하지 않는다.
"""
import os
import sys
import json
import csv
import hashlib
import argparse
from collections import defaultdict

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
    recall_score, precision_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
# 경로 (repo 기준 상대경로, source of truth = app/)
# ══════════════════════════════════════════════
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR     = os.path.join(REPO_ROOT, "app")
CFG_PATH    = os.path.join(APP_DIR, "model_config.json")
CKPT_PATH   = os.path.join(APP_DIR, "best_model.pth")

EXPORT_DIR  = os.path.join(REPO_ROOT, "training", "image_classification",
                            "outputs", "external_skindisnet")

POSITIVE_CLASSES = ["AD"]
NEGATIVE_CLASSES = ["CD", "SC", "SD", "TC"]
EXCLUDED_CLASSES = ["EC"]
ALL_CLASSES = POSITIVE_CLASSES + NEGATIVE_CLASSES + EXCLUDED_CLASSES


# ══════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ahash(path, hash_size=8):
    """단순 average-hash (perceptual similarity screening 용, 외부 라이브러리 미사용).
    64bit 정수로 반환. Hamming distance가 작을수록 시각적으로 유사한 후보(확정 중복 아님)."""
    img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


# ══════════════════════════════════════════════
# 1. FROZEN EVALUATION CONFIG 출력
# ══════════════════════════════════════════════
def print_frozen_config(cfg, ckpt_sha256, skindisnet_root):
    print("[FROZEN EVALUATION CONFIG]")
    print(f"Production checkpoint: {CKPT_PATH}  (sha256={ckpt_sha256[:16]}...)")
    print(f"Architecture: {cfg['model_name']}")
    print(f"Production threshold: {cfg['threshold']}")
    print(f"Input size: {cfg['img_size']}x{cfg['img_size']}")
    print("Normalization: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] (ImageNet, production 코드와 동일)")
    print(f"SkinDisNet source: {skindisnet_root}")
    print("Dataset subset: Preprocessed only")
    print("Positive: AD")
    print("Negative: CD / SC / SD / TC")
    print("Excluded: EC")
    print("Augmented used: NO")
    print("Training/Fine-tuning: NO")
    print("Threshold tuning: NO")
    print()


# ══════════════════════════════════════════════
# main
# ══════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Production Binary 모델을 SkinDisNet(Preprocessed)에 동결 상태로 1회 external 평가"
    )
    parser.add_argument(
        "--data-root", default=os.getenv("SKINDISNET_PATH"),
        help="SkinDisNet Preprocessed 폴더 경로 (필수, 또는 SKINDISNET_PATH 환경변수)",
    )
    parser.add_argument(
        "--dermnet-root", default=os.getenv("DERMNET_PATH"),
        help="기존 DermNet 이미지 폴더 경로 (선택, cross-dataset dedup audit용, 또는 DERMNET_PATH 환경변수)",
    )
    args = parser.parse_args()
    if not args.data_root:
        parser.error("--data-root (또는 SKINDISNET_PATH 환경변수)가 필요합니다.")
    return args


def main():
    args = parse_args()
    skindisnet_root = args.data_root
    dermnet_root = args.dermnet_root  # None이면 cross-dataset audit 생략

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # ── production config/checkpoint 로드 (동결) ──
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    ckpt_sha256 = sha256_file(CKPT_PATH)

    print_frozen_config(cfg, ckpt_sha256, skindisnet_root)

    device = torch.device("cpu")
    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=cfg["num_classes"])
    state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    load_result = model.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys, \
        f"checkpoint/architecture mismatch: {load_result}"
    model.eval()

    threshold = cfg["threshold"]
    img_size = cfg["img_size"]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── 2. SkinDisNet 파일 수집 ──
    print("[SkinDisNet 파일 수집]")
    files_by_class = {}
    for cls in ALL_CLASSES:
        folder = os.path.join(skindisnet_root, cls)
        files_by_class[cls] = list_images(folder)
        tag = "excluded" if cls in EXCLUDED_CLASSES else ""
        print(f"  {cls}: {len(files_by_class[cls])} {tag}")

    primary_files = []  # (cls, filename, label)
    for cls in POSITIVE_CLASSES:
        for fn in files_by_class[cls]:
            primary_files.append((cls, fn, 1))
    for cls in NEGATIVE_CLASSES:
        for fn in files_by_class[cls]:
            primary_files.append((cls, fn, 0))

    n_pos = sum(1 for _, _, l in primary_files if l == 1)
    n_neg = sum(1 for _, _, l in primary_files if l == 0)
    print(f"\nPrimary evaluation total N: {len(primary_files)}")
    print(f"positive N: {n_pos}")
    print(f"negative N: {n_neg}\n")

    # metadata 확인
    meta_candidates = []
    for root, _dirs, fnames in os.walk(skindisnet_root):
        for fn in fnames:
            if fn.lower().endswith((".csv", ".xlsx", ".json", ".xls")):
                meta_candidates.append(os.path.join(root, fn))
    print(f"[metadata] 발견된 CSV/Excel/JSON 파일: {len(meta_candidates)}건")
    for m in meta_candidates:
        print(f"  - {m}")
    if not meta_candidates:
        print("  metadata 없음 — patient ID/age 등은 추측하지 않고 '정보 없음'으로 보고함")
    print()

    # ── 3. Data integrity audit ──
    print("[Data integrity audit]")
    audit = {"skindisnet_internal": {}, "cross_dermnet": {}}

    # 3-A. SkinDisNet 내부 SHA-256 exact duplicate
    sha_map = defaultdict(list)  # sha256 -> [(cls, fn)]
    file_sha = {}  # (cls, fn) -> sha256
    for cls, fn, _label in primary_files:
        p = os.path.join(skindisnet_root, cls, fn)
        s = sha256_file(p)
        file_sha[(cls, fn)] = s
        sha_map[s].append(f"{cls}/{fn}")
    dup_groups = {s: v for s, v in sha_map.items() if len(v) > 1}
    audit["skindisnet_internal"]["n_files_hashed"] = len(primary_files)
    audit["skindisnet_internal"]["n_exact_duplicate_groups"] = len(dup_groups)
    audit["skindisnet_internal"]["exact_duplicate_groups"] = dup_groups
    print(f"  SkinDisNet 내부 exact(SHA-256) duplicate 그룹: {len(dup_groups)}개")
    for s, members in dup_groups.items():
        print(f"    {s[:16]}...: {members}")

    # 3-B. 단순 aHash 기반 유사도 스크리닝 (within SkinDisNet, primary set)
    # 주의: average-hash는 색/구도/병변 영역이 비슷한 서로 다른 피부 이미지도 후보로
    # 많이 잡을 수 있어, 아래 결과는 "확정된 중복"이 아니라 "유사도 스크리닝 후보"다.
    # 이 후보를 이유로 샘플을 제외하거나 평가를 바꾸지 않는다.
    file_ahash = {k: ahash(os.path.join(skindisnet_root, k[0], k[1])) for k in file_sha}
    similarity_candidates = []
    keys = list(file_ahash.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = hamming(file_ahash[keys[i]], file_ahash[keys[j]])
            if d <= 5:  # 64bit 중 5bit 이하 차이 = 시각적으로 유사한 후보 (경험적 임계값)
                similarity_candidates.append({
                    "a": f"{keys[i][0]}/{keys[i][1]}",
                    "b": f"{keys[j][0]}/{keys[j][1]}",
                    "hamming_distance": d,
                })
    audit["skindisnet_internal"]["ahash_similarity_candidate_method"] = (
        "average-hash(8x8), Hamming<=5/64 — perceptual similarity screening candidate만 탐지"
        "(경량 자체 구현, imagehash 등 외부 라이브러리 미사용). confirmed duplicate 아님."
    )
    audit["skindisnet_internal"]["n_ahash_similarity_candidates"] = len(similarity_candidates)
    audit["skindisnet_internal"]["ahash_similarity_candidates"] = similarity_candidates[:50]
    print(f"  SkinDisNet 내부 aHash similarity candidate(Hamming<=5, 확정 중복 아님): {len(similarity_candidates)}개"
          + (" (상위 50개만 audit json에 기록)" if len(similarity_candidates) > 50 else ""))

    # 3-C. 기존 DermNet과 cross-dataset overlap
    if dermnet_root and os.path.isdir(dermnet_root):
        dermnet_files = []
        for root, _dirs, fnames in os.walk(dermnet_root):
            for fn in fnames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    dermnet_files.append(os.path.join(root, fn))
        dermnet_sha = {p: sha256_file(p) for p in dermnet_files}
        dermnet_sha_set = set(dermnet_sha.values())
        dermnet_ahash = {p: ahash(p) for p in dermnet_files}

        exact_overlap = [k for k, s in file_sha.items() if s in dermnet_sha_set]
        cross_similarity_candidates = []
        for k, h in file_ahash.items():
            for dp, dh in dermnet_ahash.items():
                d = hamming(h, dh)
                if d <= 5:
                    cross_similarity_candidates.append({
                        "skindisnet": f"{k[0]}/{k[1]}",
                        "dermnet": os.path.relpath(dp, dermnet_root),
                        "hamming_distance": d,
                    })
        audit["cross_dermnet"]["available"] = True
        audit["cross_dermnet"]["n_dermnet_files"] = len(dermnet_files)
        audit["cross_dermnet"]["n_exact_sha_overlap"] = len(exact_overlap)
        audit["cross_dermnet"]["exact_sha_overlap"] = [f"{k[0]}/{k[1]}" for k in exact_overlap]
        audit["cross_dermnet"]["ahash_similarity_candidate_method"] = (
            "average-hash(8x8), Hamming<=5/64 — perceptual similarity screening candidate만 탐지, confirmed duplicate 아님"
        )
        audit["cross_dermnet"]["n_ahash_similarity_candidates"] = len(cross_similarity_candidates)
        audit["cross_dermnet"]["ahash_similarity_candidates"] = cross_similarity_candidates[:50]
        print(f"  DermNet({len(dermnet_files)}장) cross-dataset exact SHA overlap: {len(exact_overlap)}건")
        print(f"  DermNet cross-dataset aHash similarity candidate(확정 중복 아님): {len(cross_similarity_candidates)}건")
    else:
        audit["cross_dermnet"]["available"] = False
        audit["cross_dermnet"]["note"] = "cross-dataset duplicate audit unavailable because original DermNet files are not locally available"
        print("  cross-dataset duplicate audit unavailable because original DermNet files are not locally available")
    print()

    with open(os.path.join(EXPORT_DIR, "dataset_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    # ── 6. Evaluation (threshold/전처리/모델 전부 동결, augmentation 없음) ──
    print("[Evaluation — production threshold 그대로, no tuning]")
    rows = []
    probs, labels = [], []
    with torch.no_grad():
        for cls, fn, label in primary_files:
            path = os.path.join(skindisnet_root, cls, fn)
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[0, 1].item()  # index 1 = atopy (label_names[1])
            pred = 1 if prob >= threshold else 0
            probs.append(prob)
            labels.append(label)
            rows.append({
                "filename": fn,
                "disease": cls,
                "true_label": label,
                "predicted_label": pred,
                "probability_atopy": round(prob, 6),
                "patient_id": "",  # metadata 없음
            })

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted")
    f1_binary = f1_score(labels, preds, average="binary")
    f1_macro = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    sens = recall_score(labels, preds, pos_label=1)
    spec = recall_score(labels, preds, pos_label=0)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn_, tp = cm.ravel()

    metrics = {
        "n_total": int(len(labels)),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "threshold_used": threshold,
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "f1_weighted_primary": round(float(f1_weighted), 4),
        "f1_binary": round(float(f1_binary), 4),
        "f1_macro": round(float(f1_macro), 4),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "sensitivity_recall": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn_), "TP": int(tp)},
    }

    # ── 7. Disease별 breakdown (error analysis 목적) ──
    breakdown = {}
    for cls in POSITIVE_CLASSES:
        idx = [i for i, (c, _, _) in enumerate(primary_files) if c == cls]
        if idx:
            sens_cls = float(np.mean(preds[idx] == 1))
            breakdown[cls] = {"n": len(idx), "sensitivity": round(sens_cls, 4)}
    for cls in NEGATIVE_CLASSES:
        idx = [i for i, (c, _, _) in enumerate(primary_files) if c == cls]
        if idx:
            fpr_cls = float(np.mean(preds[idx] == 1))
            breakdown[cls] = {"n": len(idx), "false_positive_rate": round(fpr_cls, 4)}
    metrics["disease_breakdown"] = breakdown

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with open(os.path.join(EXPORT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(EXPORT_DIR, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "disease", "true_label", "predicted_label",
            "probability_atopy", "patient_id",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # confusion matrix 시각화 (matplotlib만 사용, seaborn 의존성 없음)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    labels_txt = ["non_atopy", "atopy"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_txt)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_txt)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"SkinDisNet external eval (threshold={threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"\n[완료] 산출물: {EXPORT_DIR}")
    return metrics, audit, len(meta_candidates)


if __name__ == "__main__":
    main()
