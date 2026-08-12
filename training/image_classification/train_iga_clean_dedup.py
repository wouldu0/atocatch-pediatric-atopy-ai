"""
train_iga_clean_dedup.py
========================
IGA 모델 최종 재학습 — exact content duplicate 제거 + leakage-free split 사용.

입력: manifests/iga_content_dedup_grouped_split_seed42.csv  (1791행)
출력 (outputs/iga_clean_dedup/ 기준):
  models/best_iga_clean_dedup.pth
  results/final_metrics.json
  results/training_curve.png
  validation_threshold_search.csv

학습 조건:
  - 모델/augmentation/loss/optimizer/scheduler: train_iga_grouped_final.py와 동일
  - 이번 작업의 목적은 split/data leakage 제거뿐; 임의 성능 튜닝 없음
  - validation Youden's J로 threshold 결정, test는 최종 1회 평가 후 미조정

주의:
  - 기존 best_iga_grouped_seed42.pth 등 원본 가중치는 덮어쓰지 않음
  - commit/push 금지; raw 파일 삭제 금지
"""

import os, sys, csv, json, random
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, average_precision_score,
)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

# ══════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_DIR = os.path.join(REPO_ROOT, "training", "image_classification", "manifests")
CLEAN_CSV    = os.path.join(MANIFEST_DIR, "iga_content_dedup_grouped_split_seed42.csv")
IGA_DATA_ROOT = r"E:\atopic"   # relative_path 앞에 붙이면 절대경로

EXPORT_DIR  = os.getenv(
    "IGA_CLEAN_EXPORT_DIR",
    os.path.join(REPO_ROOT, "training", "image_classification", "outputs", "iga_clean_dedup"),
)
MODELS_DIR  = os.path.join(EXPORT_DIR, "models")
RESULTS_DIR = os.path.join(EXPORT_DIR, "results")

# ══════════════════════════════════════════════
# 학습 설정 (train_iga_grouped_final.py와 동일)
# ══════════════════════════════════════════════
MODEL_NAME      = "tf_efficientnetv2_s"
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_EPOCHS      = 30
LEARNING_RATE   = 5e-4
WEIGHT_DECAY    = 1e-2
NUM_WORKERS     = 0
SEED            = 42
LABEL_SMOOTHING = 0.1
LABEL_NAMES     = ["mild_or_below", "moderate_severe"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rel_to_abs(rel: str) -> str:
    return os.path.join(IGA_DATA_ROOT, rel.replace("/", os.sep))


def clean_key(k):
    return k.strip().lstrip('﻿').strip('"').strip("'")


# ══════════════════════════════════════════════
# 데이터 로드 (CSV에서 직접)
# ══════════════════════════════════════════════
def load_split_csv(csv_path: str):
    train_rows, val_rows, test_rows = [], [], []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        raw_headers = next(reader)
        headers = [clean_key(h) for h in raw_headers]
        for line in reader:
            r = dict(zip(headers, line))
            r['img_path'] = rel_to_abs(r['relative_path'])
            r['label']    = int(r['label'])
            if   r['split'] == 'train': train_rows.append(r)
            elif r['split'] == 'val':   val_rows.append(r)
            elif r['split'] == 'test':  test_rows.append(r)
    return train_rows, val_rows, test_rows


# ══════════════════════════════════════════════
# Dataset / Transform (train_iga_grouped_final.py와 동일)
# ══════════════════════════════════════════════
def get_train_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class SkinDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows      = rows
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r   = self.rows[idx]
        img = Image.open(r['img_path']).convert("RGB")
        return self.transform(img), r['label']


# ══════════════════════════════════════════════
# 학습 루프 (train_iga_grouped_final.py와 동일)
# ══════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(loader, desc="  Train", ncols=80, leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(lbls).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out  = model(imgs)
            loss = criterion(out, lbls)
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(lbls).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


def inference(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            prbs = torch.softmax(out, dim=1)
            all_labels.extend(lbls.numpy())
            all_probs.extend(prbs[:, 1].cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


# ══════════════════════════════════════════════
# 메트릭
# ══════════════════════════════════════════════
def compute_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="weighted")
    auc   = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens  = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec  = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec  = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return {
        "threshold":   float(threshold),
        "n":           int(len(labels)),
        "accuracy":    round(float(acc), 4),
        "f1_weighted": round(float(f1), 4),
        "roc_auc":     round(float(auc), 4),
        "pr_auc":      round(float(pr_auc), 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision":   round(prec, 4),
        "cm":          cm.tolist(),
    }


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    set_seed(SEED)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 데이터 로드 ──
    print(f"\n[1] Clean split CSV 로드: {CLEAN_CSV}")
    train_rows, val_rows, test_rows = load_split_csv(CLEAN_CSV)

    print(f"  Train: {len(train_rows)}")
    print(f"  Val:   {len(val_rows)}")
    print(f"  Test:  {len(test_rows)}")

    for sp_name, rows in [("Train", train_rows), ("Val", val_rows), ("Test", test_rows)]:
        n1  = sum(1 for r in rows if r['label'] == 1)
        n0  = len(rows) - n1
        bids = set(r['base_id'] for r in rows)
        print(f"  {sp_name:<6}: {len(rows):>5}장  base_id {len(bids):>4}  "
              f"mild {n0} ({n0/len(rows)*100:.1f}%)  mod-sev {n1} ({n1/len(rows)*100:.1f}%)")

    # base_id overlap 최종 확인
    train_bids = set(r['base_id'] for r in train_rows)
    val_bids   = set(r['base_id'] for r in val_rows)
    test_bids  = set(r['base_id'] for r in test_rows)
    assert not (train_bids & val_bids),  "train/val base_id 중복"
    assert not (train_bids & test_bids), "train/test base_id 중복"
    assert not (val_bids   & test_bids), "val/test base_id 중복"

    train_shas = set(r['sha256'] for r in train_rows)
    val_shas   = set(r['sha256'] for r in val_rows)
    test_shas  = set(r['sha256'] for r in test_rows)
    assert not (train_shas & val_shas),  "train/val sha256 중복"
    assert not (train_shas & test_shas), "train/test sha256 중복"
    assert not (val_shas   & test_shas), "val/test sha256 중복"
    print("\n  overlap assertion all passed ✓")

    # ── class weight (train 기준) ──
    n_mild   = sum(1 for r in train_rows if r['label'] == 0)
    n_modsev = sum(1 for r in train_rows if r['label'] == 1)
    w0 = n_modsev / n_mild if n_mild > 0 else 1.0
    class_weights = torch.tensor([w0, 1.0], dtype=torch.float).to(device)
    print(f"\n  class weight: mild={w0:.4f}  mod-sev=1.0")

    # ── DataLoader ──
    train_loader = DataLoader(
        SkinDataset(train_rows, get_train_transform(IMG_SIZE)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        SkinDataset(val_rows, get_eval_transform(IMG_SIZE)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        SkinDataset(test_rows, get_eval_transform(IMG_SIZE)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    # ── 모델 / optimizer / scheduler ──
    model     = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    ckpt_path    = os.path.join(MODELS_DIR, "best_iga_clean_dedup.pth")
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n[2] 학습 ({NUM_EPOCHS} epochs)  checkpoint → {ckpt_path}")
    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:2d}/{NUM_EPOCHS}  "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"  Best val acc: {best_val_acc:.4f}")

    # ── 학습 곡선 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_x = range(1, NUM_EPOCHS + 1)
    axes[0].plot(epochs_x, history["train_loss"], label="train")
    axes[0].plot(epochs_x, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs_x, history["train_acc"],  label="train")
    axes[1].plot(epochs_x, history["val_acc"],    label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curve.png"), dpi=150)
    plt.close()

    # ── 평가 ──
    print("\n[3] 평가")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # Validation threshold (Youden's J)
    val_labels, val_probs = inference(model, val_loader, device)
    fpr_v, tpr_v, thrs_v = roc_curve(val_labels, val_probs)
    youden_j    = tpr_v - fpr_v
    best_idx_y  = int(np.argmax(youden_j))
    val_thr     = float(thrs_v[best_idx_y])

    f1_scores_v = [f1_score(val_labels, (val_probs >= t).astype(int), average="weighted")
                   for t in thrs_v]
    best_idx_f1 = int(np.argmax(f1_scores_v))
    val_thr_f1  = float(thrs_v[best_idx_f1])

    print(f"  Validation Youden's J threshold: {val_thr:.4f}")
    print(f"  Validation F1-optimal threshold: {val_thr_f1:.4f}")

    # threshold search CSV
    thr_csv = os.path.join(EXPORT_DIR, "validation_threshold_search.csv")
    with open(thr_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold","tpr","fpr","youden_j","f1_weighted"])
        w.writeheader()
        for i, t in enumerate(thrs_v):
            f1v = f1_score(val_labels, (val_probs >= t).astype(int), average="weighted")
            w.writerow({"threshold": round(float(t), 6),
                        "tpr":       round(float(tpr_v[i]), 6),
                        "fpr":       round(float(fpr_v[i]), 6),
                        "youden_j":  round(float(youden_j[i]), 6),
                        "f1_weighted": round(float(f1v), 6)})

    # Test 평가 (단 1회, 재조정 없음)
    test_labels, test_probs = inference(model, test_loader, device)
    m05      = compute_metrics(test_labels, test_probs, 0.5)
    m_val    = compute_metrics(test_labels, test_probs, val_thr)
    m_val_f1 = compute_metrics(test_labels, test_probs, val_thr_f1)

    print(f"\n  Test (thr=0.5)  : Acc={m05['accuracy']:.4f}  F1={m05['f1_weighted']:.4f}"
          f"  AUC={m05['roc_auc']:.4f}  Sens={m05['sensitivity']:.4f}  Spec={m05['specificity']:.4f}")
    print(f"  Test (thr=Youden): Acc={m_val['accuracy']:.4f}  F1={m_val['f1_weighted']:.4f}"
          f"  AUC={m_val['roc_auc']:.4f}  Sens={m_val['sensitivity']:.4f}  Spec={m_val['specificity']:.4f}")
    print(f"  Test (thr=F1opt) : Acc={m_val_f1['accuracy']:.4f}  F1={m_val_f1['f1_weighted']:.4f}"
          f"  AUC={m_val_f1['roc_auc']:.4f}  Sens={m_val_f1['sensitivity']:.4f}  Spec={m_val_f1['specificity']:.4f}")

    # ROC curve
    fpr_t, tpr_t, _ = roc_curve(test_labels, test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_t, tpr_t, label=f"Test AUC={m05['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (Test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # Confusion matrix (val threshold)
    preds_val_thr = (test_probs >= val_thr).astype(int)
    cm_img = confusion_matrix(test_labels, preds_val_thr)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_img, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Confusion Matrix (thr={val_thr:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cm_thr_val.png"), dpi=150)
    plt.close()

    # ── final_metrics.json ──
    final = {
        "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":                 MODEL_NAME,
        "checkpoint":            ckpt_path,
        "best_val_acc":          round(best_val_acc, 4),
        "val_threshold_youden":  round(val_thr, 6),
        "val_threshold_f1":      round(val_thr_f1, 6),
        "split": {
            "train": len(train_rows),
            "val":   len(val_rows),
            "test":  len(test_rows),
        },
        "grade_distribution": {
            "train": {
                "mild_or_below":   sum(1 for r in train_rows if r['label'] == 0),
                "moderate_severe": sum(1 for r in train_rows if r['label'] == 1),
            },
            "val": {
                "mild_or_below":   sum(1 for r in val_rows if r['label'] == 0),
                "moderate_severe": sum(1 for r in val_rows if r['label'] == 1),
            },
            "test": {
                "mild_or_below":   sum(1 for r in test_rows if r['label'] == 0),
                "moderate_severe": sum(1 for r in test_rows if r['label'] == 1),
            },
        },
        "test_thr_0.5":          m05,
        f"test_thr_{val_thr:.4f}": m_val,
        f"test_thr_{val_thr_f1:.4f}_f1opt": m_val_f1,
        "comparison_historical": {
            "original_IGA_image_level_random_split": {
                "note": "image2_model.py — image-level random split, test로 threshold 선택",
                "threshold": 0.38,
                "accuracy":    0.8389,
                "f1_weighted": 0.8368,
                "roc_auc":     0.8758,
                "sensitivity": 0.9058,
                "specificity": 0.6190,
                "split": "image-level random",
                "n_test": "unknown",
            },
            "corrected_base_id_grouped_split": {
                "note": "train_iga_grouped_final.py — base_id group split (leakage 있음)",
                "val_threshold_youden": 0.6438,
                "test_thr_0.5": {
                    "accuracy": 0.8398, "f1_weighted": 0.8474,
                    "roc_auc": 0.9247,
                },
                "split": "base_id group-preserving, n_train=1263 n_val=356 n_test=181",
                "note2": "cross-split SHA duplicate 4그룹 포함됨",
            },
            "clean_dedup_this_run": {
                "note": "train_iga_clean_dedup.py — base_id+sha256 connected component split",
                "val_threshold_youden": round(val_thr, 6),
                "test_thr_0.5": m05,
                f"test_thr_{val_thr:.4f}": m_val,
                "split": f"dedup n_train={len(train_rows)} n_val={len(val_rows)} n_test={len(test_rows)}",
            },
        },
    }
    metrics_path = os.path.join(RESULTS_DIR, "final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n[완료] 출력: {EXPORT_DIR}")
    print(f"  checkpoint:     {ckpt_path}")
    print(f"  final_metrics:  {metrics_path}")
    print("\n  ⚠ 배포 가중치(app/best_iga_model.pth) 교체, README 수정, commit/push는 미수행.")


if __name__ == "__main__":
    main()
