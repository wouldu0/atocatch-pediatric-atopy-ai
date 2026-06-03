"""
Model 2-A: 모델 아키텍처 비교 실험
- 동일 데이터/설정에서 모델만 변경하여 비교
- DermNet 믹싱 포함

사용법:
    python model_comparison.py
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm


# ══════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════
class Config:
    TRAIN_ROOT = r"E:\atopic\AIHub\Training\01.원천데이터"
    VAL_ROOT = r"E:\atopic\AIHub\Validation\01.원천데이터"
    DERMNET_ROOT = r"E:\atopic\dermnet_images"
    OUTPUT_DIR = r"E:\atopic\models\comparison"

    # ── 비교할 모델 목록 ──
    MODELS = [
        "efficientnet_b0",      # 현재 베이스라인
        "tf_efficientnetv2_s",     # 지난번 최종 모델
        "efficientnet_b3",
        "resnet50",
        "convnext_tiny",
    ]

    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4
    SEED = 42
    LABEL_SMOOTHING = 0.1

    NON_ATOPY_COMPOSITION = {
        "정상": 360,
        "건선": 360,
        "여드름": 360,
        "주사": 360,
        "지루": 360,
    }

    LABEL_NAMES = ["non_atopy", "atopy"]
    DERMNET_TRAIN_RATIO = 0.6
    DERMNET_ATOPY = ["atopic_dermatitis"]
    DERMNET_NON_ATOPY = ["psoriasis", "acne", "rosacea", "seborrheic_dermatitis"]


# ══════════════════════════════════════════════
# 시드 고정
# ══════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════
# 데이터 수집 (이전과 동일)
# ══════════════════════════════════════════════
def collect_aihub_samples(roots):
    disease_files = {}
    for root_dir in roots:
        if not os.path.exists(root_dir):
            continue
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            parts = folder_name.split("_")
            if len(parts) < 3:
                continue
            disease = parts[1]
            if disease not in disease_files:
                disease_files[disease] = []
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    disease_files[disease].append(os.path.join(folder_path, fname))
    return disease_files


def build_aihub_binary(disease_files, non_atopy_comp, seed):
    samples = []
    rng = random.Random(seed)
    for f in disease_files["아토피"]:
        samples.append((f, 1))
    for disease, n in non_atopy_comp.items():
        files = disease_files.get(disease, [])
        selected = rng.sample(files, min(n, len(files)))
        for f in selected:
            samples.append((f, 0))
    return samples


def collect_dermnet_split(root_dir, atopy_folders, non_atopy_folders, train_ratio, seed):
    rng = random.Random(seed)
    dermnet_train, dermnet_test = [], []

    for folder in atopy_folders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        rng.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        dermnet_train.extend([(f, 1) for f in files[:split_idx]])
        dermnet_test.extend([(f, 1) for f in files[split_idx:]])

    for folder in non_atopy_folders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        rng.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        dermnet_train.extend([(f, 0) for f in files[:split_idx]])
        dermnet_test.extend([(f, 0) for f in files[split_idx:]])

    return dermnet_train, dermnet_test


# ══════════════════════════════════════════════
# 데이터셋 / Transform
# ══════════════════════════════════════════════
class SkinDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size, is_train=True):
    if is_train:
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
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ══════════════════════════════════════════════
# 학습 / 검증 / 평가
# ══════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="    Train", ncols=80, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="    Val  ", ncols=80, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    # sensitivity / specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "acc": acc, "f1": f1, "auc": auc,
        "sensitivity": sensitivity, "specificity": specificity,
        "cm": cm, "labels": all_labels, "probs": all_probs,
    }


# ══════════════════════════════════════════════
# 단일 모델 학습 + 평가
# ══════════════════════════════════════════════
def run_experiment(model_name, train_loader, val_loader, test_loader, dermnet_loader,
                   cfg, device, output_dir):
    print(f"\n{'='*60}")
    print(f"  모델: {model_name}")
    print(f"{'='*60}")

    set_seed(cfg.SEED)
    model = timm.create_model(model_name, pretrained=True, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    best_val_acc = 0.0
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, "best_model.pth")

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{cfg.NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"  Best Val Acc: {best_val_acc:.4f}")

    # 평가
    model.load_state_dict(torch.load(best_path, weights_only=True))

    internal = evaluate(model, test_loader, device)
    external = evaluate(model, dermnet_loader, device)

    print(f"  내부 Test: Acc={internal['acc']:.4f}  F1={internal['f1']:.4f}  AUC={internal['auc']:.4f}")
    print(f"  외부 Test: Acc={external['acc']:.4f}  F1={external['f1']:.4f}  AUC={external['auc']:.4f}")
    print(f"  외부 Sensitivity={external['sensitivity']:.4f}  Specificity={external['specificity']:.4f}")

    # Confusion matrix 저장
    for name, result in [("internal", internal), ("external", external)]:
        plt.figure(figsize=(6, 5))
        sns.heatmap(result["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=cfg.LABEL_NAMES, yticklabels=cfg.LABEL_NAMES)
        plt.title(f"{model_name} - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"cm_{name}.png"), dpi=150)
        plt.close()

    return {
        "model": model_name,
        "best_val_acc": best_val_acc,
        "internal": {k: v for k, v in internal.items() if k not in ("cm", "labels", "probs")},
        "external": {k: v for k, v in external.items() if k not in ("cm", "labels", "probs")},
    }


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    cfg = Config()
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 데이터 준비 (모든 모델이 동일한 데이터 사용) ──
    print("\n[데이터 준비]")
    disease_files = collect_aihub_samples([cfg.TRAIN_ROOT, cfg.VAL_ROOT])
    aihub_samples = build_aihub_binary(disease_files, cfg.NON_ATOPY_COMPOSITION, cfg.SEED)

    paths = [s[0] for s in aihub_samples]
    labels = [s[1] for s in aihub_samples]

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=cfg.SEED
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=1/3, stratify=temp_l, random_state=cfg.SEED
    )

    dermnet_train, dermnet_test = collect_dermnet_split(
        cfg.DERMNET_ROOT, cfg.DERMNET_ATOPY, cfg.DERMNET_NON_ATOPY,
        cfg.DERMNET_TRAIN_RATIO, cfg.SEED
    )

    train_samples = list(zip(train_p, train_l)) + dermnet_train
    random.shuffle(train_samples)
    val_samples = list(zip(val_p, val_l))
    test_samples = list(zip(test_p, test_l))

    print(f"  Train: {len(train_samples)}장 (AI Hub {len(train_p)} + DermNet {len(dermnet_train)})")
    print(f"  Val:   {len(val_samples)}장")
    print(f"  Test 내부: {len(test_samples)}장")
    print(f"  Test 외부: {len(dermnet_test)}장")

    train_loader = DataLoader(
        SkinDataset(train_samples, get_transforms(cfg.IMG_SIZE, True)),
        batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(
        SkinDataset(val_samples, get_transforms(cfg.IMG_SIZE, False)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        SkinDataset(test_samples, get_transforms(cfg.IMG_SIZE, False)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    dermnet_loader = DataLoader(
        SkinDataset(dermnet_test, get_transforms(cfg.IMG_SIZE, False)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # ── 모델 순차 실험 ──
    all_results = []
    for model_name in cfg.MODELS:
        result = run_experiment(
            model_name, train_loader, val_loader, test_loader, dermnet_loader,
            cfg, device, cfg.OUTPUT_DIR
        )
        all_results.append(result)

    # ── 비교 테이블 출력 ──
    print(f"\n{'='*80}")
    print(f"  모델 비교 결과")
    print(f"{'='*80}")
    print(f"{'모델':25s} | {'Val Acc':>8s} | {'내부 Acc':>8s} | {'외부 Acc':>8s} | {'외부 AUC':>8s} | {'Sens':>6s} | {'Spec':>6s}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['model']:25s} | {r['best_val_acc']:8.4f} | {r['internal']['acc']:8.4f} | "
              f"{r['external']['acc']:8.4f} | {r['external']['auc']:8.4f} | "
              f"{r['external']['sensitivity']:6.4f} | {r['external']['specificity']:6.4f}")

    # ── 비교 차트 ──
    models = [r["model"] for r in all_results]
    ext_acc = [r["external"]["acc"] for r in all_results]
    ext_auc = [r["external"]["auc"] for r in all_results]
    ext_sens = [r["external"]["sensitivity"] for r in all_results]
    ext_spec = [r["external"]["specificity"] for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(models, ext_acc, color="steelblue")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("External Accuracy")
    axes[0].set_xlim(0, 1)

    axes[1].barh(models, ext_auc, color="coral")
    axes[1].set_xlabel("AUC")
    axes[1].set_title("External AUC")
    axes[1].set_xlim(0, 1)

    x = np.arange(len(models))
    width = 0.35
    axes[2].barh(x - width/2, ext_sens, width, label="Sensitivity", color="mediumseagreen")
    axes[2].barh(x + width/2, ext_spec, width, label="Specificity", color="mediumpurple")
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(models)
    axes[2].set_xlabel("Score")
    axes[2].set_title("Sensitivity vs Specificity")
    axes[2].set_xlim(0, 1)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"\n  비교 차트 저장: {os.path.join(cfg.OUTPUT_DIR, 'model_comparison.png')}")

    # JSON
    with open(os.path.join(cfg.OUTPUT_DIR, "comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()