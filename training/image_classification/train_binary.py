"""
Model 2-A: 아토피 vs 비아토피 이진 분류기

데이터 구성:
- 아토피: 1,800장 (전체)
- 비아토피: 1,800장 (정상 1,200 + 건선 150 + 여드름 150 + 주사 150 + 지루 150)
- 총 3,600장 → 7:2:1 stratified split

평가:
- 내부: 합성 Test set
- 외부: DermNet 실제 이미지 (아토피 vs 비아토피)

사용법:
    python train_atopy_binary.py
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
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm


# ══════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════
class Config:
    # ── 경로 ──
    TRAIN_ROOT = r"E:\atopic\AIHub\Training\01.원천데이터"
    VAL_ROOT = r"E:\atopic\AIHub\Validation\01.원천데이터"
    DERMNET_ROOT = r"E:\atopic\dermnet_images"
    OUTPUT_DIR = r"E:\atopic\models\binary"

    # ── 학습 ──
    MODEL_NAME = "efficientnet_b0"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4
    SEED = 42
    LABEL_SMOOTHING = 0.1

    # ── Split ──
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    # ── 비아토피 구성 (언더샘플링 장수) ──
    NON_ATOPY_COMPOSITION = {
        "정상": 360,
        "건선": 360,
        "여드름": 360,
        "주사": 360,
        "지루": 360,
    }

    # ── 이진 라벨 ──
    # 0 = 비아토피, 1 = 아토피
    LABEL_NAMES = ["non_atopy", "atopy"]

    # ── DermNet 매핑 ──
    DERMNET_ATOPY = ["atopic_dermatitis"]  # → 1
    DERMNET_NON_ATOPY = ["psoriasis", "acne", "rosacea", "seborrheic_dermatitis"]  # → 0


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
# 데이터 수집
# ══════════════════════════════════════════════
def collect_samples(roots):
    """폴더에서 (파일경로, 질환명) 수집"""
    disease_files = {}  # {질환명: [파일경로, ...]}

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


def build_binary_dataset(disease_files, non_atopy_comp, seed):
    """아토피 vs 비아토피 이진 데이터셋 구성"""
    samples = []

    # 아토피: 전부 사용
    atopy_files = disease_files.get("아토피", [])
    for f in atopy_files:
        samples.append((f, 1))
    print(f"  아토피: {len(atopy_files)}장 (label=1)")

    # 비아토피: 질환별 언더샘플링
    rng = random.Random(seed)
    total_non = 0
    for disease, n in non_atopy_comp.items():
        files = disease_files.get(disease, [])
        if len(files) < n:
            print(f"  [WARNING] {disease}: {len(files)}장밖에 없음 (요청: {n}장)")
            selected = files
        else:
            selected = rng.sample(files, n)
        for f in selected:
            samples.append((f, 0))
        total_non += len(selected)
        print(f"  {disease}: {len(selected)}장 (label=0)")

    print(f"  비아토피 합계: {total_non}장")
    return samples


# ══════════════════════════════════════════════
# 데이터셋
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


class DermNetBinaryDataset(Dataset):
    def __init__(self, root_dir, atopy_folders, non_atopy_folders, transform=None):
        self.transform = transform
        self.samples = []

        for folder in atopy_folders:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append((os.path.join(folder_path, fname), 1))

        for folder in non_atopy_folders:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append((os.path.join(folder_path, fname), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ══════════════════════════════════════════════
# Transform
# ══════════════════════════════════════════════
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
# 모델
# ══════════════════════════════════════════════
def build_model(model_name, num_classes=2, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


# ══════════════════════════════════════════════
# 학습 / 검증
# ══════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", ncols=80, leave=False):
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
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", ncols=80, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


# ══════════════════════════════════════════════
# 상세 평가
# ══════════════════════════════════════════════
def evaluate_detailed(model, loader, device, label_names, title="Evaluation"):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {title}", ncols=80, leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 아토피 확률

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # AUC 계산
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=label_names)}")

    cm = confusion_matrix(all_labels, all_preds)
    return cm, acc, f1, auc, all_labels, all_probs


def plot_confusion_matrix(cm, label_names, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, probs, title, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


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

    # ── 1. 데이터 수집 ──
    print("\n[1/5] 데이터 수집")
    disease_files = collect_samples([cfg.TRAIN_ROOT, cfg.VAL_ROOT])
    for disease, files in sorted(disease_files.items()):
        print(f"  {disease}: {len(files)}장 (전체)")

    # ── 2. 이진 데이터셋 구성 ──
    print(f"\n[2/5] 이진 데이터셋 구성 (아토피 vs 비아토피)")
    all_samples = build_binary_dataset(disease_files, cfg.NON_ATOPY_COMPOSITION, cfg.SEED)

    label_counts = Counter([s[1] for s in all_samples])
    print(f"\n  최종: 아토피={label_counts[1]}장, 비아토피={label_counts[0]}장")

    # ── 3. 7:2:1 Split ──
    print(f"\n[3/5] 7:2:1 Stratified Split")
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=cfg.SEED
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=1/3, stratify=temp_l, random_state=cfg.SEED
    )

    train_samples = list(zip(train_p, train_l))
    val_samples = list(zip(val_p, val_l))
    test_samples = list(zip(test_p, test_l))

    print(f"  Train: {len(train_samples)}장")
    print(f"  Val:   {len(val_samples)}장")
    print(f"  Test:  {len(test_samples)}장")

    # 데이터셋 + 로더
    train_dataset = SkinDataset(train_samples, transform=get_transforms(cfg.IMG_SIZE, is_train=True))
    val_dataset = SkinDataset(val_samples, transform=get_transforms(cfg.IMG_SIZE, is_train=False))
    test_dataset = SkinDataset(test_samples, transform=get_transforms(cfg.IMG_SIZE, is_train=False))

    dermnet_dataset = DermNetBinaryDataset(
        cfg.DERMNET_ROOT, cfg.DERMNET_ATOPY, cfg.DERMNET_NON_ATOPY,
        transform=get_transforms(cfg.IMG_SIZE, is_train=False),
    )
    print(f"  DermNet: {len(dermnet_dataset)}장")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    dermnet_loader = DataLoader(dermnet_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # ── 4. 모델 + 학습 ──
    print(f"\n[4/5] 모델: {cfg.MODEL_NAME}")
    model = build_model(cfg.MODEL_NAME, num_classes=2, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    best_val_acc = 0.0
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_binary_model.pth")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n  학습 시작 (Epochs: {cfg.NUM_EPOCHS})")
    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{cfg.NUM_EPOCHS}]  lr={optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ★ Best model 저장 (Val Acc: {val_acc:.4f})")

    # 학습 곡선
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "training_curve.png"), dpi=150)
    plt.close()

    # ── 5. 평가 ──
    print(f"\n[5/5] 평가")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # 내부 Test
    cm_test, acc_test, f1_test, auc_test, labels_test, probs_test = evaluate_detailed(
        model, test_loader, device, cfg.LABEL_NAMES,
        title="내부 평가 (합성 Test Set)"
    )
    plot_confusion_matrix(cm_test, cfg.LABEL_NAMES,
        "Internal Test - Atopy Binary", os.path.join(cfg.OUTPUT_DIR, "cm_internal.png"))
    plot_roc_curve(labels_test, probs_test,
        "ROC - Internal Test", os.path.join(cfg.OUTPUT_DIR, "roc_internal.png"))

    # 외부 DermNet
    cm_ext, acc_ext, f1_ext, auc_ext, labels_ext, probs_ext = evaluate_detailed(
        model, dermnet_loader, device, cfg.LABEL_NAMES,
        title="외부 평가 (DermNet)"
    )
    plot_confusion_matrix(cm_ext, cfg.LABEL_NAMES,
        "External Test - DermNet", os.path.join(cfg.OUTPUT_DIR, "cm_external.png"))
    if auc_ext > 0:
        plot_roc_curve(labels_ext, probs_ext,
            "ROC - External Test (DermNet)", os.path.join(cfg.OUTPUT_DIR, "roc_external.png"))

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print(f"  Model 2-A 결과 요약 (아토피 vs 비아토피)")
    print(f"{'='*60}")
    print(f"  데이터: 아토피 {label_counts[1]}장 vs 비아토피 {label_counts[0]}장")
    print(f"  Train/Val/Test: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")
    print(f"  DermNet: {len(dermnet_dataset)}장")
    print(f"")
    print(f"  내부 Test:  Acc={acc_test:.4f}  F1={f1_test:.4f}  AUC={auc_test:.4f}")
    print(f"  외부 Test:  Acc={acc_ext:.4f}  F1={f1_ext:.4f}  AUC={auc_ext:.4f}")
    print(f"  도메인 갭:  Acc {acc_test - acc_ext:.4f}  AUC {auc_test - auc_ext:.4f}")
    print(f"\n  모델: {best_model_path}")

    # JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "task": "binary_atopy_classification",
        "model": cfg.MODEL_NAME,
        "data": {
            "atopy": label_counts[1],
            "non_atopy": label_counts[0],
            "non_atopy_composition": cfg.NON_ATOPY_COMPOSITION,
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "dermnet": len(dermnet_dataset),
        },
        "internal_test": {"accuracy": acc_test, "f1": f1_test, "auc": auc_test},
        "external_test": {"accuracy": acc_ext, "f1": f1_ext, "auc": auc_ext},
        "domain_gap": {"accuracy": acc_test - acc_ext, "auc": auc_test - auc_ext},
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()