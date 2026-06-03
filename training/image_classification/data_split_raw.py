"""
피부질환 6-class 분류기: 학습 + 평가 통합 파이프라인

데이터 구성:
- AI Hub Training(9,600장) + Validation(1,200장) 합친 뒤 7:2:1 재분할
  → Train 7,560장 / Val 2,160장 / Test 1,080장
- 외부 평가: DermNet 실제 이미지 265장 (5-class, normal 제외)

평가 구조:
- Val: 학습 중 모니터링 (best model 선택)
- Test (내부): 합성 데이터 최종 성능
- Test (외부): DermNet 실제 이미지 → 도메인 갭 측정

사용법:
    python train_and_evaluate.py

필요 패키지: torch, torchvision, timm, pandas, scikit-learn, matplotlib, seaborn, tqdm
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
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm


# ══════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════
class Config:
    # ── 경로 (네 환경에 맞게 수정) ──
    TRAIN_ROOT = r"E:\atopic\AI Hub\Training"
    VAL_ROOT = r"E:\atopic\AI Hub\Validation"
    DERMNET_ROOT = r"E:\atopic\dermnet_images"
    OUTPUT_DIR = r"E:\atopic\models"

    # ── 학습 하이퍼파라미터 ──
    MODEL_NAME = "efficientnet_b0"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 20           # 30에서 20으로 줄여보세요.
    LEARNING_RATE = 5e-4      # 1e-3에서 절반으로 낮춰보세요. (안정적 학습)
    WEIGHT_DECAY = 1e-2       # 1e-4에서 대폭 올려보세요. (과적합 강하게 방지)
    NUM_WORKERS = 4
    SEED = 42

    # ── Split 비율 ──
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    # ── 6-class 라벨 매핑 (한글 → 숫자) ──
    LABEL_MAP = {
        "정상": 0,
        "아토피": 1,
        "건선": 2,
        "여드름": 3,
        "주사": 4,
        "지루": 5,
    }

    LABEL_NAMES = ["normal", "atopy", "psoriasis", "acne", "rosacea", "seborrheic"]

    # ── DermNet 폴더 → 라벨 매핑 ──
    DERMNET_MAP = {
        "atopic_dermatitis": 1,
        "psoriasis": 2,
        "acne": 3,
        "rosacea": 4,
        "seborrheic_dermatitis": 5,
    }


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
# 데이터 수집 + 7:2:1 Split
# ══════════════════════════════════════════════
def collect_all_samples(roots, label_map):
    """여러 루트 폴더에서 (파일경로, 라벨) 리스트 수집"""
    samples = []
    for root_dir in roots:
        if not os.path.exists(root_dir):
            print(f"  [WARNING] 경로 없음: {root_dir}")
            continue
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            parts = folder_name.split("_")
            if len(parts) < 3:
                continue

            disease_name = parts[1]
            if disease_name not in label_map:
                continue

            label = label_map[disease_name]

            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    fpath = os.path.join(folder_path, fname)
                    samples.append((fpath, label))

    return samples


def split_dataset(samples, train_ratio, val_ratio, test_ratio, seed):
    """stratified 7:2:1 split"""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # train vs (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    # val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))

    return train_samples, val_samples, test_samples


# ══════════════════════════════════════════════
# 데이터셋 클래스
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


class DermNetDataset(Dataset):
    def __init__(self, root_dir, dermnet_map, transform=None):
        self.transform = transform
        self.samples = []

        for folder_name, label in dermnet_map.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"  [WARNING] 폴더 없음: {folder_path}")
                continue

            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    fpath = os.path.join(folder_path, fname)
                    self.samples.append((fpath, label))

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
            # 1. 더 과감한 크롭 (병변이 작게 찍힌 경우 대비)
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30), # 회전각 확대
            # 2. 색감 변형 강화 (DermNet의 다양한 조명 대비)
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            # 3. 화질 저하 시뮬레이션 (DermNet의 노이즈 대비)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        # 평가 시에도 Resize만 하기보다 CenterCrop을 섞는 것이 안정적일 수 있음
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ══════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════
def build_model(model_name, num_classes, pretrained=True):
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

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {title}", ncols=80, leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    unique_labels = sorted(set(all_labels.tolist() + all_preds.tolist()))
    target_names = [label_names[i] for i in unique_labels]

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names)}")

    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    return cm, unique_labels, target_names, acc, f1


def plot_confusion_matrix(cm, target_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → 저장: {save_path}")


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
    all_samples = collect_all_samples([cfg.TRAIN_ROOT, cfg.VAL_ROOT], cfg.LABEL_MAP)
    print(f"  전체 AI Hub 데이터: {len(all_samples):,d}장")

    label_counts = Counter([s[1] for s in all_samples])
    for label_idx in sorted(label_counts):
        print(f"    {cfg.LABEL_NAMES[label_idx]:12s}: {label_counts[label_idx]:,d}장")

    # ── 2. 7:2:1 Split ──
    print(f"\n[2/5] 7:2:1 Stratified Split")
    train_samples, val_samples, test_samples = split_dataset(
        all_samples, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.TEST_RATIO, cfg.SEED
    )
    print(f"  Train: {len(train_samples):,d}장")
    print(f"  Val:   {len(val_samples):,d}장")
    print(f"  Test:  {len(test_samples):,d}장")

    # 데이터셋 + 로더 생성
    train_dataset = SkinDataset(train_samples, transform=get_transforms(cfg.IMG_SIZE, is_train=True))
    val_dataset = SkinDataset(val_samples, transform=get_transforms(cfg.IMG_SIZE, is_train=False))
    test_dataset = SkinDataset(test_samples, transform=get_transforms(cfg.IMG_SIZE, is_train=False))
    dermnet_dataset = DermNetDataset(
        cfg.DERMNET_ROOT, cfg.DERMNET_MAP,
        transform=get_transforms(cfg.IMG_SIZE, is_train=False),
    )
    print(f"  DermNet (외부): {len(dermnet_dataset):,d}장")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    dermnet_loader = DataLoader(dermnet_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # ── 3. 모델 ──
    print(f"\n[3/5] 모델 초기화: {cfg.MODEL_NAME}")
    model = build_model(cfg.MODEL_NAME, num_classes=6, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    # ── 4. 학습 ──
    print(f"\n[4/5] 학습 시작 (Epochs: {cfg.NUM_EPOCHS})")
    best_val_acc = 0.0
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_6class_model.pth")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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
    cm_test, _, names_test, acc_test, f1_test = evaluate_detailed(
        model, test_loader, device, cfg.LABEL_NAMES,
        title="내부 평가 (합성 Test Set)"
    )
    plot_confusion_matrix(
        cm_test, names_test, "Internal Test (Synthetic)",
        os.path.join(cfg.OUTPUT_DIR, "cm_internal_test.png"),
    )

    # 외부 DermNet
    cm_ext, _, names_ext, acc_ext, f1_ext = evaluate_detailed(
        model, dermnet_loader, device, cfg.LABEL_NAMES,
        title="외부 평가 (DermNet 실제 이미지)"
    )
    plot_confusion_matrix(
        cm_ext, names_ext, "External Test (DermNet Real Images)",
        os.path.join(cfg.OUTPUT_DIR, "cm_external_test.png"),
    )

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print(f"  최종 결과 요약")
    print(f"{'='*60}")
    print(f"  Train:  {len(train_samples):,d}장 (70%)")
    print(f"  Val:    {len(val_samples):,d}장 (20%)")
    print(f"  Test:   {len(test_samples):,d}장 (10%)")
    print(f"  DermNet:{len(dermnet_dataset):,d}장 (외부)")
    print(f"")
    print(f"  Best Val Acc:             {best_val_acc:.4f}")
    print(f"  내부 Test:  Acc={acc_test:.4f}  F1={f1_test:.4f}")
    print(f"  외부 Test:  Acc={acc_ext:.4f}  F1={f1_ext:.4f}")
    print(f"  도메인 갭:  {acc_test - acc_ext:.4f}")
    print(f"\n  모델: {best_model_path}")

    # JSON 저장
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg.MODEL_NAME,
        "epochs": cfg.NUM_EPOCHS,
        "data": {
            "total": len(all_samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "dermnet": len(dermnet_dataset),
        },
        "best_val_acc": best_val_acc,
        "internal_test": {"accuracy": acc_test, "f1_weighted": f1_test},
        "external_test": {"accuracy": acc_ext, "f1_weighted": f1_ext},
        "domain_gap": acc_test - acc_ext,
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()