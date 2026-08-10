"""
Model 2-B: IGA 중증도 이진 분류기
- mild 이하 (Clear + Almost Clear + Mild) vs moderate-severe (Moderate + Severe)
- 모델: tf_EfficientNetV2-S (Model 2-A와 동일)
- 불균형 처리: Class Weight

사용법:
    python train_iga_severity.py

── 출처 검증 (포트폴리오 정리 중 추가) ──────────────────────
배포된 best_iga_model.pth의 학습 스크립트를 찾지 못해 README에 "학습 스크립트
없음"으로 명시해뒀었는데, 별도 로컬 백업에서 이 스크립트를 다시 발견했다.
아래 근거로 실제 배포 모델의 학습 스크립트임을 확인:
  - 산출물 경로(E:\atopic\models\iga_severity)에 남아있던 model_config.json의
    성능 수치(accuracy 0.8444, f1 0.8431, auc 0.8758, sensitivity 0.9058,
    specificity 0.6429)가 app/model_config2.json과 정확히 일치
  - 아키텍처(tf_efficientnetv2_s, num_classes=2)가 best_iga_model.pth의
    strict state_dict load 결과와 일치
  - eval_iga_threshold_search.py / eval_iga_final.py가 이 스크립트와 동일한
    SEED=42, 7:2:1 stratified split을 재현해 이어지는 파이프라인임을 확인
가중치 자체를 재학습해서 bit-exact 비교한 것은 아니고(GPU 없이는 30 epoch
재현 불가), 산출물 정합성으로 검증했다는 점은 명확히 밝혀둔다.
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
    LABEL_ROOTS = {
        "train": r"E:\atopic\AIHub\Training\02.라벨링데이터",
        "val":   r"E:\atopic\AIHub\Validation\02.라벨링데이터",
    }
    IMG_ROOTS = {
        "train": r"E:\atopic\AIHub\Training\01.원천데이터",
        "val":   r"E:\atopic\AIHub\Validation\01.원천데이터",
    }
    OUTPUT_DIR = r"E:\atopic\models\iga_severity"

    MODEL_NAME = "tf_efficientnetv2_s"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 0  # Windows 멀티프로세싱 문제 방지
    SEED = 42
    LABEL_SMOOTHING = 0.1

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    IGA_MAP = {
        "Clear":        0,
        "Almost Clear": 0,
        "Mild":         0,
        "Moderate":     1,
        "Severe":       1,
    }

    LABEL_NAMES = ["mild_or_below", "moderate_severe"]


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
def collect_samples(cfg):
    samples = []
    iga_counts = Counter()

    for split in ["train", "val"]:
        label_root = cfg.LABEL_ROOTS[split]
        img_root = cfg.IMG_ROOTS[split]
        prefix = "TS" if split == "train" else "VS"
        label_prefix = "TL" if split == "train" else "VL"

        for view in ["정면", "측면"]:
            label_folder = os.path.join(label_root, f"{label_prefix}_아토피_{view}")
            img_folder = os.path.join(img_root, f"{prefix}_아토피_{view}")

            if not os.path.exists(label_folder) or not os.path.exists(img_folder):
                print(f"[SKIP] {label_folder}")
                continue

            for fname in os.listdir(label_folder):
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(label_folder, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                ann = data['annotations'][0]
                identifier = ann['identifier']
                iga = ann['diagnosis_info']['easi_score']['iga_grade']
                label = cfg.IGA_MAP.get(iga, -1)
                if label == -1:
                    continue

                img_path = os.path.join(img_folder, f"{identifier}.png")
                if os.path.exists(img_path):
                    samples.append((img_path, label))
                    iga_counts[iga] += 1

    return samples, iga_counts


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
# 학습 / 검증
# ══════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    running_loss, correct, total = 0.0, 0, 0
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
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {title}", ncols=80, leave=False):
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
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=label_names)}")

    return cm, acc, f1, auc, sensitivity, specificity, all_labels, all_probs


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
    all_samples, iga_counts = collect_samples(cfg)

    print(f"  총 샘플: {len(all_samples)}장")
    for grade in ["Clear", "Almost Clear", "Mild", "Moderate", "Severe"]:
        print(f"    {grade:15s}: {iga_counts[grade]:4d}장")

    label_list = [s[1] for s in all_samples]
    n_mild = label_list.count(0)
    n_mod_sev = label_list.count(1)
    print(f"\n  mild 이하 (0):       {n_mild}장")
    print(f"  moderate-severe (1): {n_mod_sev}장")
    print(f"  불균형 비율:         1:{n_mod_sev/n_mild:.1f}")

    # ── 2. 7:2:1 Split ──
    print(f"\n[2/5] 7:2:1 Stratified Split")
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=cfg.SEED
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=1/3, stratify=temp_l, random_state=cfg.SEED
    )

    print(f"  Train: {len(train_p)}장 (mild:{train_l.count(0)} / mod-sev:{train_l.count(1)})")
    print(f"  Val:   {len(val_p)}장 (mild:{val_l.count(0)} / mod-sev:{val_l.count(1)})")
    print(f"  Test:  {len(test_p)}장 (mild:{test_l.count(0)} / mod-sev:{test_l.count(1)})")

    train_dataset = SkinDataset(list(zip(train_p, train_l)), get_transforms(cfg.IMG_SIZE, True))
    val_dataset = SkinDataset(list(zip(val_p, val_l)), get_transforms(cfg.IMG_SIZE, False))
    test_dataset = SkinDataset(list(zip(test_p, test_l)), get_transforms(cfg.IMG_SIZE, False))

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # ── 3. 모델 ──
    print(f"\n[3/5] 모델: {cfg.MODEL_NAME}")
    model = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=2).to(device)

    # Class weight (불균형 처리)
    weight = torch.tensor([n_mod_sev / n_mild, 1.0]).to(device)
    print(f"  Class weight: mild={weight[0]:.2f}, mod-sev={weight[1]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    # ── 4. 학습 ──
    print(f"\n[4/5] 학습 시작 (Epochs: {cfg.NUM_EPOCHS})")
    best_val_acc = 0.0
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_iga_model.pth")
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

    cm, acc, f1, auc, sens, spec, labels_test, probs_test = evaluate_detailed(
        model, test_loader, device, cfg.LABEL_NAMES,
        title="IGA 중증도 분류 (Test Set)"
    )
    plot_confusion_matrix(cm, cfg.LABEL_NAMES,
        "IGA Severity - Test Set",
        os.path.join(cfg.OUTPUT_DIR, "cm_test.png"))
    plot_roc_curve(labels_test, probs_test,
        "ROC - IGA Severity",
        os.path.join(cfg.OUTPUT_DIR, "roc_test.png"))

    # ── 최종 요약 ──
    print(f"\n{'='*60}")
    print(f"  Model 2-B 결과 (IGA 중증도)")
    print(f"{'='*60}")
    print(f"  데이터: mild이하 {n_mild}장 / moderate-severe {n_mod_sev}장")
    print(f"  Train/Val/Test: {len(train_p)}/{len(val_p)}/{len(test_p)}")
    print(f"  Best Val Acc: {best_val_acc:.4f}")
    print(f"  Test Acc:     {acc:.4f}")
    print(f"  Test F1:      {f1:.4f}")
    print(f"  Test AUC:     {auc:.4f}")
    print(f"  Sensitivity:  {sens:.4f}")
    print(f"  Specificity:  {spec:.4f}")

    # JSON 저장
    results = {
        "timestamp": datetime.now().isoformat(),
        "task": "iga_severity_binary",
        "model": cfg.MODEL_NAME,
        "data": {"mild_or_below": n_mild, "moderate_severe": n_mod_sev,
                 "train": len(train_p), "val": len(val_p), "test": len(test_p)},
        "best_val_acc": best_val_acc,
        "test": {"accuracy": acc, "f1": f1, "auc": auc,
                 "sensitivity": sens, "specificity": spec},
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()