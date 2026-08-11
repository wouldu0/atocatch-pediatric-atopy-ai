"""
train_iga_grouped_final.py
===========================
IGA 중증도 모델 corrected methodology 재학습 (seed=42)
— 현재 배포 모델(app/best_iga_model.pth)의 실제 학습 스크립트.

변경 사항 (train_iga_severity.py 대비):
  1. image-level random split → base-id group-preserving split
  2. test-set threshold 선택 → validation-set threshold 선택 후 test 고정 적용

유지 사항 (train_iga_severity.py에서 그대로):
  - tf_efficientnetv2_s, pretrained=True
  - IMG_SIZE=224, BATCH_SIZE=32, NUM_EPOCHS=30
  - LR=5e-4, weight_decay=1e-2, label_smoothing=0.1
  - AdamW + CosineAnnealingLR(T_max=30)
  - class weight = [n_mod_sev/n_mild, 1.0]
  - augmentation 동일
  - checkpoint 기준: best val_acc
  - seed=42

입력 경로: AI Hub raw 데이터 루트는 사용자 환경에 따라 다르므로, 아래 LABEL_ROOTS/
  IMG_ROOTS에 본인 환경에 맞는 절대경로를 직접 지정한다 (레포에 포함되지 않는
  외부 대용량 데이터라 다른 학습 스크립트들과 동일하게 절대경로로 유지).

출력 (EXPORT_DIR 기준 — 기본: training/image_classification/outputs/iga_grouped_final/,
  IGA_EXPORT_DIR 환경변수로 재정의 가능):
  iga_grouped_split_seed42.csv    — 이미지별 split 정보 (1800행 기준)
  models/best_iga_corrected.pth   — 최종 checkpoint (val acc 기준)
  results/final_metrics.json      — 전체 성능 + 기존 모델 비교
  results/training_curve.png / roc_curve.png / pr_curve.png / cm_thr05.png / cm_thr_val.png
  validation_threshold_search.csv — validation threshold 후보별 metric

주의:
  - 기존 IGA 모델(app/의 best_iga_model.pth 등)은 이 스크립트를 실행해도 건드리지 않는다.
  - 재현성: seed=42 — 동일 Python/PyTorch/timm 버전 + 동일 파일 순서에서만 보장.

── 배포 반영 (포트폴리오 정리 중) ──────────────────────────
이 스크립트가 만든 best_iga_grouped_seed42.pth를 app/best_iga_model.pth로,
model/iga_corrected_config.json의 수치를 app/model_config2.json으로 그대로
반영해 서비스 모델을 교체했다. 이진분류 모델의 base-id leakage 수정과 같은
근거(check_aihub_subject_leakage.py에서 쓴 것과 동일한 base-id 추출 정규식,
동일 group-preserving 원칙)로 적용했으며, 여기에 더해 threshold를 test set이
아닌 validation에서만 선택하도록 함께 고쳤다. seed=42 1회 실행 기준이라는
한계는 README에 명시했다.
"""

import os, re, json, csv, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    average_precision_score, precision_recall_curve,
)
from tqdm import tqdm
import timm

# ══════════════════════════════════════════════
# 설정 (image2_model.py와 동일한 값)
# ══════════════════════════════════════════════
LABEL_ROOTS = {
    "train": r"E:\atopic\AIHub\Training\02.라벨링데이터",
    "val":   r"E:\atopic\AIHub\Validation\02.라벨링데이터",
}
IMG_ROOTS = {
    "train": r"E:\atopic\AIHub\Training\01.원천데이터",
    "val":   r"E:\atopic\AIHub\Validation\01.원천데이터",
}

MODEL_NAME      = "tf_efficientnetv2_s"
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_EPOCHS      = 30
LEARNING_RATE   = 5e-4
WEIGHT_DECAY    = 1e-2
NUM_WORKERS     = 0
SEED            = 42
LABEL_SMOOTHING = 0.1
TRAIN_RATIO     = 0.7
VAL_RATIO       = 0.2
LABEL_NAMES     = ["mild_or_below", "moderate_severe"]
IGA_MAP         = {"Clear": 0, "Almost Clear": 0, "Mild": 0, "Moderate": 1, "Severe": 1}

# 출력 경로 (repo 기준 상대경로, IGA_EXPORT_DIR 환경변수로 재정의 가능)
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_DIR      = os.getenv(
    "IGA_EXPORT_DIR",
    os.path.join(REPO_ROOT, "training", "image_classification", "outputs", "iga_grouped_final"),
)
MODELS_DIR      = os.path.join(EXPORT_DIR, "models")
RESULTS_DIR     = os.path.join(EXPORT_DIR, "results")

BASE_ID_PATTERN = re.compile(r'^(.+?)_P\d+_L\d+$')


# ══════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_base_id(identifier: str) -> str:
    m = BASE_ID_PATTERN.match(identifier)
    return m.group(1) if m else identifier


# ══════════════════════════════════════════════
# 1. 데이터 수집 (image2_model.py collect_samples 동일)
# ══════════════════════════════════════════════
def collect_samples():
    """
    아토피 정면+측면 전체 수집.
    반환: list of dict {img_path, label, iga_grade, identifier, base_id, view}
    """
    samples = []
    for split in ["train", "val"]:
        prefix       = "TS" if split == "train" else "VS"
        label_prefix = "TL" if split == "train" else "VL"
        for view in ["정면", "측면"]:
            label_folder = os.path.join(LABEL_ROOTS[split], f"{label_prefix}_아토피_{view}")
            img_folder   = os.path.join(IMG_ROOTS[split],   f"{prefix}_아토피_{view}")
            if not os.path.exists(label_folder) or not os.path.exists(img_folder):
                print(f"  [SKIP] {label_folder}")
                continue
            for fname in os.listdir(label_folder):
                if not fname.endswith('.json'):
                    continue
                with open(os.path.join(label_folder, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ann        = data['annotations'][0]
                identifier = ann['identifier']
                iga        = ann['diagnosis_info']['easi_score']['iga_grade']
                label      = IGA_MAP.get(iga, -1)
                if label == -1:
                    continue
                img_path = os.path.join(img_folder, f"{identifier}.png")
                if not os.path.exists(img_path):
                    continue
                samples.append({
                    "img_path":   img_path,
                    "label":      label,
                    "iga_grade":  iga,
                    "identifier": identifier,
                    "base_id":    extract_base_id(identifier),
                    "view":       view,
                })
    return samples


# ══════════════════════════════════════════════
# 2. Base-id group-preserving split
# ══════════════════════════════════════════════
def group_preserving_split(samples, seed=42, train_ratio=0.7, val_ratio=0.2):
    """
    동일 base_id의 모든 이미지를 하나의 split에만 배치.
    base_id 단위로 train_ratio:val_ratio:(1-train-val) 분할.
    각 이미지의 label은 그대로 유지 (base_id에 대표 label 부여 안 함).
    """
    # base_id → [index] 그룹핑
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s['base_id']].append(i)

    base_ids = sorted(groups.keys())  # 재현성을 위해 정렬 후 shuffle
    rng = random.Random(seed)
    rng.shuffle(base_ids)

    n = len(base_ids)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    # test = 나머지

    train_bids = set(base_ids[:n_train])
    val_bids   = set(base_ids[n_train:n_train + n_val])
    test_bids  = set(base_ids[n_train + n_val:])

    train_idx = [i for bid in train_bids for i in groups[bid]]
    val_idx   = [i for bid in val_bids   for i in groups[bid]]
    test_idx  = [i for bid in test_bids  for i in groups[bid]]

    return train_idx, val_idx, test_idx, train_bids, val_bids, test_bids


# ══════════════════════════════════════════════
# 3. Split 검증 및 출력
# ══════════════════════════════════════════════
def validate_and_print_split(samples, train_idx, val_idx, test_idx,
                              train_bids, val_bids, test_bids):
    def split_stats(name, idx, bids):
        rows = [samples[i] for i in idx]
        labels  = [r['label']   for r in rows]
        views   = [r['view']    for r in rows]
        n0 = labels.count(0)
        n1 = labels.count(1)
        nf = views.count('정면')
        ns = views.count('측면')
        print(f"\n  [{name}]")
        print(f"    이미지 수:        {len(rows)}")
        print(f"    unique base_id:   {len(bids)}")
        print(f"    mild_or_below(0): {n0} ({n0/len(rows)*100:.1f}%)")
        print(f"    moderate_sev(1):  {n1} ({n1/len(rows)*100:.1f}%)")
        print(f"    정면:             {nf} / 측면: {ns}")
        return rows

    print("\n" + "="*60)
    print("  Split 결과 검증")
    print("="*60)
    split_stats("Train",      train_idx, train_bids)
    split_stats("Validation", val_idx,   val_bids)
    split_stats("Test",       test_idx,  test_bids)

    tv = train_bids & val_bids
    tt = train_bids & test_bids
    vt = val_bids   & test_bids
    print(f"\n  Train∩Val  overlap:  {len(tv)} base-id")
    print(f"  Train∩Test overlap:  {len(tt)} base-id")
    print(f"  Val∩Test   overlap:  {len(vt)} base-id")

    assert len(tv) == 0, f"Train∩Val 중복 {len(tv)}개 — split 오류"
    assert len(tt) == 0, f"Train∩Test 중복 {len(tt)}개 — split 오류"
    assert len(vt) == 0, f"Val∩Test 중복 {len(vt)}개 — split 오류"
    print("  → 모두 0 ✓ 재학습 진행")


def save_split_csv(samples, train_idx, val_idx, test_idx, save_path):
    rows = []
    for idx_list, split_name in [(train_idx, "train"), (val_idx, "val"), (test_idx, "test")]:
        for i in idx_list:
            s = samples[i]
            rows.append({
                "filepath":   s['img_path'],
                "identifier": s['identifier'],
                "base_id":    s['base_id'],
                "label":      s['label'],
                "iga_grade":  s['iga_grade'],
                "view":       s['view'],
                "split":      split_name,
            })
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Split CSV 저장: {save_path}")


# ══════════════════════════════════════════════
# 4. Dataset / Transform (image2_model.py 동일)
# ══════════════════════════════════════════════
class SkinDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['img_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, s['label']


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
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
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ══════════════════════════════════════════════
# 5. 학습 / 검증 루프 (image2_model.py 동일)
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
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", ncols=80, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def get_probs(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels_b in loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            all_labels.extend(labels_b.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


# ══════════════════════════════════════════════
# 6. 평가 유틸
# ══════════════════════════════════════════════
def metrics_at(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="weighted")
    auc   = roc_auc_score(labels, probs)
    prauc = average_precision_score(labels, probs)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "threshold":   round(float(threshold), 4),
        "accuracy":    round(float(acc),   4),
        "f1_weighted": round(float(f1),    4),
        "roc_auc":     round(float(auc),   4),
        "pr_auc":      round(float(prauc), 4),
        "sensitivity": round(float(sens),  4),
        "specificity": round(float(spec),  4),
        "precision":   round(float(prec),  4),
        "recall":      round(float(rec),   4),
    }


def find_val_threshold(val_labels, val_probs):
    """Validation set에서 Youden's J 최적 threshold 탐색"""
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    j = tpr - fpr
    best_idx     = int(np.argmax(j))
    youden_thr   = float(thresholds[best_idx])

    # F1-optimal도 함께 계산
    f1_scores = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1_scores.append(f1_score(val_labels, preds, average="weighted"))
    best_f1_idx = int(np.argmax(f1_scores))
    f1_thr = float(thresholds[best_f1_idx])

    return youden_thr, f1_thr, thresholds, fpr, tpr, j, f1_scores


def save_threshold_csv(thresholds, fpr, tpr, j_scores, f1_scores,
                       val_labels, val_probs, save_path):
    auc = roc_auc_score(val_labels, val_probs)
    rows = []
    for i, t in enumerate(thresholds):
        preds = (val_probs >= t).astype(int)
        acc  = accuracy_score(val_labels, preds)
        sens = tpr[i]
        spec = 1 - fpr[i]
        rows.append({
            "threshold": round(float(t),        4),
            "accuracy":  round(float(acc),       4),
            "f1":        round(float(f1_scores[i]), 4),
            "sensitivity": round(float(sens),    4),
            "specificity": round(float(spec),    4),
            "youden_j":  round(float(j_scores[i]), 4),
            "auc":       round(float(auc),       4),
        })
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Threshold CSV 저장: {save_path}")


# ══════════════════════════════════════════════
# 7. 시각화
# ══════════════════════════════════════════════
def plot_training_curve(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc(labels, probs, selected_thr, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    idx = np.argmin(np.abs(_ - selected_thr)) if len(_) > 0 else 0
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.scatter(fpr[idx], tpr[idx], color='red', zorder=5,
                label=f"Val threshold = {selected_thr:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - IGA Corrected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    prauc = average_precision_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {prauc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve - IGA Corrected")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cm(labels, probs, threshold, save_path, title):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════
# 8. SUMMARY 생성
# ══════════════════════════════════════════════
def write_summary(summary_data, save_path):
    d = summary_data
    lines = [
        "# IGA Corrected Methodology — Summary for GitHub",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 사용 데이터",
        "",
        f"- 대상: 아토피 정면 + 측면 이미지",
        f"- 총 이미지 수: {d['n_total']}장",
        f"- unique base_id: {d['n_base_ids']}개",
        f"- IGA 이진 분류: mild_or_below (Clear/Almost Clear/Mild=0) vs moderate_severe (Moderate/Severe=1)",
        "",
        "---",
        "",
        "## 2. 기존 문제",
        "",
        "### 2-1. base-id group overlap (image-level random split)",
        "",
        "- 기존 7:2:1 image-level random split에서:",
        "  - Train∩Test 중복 base-id: 11개 그룹",
        "  - Test 180장 중 23장 (12.8%)이 Train과 동일 base-id 공유",
        "  - 동일 base-id의 P/L suffix만 다른 이미지들이 split에 걸쳐 배치됨",
        "- 참고: 이 데이터는 합성 데이터이며, base-id가 실제 동일 환자임을 의미하지 않음",
        "  - 동일 base-id 내에서도 IGA label이 서로 다를 수 있음",
        "",
        "### 2-2. Test-set threshold selection bias",
        "",
        "- 기존 image2_thr.py: test set에서 Youden's J / F1 threshold 탐색",
        "- 기존 image2_finalB.py: 동일한 test set에서 threshold=0.38 적용 후 성능 보고",
        "- ROC-AUC = 0.8758 (threshold 무관), Accuracy = 83.9% (threshold=0.38, test→test)",
        "- Audit (val→test, 기존 random split): threshold=0.7235, Accuracy=78.3%",
        "",
        "---",
        "",
        "## 3. 새 base-id group-preserving split (seed=42)",
        "",
        f"| Split      | 이미지 수 | unique base_id | mild_or_below | moderate_severe | 정면 | 측면 |",
        f"|------------|-----------|----------------|---------------|-----------------|------|------|",
        f"| Train      | {d['train_n']:<9} | {d['train_bids']:<14} | {d['train_n0']} ({d['train_n0']/d['train_n']*100:.1f}%) | {d['train_n1']} ({d['train_n1']/d['train_n']*100:.1f}%) | {d['train_front']} | {d['train_side']} |",
        f"| Validation | {d['val_n']:<9} | {d['val_bids']:<14} | {d['val_n0']} ({d['val_n0']/d['val_n']*100:.1f}%) | {d['val_n1']} ({d['val_n1']/d['val_n']*100:.1f}%) | {d['val_front']} | {d['val_side']} |",
        f"| Test       | {d['test_n']:<9} | {d['test_bids']:<14} | {d['test_n0']} ({d['test_n0']/d['test_n']*100:.1f}%) | {d['test_n1']} ({d['test_n1']/d['test_n']*100:.1f}%) | {d['test_front']} | {d['test_side']} |",
        "",
        "- Train∩Val overlap: 0 ✓",
        "- Train∩Test overlap: 0 ✓",
        "- Val∩Test overlap: 0 ✓",
        "",
        "---",
        "",
        "## 4. 학습 설정 (image2_model.py와 동일)",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| 모델 | tf_efficientnetv2_s |",
        "| pretrained | True (ImageNet) |",
        "| 입력 크기 | 224×224 |",
        "| Optimizer | AdamW |",
        "| LR | 5e-4 |",
        "| Weight Decay | 1e-2 |",
        "| Scheduler | CosineAnnealingLR(T_max=30) |",
        "| Epochs | 30 |",
        "| Batch Size | 32 |",
        "| Loss | CrossEntropyLoss + class weight + label smoothing=0.1 |",
        f"| Class weight | mild={d['class_w_mild']:.3f}, mod-sev=1.0 |",
        "| Checkpoint 기준 | best val_acc |",
        "| Seed | 42 |",
        "",
        "---",
        "",
        "## 5. Validation에서 선택한 Threshold",
        "",
        f"- Youden's J 최적: {d['youden_thr']:.4f}",
        f"- F1-optimal: {d['f1_thr']:.4f}",
        f"- **최종 선택: {d['selected_thr']:.4f} (Youden's J)**",
        "- Test 평가에는 이 threshold를 고정 적용 (test에서 탐색 안 함)",
        "",
        "---",
        "",
        "## 6. Untouched Test 최종 성능",
        "",
        f"ROC-AUC: **{d['test_auc']:.4f}** (threshold 무관)",
        f"PR-AUC:  **{d['test_prauc']:.4f}**",
        "",
        "| Threshold | Accuracy | F1(weighted) | Sensitivity | Specificity |",
        "|-----------|----------|-------------|-------------|-------------|",
        f"| 0.5 (기본) | {d['test_05']['accuracy']:.4f} | {d['test_05']['f1_weighted']:.4f} | {d['test_05']['sensitivity']:.4f} | {d['test_05']['specificity']:.4f} |",
        f"| {d['selected_thr']:.4f} (val 선택) | {d['test_sel']['accuracy']:.4f} | {d['test_sel']['f1_weighted']:.4f} | {d['test_sel']['sensitivity']:.4f} | {d['test_sel']['specificity']:.4f} |",
        "",
        "---",
        "",
        "## 7. 기존 결과와 비교",
        "",
        "| 평가 방식 | Split | Threshold 선택 | Accuracy | F1 | ROC-AUC | Sensitivity | Specificity |",
        "|---|---|---|---|---|---|---|---|",
        "| 기존 IGA (image2_finalB.py) | image-level random | Test (0.38) | 0.8389 | 0.8368 | 0.8758 | 0.9058 | 0.6190 |",
        "| 기존 split, val threshold (audit) | image-level random | Validation (0.7235) | 0.7833 | 0.7974 | 0.8758 | 0.7681 | 0.8333 |",
        f"| **Corrected IGA (thr=0.5)** | base-id group-preserving | — | {d['test_05']['accuracy']:.4f} | {d['test_05']['f1_weighted']:.4f} | {d['test_auc']:.4f} | {d['test_05']['sensitivity']:.4f} | {d['test_05']['specificity']:.4f} |",
        f"| **Corrected IGA (val thr={d['selected_thr']:.4f})** | base-id group-preserving | Validation | {d['test_sel']['accuracy']:.4f} | {d['test_sel']['f1_weighted']:.4f} | {d['test_auc']:.4f} | {d['test_sel']['sensitivity']:.4f} | {d['test_sel']['specificity']:.4f} |",
        "",
        "※ 주의: Corrected IGA는 test set 자체가 달라졌으므로 기존 수치와 직접 비교는 제한적.",
        "",
        "---",
        "",
        "## 8. 결과 해석",
        "",
        "- 기존 83.9%는 (a) image-level random split에서 동일 base-id의 연관 이미지가",
        "  train/test에 걸쳐 배치된 상태에서, (b) test set에서 threshold를 선택한 뒤",
        "  동일 test set에서 성능을 보고한 retrospective internal result.",
        "- Corrected IGA는 split independence를 강제하고 (Train∩Test = 0),",
        "  validation에서 threshold를 선택 후 untouched test를 평가한 결과.",
        "- ROC-AUC 비교가 가장 fair: threshold 및 split 방식 무관하게 비교 가능.",
        "- 이 데이터는 합성 데이터이며 base-id group이 실제 동일 환자임을 단정할 수 없음.",
        "  따라서 leakage의 실질적 영향은 불확실하나, 방법론적 엄밀성을 위해 corrected split 채택.",
        "",
        "---",
        "",
        "## 9. 생성 파일 목록",
        "",
        "```",
        "IGA_CORRECTED_EXPORT/",
        "├── train_iga_corrected.py          # 이 실험 전체 코드",
        "├── iga_grouped_split_seed42.csv    # 이미지별 split 정보",
        "├── validation_threshold_search.csv # val threshold 탐색 결과",
        "├── models/",
        "│   ├── best_iga_corrected.pth      # 새 모델 checkpoint",
        "│   └── model_config.json           # 모델/학습 설정",
        "├── results/",
        "│   ├── final_metrics.json          # 최종 성능 수치",
        "│   ├── training_curve.png",
        "│   ├── roc_curve.png",
        "│   ├── pr_curve.png",
        "│   ├── cm_thr05.png                # confusion matrix (thr=0.5)",
        "│   └── cm_thr_val.png              # confusion matrix (val threshold)",
        "└── SUMMARY_FOR_GITHUB.md           # 이 파일",
        "```",
        "",
        "---",
        "",
        "## 10. 남은 한계",
        "",
        "- 단일 seed=42 실험 (multi-seed 안정성 미확인)",
        "- 합성 데이터 특성상 base-id group이 실제 leakage를 의미하는지 불확실",
        "- lesion segmentation mask 미포함 (AI Hub에서 별도 다운로드 필요)",
        "- 외부 데이터셋 검증 없음",
    ]
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  SUMMARY 저장: {save_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    set_seed(SEED)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. 데이터 수집 ──
    print("\n[1/8] 데이터 수집")
    samples = collect_samples()
    print(f"  총 수집: {len(samples)}장")

    labels_all = [s['label'] for s in samples]
    n_mild    = labels_all.count(0)
    n_mod_sev = labels_all.count(1)
    n_base_ids = len(set(s['base_id'] for s in samples))
    print(f"  mild_or_below(0): {n_mild}장")
    print(f"  moderate_sev(1):  {n_mod_sev}장")
    print(f"  unique base_id:   {n_base_ids}개")

    # ── 2. Group-preserving split ──
    print("\n[2/8] Base-id group-preserving split")
    train_idx, val_idx, test_idx, train_bids, val_bids, test_bids = \
        group_preserving_split(samples, seed=SEED,
                               train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)

    validate_and_print_split(samples, train_idx, val_idx, test_idx,
                             train_bids, val_bids, test_bids)

    split_csv_path = os.path.join(EXPORT_DIR, "iga_grouped_split_seed42.csv")
    save_split_csv(samples, train_idx, val_idx, test_idx, split_csv_path)

    train_samples = [samples[i] for i in train_idx]
    val_samples   = [samples[i] for i in val_idx]
    test_samples  = [samples[i] for i in test_idx]

    # split 통계 변수 저장 (summary용)
    def split_counts(rows):
        labels_ = [r['label'] for r in rows]
        views_  = [r['view']  for r in rows]
        return (len(rows), labels_.count(0), labels_.count(1),
                views_.count('정면'), views_.count('측면'))

    tr_n, tr_n0, tr_n1, tr_f, tr_s = split_counts(train_samples)
    va_n, va_n0, va_n1, va_f, va_s = split_counts(val_samples)
    te_n, te_n0, te_n1, te_f, te_s = split_counts(test_samples)

    # ── 3. DataLoader ──
    print("\n[3/8] DataLoader 구성")
    train_loader = DataLoader(
        SkinDataset(train_samples, get_transforms(True)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(
        SkinDataset(val_samples, get_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        SkinDataset(test_samples, get_transforms(False)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ── 4. 모델 / 학습 설정 (image2_model.py 동일) ──
    print(f"\n[4/8] 모델: {MODEL_NAME} (pretrained=True)")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2).to(device)

    class_w_mild = n_mod_sev / n_mild
    weight = torch.tensor([class_w_mild, 1.0]).to(device)
    print(f"  Class weight: mild={class_w_mild:.3f}, mod-sev=1.000")

    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── 5. 학습 ──
    print(f"\n[5/8] 학습 (Epochs: {NUM_EPOCHS})")
    best_val_acc = 0.0
    checkpoint_path = os.path.join(MODELS_DIR, "best_iga_corrected.pth")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]  lr={optimizer.param_groups[0]['lr']:.6f}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"  Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")
        print(f"  Val   Loss: {va_loss:.4f} | Val   Acc: {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ★ Best model 저장 (Val Acc: {va_acc:.4f})")

    plot_training_curve(history, os.path.join(RESULTS_DIR, "training_curve.png"))
    print(f"\n  Best Val Acc: {best_val_acc:.4f}")

    # ── 6. Validation threshold 선택 ──
    print("\n[6/8] Validation threshold 탐색")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    val_labels, val_probs = get_probs(model, val_loader, device)

    youden_thr, f1_thr, thr_arr, fpr_arr, tpr_arr, j_arr, f1_arr = \
        find_val_threshold(val_labels, val_probs)

    selected_thr = youden_thr  # Youden's J 채택 (기존 image2_thr.py와 동일 기준)
    print(f"  Youden's J 최적: {youden_thr:.4f}")
    print(f"  F1-optimal:      {f1_thr:.4f}")
    print(f"  → 선택 threshold: {selected_thr:.4f} (Youden's J, 기존 코드 동일 기준)")

    thr_csv_path = os.path.join(EXPORT_DIR, "validation_threshold_search.csv")
    save_threshold_csv(thr_arr, fpr_arr, tpr_arr, j_arr, f1_arr,
                       val_labels, val_probs, thr_csv_path)

    # ── 7. Test 평가 (threshold 고정, 탐색 없음) ──
    print("\n[7/8] Untouched Test 평가")
    test_labels, test_probs = get_probs(model, test_loader, device)

    res_05  = metrics_at(test_labels, test_probs, 0.5)
    res_sel = metrics_at(test_labels, test_probs, selected_thr)

    print(f"\n  ROC-AUC: {res_05['roc_auc']:.4f}  PR-AUC: {res_05['pr_auc']:.4f}")
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │              Test 최종 성능 비교                          │")
    print("  ├────────────┬────────┬────────┬────────┬────────┬─────────┤")
    print("  │ threshold  │  Acc   │   F1   │  Sens  │  Spec  │ PR-AUC  │")
    print("  ├────────────┼────────┼────────┼────────┼────────┼─────────┤")

    def row_str(r):
        return (f"  │ {r['threshold']:10.4f} │ {r['accuracy']:6.4f} │ {r['f1_weighted']:6.4f} │ "
                f"{r['sensitivity']:6.4f} │ {r['specificity']:6.4f} │ {r['pr_auc']:7.4f} │")

    print(row_str(res_05))
    print(row_str(res_sel))
    print("  └────────────┴────────┴────────┴────────┴────────┴─────────┘")

    print(f"\n  [threshold={selected_thr:.4f} classification report]")
    preds_sel = (test_probs >= selected_thr).astype(int)
    print(classification_report(test_labels, preds_sel, target_names=LABEL_NAMES))

    # 시각화
    plot_roc(test_labels, test_probs, selected_thr,
             os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_pr(test_labels, test_probs,
            os.path.join(RESULTS_DIR, "pr_curve.png"))
    plot_cm(test_labels, test_probs, 0.5,
            os.path.join(RESULTS_DIR, "cm_thr05.png"),
            "IGA Corrected — Test (thr=0.5)")
    plot_cm(test_labels, test_probs, selected_thr,
            os.path.join(RESULTS_DIR, f"cm_thr_val.png"),
            f"IGA Corrected — Test (val thr={selected_thr:.4f})")

    # ── 8. 결과 저장 ──
    print("\n[8/8] 결과 저장")

    model_config = {
        "timestamp":      datetime.now().isoformat(),
        "model":          MODEL_NAME,
        "pretrained":     True,
        "img_size":       IMG_SIZE,
        "batch_size":     BATCH_SIZE,
        "num_epochs":     NUM_EPOCHS,
        "learning_rate":  LEARNING_RATE,
        "weight_decay":   WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "optimizer":      "AdamW",
        "scheduler":      "CosineAnnealingLR(T_max=30)",
        "loss":           "CrossEntropyLoss",
        "class_weight":   {"mild_or_below": round(class_w_mild, 4), "moderate_severe": 1.0},
        "checkpoint_criterion": "best_val_acc",
        "seed":           SEED,
        "split":          "base-id group-preserving",
        "threshold_selection": "validation (Youden's J)",
    }
    with open(os.path.join(MODELS_DIR, "model_config.json"), 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    final_metrics = {
        "timestamp":      datetime.now().isoformat(),
        "split":          "base-id group-preserving (seed=42)",
        "n_total":        len(samples),
        "n_train":        tr_n, "n_val": va_n, "n_test": te_n,
        "best_val_acc":   round(best_val_acc, 4),
        "selected_threshold": round(selected_thr, 4),
        "threshold_method":   "Youden's J on validation set",
        "test_thr05":     res_05,
        "test_thr_val":   res_sel,
        "comparison": {
            "original_IGA_test_thr": {"split": "image-level random", "threshold": 0.38,
                                       "accuracy": 0.8389, "f1": 0.8368, "roc_auc": 0.8758,
                                       "sensitivity": 0.9058, "specificity": 0.6190},
            "audit_val_thr_old_split": {"split": "image-level random", "threshold": 0.7235,
                                         "accuracy": 0.7833, "f1": 0.7974, "roc_auc": 0.8758,
                                         "sensitivity": 0.7681, "specificity": 0.8333},
            "corrected_IGA_thr05": {**res_05, "split": "base-id group-preserving"},
            "corrected_IGA_val_thr": {**res_sel, "split": "base-id group-preserving"},
        }
    }
    with open(os.path.join(RESULTS_DIR, "final_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    print(f"  final_metrics.json 저장")

    # SUMMARY_FOR_GITHUB.md
    summary_data = {
        "n_total": len(samples), "n_base_ids": n_base_ids,
        "train_n": tr_n, "train_bids": len(train_bids),
        "train_n0": tr_n0, "train_n1": tr_n1, "train_front": tr_f, "train_side": tr_s,
        "val_n": va_n, "val_bids": len(val_bids),
        "val_n0": va_n0, "val_n1": va_n1, "val_front": va_f, "val_side": va_s,
        "test_n": te_n, "test_bids": len(test_bids),
        "test_n0": te_n0, "test_n1": te_n1, "test_front": te_f, "test_side": te_s,
        "class_w_mild": class_w_mild,
        "youden_thr": youden_thr, "f1_thr": f1_thr, "selected_thr": selected_thr,
        "test_auc": res_05['roc_auc'], "test_prauc": res_05['pr_auc'],
        "test_05": res_05, "test_sel": res_sel,
    }
    write_summary(summary_data, os.path.join(EXPORT_DIR, "SUMMARY_FOR_GITHUB.md"))

    # 스크립트 자체도 export에 복사
    import shutil
    shutil.copy(__file__, os.path.join(EXPORT_DIR, "train_iga_corrected.py"))

    print(f"\n{'='*60}")
    print("  완료 — IGA_CORRECTED_EXPORT/ 에 모든 결과 저장됨")
    print(f"{'='*60}")
    print(f"  ROC-AUC:   {res_05['roc_auc']:.4f}")
    print(f"  thr=0.5  → Acc={res_05['accuracy']:.4f} / F1={res_05['f1_weighted']:.4f}")
    print(f"  thr={selected_thr:.4f} → Acc={res_sel['accuracy']:.4f} / F1={res_sel['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
