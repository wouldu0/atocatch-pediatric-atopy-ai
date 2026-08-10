import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import timm

# ── 설정 ──
MODEL_PATH = r"E:\atopic\models\iga_severity\best_iga_model.pth"
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
LABEL_NAMES = ["mild_or_below", "moderate_severe"]

IGA_MAP = {
    "Clear": 0, "Almost Clear": 0, "Mild": 0,
    "Moderate": 1, "Severe": 1,
}

LABEL_ROOTS = {
    "train": r"E:\atopic\AIHub\Training\02.라벨링데이터",
    "val":   r"E:\atopic\AIHub\Validation\02.라벨링데이터",
}
IMG_ROOTS = {
    "train": r"E:\atopic\AIHub\Training\01.원천데이터",
    "val":   r"E:\atopic\AIHub\Validation\01.원천데이터",
}

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

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    # ── 데이터 수집 ──
    all_samples = []
    for split in ["train", "val"]:
        prefix = "TS" if split == "train" else "VS"
        label_prefix = "TL" if split == "train" else "VL"
        for view in ["정면", "측면"]:
            label_folder = os.path.join(LABEL_ROOTS[split], f"{label_prefix}_아토피_{view}")
            img_folder = os.path.join(IMG_ROOTS[split], f"{prefix}_아토피_{view}")
            if not os.path.exists(label_folder) or not os.path.exists(img_folder):
                continue
            for fname in os.listdir(label_folder):
                if not fname.endswith('.json'):
                    continue
                with open(os.path.join(label_folder, fname), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ann = data['annotations'][0]
                identifier = ann['identifier']
                iga = ann['diagnosis_info']['easi_score']['iga_grade']
                label = IGA_MAP.get(iga, -1)
                if label == -1:
                    continue
                img_path = os.path.join(img_folder, f"{identifier}.png")
                if os.path.exists(img_path):
                    all_samples.append((img_path, label))

    # ── 동일한 SEED로 7:2:1 split → test set만 사용 ──
    random.seed(SEED)
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    from sklearn.model_selection import train_test_split
    _, temp_p, _, temp_l = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    _, test_p, _, test_l = train_test_split(
        temp_p, temp_l, test_size=1/3, stratify=temp_l, random_state=SEED
    )

    print(f"Test set: {len(test_p)}장 (mild:{test_l.count(0)} / mod-sev:{test_l.count(1)})")

    test_loader = DataLoader(
        SkinDataset(list(zip(test_p, test_l)), eval_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    # ── 모델 로드 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # ── 확률값 추출 ──
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels_batch.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── Threshold 최적화 ──
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)

    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    f1_scores = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1_scores.append(f1_score(all_labels, preds, average="weighted"))
    best_idx_f1 = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_idx_f1]

    print(f"\nAUC: {auc:.4f}")
    print(f"Youden's J 최적 threshold: {best_threshold:.4f}")
    print(f"F1 최적 threshold:         {best_threshold_f1:.4f}")

    # ── threshold별 성능 비교 ──
    print(f"\n{'threshold':>10s} | {'Acc':>6s} | {'F1':>6s} | {'Sens':>6s} | {'Spec':>6s}")
    print("-" * 55)

    test_thresholds = sorted(set([0.3, 0.4, 0.5, 0.6, 0.7, round(best_threshold, 4), round(best_threshold_f1, 4)]))
    for t in test_thresholds:
        preds = (all_probs >= t).astype(int)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, average="weighted")
        cm = confusion_matrix(all_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        marker = " ← Youden" if round(t, 4) == round(best_threshold, 4) else ""
        marker += " ← F1" if round(t, 4) == round(best_threshold_f1, 4) else ""
        print(f"{t:10.4f} | {acc:6.4f} | {f1:6.4f} | {sens:6.4f} | {spec:6.4f}{marker}")

    # ── 시각화 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax1.scatter(fpr[best_idx], tpr[best_idx], color="red", zorder=5,
                label=f"Youden threshold={best_threshold:.4f}")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve - IGA Severity")
    ax1.legend()

    ax2.plot(thresholds, f1_scores)
    ax2.axvline(best_threshold_f1, color="red", linestyle="--",
                label=f"Best F1 threshold={best_threshold_f1:.4f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score by Threshold")
    ax2.legend()

    plt.tight_layout()
    save_path = r"E:\atopic\models\iga_severity\threshold_optimization.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n시각화 저장: {save_path}")