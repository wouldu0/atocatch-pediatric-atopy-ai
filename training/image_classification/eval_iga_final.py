import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import timm

MODEL_PATH = r"E:\atopic\models\iga_severity\best_iga_model.pth"
THRESHOLD = 0.38
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

    # ── 동일한 SEED로 test set 추출 ──
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

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

    # ── Threshold 0.38 적용 ──
    all_preds = (all_probs >= THRESHOLD).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\n{'='*60}")
    print(f"  Model 2-B 최종: tf_EfficientNetV2-S + Threshold {THRESHOLD}")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=LABEL_NAMES)}")

    # confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(f"Model 2-B Final (threshold={THRESHOLD})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    save_path = r"E:\atopic\models\iga_severity\final_cm.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Confusion Matrix 저장: {save_path}")