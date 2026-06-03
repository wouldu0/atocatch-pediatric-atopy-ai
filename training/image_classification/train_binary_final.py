import os
import random
import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── 설정 ──
MODEL_PATH = r"E:\atopic\models\comparison\tf_efficientnetv2_s\best_model.pth"
DERMNET_ROOT = r"E:\atopic\dermnet_images"
THRESHOLD = 0.29
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
LABEL_NAMES = ["non_atopy", "atopy"]
DERMNET_TRAIN_RATIO = 0.6

# ── 데이터셋 ──
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
    # ── DermNet 평가용 (40%) ──
    rng = random.Random(SEED)
    dermnet_test = []
    for folder, label in [("atopic_dermatitis", 1), ("psoriasis", 0),
                           ("acne", 0), ("rosacea", 0), ("seborrheic_dermatitis", 0)]:
        folder_path = os.path.join(DERMNET_ROOT, folder)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
        rng.shuffle(files)
        split_idx = int(len(files) * DERMNET_TRAIN_RATIO)
        for f in files[split_idx:]:
            dermnet_test.append((f, label))

    print(f"외부 평가셋: {len(dermnet_test)}장")
    print(f"  아토피: {sum(1 for _, l in dermnet_test if l == 1)}장")
    print(f"  비아토피: {sum(1 for _, l in dermnet_test if l == 0)}장")

    dermnet_loader = DataLoader(
        SkinDataset(dermnet_test, eval_transform),
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
        for images, labels in dermnet_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── threshold 0.29 적용 ──
    all_preds = (all_probs >= THRESHOLD).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\n{'='*60}")
    print(f"  최종 모델: tf_EfficientNetV2-S + Threshold {THRESHOLD}")
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
    plt.title(f"Final Model - tf_EfficientNetV2-S (threshold={THRESHOLD})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    save_path = r"E:\atopic\models\comparison\final_cm_external.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Confusion Matrix 저장: {save_path}")