import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

# ── 설정 ──
MODEL_PATH = r"E:\atopic\models\comparison\tf_efficientnetv2_s\best_model.pth"
DERMNET_ROOT = r"E:\atopic\dermnet_images"
DERMNET_TRAIN_RATIO = 0.6
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
LABEL_NAMES = ["non_atopy", "atopy"]

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

# ── DermNet 평가용만 추출 (train_ratio=0.6이면 나머지 40%가 평가용) ──
import random
rng = random.Random(SEED)

dermnet_test = []
for folder, label in [("atopic_dermatitis", 1), ("psoriasis", 0),
                       ("acne", 0), ("rosacea", 0), ("seborrheic_dermatitis", 0)]:
    folder_path = os.path.join(DERMNET_ROOT, folder)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    rng.shuffle(files)
    split_idx = int(len(files) * DERMNET_TRAIN_RATIO)
    for f in files[split_idx:]:  # 40% = 평가용
        dermnet_test.append((f, label))

print(f"외부 평가셋: {len(dermnet_test)}장")

dermnet_loader = DataLoader(
    SkinDataset(dermnet_test, eval_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

# ── 모델 로드 + 확률값 추출 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

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

# ── ROC curve로 최적 threshold 탐색 ──
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
auc = roc_auc_score(all_labels, all_probs)

# Youden's J (sensitivity + specificity - 1 최대화)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold_youden = thresholds[best_idx]

# F1 최대화 threshold
f1_scores = []
for t in thresholds:
    preds = (all_probs >= t).astype(int)
    f1_scores.append(f1_score(all_labels, preds, average="weighted"))
best_idx_f1 = np.argmax(f1_scores)
best_threshold_f1 = thresholds[best_idx_f1]

print(f"\nAUC: {auc:.4f}")
print(f"Youden's J 최적 threshold: {best_threshold_youden:.4f}")
print(f"F1 최적 threshold:         {best_threshold_f1:.4f}")

# ── threshold별 성능 비교 ──
print(f"\n{'threshold':>10s} | {'Acc':>6s} | {'F1':>6s} | {'Sens':>6s} | {'Spec':>6s}")
print("-" * 50)

for t in [0.3, 0.35, 0.4, 0.45, 0.5, best_threshold_youden, best_threshold_f1]:
    t = round(t, 4)
    preds = (all_probs >= t).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average="weighted")
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    marker = " ← Youden" if t == round(best_threshold_youden, 4) else (" ← F1" if t == round(best_threshold_f1, 4) else "")
    print(f"{t:10.4f} | {acc:6.4f} | {f1:6.4f} | {sens:6.4f} | {spec:6.4f}{marker}")

# ── 시각화 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve
ax1.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
ax1.scatter(fpr[best_idx], tpr[best_idx], color="red", zorder=5,
            label=f"Youden threshold={best_threshold_youden:.4f}")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend()

# F1 vs threshold
ax2.plot(thresholds, f1_scores)
ax2.axvline(best_threshold_f1, color="red", linestyle="--",
            label=f"Best F1 threshold={best_threshold_f1:.4f}")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("F1 Score")
ax2.set_title("F1 Score by Threshold")
ax2.legend()

plt.tight_layout()
plt.savefig(r"E:\atopic\models\comparison\threshold_optimization.png", dpi=150)
plt.close()
print(f"\n시각화 저장: E:\\atopic\\models\\comparison\\threshold_optimization.png")