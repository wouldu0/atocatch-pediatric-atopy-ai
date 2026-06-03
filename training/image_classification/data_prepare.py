import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── 데이터셋 클래스 ──
class SkinDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# ── Transform ──
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── 데이터셋 생성 ──
train_dataset = SkinDataset(train_p, train_l, transform=train_transform)
val_dataset = SkinDataset(val_p, val_l, transform=eval_transform)
test_dataset = SkinDataset(test_p, test_l, transform=eval_transform)

# ── DermNet 외부 평가셋 ──
DERMNET_ROOT = r"E:\atopic\dermnet_images"
dermnet_paths = []
dermnet_labels = []

# 아토피 = 1
atopy_dir = os.path.join(DERMNET_ROOT, "atopic_dermatitis")
if os.path.isdir(atopy_dir):
    for f in os.listdir(atopy_dir):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            dermnet_paths.append(os.path.join(atopy_dir, f))
            dermnet_labels.append(1)

# 비아토피 = 0
for folder in ["psoriasis", "acne", "rosacea", "seborrheic_dermatitis"]:
    d = os.path.join(DERMNET_ROOT, folder)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                dermnet_paths.append(os.path.join(d, f))
                dermnet_labels.append(0)

dermnet_dataset = SkinDataset(dermnet_paths, dermnet_labels, transform=eval_transform)

# ── 로더 ──
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
dermnet_loader = DataLoader(dermnet_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ── 확인 ──
print(f"Train:   {len(train_dataset)}장 → {len(train_loader)} batches")
print(f"Val:     {len(val_dataset)}장 → {len(val_loader)} batches")
print(f"Test:    {len(test_dataset)}장 → {len(test_loader)} batches")
print(f"DermNet: {len(dermnet_dataset)}장 (아토피:{dermnet_labels.count(1)} / 비아토피:{dermnet_labels.count(0)})")

# 이미지 한 장 로드 테스트
img, label = train_dataset[0]
print(f"\n샘플 이미지 shape: {img.shape}, label: {label}")