"""
Grad-CAM 시각화
- Model 2-A (아토피 여부) + Model 2-B (IGA 중증도) 둘 다 적용
- 이미지 폴더 또는 단일 이미지에 적용 가능

사용법:
    python gradcam.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# ══════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════
MODEL_A_PATH = r"E:\atopic\models\comparison\tf_efficientnetv2_s\best_model.pth"
MODEL_B_PATH = r"E:\atopic\models\iga_severity\best_iga_model.pth"
OUTPUT_DIR   = r"E:\atopic\models\gradcam"

# 시각화할 이미지 경로 (테스트용 몇 장)
TEST_IMAGES = [
    # 아토피 이미지 몇 장
    r"E:\atopic\dermnet_images\atopic_dermatitis",
    # 비아토피 이미지 몇 장
    r"E:\atopic\dermnet_images\psoriasis",
]

THRESHOLD_A = 0.29
THRESHOLD_B = 0.38
IMG_SIZE    = 224
NUM_IMAGES  = 3  # 폴더당 몇 장 시각화할지

LABEL_NAMES_A = ["non_atopy", "atopy"]
LABEL_NAMES_B = ["mild_or_below", "moderate_severe"]


# ══════════════════════════════════════════════
# Grad-CAM 구현
# ══════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Grad-CAM 계산
        gradients = self.gradients[0]           # (C, H, W)
        activations = self.activations[0]       # (C, H, W)

        weights = gradients.mean(dim=(1, 2))    # (C,)
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = torch.relu(cam)

        # 정규화
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), probs.detach().cpu().numpy()[0], class_idx


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════
def overlay_gradcam(original_img, cam, alpha=0.5):
    """원본 이미지에 Grad-CAM heat map 오버레이"""
    # cam을 원본 이미지 크기로 리사이즈
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            original_img.size, Image.BILINEAR
        )
    ) / 255.0

    # colormap 적용 (jet: 파랑→초록→빨강)
    colormap = cm.get_cmap("jet")
    heatmap = colormap(cam_resized)[:, :, :3]  # RGBA → RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    # 원본 이미지와 합성
    orig_array = np.array(original_img)
    overlay = (alpha * heatmap + (1 - alpha) * orig_array).astype(np.uint8)

    return overlay, heatmap


def visualize_single(image_path, gradcam_a, gradcam_b, threshold_a, threshold_b,
                     label_names_a, label_names_b, transform, device, save_path):
    """단일 이미지에 대해 Model A + B Grad-CAM 시각화"""
    original_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # Model 2-A
    cam_a, probs_a, pred_a = gradcam_a.generate(input_tensor, class_idx=1)  # 아토피 클래스
    prob_atopy = probs_a[1]
    label_a = "atopy" if prob_atopy >= threshold_a else "non_atopy"
    overlay_a, heatmap_a = overlay_gradcam(original_img, cam_a)

    # Model 2-B
    cam_b, probs_b, pred_b = gradcam_b.generate(input_tensor, class_idx=1)  # moderate-severe
    prob_mod_sev = probs_b[1]
    label_b = "moderate_severe" if prob_mod_sev >= threshold_b else "mild_or_below"
    overlay_b, heatmap_b = overlay_gradcam(original_img, cam_b)

    # 시각화: 2행 3열
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(os.path.basename(image_path), fontsize=12, y=0.98)

    # 행 1: Model 2-A
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(heatmap_a)
    axes[0, 1].set_title("Grad-CAM Heat Map\n(Model 2-A)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(overlay_a)
    color_a = "red" if label_a == "atopy" else "green"
    axes[0, 2].set_title(
        f"Model 2-A: {label_a}\n아토피 확률: {prob_atopy:.3f} (threshold={threshold_a})",
        fontsize=10, color=color_a
    )
    axes[0, 2].axis("off")

    # 행 2: Model 2-B
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title("Original", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(heatmap_b)
    axes[1, 1].set_title("Grad-CAM Heat Map\n(Model 2-B)", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(overlay_b)
    color_b = "red" if label_b == "moderate_severe" else "green"
    axes[1, 2].set_title(
        f"Model 2-B: {label_b}\n중증 확률: {prob_mod_sev:.3f} (threshold={threshold_b})",
        fontsize=10, color=color_b
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path}")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 모델 로드 ──
    print("\n[1/3] 모델 로드")

    model_a = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
    model_a.load_state_dict(torch.load(MODEL_A_PATH, weights_only=True))
    model_a.eval()

    model_b = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
    model_b.load_state_dict(torch.load(MODEL_B_PATH, weights_only=True))
    model_b.eval()

    # ── target layer 찾기 (EfficientNetV2-S 마지막 conv layer) ──
    # timm EfficientNetV2-S의 마지막 conv block
    target_layer_a = model_a.blocks[-1][-1].conv_pw
    target_layer_b = model_b.blocks[-1][-1].conv_pw

    gradcam_a = GradCAM(model_a, target_layer_a)
    gradcam_b = GradCAM(model_b, target_layer_b)

    print("  Model 2-A 로드 완료")
    print("  Model 2-B 로드 완료")

    # ── Transform ──
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── 이미지 수집 ──
    print("\n[2/3] 이미지 수집")
    image_paths = []
    for path in TEST_IMAGES:
        if os.path.isdir(path):
            files = [
                os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ]
            image_paths.extend(files[:NUM_IMAGES])
        elif os.path.isfile(path):
            image_paths.append(path)

    print(f"  총 {len(image_paths)}장")

    # ── Grad-CAM 시각화 ──
    print("\n[3/3] Grad-CAM 시각화")
    for i, img_path in enumerate(image_paths):
        fname = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{i+1:02d}_{fname}.png")
        print(f"\n  [{i+1}/{len(image_paths)}] {os.path.basename(img_path)}")
        try:
            visualize_single(
                img_path, gradcam_a, gradcam_b,
                THRESHOLD_A, THRESHOLD_B,
                LABEL_NAMES_A, LABEL_NAMES_B,
                transform, device, save_path
            )
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n완료! 결과 저장: {OUTPUT_DIR}")