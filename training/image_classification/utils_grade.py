# gradcam_module.py
# Streamlit에서 import해서 쓰는 용도

import numpy as np
import matplotlib.cm as cm
import torch
from torchvision import transforms
from PIL import Image
import timm
import io
import base64


# ── 모델 로드 (한 번만 로드하도록 캐싱) ──
def load_models(model_a_path, model_b_path, device):
    model_a = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
    model_a.load_state_dict(torch.load(model_a_path, weights_only=True, map_location=device))
    model_a.eval()

    model_b = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
    model_b.load_state_dict(torch.load(model_b_path, weights_only=True, map_location=device))
    model_b.eval()

    return model_a, model_b


# ── Grad-CAM ──
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach())
        )

    def generate(self, input_tensor, class_idx=1):
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients[0].mean(dim=(1, 2))
        cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)
        cam = torch.relu(cam).cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, probs


def overlay_heatmap(original_img, cam, alpha=0.5):
    """원본 PIL 이미지에 Grad-CAM 오버레이 → PIL 이미지 반환"""
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            original_img.size, Image.BILINEAR
        )
    ) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap = (colormap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    orig_array = np.array(original_img)
    overlay = (alpha * heatmap + (1 - alpha) * orig_array).astype(np.uint8)

    return Image.fromarray(overlay), Image.fromarray(heatmap)


# ── 메인 추론 함수 (Streamlit에서 이것만 호출하면 됨) ──
def predict_with_gradcam(
    pil_image,
    model_a, model_b,
    threshold_a=0.29, threshold_b=0.38,
    img_size=224,
    device=None
):
    """
    입력: PIL Image
    출력: 딕셔너리
        - atopy: bool
        - atopy_prob: float
        - severity: str or None (아토피가 아니면 None)
        - severity_prob: float or None
        - gradcam_a_overlay: PIL Image
        - gradcam_b_overlay: PIL Image or None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # ── Model 2-A: 아토피 여부 ──
    gradcam_a = GradCAM(model_a, model_a.blocks[-1][-1].conv_pw)
    cam_a, probs_a = gradcam_a.generate(input_tensor, class_idx=1)
    prob_atopy = float(probs_a[1])
    is_atopy = prob_atopy >= threshold_a
    overlay_a, heatmap_a = overlay_heatmap(pil_image, cam_a)

    result = {
        "atopy": is_atopy,
        "atopy_prob": prob_atopy,
        "atopy_label": "아토피 의심" if is_atopy else "아토피 아님",
        "gradcam_a_overlay": overlay_a,
        "gradcam_a_heatmap": heatmap_a,
        "severity": None,
        "severity_prob": None,
        "severity_label": None,
        "gradcam_b_overlay": None,
        "gradcam_b_heatmap": None,
    }

    # ── Model 2-B: 중증도 (아토피인 경우만) ──
    if is_atopy:
        gradcam_b = GradCAM(model_b, model_b.blocks[-1][-1].conv_pw)
        cam_b, probs_b = gradcam_b.generate(input_tensor, class_idx=1)
        prob_severe = float(probs_b[1])
        is_severe = prob_severe >= threshold_b
        overlay_b, heatmap_b = overlay_heatmap(pil_image, cam_b)

        result.update({
            "severity": "moderate_severe" if is_severe else "mild_or_below",
            "severity_prob": prob_severe,
            "severity_label": "중등증-중증 (전문의 진료 권장)" if is_severe else "경증 (보습제 관리 권장)",
            "gradcam_b_overlay": overlay_b,
            "gradcam_b_heatmap": heatmap_b,
        })

    return result