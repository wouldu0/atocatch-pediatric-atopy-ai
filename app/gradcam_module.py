"""
gradcam_module.py — AtoCatch Grad-CAM 모듈
EfficientNetV2 기반 아토피 여부 + IGA 중증도 모델에 Grad-CAM 적용

사용법:
    from gradcam_module import load_models, predict_with_gradcam
"""

import torch
import torch.nn.functional as F
import numpy as np
import timm
from PIL import Image
import torchvision.transforms as transforms
import cv2


# ── 전처리 ──────────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── 모델 로드 ────────────────────────────────────────────────
def load_models(model_a_path: str, model_b_path: str, device=None):
    """
    아토피 여부(model_a)와 IGA 중증도(model_b) 모델을 로드합니다.

    Args:
        model_a_path: best_model.pth 경로
        model_b_path: best_iga_model.pth 경로
        device: torch.device (None이면 자동 선택)

    Returns:
        model_a, model_b (eval 모드)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load(path):
        m = timm.create_model("tf_efficientnetv2_s",
                               pretrained=False, num_classes=2)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        m.to(device)
        return m

    model_a = _load(model_a_path)
    model_b = _load(model_b_path)
    return model_a, model_b


# ── Grad-CAM 핵심 ────────────────────────────────────────────
class GradCAM:
    """
    EfficientNetV2 전용 Grad-CAM.
    마지막 Conv 블록(blocks[-1])의 출력을 타겟 레이어로 사용합니다.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # EfficientNetV2-S: timm 기준 마지막 feature 블록
        target_layer = self.model.blocks[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(
            target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(
            target_layer.register_full_backward_hook(backward_hook))

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Grad-CAM 히트맵(0~1, float32) 반환.

        Args:
            tensor: 전처리된 이미지 텐서 [1, C, H, W]
            class_idx: 관심 클래스 인덱스 (1 = 양성)

        Returns:
            cam: [H_orig, W_orig] numpy 배열 (0~1)
        """
        self.model.zero_grad()
        output = self.model(tensor)
        score = output[0, class_idx]
        score.backward()

        # 가중 평균
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1,C,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1,1,H,W]
        cam = F.relu(cam)

        # 정규화
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


# ── 오버레이 생성 ─────────────────────────────────────────────
def _make_overlay(original_pil: Image.Image,
                  cam: np.ndarray,
                  alpha: float = 0.45) -> Image.Image:
    """
    Grad-CAM 히트맵을 원본 이미지에 오버레이한 PIL Image 반환.

    Args:
        original_pil: 원본 PIL Image (RGB)
        cam: Grad-CAM 배열 (0~1)
        alpha: 히트맵 투명도 (0=원본만, 1=히트맵만)

    Returns:
        overlay PIL Image (RGB)
    """
    # 원본 크기에 맞게 리사이즈
    w, h = original_pil.size
    cam_resized = cv2.resize(cam, (w, h))

    # JET 컬러맵 적용
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 원본과 합성
    orig_np = np.array(original_pil.convert("RGB")).astype(np.float32)
    overlay = orig_np * (1 - alpha) + heatmap.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# ── 메인 추론 함수 ────────────────────────────────────────────
def predict_with_gradcam(
    image: Image.Image,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    device=None,
    atopy_threshold: float = 0.2739,
    severity_threshold: float = 0.38,
    overlay_alpha: float = 0.45,
) -> dict:
    """
    아토피 여부 + IGA 중증도 예측 + Grad-CAM 히트맵 생성.

    Args:
        image: PIL Image (RGB)
        model_a: 아토피 여부 모델
        model_b: IGA 중증도 모델
        device: torch.device
        atopy_threshold: 아토피 의심 임계값
        severity_threshold: 중등도·중증 임계값
        overlay_alpha: 히트맵 투명도

    Returns:
        dict:
            atopy_prob        (float)  아토피 확률
            atopy_label       (str)    "아토피 의심" / "아토피 낮음"
            is_atopy          (bool)
            gradcam_a_cam     (np.ndarray)  raw cam
            gradcam_a_overlay (PIL Image)   오버레이 이미지

            severity          (bool | None)  None = 분석 안 함
            severity_prob     (float | None)
            severity_label    (str | None)   "중등도·중증" / "경증 이하"
            gradcam_b_cam     (np.ndarray | None)
            gradcam_b_overlay (PIL Image | None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = IMG_TRANSFORM(image).unsqueeze(0).to(device)

    result = {}

    # ── 1단계: 아토피 여부 ──
    gcam_a = GradCAM(model_a)
    try:
        with torch.enable_grad():
            tensor_a = tensor.clone().requires_grad_(True)
            with torch.no_grad():
                pass  # forward는 GradCAM 내부에서
            output_a = model_a(tensor_a)
            probs_a = torch.softmax(output_a, dim=1)[0]
        atopy_prob = float(probs_a[1])

        # Grad-CAM
        tensor_gc = IMG_TRANSFORM(image).unsqueeze(0).to(device).requires_grad_(True)
        cam_a = gcam_a.generate(tensor_gc, class_idx=1)
        overlay_a = _make_overlay(image, cam_a, alpha=overlay_alpha)
    finally:
        gcam_a.remove_hooks()

    is_atopy = atopy_prob >= atopy_threshold
    result.update({
        "atopy_prob":        atopy_prob,
        "atopy_label":       "아토피 의심" if is_atopy else "아토피 낮음",
        "is_atopy":          is_atopy,
        "gradcam_a_cam":     cam_a,
        "gradcam_a_overlay": overlay_a,
    })

    # ── 2단계: IGA 중증도 (아토피 의심 시에만) ──
    if is_atopy:
        gcam_b = GradCAM(model_b)
        try:
            tensor_gc2 = IMG_TRANSFORM(image).unsqueeze(0).to(device).requires_grad_(True)
            cam_b = gcam_b.generate(tensor_gc2, class_idx=1)
            overlay_b = _make_overlay(image, cam_b, alpha=overlay_alpha)

            with torch.no_grad():
                output_b = model_b(tensor)
                probs_b = torch.softmax(output_b, dim=1)[0]
            severity_prob = float(probs_b[1])
        finally:
            gcam_b.remove_hooks()

        is_severe = severity_prob >= severity_threshold
        result.update({
            "severity":          is_severe,
            "severity_prob":     severity_prob,
            "severity_label":    "중등도·중증" if is_severe else "경증 이하",
            "gradcam_b_cam":     cam_b,
            "gradcam_b_overlay": overlay_b,
        })
    else:
        result.update({
            "severity":          None,
            "severity_prob":     None,
            "severity_label":    None,
            "gradcam_b_cam":     None,
            "gradcam_b_overlay": None,
        })

    return result
