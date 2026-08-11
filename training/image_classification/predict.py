"""
단일 이미지에서 아토피 유무를 예측하는 추론 스크립트.
모델 가중치: app/best_model.pth
설정:       app/model_config.json (threshold, model_name, img_size)
"""

import os
import json
import torch
import timm
from torchvision import transforms
from PIL import Image

# ── repo 루트 기준 경로 ──
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_APP_DIR   = os.path.join(_REPO_ROOT, "app")

_CFG_PATH   = os.path.join(_APP_DIR, "model_config.json")
_MODEL_PATH = os.path.join(_APP_DIR, "best_model.pth")

# ── model_config.json에서 설정 로드 ──
with open(_CFG_PATH, encoding="utf-8") as _f:
    _cfg = json.load(_f)

THRESHOLD   = _cfg["threshold"]
MODEL_NAME  = _cfg.get("model_name", "tf_efficientnetv2_s")
IMG_SIZE    = _cfg.get("img_size", 224)
LABEL_NAMES = _cfg.get("label_names", ["non_atopy", "atopy"])

# ── 모델 로드 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2).to(device)
model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image_path: str) -> dict:
    """이미지 파일 경로를 받아 {'label', 'atopy_prob', 'threshold'} 반환."""
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out  = model(tensor)
        prob = torch.softmax(out, dim=1)[0, 1].item()
    label = LABEL_NAMES[1] if prob >= THRESHOLD else LABEL_NAMES[0]
    return {"label": label, "atopy_prob": prob, "threshold": THRESHOLD}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        result = predict(path)
        print(f"예측: {result['label']}")
        print(f"아토피 확률: {result['atopy_prob']:.4f}")
        print(f"threshold: {result['threshold']}")
    else:
        print("사용법: python predict.py [이미지경로]")
