import torch
import timm
from torchvision import transforms
from PIL import Image

# ── 설정 ──
MODEL_PATH = r"E:\atopic\models\binary_grouped_split\tf_efficientnetv2_s\best_model.pth"
THRESHOLD = 0.2739  # Youden's J 최적값 (base-id 그룹 보존 split으로 재학습한 모델 기준)
IMG_SIZE = 224
LABEL_NAMES = ["non_atopy", "atopy"]

# ── 모델 로드 ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0, 1].item()  # 아토피 확률

    label = "atopy" if prob >= THRESHOLD else "non_atopy"
    return {"label": label, "atopy_prob": prob, "threshold": THRESHOLD}

# ── 테스트 ──
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