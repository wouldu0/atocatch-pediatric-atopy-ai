"""
base-id 그룹 leakage 수정 재학습
==============================================================
배경: check_aihub_subject_leakage.py / make_grouped_split.py 검증으로
data_split.py / eval_comparison.py의 순수 랜덤 split이 AI Hub 전 클래스에서
base-id 그룹(같은 합성 피험자의 P/L 변형)을 train/val/test에 걸쳐 쪼개는
leakage를 만든다는 게 확인됨.

이 스크립트는 eval_comparison.py의 학습/평가 루프(train_one_epoch, validate,
evaluate, run_experiment)를 그대로 import해서 재사용하고, 데이터 split
부분만 make_grouped_split.py의 grouped_train_val_test_split()으로 교체한다.
모델 구조/augmentation/loss/optimizer는 전혀 건드리지 않음.

배포된 기존 모델(comparison/tf_efficientnetv2_s/best_model.pth)과 backup/ 안의
산출물은 절대 건드리지 않고, 결과는 전부 별도 OUTPUT_DIR에 새로 저장한다.

경로: eval_comparison.py / make_grouped_split.py는 이 스크립트와 같은 폴더에
있다는 전제로 저장소 루트 기준 상대경로를 쓴다 (개인 PC 절대경로 없음).
AI Hub/DermNet 원본 데이터 루트만 Config()(eval_comparison.py) 쪽에서
본인 환경에 맞게 수정할 것 — 이 저장소에 포함되지 않는 외부 대용량 데이터라
상대경로로 고정할 수 없다.
"""
import os
import sys
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # training/image_classification/
sys.path.insert(0, str(BASE_DIR))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, confusion_matrix

from eval_comparison import (
    Config, set_seed, collect_aihub_samples, build_aihub_binary,
    collect_dermnet_split, SkinDataset, get_transforms, run_experiment,
)
from make_grouped_split import grouped_train_val_test_split

OUTPUT_DIR = str(BASE_DIR / "outputs" / "binary_grouped_split")  # ← 필요하면 본인 경로로 수정
MODEL_NAME = "tf_efficientnetv2_s"


def main():
    cfg = Config()
    cfg.OUTPUT_DIR = OUTPUT_DIR
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 데이터 준비: grouped split으로 교체 ──
    print("\n[데이터 준비 - grouped split]")
    disease_files = collect_aihub_samples([cfg.TRAIN_ROOT, cfg.VAL_ROOT])
    aihub_samples = build_aihub_binary(disease_files, cfg.NON_ATOPY_COMPOSITION, cfg.SEED)

    paths = [s[0] for s in aihub_samples]
    labels = [s[1] for s in aihub_samples]

    train_p, val_p, test_p, train_l, val_l, test_l = grouped_train_val_test_split(
        paths, labels, test_size=0.3, val_ratio_of_temp=1 / 3, seed=cfg.SEED
    )

    dermnet_train, dermnet_test = collect_dermnet_split(
        cfg.DERMNET_ROOT, cfg.DERMNET_ATOPY, cfg.DERMNET_NON_ATOPY,
        cfg.DERMNET_TRAIN_RATIO, cfg.SEED
    )

    train_samples = list(zip(train_p, train_l)) + dermnet_train
    random.Random(cfg.SEED).shuffle(train_samples)
    val_samples = list(zip(val_p, val_l))
    test_samples = list(zip(test_p, test_l))

    print(f"  Train: {len(train_samples)}장 (AI Hub {len(train_p)} + DermNet {len(dermnet_train)})")
    print(f"  Val:   {len(val_samples)}장")
    print(f"  Test 내부: {len(test_samples)}장")
    print(f"  Test 외부(DermNet holdout): {len(dermnet_test)}장")

    train_loader = DataLoader(
        SkinDataset(train_samples, get_transforms(cfg.IMG_SIZE, True)),
        batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(
        SkinDataset(val_samples, get_transforms(cfg.IMG_SIZE, False)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        SkinDataset(test_samples, get_transforms(cfg.IMG_SIZE, False)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    dermnet_loader = DataLoader(
        SkinDataset(dermnet_test, get_transforms(cfg.IMG_SIZE, False)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # ── 학습 (eval_comparison.py의 run_experiment 그대로 재사용) ──
    result = run_experiment(
        MODEL_NAME, train_loader, val_loader, test_loader, dermnet_loader,
        cfg, device, cfg.OUTPUT_DIR
    )

    with open(os.path.join(cfg.OUTPUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n[학습 완료] internal:", result["internal"])
    print("[학습 완료] external(threshold 0.5 기준):", result["external"])

    # ── threshold 재탐색 (utils_threshold.py 로직 재사용, 새 체크포인트에 적용) ──
    print("\n[threshold 재탐색 - utils_threshold.py 로직]")
    best_path = os.path.join(cfg.OUTPUT_DIR, MODEL_NAME, "best_model.pth")

    import timm
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels_b in dermnet_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels_b.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    best_threshold_youden = float(thresholds[best_idx])

    f1_scores = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1_scores.append(f1_score(all_labels, preds, average="weighted"))
    best_idx_f1 = int(np.argmax(f1_scores))
    best_threshold_f1 = float(thresholds[best_idx_f1])

    def metrics_at(t):
        preds = (all_probs >= t).astype(int)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, average="weighted")
        cm = confusion_matrix(all_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        return {"threshold": float(t), "acc": float(acc), "f1": float(f1),
                "auc": float(auc), "sensitivity": float(sens), "specificity": float(spec)}

    youden_metrics = metrics_at(best_threshold_youden)
    f1_metrics = metrics_at(best_threshold_f1)

    print(f"AUC: {auc:.4f}")
    print(f"Youden's J 최적 threshold: {best_threshold_youden:.4f} -> {youden_metrics}")
    print(f"F1 최적 threshold: {best_threshold_f1:.4f} -> {f1_metrics}")

    final = {
        "internal": result["internal"],
        "external_default_argmax": result["external"],
        "external_youden": youden_metrics,
        "external_f1_optimal": f1_metrics,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test_internal": len(test_samples),
        "n_test_external": len(dermnet_test),
    }
    with open(os.path.join(cfg.OUTPUT_DIR, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("\n[전체 완료] 최종 요약을 final_summary.json 에 저장했습니다.")
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
