"""
recover_and_validate_split.py
==============================
binary 최종 모델(grouped split)의 split 복원 가능성 검증 스크립트.

수행 작업:
1. eval_comparison.py + make_grouped_split.py 로직을 그대로 재현해
   train/val/test/dermnet_holdout 후보 split을 생성
2. 각 샘플에 source, split, label, relative_path, filename, base_id, sha256을 기록한
   manifests/binary_split_candidate_seed42.csv 저장 (절대경로 미포함)
3. 후보 internal test + DermNet holdout에 binary grouped split 모델을 추론해
   E:\\atopic\\models\\binary_grouped_split\\final_summary.json 수치와 대조
4. 일치하면 recovered_final, 불일치하면 reconstructed_split_seed42 판정

실행:
    python training/image_classification/recover_and_validate_split.py \
        --model_path E:\\atopic\\models\\binary_grouped_split\\tf_efficientnetv2_s\\best_model.pth \
        --aihub_train  E:\\atopic\\AIHub\\Training\\01.원천데이터 \
        --aihub_val    E:\\atopic\\AIHub\\Validation\\01.원천데이터 \
        --dermnet      E:\\atopic\\dermnet_images

⚠ 기존 eval_comparison.py / make_grouped_split.py의 정렬·샘플링·split 동작을
  임의로 변경하지 않는다.
"""
import os, sys, re, csv, json, hashlib, argparse, random
import numpy as np
import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, average_precision_score,
)

REPO_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_TRAIN_DIR)
from eval_comparison import (
    Config, set_seed, collect_aihub_samples, build_aihub_binary,
    collect_dermnet_split,
)
from make_grouped_split import grouped_train_val_test_split, extract_base_id

IMG_SIZE = 224
BATCH_SIZE = 32
MANIFEST_DIR = os.path.join(REPO_TRAIN_DIR, "manifests")


# ── sha256 헬퍼 ──
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def to_relative(abs_path: str, data_root: str) -> str:
    """E:\\atopic\\AIHub\\Training\\...  →  AIHub/Training/..."""
    abs_path = os.path.normpath(abs_path)
    data_root = os.path.normpath(data_root)
    try:
        rel = os.path.relpath(abs_path, data_root)
        return rel.replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


# ── 추론 헬퍼 ──
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SkinDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def run_inference(model, samples, device):
    loader = DataLoader(SkinDataset(samples, eval_transform),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            prbs = torch.softmax(out, dim=1)
            all_labels.extend(lbls.numpy())
            all_probs.extend(prbs[:, 1].cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def mets(labels, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    auc = roc_auc_score(labels, probs)
    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"acc": round(float(acc), 4), "f1": round(float(f1), 4),
            "auc": round(float(auc), 4), "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4), "cm": cm.tolist(),
            "n": len(labels), "n_pos": int(labels.sum())}


def youden_threshold(labels, probs):
    fpr, tpr, thrs = roc_curve(labels, probs)
    idx = int(np.argmax(tpr - fpr))
    return float(thrs[idx])


def compare(actual, target, tol=0.001):
    diffs = {k: abs(actual.get(k, 0) - target.get(k, 0))
             for k in target if k in actual}
    return all(v <= tol for v in diffs.values()), diffs


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
        default=r"E:\atopic\models\binary_grouped_split\tf_efficientnetv2_s\best_model.pth")
    parser.add_argument("--aihub_train",
        default=r"E:\atopic\AIHub\Training\01.원천데이터")
    parser.add_argument("--aihub_val",
        default=r"E:\atopic\AIHub\Validation\01.원천데이터")
    parser.add_argument("--dermnet",
        default=r"E:\atopic\dermnet_images")
    parser.add_argument("--tol", type=float, default=0.001,
        help="metric tolerance for 'match' verdict")
    args = parser.parse_args()

    # 참조값 (final_summary.json)
    REFERENCE_INTERNAL = {
        "acc": 0.9511, "f1": 0.9511, "auc": 0.9874,
    }
    REFERENCE_EXTERNAL_YOUDEN = {
        "threshold": 0.27393072843551636,
        "acc": 0.7963, "f1": 0.8093, "auc": 0.8391,
        "sensitivity": 0.8846, "specificity": 0.7683,
    }
    REFERENCE_N = {"n_train": 2663, "n_val": 726,
                   "n_test_internal": 368, "n_test_external": 108}

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    set_seed(42)

    # ── 1. 데이터 수집 ──
    print("[1] AI Hub + DermNet 데이터 수집...")
    cfg = Config()
    cfg.TRAIN_ROOT   = args.aihub_train
    cfg.VAL_ROOT     = args.aihub_val
    cfg.DERMNET_ROOT = args.dermnet

    disease_files = collect_aihub_samples([cfg.TRAIN_ROOT, cfg.VAL_ROOT])
    aihub_samples = build_aihub_binary(disease_files, cfg.NON_ATOPY_COMPOSITION, cfg.SEED)

    paths  = [s[0] for s in aihub_samples]
    labels = [s[1] for s in aihub_samples]

    print(f"  AI Hub 총 {len(paths)}장  (아토피 {sum(labels)}장 / 비아토피 {len(labels)-sum(labels)}장)")

    # ── 2. grouped split ──
    print("\n[2] grouped split (seed=42, test_size=0.3, val_ratio_of_temp=1/3)...")
    train_p, val_p, test_p, train_l, val_l, test_l = grouped_train_val_test_split(
        paths, labels, test_size=0.3, val_ratio_of_temp=1/3, seed=cfg.SEED
    )

    dermnet_train, dermnet_test = collect_dermnet_split(
        cfg.DERMNET_ROOT, cfg.DERMNET_ATOPY, cfg.DERMNET_NON_ATOPY,
        cfg.DERMNET_TRAIN_RATIO, cfg.SEED
    )

    train_samples = list(zip(train_p, train_l)) + dermnet_train
    random.Random(cfg.SEED).shuffle(train_samples)
    val_samples  = list(zip(val_p, val_l))
    test_samples = list(zip(test_p, test_l))

    print(f"  Train {len(train_samples)} (AIHub {len(train_p)} + DermNet {len(dermnet_train)})")
    print(f"  Val   {len(val_samples)}")
    print(f"  Test internal {len(test_samples)}")
    print(f"  DermNet holdout {len(dermnet_test)}")

    # split count check
    n_match = (len(train_samples) == REFERENCE_N["n_train"] and
               len(val_samples)   == REFERENCE_N["n_val"]   and
               len(test_samples)  == REFERENCE_N["n_test_internal"] and
               len(dermnet_test)  == REFERENCE_N["n_test_external"])
    print(f"\n  참조 카운트 일치: {'✓' if n_match else '✗'}")
    if not n_match:
        print(f"    참조: train={REFERENCE_N['n_train']} val={REFERENCE_N['n_val']} "
              f"test_int={REFERENCE_N['n_test_internal']} test_ext={REFERENCE_N['n_test_external']}")

    # ── 3. 매니페스트 생성 ──
    print("\n[3] 매니페스트 생성 (sha256 포함, 절대경로 미포함)...")

    # data_root: E:\atopic
    data_root = os.path.dirname(os.path.normpath(args.aihub_train).split("AIHub")[0]
                                + "AIHub")
    data_root = os.path.dirname(args.aihub_train)  # E:\atopic\AIHub
    data_root = os.path.dirname(data_root)           # E:\atopic

    manifest_rows = []

    def add_rows(samples, split_tag, source_tag):
        for path, lbl in samples:
            fname = os.path.basename(path)
            stem  = os.path.splitext(fname)[0]
            bid   = extract_base_id(fname)
            rel   = to_relative(path, data_root)
            try:
                sha = sha256_file(path)
            except Exception:
                sha = "ERROR"
            manifest_rows.append({
                "source": source_tag,
                "split":  split_tag,
                "label":  lbl,
                "relative_path": rel,
                "filename": fname,
                "base_id": bid,
                "sha256": sha,
            })

    # AI Hub partitions
    add_rows(list(zip(train_p, train_l)), "train", "aihub")
    add_rows(val_samples,  "val",  "aihub")
    add_rows(test_samples, "test", "aihub")

    # DermNet train / holdout
    add_rows(dermnet_train, "train", "dermnet")
    add_rows(dermnet_test,  "dermnet_holdout", "dermnet")

    manifest_path = os.path.join(MANIFEST_DIR, "binary_split_candidate_seed42.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    abs_count = sum(1 for r in manifest_rows if r["relative_path"].startswith(("E:", "C:", "D:")))
    print(f"  저장: {manifest_path}")
    print(f"  총 행: {len(manifest_rows)} | 절대경로 잔존: {abs_count}")
    print(f"  SHA256 오류: {sum(1 for r in manifest_rows if r['sha256']=='ERROR')}")

    # split별 요약
    for sp in ["train", "val", "test", "dermnet_holdout"]:
        rows = [r for r in manifest_rows if r["split"] == sp]
        pos  = sum(1 for r in rows if int(r["label"]) == 1)
        print(f"  {sp:<18} {len(rows):>5}장  양성 {pos}")

    # ── 4. 모델 로드 ──
    print(f"\n[4] 모델 로드: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))

    # ── 5. 내부 test 추론 ──
    print("\n[5] 내부 test 추론...")
    int_labels, int_probs = run_inference(model, test_samples, device)
    int_metrics = mets(int_labels, int_probs, thr=0.5)
    print(f"  Acc={int_metrics['acc']:.4f}  F1={int_metrics['f1']:.4f}  AUC={int_metrics['auc']:.4f}")
    print(f"  n={int_metrics['n']}  n_pos={int_metrics['n_pos']}")

    # ── 6. DermNet holdout 추론 ──
    print("\n[6] DermNet holdout 추론...")
    ext_labels, ext_probs = run_inference(model, dermnet_test, device)
    youden_thr = youden_threshold(ext_labels, ext_probs)
    ext_metrics_05     = mets(ext_labels, ext_probs, thr=0.5)
    ext_metrics_youden = mets(ext_labels, ext_probs, thr=youden_thr)
    print(f"  Youden threshold: {youden_thr:.6f}  (참조: {REFERENCE_EXTERNAL_YOUDEN['threshold']:.6f})")
    print(f"  AUC={ext_metrics_youden['auc']:.4f}  Acc={ext_metrics_youden['acc']:.4f}  F1={ext_metrics_youden['f1']:.4f}")
    print(f"  Sensitivity={ext_metrics_youden['sensitivity']:.4f}  Specificity={ext_metrics_youden['specificity']:.4f}")
    print(f"  n={ext_metrics_youden['n']}  n_pos={ext_metrics_youden['n_pos']}")

    # ── 7. 기존 수치와 비교 ──
    print(f"\n[7] 참조 수치 대조 (허용 오차 ±{args.tol})")
    print("  내부 test (threshold=0.5 기준 — AUC/Acc/F1만 비교):")
    int_ok, int_diff = compare(
        {"acc": int_metrics["acc"], "f1": int_metrics["f1"], "auc": int_metrics["auc"]},
        {"acc": REFERENCE_INTERNAL["acc"], "f1": REFERENCE_INTERNAL["f1"],
         "auc": REFERENCE_INTERNAL["auc"]},
        tol=args.tol
    )
    for k, v in int_diff.items():
        mark = "✓" if v <= args.tol else "✗"
        print(f"    {mark} {k}: actual={int_metrics.get(k, '?'):.4f}  "
              f"ref={REFERENCE_INTERNAL.get(k, '?'):.4f}  diff={v:.4f}")

    print("  외부 test (Youden threshold — AUC/Acc/F1/Sens/Spec 비교):")
    ext_compare_actual = {k: ext_metrics_youden[k]
                          for k in ("acc", "f1", "auc", "sensitivity", "specificity")}
    ext_compare_ref    = {k: REFERENCE_EXTERNAL_YOUDEN[k]
                          for k in ("acc", "f1", "auc", "sensitivity", "specificity")}
    ext_ok, ext_diff = compare(ext_compare_actual, ext_compare_ref, tol=args.tol)
    for k, v in ext_diff.items():
        mark = "✓" if v <= args.tol else "✗"
        print(f"    {mark} {k}: actual={ext_compare_actual.get(k, '?'):.4f}  "
              f"ref={ext_compare_ref.get(k, '?'):.4f}  diff={v:.4f}")

    thr_diff = abs(youden_thr - REFERENCE_EXTERNAL_YOUDEN["threshold"])
    thr_mark = "✓" if thr_diff <= args.tol else "✗"
    print(f"    {thr_mark} threshold: actual={youden_thr:.6f}  "
          f"ref={REFERENCE_EXTERNAL_YOUDEN['threshold']:.6f}  diff={thr_diff:.6f}")

    # ── 8. 최종 판정 ──
    all_match = int_ok and ext_ok and n_match
    verdict = "RECOVERED_FINAL_SPLIT" if all_match else "RECONSTRUCTED_SPLIT_SEED42"
    print(f"\n{'='*60}")
    print(f"  최종 판정: {verdict}")
    if not all_match:
        reasons = []
        if not n_match:  reasons.append("샘플 수 불일치")
        if not int_ok:   reasons.append("내부 test 성능 불일치")
        if not ext_ok:   reasons.append("외부 test 성능 불일치")
        print(f"  불일치 원인: {', '.join(reasons)}")
        print("  → manifests/에 reconstructed_split_seed42로 보존. seed=42 단일 재현 시도.")
        print("  → GroupShuffleSplit의 비결정성 또는 파일 목록 정렬 차이 가능성.")
    print(f"{'='*60}")

    # ── 9. 결과 JSON 저장 ──
    result = {
        "verdict": verdict,
        "tol": args.tol,
        "n_match": n_match,
        "sample_counts": {
            "train_aihub": len(train_p),
            "train_dermnet": len(dermnet_train),
            "train_total": len(train_samples),
            "val": len(val_samples),
            "test_internal": len(test_samples),
            "dermnet_holdout": len(dermnet_test),
        },
        "reference_n": REFERENCE_N,
        "internal_test": {
            "threshold": 0.5,
            "actual": int_metrics,
            "reference": REFERENCE_INTERNAL,
            "diffs": int_diff,
            "match": int_ok,
        },
        "dermnet_holdout": {
            "youden_threshold_actual": youden_thr,
            "youden_threshold_ref": REFERENCE_EXTERNAL_YOUDEN["threshold"],
            "actual": ext_metrics_youden,
            "reference": {k: REFERENCE_EXTERNAL_YOUDEN[k]
                          for k in ("acc", "f1", "auc", "sensitivity", "specificity")},
            "diffs": ext_diff,
            "match": ext_ok,
        },
        "manifest_path": manifest_path.replace("\\", "/"),
    }
    result_json = os.path.join(MANIFEST_DIR, "binary_split_validation_result.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {result_json}")


if __name__ == "__main__":
    main()
