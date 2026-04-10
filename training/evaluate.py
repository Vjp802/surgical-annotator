"""
Evaluate a fine-tuned PaliGemma 2 checkpoint against a COCO export.

Usage:
    python evaluate.py \\
        --checkpoint ./checkpoints/best_model \\
        --coco_exports ../exports/*.json \\
        --images_dir ../uploads/images
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

from dataset import SurgicalOrganDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

PROMPT = "Identify the highlighted surgical organ. Answer with the organ name only:"


# ======================================================================
# Inference
# ======================================================================

@torch.no_grad()
def run_inference(model, processor, loader, device, organ_list: list[str]) -> tuple[list, list]:
    """Returns (y_true_names, y_pred_names)."""
    model.eval()
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        images      = batch["pixel_values"].to(device)
        label_names = batch["label_names"]

        inputs = processor(
            images=None,
            text=[PROMPT] * len(label_names),
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        inputs["pixel_values"] = images

        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

        input_len = inputs["input_ids"].shape[1]
        pred_ids  = generated[:, input_len:]
        preds     = processor.batch_decode(pred_ids, skip_special_tokens=True)

        for pred, gt in zip(preds, label_names):
            pred_clean = pred.strip().lower().replace("-", "_").replace(" ", "_")
            # Snap to nearest known organ if exact match fails
            if pred_clean not in organ_list:
                pred_clean = _snap_to_organ(pred_clean, organ_list)
            y_pred.append(pred_clean)
            y_true.append(gt.strip().lower())

    return y_true, y_pred


def _snap_to_organ(pred: str, organ_list: list[str]) -> str:
    """Return the organ name that shares the most characters with pred."""
    best, best_score = "unknown", 0
    for organ in organ_list:
        score = sum(c in organ for c in pred)
        if score > best_score:
            best, best_score = organ, score
    return best if best_score > 0 else "unknown"


# ======================================================================
# Reporting
# ======================================================================

def print_report(y_true, y_pred, organ_list: list[str], output_dir: Path):
    labels_present = sorted(set(y_true) | set(y_pred))

    # Overall accuracy
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)

    # Baselines
    from collections import Counter
    most_common = Counter(y_true).most_common(1)[0][0]
    majority_acc = sum(t == most_common for t in y_true) / len(y_true)
    random_acc   = 1.0 / len(set(y_true)) if y_true else 0.0

    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info("Overall accuracy : %.3f  (%d/%d)", accuracy, int(accuracy * len(y_true)), len(y_true))
    logger.info("Baseline — random: %.3f", random_acc)
    logger.info("Baseline — majority class (%s): %.3f", most_common, majority_acc)
    logger.info("Improvement over random:   +%.3f", accuracy - random_acc)
    logger.info("Improvement over majority: +%.3f", accuracy - majority_acc)

    # Per-class report
    report = classification_report(y_true, y_pred, labels=labels_present, zero_division=0, output_dict=True)
    logger.info("\nPer-class metrics:")
    logger.info("  %-22s  %6s  %6s  %6s  %6s", "Class", "Prec", "Rec", "F1", "N")
    for cls in sorted(labels_present):
        m = report.get(cls, {})
        f1   = m.get("f1-score",  0.0)
        flag = " ⚠ needs more data" if f1 < 0.7 else ""
        logger.info(
            "  %-22s  %6.3f  %6.3f  %6.3f  %6d%s",
            cls, m.get("precision", 0), m.get("recall", 0), f1,
            int(m.get("support", 0)), flag,
        )

    low_f1 = [cls for cls in labels_present if report.get(cls, {}).get("f1-score", 1.0) < 0.7]
    if low_f1:
        logger.warning(
            "\nClasses with F1 < 0.7 (collect more annotated data): %s",
            ", ".join(low_f1),
        )

    # --- Confusion matrix ---
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    fig, ax = plt.subplots(figsize=(max(8, len(labels_present)), max(6, len(labels_present) - 2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels_present)),
        yticks=range(len(labels_present)),
        xticklabels=labels_present,
        yticklabels=labels_present,
        ylabel="True label",
        xlabel="Predicted label",
        title="Organ Classification — Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("\nConfusion matrix saved to: %s", cm_path)

    # --- metrics.json ---
    metrics = {
        "accuracy": accuracy,
        "baselines": {"random": random_acc, "majority_class": majority_acc},
        "improvement": {
            "over_random":   accuracy - random_acc,
            "over_majority": accuracy - majority_acc,
        },
        "per_class": {
            cls: {
                "precision": report.get(cls, {}).get("precision", 0),
                "recall":    report.get(cls, {}).get("recall",    0),
                "f1":        report.get(cls, {}).get("f1-score",  0),
                "support":   int(report.get(cls, {}).get("support", 0)),
            }
            for cls in labels_present
        },
        "low_f1_classes": low_f1,
        "n_samples": len(y_true),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Full metrics saved to: %s", metrics_path)
    return metrics


# ======================================================================
# Main
# ======================================================================

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # --- Load dataset ---
    coco_paths = []
    for pattern in args.coco_exports:
        coco_paths.extend(glob.glob(pattern))
    if not coco_paths:
        logger.error("No COCO files found: %s", args.coco_exports)
        sys.exit(1)

    dataset = SurgicalOrganDataset(coco_paths, args.images_dir)

    # Load organ list from checkpoint if available
    organ_list_path = Path(args.checkpoint) / "organ_list.json"
    if organ_list_path.exists():
        organ_list = json.loads(organ_list_path.read_text())
        logger.info("Loaded organ list from checkpoint (%d organs)", len(organ_list))
    else:
        organ_list = list(dataset.label_to_id.keys())

    _, _, test_set = dataset.split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
    )

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b),
    )

    # --- Load model ---
    logger.info("Loading checkpoint from %s...", args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    base = PaliGemmaForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, args.checkpoint)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded.")

    # --- Run inference ---
    y_true, y_pred = run_inference(model, processor, loader, device, organ_list)

    if not y_true:
        logger.error("Test set is empty.")
        sys.exit(1)

    # --- Report ---
    output_dir = Path(args.output_dir)
    print_report(y_true, y_pred, organ_list, output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned organ classifier")
    p.add_argument("--checkpoint",     required=True,  help="Path to fine-tuned LoRA checkpoint")
    p.add_argument("--coco_exports",   nargs="+", required=True)
    p.add_argument("--images_dir",     required=True)
    p.add_argument("--output_dir",     default="./evaluation_results")
    p.add_argument("--base_model_id",  default="google/paligemma2-3b-pt-224")
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--train_ratio",    type=float, default=0.8)
    p.add_argument("--val_ratio",      type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
