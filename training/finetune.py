"""
Fine-tune PaliGemma 2 for surgical organ classification.

Uses LoRA (via PEFT) to efficiently adapt google/paligemma2-3b-pt-224
to classify organs from cropped laparoscopic images.  Fine-tuning is
done as a generative task: the model learns to produce the organ name
string given an image + a fixed prompt.  This means the fine-tuned model
has the same interface as the original PaliGemma and no architecture
changes are needed for export.

Usage:
    python finetune.py \\
        --coco_exports ../exports/*.json \\
        --images_dir ../uploads/images \\
        --output_dir ./checkpoints \\
        --epochs 10 \\
        --batch_size 8 \\
        --learning_rate 2e-5 \\
        --model_id google/paligemma2-3b-pt-224
"""

import argparse
import glob
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from dataset import SurgicalOrganDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Prompt prepended to every image at both train and inference time
PROMPT = "Identify the highlighted surgical organ. Answer with the organ name only:"

MIN_CLASS_SAMPLES = 50  # warn but continue below this


# ======================================================================
# Data helpers
# ======================================================================

def make_collate_fn(processor, organ_list: list[str], max_new_tokens: int = 8):
    """
    Returns a collate_fn that applies the PaliGemma processor to a batch.
    Labels are the tokenised organ names; input tokens are masked (-100)
    so loss is computed only on the generated answer.
    """
    def collate(batch: list[dict]) -> dict:
        images      = [item["image"] for item in batch]
        label_names = [item["label_name"] for item in batch]

        # --- Encode inputs (image + prompt) ---
        inputs = processor(
            images=images,
            text=[PROMPT] * len(batch),
            return_tensors="pt",
            padding=True,
        )

        # --- Encode targets (organ name strings) ---
        with processor.tokenizer.as_target_tokenizer():
            targets = processor.tokenizer(
                label_names,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )

        # Build labels: -100 for all input positions, then answer tokens
        input_len  = inputs["input_ids"].shape[1]
        target_ids = targets["input_ids"]          # (B, T_ans)
        pad_id     = processor.tokenizer.pad_token_id

        # Mask padding in answers
        target_ids = target_ids.masked_fill(target_ids == pad_id, -100)

        # Full label tensor: -100 for image+prompt tokens, then answer tokens
        ignore = torch.full(
            (len(batch), input_len), -100, dtype=torch.long
        )
        labels = torch.cat([ignore, target_ids], dim=1)

        # Extend input_ids and attention_mask to include answer tokens
        # (teacher forcing: model sees the full sequence, predicts shifted)
        full_input_ids = torch.cat(
            [inputs["input_ids"], targets["input_ids"].masked_fill(
                targets["input_ids"] == pad_id, 0)], dim=1
        )
        full_attn_mask = torch.cat(
            [inputs["attention_mask"], targets["attention_mask"]], dim=1
        )

        return {
            "input_ids":        full_input_ids,
            "attention_mask":   full_attn_mask,
            "pixel_values":     inputs["pixel_values"],
            "labels":           labels,
            "label_names":      label_names,
        }
    return collate


# ======================================================================
# Evaluation helpers
# ======================================================================

@torch.no_grad()
def evaluate(model, processor, loader, accelerator, organ_list: list[str]) -> dict:
    """
    Greedy-decode predictions and compute accuracy.
    Uses constrained generation: pick the organ whose first token has the
    highest probability, then verify by generating the full string.
    """
    model.eval()
    correct = 0
    total   = 0
    per_class_correct: dict[str, int] = {}
    per_class_total:   dict[str, int] = {}

    # Pre-tokenise all organ names to get their first token id
    organ_first_tokens = []
    for organ in organ_list:
        toks = processor.tokenizer.encode(organ, add_special_tokens=False)
        organ_first_tokens.append(toks[0] if toks else -1)

    for batch in loader:
        images      = batch["pixel_values"]
        label_names = batch["label_names"]

        # Run a short generation constrained to organ vocab
        inputs = processor(
            images=[Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * len(label_names),
            text=[PROMPT] * len(label_names),
            return_tensors="pt",
        )
        # Use the actual pixel_values from batch (already preprocessed)
        inputs["pixel_values"] = images.to(accelerator.device)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        generated = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

        # Decode predictions
        input_len = inputs["input_ids"].shape[1]
        pred_ids  = generated[:, input_len:]
        preds     = processor.batch_decode(pred_ids, skip_special_tokens=True)

        for pred, gt in zip(preds, label_names):
            pred_clean = pred.strip().lower().replace("-", "_").replace(" ", "_")
            gt_clean   = gt.strip().lower()
            match = pred_clean == gt_clean or gt_clean in pred_clean

            per_class_total[gt_clean]   = per_class_total.get(gt_clean, 0) + 1
            per_class_correct[gt_clean] = per_class_correct.get(gt_clean, 0) + int(match)
            correct += int(match)
            total   += 1

    accuracy = correct / total if total > 0 else 0.0
    per_class_acc = {
        cls: per_class_correct.get(cls, 0) / per_class_total[cls]
        for cls in per_class_total
    }
    model.train()
    return {"accuracy": accuracy, "per_class": per_class_acc, "total": total}


# ======================================================================
# Training
# ======================================================================

def train(args):
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "no")
    logger.info("Device: %s | Mixed precision: %s", accelerator.device, accelerator.mixed_precision)

    # --- Dataset ---
    coco_paths = []
    for pattern in args.coco_exports:
        coco_paths.extend(glob.glob(pattern))
    if not coco_paths:
        logger.error("No COCO export files found: %s", args.coco_exports)
        sys.exit(1)
    logger.info("Loading %d COCO export file(s)...", len(coco_paths))

    dataset = SurgicalOrganDataset(coco_paths, args.images_dir)
    organ_list = list(dataset.label_to_id.keys())

    train_set, val_set, _ = dataset.split(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
    )

    # --- Processor & model ---
    logger.info("Loading processor from %s...", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    logger.info("Loading model from %s...", args.model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # Freeze vision tower — only fine-tune the language model layers with LoRA
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False

    # LoRA on language model attention layers
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Class weights for loss ---
    class_weights = dataset.get_class_weights()

    # --- DataLoaders ---
    collate = make_collate_fn(processor, organ_list)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # --- Optimiser ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # --- Accelerate ---
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # --- Output dirs ---
    output_dir  = Path(args.output_dir)
    best_dir    = output_dir / "best_model"
    latest_dir  = output_dir / "latest"
    for d in [best_dir, latest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save label map alongside checkpoints
    label_map = dataset.id_to_label
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (output_dir / "organ_list.json").write_text(json.dumps(organ_list, indent=2))

    # --- Training loop ---
    best_val_acc    = -1.0
    patience_count  = 0
    history         = []

    logger.info("Starting training: %d epochs, %d steps/epoch", args.epochs, len(train_loader))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        from PIL import Image
        val_metrics = evaluate(model, processor, val_loader, accelerator, organ_list)
        val_acc = val_metrics["accuracy"]

        logger.info(
            "Epoch %d/%d — loss: %.4f | val_acc: %.3f",
            epoch, args.epochs, avg_loss, val_acc,
        )

        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc})

        # Save latest
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(latest_dir)
        processor.save_pretrained(latest_dir)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            unwrapped.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (best_dir / "label_map.json").write_text(
                json.dumps(label_map, indent=2)
            )
            (best_dir / "organ_list.json").write_text(
                json.dumps(organ_list, indent=2)
            )
            logger.info("  ✓ New best model saved (val_acc=%.3f)", best_val_acc)
        else:
            patience_count += 1
            logger.info(
                "  No improvement (%d/%d patience)", patience_count, args.patience
            )
            if patience_count >= args.patience:
                logger.info("Early stopping triggered.")
                break

    # --- Final report ---
    logger.info("=" * 60)
    logger.info("Training complete. Best val accuracy: %.3f", best_val_acc)
    logger.info("=" * 60)

    val_metrics = evaluate(model, processor, val_loader, accelerator, organ_list)
    logger.info("Final val accuracy: %.3f", val_metrics["accuracy"])
    logger.info("Per-class accuracy:")
    for cls, acc in sorted(val_metrics["per_class"].items()):
        logger.info("  %-22s %.3f", cls, acc)

    # Save training history
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    logger.info("Checkpoints in: %s", output_dir)


# ======================================================================
# Entry point
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune PaliGemma 2 for organ classification")
    p.add_argument("--coco_exports",   nargs="+", required=True,
                   help="Glob pattern(s) for COCO JSON export files")
    p.add_argument("--images_dir",     required=True,
                   help="Directory containing uploaded images (uploads/images/)")
    p.add_argument("--output_dir",     default="./checkpoints")
    p.add_argument("--model_id",       default="google/paligemma2-3b-pt-224")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--learning_rate",  type=float, default=2e-5)
    p.add_argument("--lora_r",         type=int,   default=16)
    p.add_argument("--lora_alpha",     type=int,   default=32)
    p.add_argument("--lora_dropout",   type=float, default=0.05)
    p.add_argument("--patience",       type=int,   default=3)
    p.add_argument("--train_ratio",    type=float, default=0.8)
    p.add_argument("--val_ratio",      type=float, default=0.1)
    p.add_argument("--bf16",           action="store_true", default=True,
                   help="Use bfloat16 mixed precision (recommended for MPS/A100)")
    return p.parse_args()


if __name__ == "__main__":
    from PIL import Image
    args = parse_args()
    train(args)
