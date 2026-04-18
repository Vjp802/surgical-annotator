"""
Fine-tune PaliGemma 2 for surgical organ classification.

Uses LoRA (via PEFT) to efficiently adapt google/paligemma2-3b-pt-224
to classify organs from cropped laparoscopic images.  Fine-tuning is
done as a generative task: the model learns to produce the organ name
string given an image + a fixed prompt.  This means the fine-tuned model
has the same interface as the original PaliGemma and no architecture
changes are needed for export.

Single-dataset usage (legacy):
    python finetune.py \\
        --coco_exports ../exports/*.json \\
        --images_dir ../uploads/images \\
        --output_dir ./checkpoints

Multi-dataset usage with per-dataset weights:
    python finetune.py \\
        --datasets \\
            ./datasets/dsad/annotations.json:./datasets/dsad/images:3.0 \\
            ./exports/surgeon_corrections.json:./uploads/images:5.0 \\
            ./datasets/cholec80/annotations.json:./datasets/cholec80/images:1.0 \\
        --output_dir ./checkpoints

Dataset weight guidance:
    dsad:                 3.0  (colorectal-specific, expert labels)
    surgeon_corrections:  5.0  (ground truth, highest priority)
    cholec_seg8k:         1.0  (general surgical)
    surgisr4k:            0.0  (no labels — skip)
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
# Weighted multi-dataset support
# ======================================================================

class WeightedSurgicalDataset(torch.utils.data.Dataset):
    """
    Combines multiple COCO JSON datasets with per-dataset sampling weights.

    Higher weight → that dataset is sampled more frequently, which lets you
    up-weight high-quality or domain-specific datasets (e.g. DSAD for
    colorectal procedures) and down-weight noisier or less relevant ones.

    Recommended weights for colorectal surgery fine-tuning:
        dsad:                 3.0  (colorectal-specific, expert pixel labels)
        surgeon_corrections:  5.0  (your own ground-truth annotations)
        cholec_seg8k:         1.0  (general laparoscopic, lower priority)
        surgisr4k:            0.0  (no labels — exclude)

    How it works
    ------------
    Each dataset i has n_i samples and weight w_i.  The effective size of
    the combined dataset is set to  sum(w_i * n_i) / max(w_i), i.e. the
    "heaviest" dataset is fully traversed once per epoch and lighter ones
    are proportionally sub-sampled.  Indices are drawn from a pre-built
    probability table so PyTorch's default sequential / shuffle samplers
    work without modification.
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        transform=None,
    ):
        """
        Args:
            dataset_configs: List of dicts, each with:
                - coco_json_path (str): Path to COCO JSON file.
                - images_dir     (str): Directory containing the images.
                - weight       (float): Sampling weight for this dataset.
                - name          (str):  Human-readable name for logging.
            transform: Optional image transform applied after crop/overlay.
        """
        if not dataset_configs:
            raise ValueError("dataset_configs must contain at least one entry.")

        self.transform = transform
        self._sub_datasets: list[SurgicalOrganDataset] = []
        self._weights:      list[float]                = []
        self._names:        list[str]                  = []

        # ---- Load sub-datasets ----------------------------------------
        for cfg in dataset_configs:
            coco_json = cfg["coco_json_path"]
            images_dir = cfg["images_dir"]
            weight     = float(cfg.get("weight", 1.0))
            name       = cfg.get("name", os.path.basename(coco_json))

            if weight <= 0:
                logger.info("Skipping dataset '%s' (weight=%.2f)", name, weight)
                continue

            logger.info(
                "Loading dataset '%s' from %s (weight=%.2f)...",
                name, coco_json, weight,
            )
            try:
                ds = SurgicalOrganDataset([coco_json], images_dir)
            except Exception as e:
                logger.error("Failed to load '%s': %s — skipping.", name, e)
                continue

            self._sub_datasets.append(ds)
            self._weights.append(weight)
            self._names.append(name)

            from collections import Counter
            counts = Counter(s["label_name"] for s in ds.samples)
            logger.info(
                "  '%s': %d samples, %d classes",
                name, len(ds), len(ds.label_to_id),
            )
            for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                logger.info("    %-22s %4d", lbl, cnt)

        if not self._sub_datasets:
            raise ValueError(
                "No valid sub-datasets loaded. Check your dataset_configs."
            )

        # ---- Build unified label → id mapping -------------------------
        all_labels = sorted({
            lbl
            for ds in self._sub_datasets
            for lbl in ds.label_to_id
        })
        self.label_to_id: dict[str, int] = {lbl: i for i, lbl in enumerate(all_labels)}
        self.id_to_label: dict[int, str] = {i: lbl for lbl, i in self.label_to_id.items()}

        # Patch each sub-dataset's label mapping to use the unified ids
        for ds in self._sub_datasets:
            ds.label_to_id = self.label_to_id
            ds.id_to_label = self.id_to_label
            for s in ds.samples:
                # Ensure every sample's label_name is in the unified map
                if s["label_name"] not in self.label_to_id:
                    s["label_name"] = "unknown"

        # ---- Build weighted index table --------------------------------
        # Effective length = max_w dataset fully traversed once,
        # others sub-sampled proportionally.
        max_w = max(self._weights)
        self._index_table: list[tuple[int, int]] = []  # (ds_idx, sample_idx)

        rng = np.random.default_rng(42)
        for ds_i, (ds, w) in enumerate(zip(self._sub_datasets, self._weights)):
            ratio      = w / max_w                      # fraction of this ds to include
            n_take     = max(1, round(len(ds) * ratio))
            chosen_idx = rng.choice(len(ds), size=n_take, replace=n_take > len(ds))
            for s_idx in chosen_idx:
                self._index_table.append((ds_i, int(s_idx)))

        rng.shuffle(self._index_table)  # mix datasets

        logger.info(
            "WeightedSurgicalDataset: %d effective samples from %d dataset(s), %d classes",
            len(self._index_table),
            len(self._sub_datasets),
            len(self.label_to_id),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index_table)

    def __getitem__(self, idx: int) -> dict:
        ds_i, s_i = self._index_table[idx]
        item = self._sub_datasets[ds_i][s_i]
        if self.transform is not None:
            item["image"] = self.transform(item["image"])
        return item

    # ------------------------------------------------------------------
    def get_class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights across ALL sub-datasets combined,
        accounting for the effective sampling weight of each dataset.

        Returns a 1-D tensor of shape (n_classes,) for use with
        CrossEntropyLoss(weight=...).
        """
        from collections import Counter
        max_w = max(self._weights)
        combined_counts: Counter = Counter()

        for ds, w in zip(self._sub_datasets, self._weights):
            ratio = w / max_w
            counts = Counter(s["label_name"] for s in ds.samples)
            for lbl, cnt in counts.items():
                combined_counts[lbl] += cnt * ratio

        total   = sum(combined_counts.values())
        n_cls   = len(self.label_to_id)
        weights = torch.zeros(n_cls)
        for lbl, lbl_id in self.label_to_id.items():
            cnt = combined_counts.get(lbl, 1)
            weights[lbl_id] = total / (n_cls * max(cnt, 1))
        return weights

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio:   float = 0.1,
        test_ratio:  float = 0.1,
        seed: int = 42,
    ):
        """
        Random split of the index table into train / val / test subsets.
        Returns three torch.utils.data.Subset objects.
        """
        from torch.utils.data import Subset
        from sklearn.model_selection import train_test_split
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        indices = list(range(len(self)))
        train_idx, temp_idx = train_test_split(
            indices, test_size=1 - train_ratio, random_state=seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed
        )
        logger.info(
            "Split: %d train / %d val / %d test",
            len(train_idx), len(val_idx), len(test_idx),
        )
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)


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

def _parse_dataset_config(spec: str) -> dict:
    """
    Parse a dataset spec string in the form:
        path/to/coco.json:path/to/images[:weight]

    Weight defaults to 1.0 if omitted.
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid dataset spec '{spec}'. "
            "Expected format: coco.json:images_dir[:weight]"
        )
    coco_json  = parts[0]
    images_dir = parts[1]
    weight     = float(parts[2]) if len(parts) >= 3 else 1.0
    return {
        "coco_json_path": coco_json,
        "images_dir":     images_dir,
        "weight":         weight,
        "name":           os.path.basename(os.path.dirname(coco_json)) or os.path.basename(coco_json),
    }


def train(args):
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "no")
    logger.info("Device: %s | Mixed precision: %s", accelerator.device, accelerator.mixed_precision)

    # --- Dataset ---
    # Prefer --datasets (multi-dataset with weights) over legacy --coco_exports
    if getattr(args, "datasets", None):
        logger.info("Building WeightedSurgicalDataset from %d spec(s)...", len(args.datasets))
        dataset_configs = [_parse_dataset_config(spec) for spec in args.datasets]
        dataset = WeightedSurgicalDataset(dataset_configs)
    else:
        # Legacy: single images_dir, flat weight=1.0 per export file
        coco_paths = []
        for pattern in (args.coco_exports or []):
            coco_paths.extend(glob.glob(pattern))
        if not coco_paths:
            logger.error(
                "No COCO export files found. "
                "Provide --datasets or --coco_exports."
            )
            sys.exit(1)
        logger.info("Loading %d COCO export file(s) (legacy mode)...", len(coco_paths))
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
    p = argparse.ArgumentParser(
        description="Fine-tune PaliGemma 2 for organ classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Multi-dataset inputs (preferred) ----
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Space-separated list of dataset configs in the format: "
            "path/to/coco.json:path/to/images[:weight]  "
            "Example:  --datasets "
            "./datasets/dsad/annotations.json:./datasets/dsad/images:3.0 "
            "./exports/surgeon_corrections.json:./uploads/images:5.0"
        ),
    )

    # ---- Legacy single-dataset inputs ----
    p.add_argument(
        "--coco_exports",
        nargs="+",
        default=None,
        help="Legacy: glob pattern(s) for COCO JSON export files (weight=1.0 each)",
    )
    p.add_argument(
        "--images_dir",
        default=None,
        help="Legacy: directory containing uploaded images (used with --coco_exports)",
    )

    # ---- General options ----
    p.add_argument("--output_dir",    default="./checkpoints")
    p.add_argument("--model_id",      default="google/paligemma2-3b-pt-224")
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lora_r",        type=int,   default=16)
    p.add_argument("--lora_alpha",    type=int,   default=32)
    p.add_argument("--lora_dropout",  type=float, default=0.05)
    p.add_argument("--patience",      type=int,   default=3)
    p.add_argument("--train_ratio",   type=float, default=0.8)
    p.add_argument("--val_ratio",     type=float, default=0.1)
    p.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 mixed precision (recommended for A100/H100 and Apple MPS)",
    )

    args = p.parse_args()

    # Validate: must supply either --datasets or --coco_exports
    if not args.datasets and not args.coco_exports:
        p.error(
            "You must provide either --datasets or --coco_exports. "
            "Run with -h for usage examples."
        )
    if args.coco_exports and not args.images_dir:
        p.error("--images_dir is required when using --coco_exports.")

    return args


if __name__ == "__main__":
    from PIL import Image
    args = parse_args()
    train(args)
