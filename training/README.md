# Fine-Tuning Pipeline — Surgical Organ Classifier

Fine-tunes **PaliGemma 2** (`google/paligemma2-3b-pt-224`) on your own
annotated laparoscopic images to produce a domain-specific organ classifier
that replaces the general-purpose Ollama VLM.  Training uses **LoRA** via
PEFT so it fits on an M2 Pro (or any machine with ≥16 GB RAM) without a GPU.

---

## Prerequisites

| Requirement | Detail |
|---|---|
| Annotated data | ≥ 50 approved annotations per organ class (absolute minimum); ≥ 500 for reliable results |
| Disk space | ~6 GB for base model weights + ~500 MB for LoRA adapter |
| RAM | ≥ 16 GB (M2 Pro / M3 / any CUDA GPU) |
| HuggingFace token | Set `HF_TOKEN` in `.env` — needed to download `google/paligemma2-3b-pt-224` |

> **Data tip:** Use the annotation app to segment and approve organ regions,
> then click **Export COCO** to generate the JSON files used here.
> Aim for a balanced class distribution — the training script applies
> inverse-frequency class weights automatically, but extreme imbalance
> (e.g., 500 liver vs 5 gallbladder) will still hurt minority classes.

---

## Step-by-step

### 1. Install training dependencies

```bash
cd training/
pip install -r requirements_training.txt
```

These are separate from the app's `environment.yml` to keep the inference
environment lean.

### 2. Export annotated data from the app

Open the annotation app, approve your annotations, then click
**Export COCO** in the header.  The JSON file is saved to `exports/`.

### 3. Fine-tune

```bash
python finetune.py \
    --coco_exports ../exports/*.json \
    --images_dir ../uploads/images \
    --output_dir ./checkpoints \
    --epochs 10 \
    --batch_size 8 \
    --learning_rate 2e-5
```

**On an M2 Pro with ~1 000 annotations and the default settings, expect
30–60 minutes.**  Progress is logged every epoch:

```
Epoch 1/10 — loss: 2.341 | val_acc: 0.421
Epoch 2/10 — loss: 1.876 | val_acc: 0.613
...
✓ New best model saved (val_acc=0.847)
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 10 | Increase to 20 for small datasets |
| `--batch_size` | 8 | Reduce to 4 if you run out of memory |
| `--lora_r` | 16 | Increase to 32 for more capacity |
| `--patience` | 3 | Early stopping patience (epochs) |
| `--bf16` | true | Disable with `--no-bf16` on older hardware |

Checkpoints are written to:
- `checkpoints/best_model/` — best val accuracy (use this for export)
- `checkpoints/latest/`     — most recent epoch

### 4. Evaluate

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_model \
    --coco_exports ../exports/*.json \
    --images_dir ../uploads/images
```

Outputs to `evaluation_results/`:
- `confusion_matrix.png` — visual breakdown of correct vs wrong predictions
- `metrics.json` — full per-class precision, recall, F1

**Reading the confusion matrix:**
- The diagonal (top-left to bottom-right) shows correct predictions.
- Off-diagonal cells show confusions — e.g., if row "liver" has a high value
  in column "spleen", the model often mistakes liver for spleen.
- Dark off-diagonal cells indicate organ pairs that need more training data
  or cleaner annotations.

Classes with **F1 < 0.7** are flagged in the log and `metrics.json`.
Collect more approved annotations for those classes and retrain.

### 5. Export for production

```bash
python export_model.py \
    --checkpoint ./checkpoints/best_model \
    --output ./production_model
```

This:
1. Merges the LoRA adapter into the base model weights
2. Saves the merged model to `production_model/` (~3 GB)
3. Writes `app/services/vlm_service_finetuned.py`

### 6. Activate in the annotation app

Set the environment variable before starting the app:

```bash
export VLM_BACKEND=finetuned
export FINETUNED_MODEL_PATH=./training/production_model
```

Or edit `app/config.py` directly:

```python
VLM_BACKEND = "finetuned"
FINETUNED_MODEL_PATH = "./training/production_model"
```

Restart the app — the header VLM status dot should turn green and organ
identification will now use your fine-tuned model with no Ollama dependency.

---

## Backend options

| `VLM_BACKEND` | Description |
|---|---|
| `"ollama"` | Default — calls local Ollama (gemma4:e4b) |
| `"hf"` | HuggingFace Inference API (used on HF Spaces) |
| `"finetuned"` | Local fine-tuned PaliGemma 2 (this pipeline) |

---

## Notes

- The exported model is **~3 GB** in float32 and runs fully locally with
  no internet or Ollama dependency — suitable for air-gapped environments.
- On M2 Pro, single-image inference takes roughly **200–500 ms** (CPU/MPS).
  On a CUDA GPU expect **20–50 ms**.
- Re-running `export_model.py` overwrites `app/services/vlm_service_finetuned.py`.
  This is intentional — the service file is generated artefact, not hand-edited code.
- The training script splits your data 80/10/10 (train/val/test).
  The test split is held out and only used in `evaluate.py`.
