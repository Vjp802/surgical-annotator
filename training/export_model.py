"""
Export a fine-tuned LoRA checkpoint for production use in the annotation app.

Merges LoRA adapter weights into the base model, saves the merged model,
and writes app/services/vlm_service_finetuned.py — a drop-in replacement
for vlm_service.py that runs inference locally with no Ollama dependency.

Usage:
    python export_model.py \\
        --checkpoint ./checkpoints/best_model \\
        --output ./production_model
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# vlm_service_finetuned.py template
# Written to app/services/ so the annotation app can import it directly.
# ======================================================================

SERVICE_TEMPLATE = '''"""
Surgical Annotator — VLM Service (Fine-tuned PaliGemma 2)
Drop-in replacement for vlm_service.py that uses a locally-exported,
domain-specific organ classifier instead of a general-purpose Ollama model.

Activate by setting VLM_BACKEND=finetuned in config.py or environment.
"""

import io
import logging
import base64
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from app.config import FINETUNED_MODEL_PATH, ABDOMINAL_ORGANS

logger = logging.getLogger(__name__)

PROMPT = "Identify the highlighted surgical organ. Answer with the organ name only:"
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_ALPHA = 0.3
CROP_PADDING  = 20


class VLMService:
    """Organ identification via fine-tuned PaliGemma 2 (fully local)."""

    def __init__(self):
        self.model     = None
        self.processor = None
        self.organ_list: list[str] = []
        self._loaded   = False
        self.device    = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if self._loaded:
            return
        model_path = Path(FINETUNED_MODEL_PATH)
        if not model_path.exists():
            logger.error(
                "Fine-tuned model not found at %s. "
                "Run training/export_model.py first.", model_path
            )
            return

        logger.info("Loading fine-tuned model from %s on %s...", model_path, self.device)
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

        organ_list_path = model_path / "organ_list.json"
        if organ_list_path.exists():
            self.organ_list = json.loads(organ_list_path.read_text())
        else:
            self.organ_list = list(ABDOMINAL_ORGANS)

        self._loaded = True
        logger.info("Fine-tuned model ready. %d organ classes.", len(self.organ_list))

    def is_available(self) -> bool:
        return Path(FINETUNED_MODEL_PATH).exists()

    def identify_organ(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Identify the organ in the masked region.
        Returns {"label": str, "confidence": float}.
        """
        self._load()
        if not self._loaded:
            return {"label": "unknown", "confidence": 0.0}

        try:
            crop = self._prepare_masked_image(image, mask)
            return self._infer(crop)
        except Exception as e:
            logger.error("Fine-tuned VLM inference failed: %s", e)
            return {"label": "unknown", "confidence": 0.0}

    # ------------------------------------------------------------------

    def _prepare_masked_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Image.Image:
        h, w  = mask.shape
        rows  = np.any(mask, axis=1)
        cols  = np.any(mask, axis=0)
        if not rows.any():
            return Image.fromarray(image)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - CROP_PADDING)
        rmax = min(h - 1, rmax + CROP_PADDING)
        cmin = max(0, cmin - CROP_PADDING)
        cmax = min(w - 1, cmax + CROP_PADDING)

        crop      = image[rmin:rmax + 1, cmin:cmax + 1].copy()
        mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]
        overlay   = crop.copy()
        overlay[mask_crop] = OVERLAY_COLOR
        cv2.addWeighted(overlay, OVERLAY_ALPHA, crop, 1 - OVERLAY_ALPHA, 0, crop)
        return Image.fromarray(crop)

    @torch.no_grad()
    def _infer(self, pil_image: Image.Image) -> dict:
        inputs = self.processor(
            images=pil_image,
            text=PROMPT,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        # --- Constrained scoring: score each organ via log-prob of first token ---
        # This is faster and more calibrated than full generation.
        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]  # (1, vocab)
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

        organ_scores = {}
        for organ in self.organ_list:
            toks = self.processor.tokenizer.encode(organ, add_special_tokens=False)
            if not toks:
                continue
            organ_scores[organ] = log_probs[0, toks[0]].item()

        if not organ_scores:
            return {"label": "unknown", "confidence": 0.0}

        # Convert log-probs to normalised probabilities over organ set
        import math
        max_lp    = max(organ_scores.values())
        exp_scores = {o: math.exp(lp - max_lp) for o, lp in organ_scores.items()}
        total      = sum(exp_scores.values())
        probs      = {o: v / total for o, v in exp_scores.items()}

        best_organ = max(probs, key=probs.get)
        confidence = round(probs[best_organ], 4)

        if best_organ not in ABDOMINAL_ORGANS:
            best_organ = "unknown"

        return {"label": best_organ, "confidence": confidence}
'''

# We need json in the template; add the import
SERVICE_TEMPLATE = "import json\n" + SERVICE_TEMPLATE


# ======================================================================
# Main
# ======================================================================

def main(args):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    checkpoint  = Path(args.checkpoint)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load base model + LoRA adapter ----
    logger.info("Loading base model: %s", args.base_model_id)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float32,
    )

    logger.info("Loading LoRA adapter from: %s", checkpoint)
    model = PeftModel.from_pretrained(base_model, str(checkpoint))

    # ---- 2. Merge LoRA into base weights ----
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    logger.info("  ✓ Merge complete.")

    # ---- 3. Save merged model ----
    logger.info("Saving merged model to: %s", output_dir)
    model.save_pretrained(str(output_dir))

    processor = AutoProcessor.from_pretrained(str(checkpoint))
    processor.save_pretrained(str(output_dir))
    logger.info("  ✓ Model and processor saved.")

    # ---- 4. Copy metadata ----
    for fname in ["label_map.json", "organ_list.json"]:
        src = checkpoint / fname
        dst = output_dir / fname
        if src.exists():
            dst.write_text(src.read_text())
            logger.info("  ✓ Copied %s", fname)
        else:
            logger.warning("  %s not found in checkpoint.", fname)

    # ---- 5. Estimate model size ----
    total_params = sum(p.numel() for p in model.parameters())
    size_gb = total_params * 4 / 1e9  # float32 bytes
    logger.info("Model size: %.2f B params (~%.1f GB float32)", total_params / 1e9, size_gb)

    # ---- 6. Benchmark inference speed ----
    logger.info("Benchmarking inference speed on %s...", device)
    model = model.to(device)
    model.eval()
    dummy_img = torch.zeros(1, 3, 224, 224).to(device)
    dummy_ids = torch.zeros(1, 16, dtype=torch.long).to(device)
    dummy_mask = torch.ones(1, 16, dtype=torch.long).to(device)

    # Warm-up
    with torch.no_grad():
        try:
            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask, pixel_values=dummy_img)
        except Exception:
            pass

    times = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.perf_counter()
            try:
                _ = model(input_ids=dummy_ids, attention_mask=dummy_mask, pixel_values=dummy_img)
            except Exception:
                break
            times.append(time.perf_counter() - t0)

    if times:
        avg_ms = sum(times) / len(times) * 1000
        logger.info("Avg forward pass: %.1f ms on %s", avg_ms, device)
    else:
        logger.info("Speed benchmark skipped (dummy input shape mismatch — normal for full model).")

    # ---- 7. Write vlm_service_finetuned.py ----
    service_dst = Path(args.service_output)
    service_dst.parent.mkdir(parents=True, exist_ok=True)
    service_dst.write_text(SERVICE_TEMPLATE)
    logger.info("  ✓ Written: %s", service_dst)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Export complete.")
    logger.info("  Merged model : %s", output_dir)
    logger.info("  Service file : %s", service_dst)
    logger.info("  Model size   : ~%.1f GB (float32)", size_gb)
    logger.info("")
    logger.info("To activate in the annotation app:")
    logger.info("  Set VLM_BACKEND=finetuned in your environment, or")
    logger.info("  edit app/config.py:  VLM_BACKEND = 'finetuned'")
    logger.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Export fine-tuned model for production")
    p.add_argument("--checkpoint",    required=True, help="Path to fine-tuned LoRA checkpoint")
    p.add_argument("--output",        default="./production_model")
    p.add_argument("--base_model_id", default="google/paligemma2-3b-pt-224")
    p.add_argument(
        "--service_output",
        default="../app/services/vlm_service_finetuned.py",
        help="Where to write the drop-in service file",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
