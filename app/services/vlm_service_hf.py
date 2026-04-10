"""
Surgical Annotator — VLM Service (HuggingFace Spaces)
Drop-in replacement for vlm_service.py that uses HuggingFace InferenceClient
instead of local Ollama. Activated automatically when IS_HF_SPACES is True.
"""

import base64
import io
import json
import logging

import cv2
import numpy as np
from PIL import Image

from app.config import HF_TOKEN, HF_VLM_MODEL, ABDOMINAL_ORGANS

logger = logging.getLogger(__name__)

ORGAN_PROMPT = """You are a surgical anatomy expert reviewing a laparoscopic colorectal surgery image.
The highlighted region is the organ of interest. Identify it.
Respond ONLY in this exact JSON format with no other text:
{"organ": "<organ name>", "confidence": <0.0-1.0>}
Choose only from: liver, gallbladder, stomach, spleen, pancreas,
colon, small_intestine, appendix, kidney, adrenal_gland, omentum,
mesentery, diaphragm, bladder, uterus, ovary, peritoneum, unknown"""


class VLMService:
    """Organ identification via HuggingFace Inference API (for HF Spaces deployment)."""

    def __init__(self):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=HF_VLM_MODEL, token=HF_TOKEN or None)
        self.model = HF_VLM_MODEL

    def is_available(self) -> bool:
        try:
            # Quick connectivity check
            self.client.get_model_status(self.model)
            return True
        except Exception:
            return False

    def identify_organ(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Identify the organ in the masked region via HF Inference API.
        Returns {"label": str, "confidence": float}.
        """
        try:
            img_b64 = self._prepare_masked_image(image, mask)
            raw = self._call_hf(img_b64)
            return self._parse_response(raw)
        except Exception as e:
            logger.error("HF VLM identification failed: %s", e)
            return {"label": "unknown", "confidence": 0.0}

    # -- internals --

    def _prepare_masked_image(
        self, image: np.ndarray, mask: np.ndarray,
        padding: int = 20, overlay_color=(0, 255, 0), overlay_alpha=0.3,
    ) -> str:
        h, w = mask.shape
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return self._encode_b64(image)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin, rmax = max(0, rmin - padding), min(h - 1, rmax + padding)
        cmin, cmax = max(0, cmin - padding), min(w - 1, cmax + padding)

        crop = image[rmin:rmax + 1, cmin:cmax + 1].copy()
        mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]

        overlay = crop.copy()
        overlay[mask_crop] = overlay_color
        cv2.addWeighted(overlay, overlay_alpha, crop, 1 - overlay_alpha, 0, crop)
        return self._encode_b64(crop)

    @staticmethod
    def _encode_b64(image: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_hf(self, image_b64: str) -> str:
        """Call HuggingFace Inference API with the image."""
        image_bytes = base64.b64decode(image_b64)

        response = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ORGAN_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=150,
        )

        return response.choices[0].message.content or ""

    def _parse_response(self, raw: str) -> dict:
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = "\n".join(
                    l for l in text.split("\n") if not l.strip().startswith("```")
                ).strip()
            parsed = json.loads(text)
            organ = parsed.get("organ", "unknown").lower().strip()
            conf = float(parsed.get("confidence", 0.0))
            if organ not in ABDOMINAL_ORGANS:
                organ = "unknown"
            return {"label": organ, "confidence": max(0.0, min(1.0, conf))}
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse HF VLM response: %s — raw: %s", e, raw[:200])
            return {"label": "unknown", "confidence": 0.0}
