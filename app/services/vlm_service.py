"""
Surgical Annotator — VLM Service (Ollama)
Organ identification via local Ollama running gemma4:e4b.
"""

import base64
import io
import json
import logging

import cv2
import numpy as np
import requests
from PIL import Image

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, ABDOMINAL_ORGANS

logger = logging.getLogger(__name__)

ORGAN_PROMPT = """You are a surgical anatomy expert reviewing a laparoscopic colorectal surgery image.
The highlighted region is the organ of interest. Identify it.
Respond ONLY in this exact JSON format with no other text:
{"organ": "<organ name>", "confidence": <0.0-1.0>}
Choose only from: liver, gallbladder, stomach, spleen, pancreas,
colon, small_intestine, appendix, kidney, adrenal_gland, omentum,
mesentery, diaphragm, bladder, uterus, ovary, peritoneum, unknown"""


class VLMService:
    """Organ identification via Ollama (gemma4:e4b)."""

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def identify_organ(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Identify the organ in the masked region.
        Returns {"label": str, "confidence": float}.
        """
        try:
            img_b64 = self._prepare_masked_image(image, mask)
            raw = self._call_ollama(img_b64)
            return self._parse_response(raw)
        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            return {"label": "unknown", "confidence": 0.0}
        except requests.Timeout:
            logger.error("Ollama request timed out.")
            return {"label": "unknown", "confidence": 0.0}
        except Exception as e:
            logger.error("VLM identification failed: %s", e)
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

    def _call_ollama(self, image_b64: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": ORGAN_PROMPT,
                "images": [image_b64],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

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
            logger.warning("Failed to parse VLM response: %s — raw: %s", e, raw[:200])
            return {"label": "unknown", "confidence": 0.0}
