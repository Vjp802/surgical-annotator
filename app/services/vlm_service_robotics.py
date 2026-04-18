"""
VLM backend using Gemini Robotics-ER 1.6 via Gemini API.
Provides enhanced spatial reasoning for organ identification
compared to general-purpose VLMs.

Requires:
    pip install google-generativeai
    GEMINI_API_KEY environment variable set

Activated via:
    export VLM_BACKEND=robotics_er
"""

import json
import logging

import cv2
import numpy as np
from PIL import Image

import google.generativeai as genai

from app.config import ABDOMINAL_ORGANS, GEMINI_API_KEY

logger = logging.getLogger(__name__)

ORGAN_PROMPT = """You are analyzing a laparoscopic colorectal \
surgery image. The highlighted region (colored overlay) is the \
anatomical structure to identify.

Use spatial reasoning to identify the organ considering:
- Its position in the frame (upper/lower, left/right)
- Its relationship to surrounding structures
- Its visual texture, color, and shape in laparoscopic light
- Common anatomical positions in colorectal surgery

Key visual features:
- Liver: large smooth dark red-brown, upper right quadrant
- Gallbladder: small pear-shaped green-yellow, under liver
- Colon: tubular pink-white with haustra folds
- Small intestine: narrower pink tubes, more mobile
- Stomach: large smooth organ, upper left
- Omentum: fatty lacy yellow tissue
- Mesentery: thin membrane with visible vessels
- Surgical instruments: metal/silver, label as unknown

Respond ONLY in this exact JSON format, no other text:
{
  "organ": "<organ name>",
  "confidence": <0.0-1.0>,
  "spatial_context": "<one sentence describing location and \
relationship to surrounding structures>"
}

If confidence is below 0.5 use organ: "unknown".
Choose organ only from: liver, gallbladder, stomach, spleen, \
pancreas, colon, small_intestine, appendix, kidney, adrenal_gland, \
omentum, mesentery, diaphragm, bladder, uterus, ovary, \
peritoneum, unknown"""


class VLMService:
    """Organ identification via Gemini Robotics-ER 1.6."""

    def __init__(self):
        if not GEMINI_API_KEY:
            logger.warning(
                "GEMINI_API_KEY not set. Robotics-ER backend will "
                "return unknown for all identifications."
            )
            self.client = None
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            self.client = genai.GenerativeModel(
                "gemini-robotics-er-1.6-preview"
            )
            logger.info("Gemini Robotics-ER 1.6 backend initialized")

    def is_available(self) -> bool:
        """True if the API key is configured and client is initialized."""
        return self.client is not None

    def identify_organ(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> dict:
        """
        Identify organ using Gemini Robotics-ER 1.6.

        Sends the full frame with masked region highlighted.
        Uses the model's spatial reasoning to identify the organ
        in context of surrounding anatomy — not just the crop.

        Returns:
            {
                "label": str,           # organ name
                "confidence": float,    # 0.0-1.0
                "spatial_context": str  # brief spatial description
                                        # e.g. "upper right quadrant,
                                        # adjacent to liver"
            }
        """
        if not self.client:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "spatial_context": "API key not configured",
            }

        try:
            # 1. Prepare full frame with colored mask overlay
            #    (send full frame, not just crop — use spatial reasoning)
            full_frame = self._apply_mask_overlay(image, mask)
            pil_image = Image.fromarray(full_frame)

            # 2. Also prepare cropped region for detail
            crop = self._crop_masked_region(image, mask, padding=30)
            pil_crop = Image.fromarray(crop)

            # 3. Send both full frame and crop to leverage
            #    multi-view spatial reasoning
            response = self.client.generate_content(
                [ORGAN_PROMPT, pil_image, pil_crop]
            )

            # 4. Parse response
            text = response.text.strip()
            # Strip markdown code fences if present
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)

            organ = result.get("organ", "unknown").lower().strip()
            conf = float(result.get("confidence", 0.0))

            if organ not in ABDOMINAL_ORGANS:
                organ = "unknown"

            return {
                "label": organ,
                "confidence": max(0.0, min(1.0, conf)),
                "spatial_context": result.get("spatial_context", ""),
            }

        except Exception as e:
            logger.error("Robotics-ER identification failed: %s", e)
            return {
                "label": "unknown",
                "confidence": 0.0,
                "spatial_context": "",
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_mask_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple = (255, 165, 0),   # orange overlay
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Apply semi-transparent colored overlay on masked region.

        Returns the full frame (not cropped) so the model can use
        spatial context from surrounding anatomy.
        """
        vis = image.copy()
        overlay = vis.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        return vis

    @staticmethod
    def _crop_masked_region(
        image: np.ndarray,
        mask: np.ndarray,
        padding: int = 30,
    ) -> np.ndarray:
        """Crop image to mask bounding box with padding.

        Returns the cropped region with mask overlay for detail view.
        """
        h, w = mask.shape
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any():
            return image.copy()

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - padding)
        rmax = min(h - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(w - 1, cmax + padding)

        crop = image[rmin:rmax + 1, cmin:cmax + 1].copy()
        mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]

        # Apply overlay on crop too for clarity
        overlay = crop.copy()
        overlay[mask_crop] = (255, 165, 0)
        cv2.addWeighted(overlay, 0.4, crop, 0.6, 0, crop)

        return crop
