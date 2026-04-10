"""
Surgical Annotator — SAM 2 Service
Image segmentation and video mask propagation via SAM 2.1.
"""

import io
import base64
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from app.config import DEVICE, SAM_IMAGE_MODEL_ID, SAM_VIDEO_MODEL_ID

logger = logging.getLogger(__name__)


class SAMService:
    """SAM 2.1 wrapper for single-image segmentation and video tracking."""

    def __init__(self):
        self.image_predictor = None
        self.video_predictor = None
        self.device = DEVICE

    def load_models(self) -> None:
        """Load SAM 2.1 image and video predictors."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        logger.info("Loading SAM 2.1 image predictor on device '%s'...", self.device)
        self.image_predictor = SAM2ImagePredictor.from_pretrained(
            SAM_IMAGE_MODEL_ID, device=self.device
        )
        logger.info("  ✓ Image predictor ready.")

        logger.info("Loading SAM 2.1 video predictor...")
        self.video_predictor = SAM2VideoPredictor.from_pretrained(
            SAM_VIDEO_MODEL_ID, device=self.device
        )
        logger.info("  ✓ Video predictor ready.")

    def is_ready(self) -> bool:
        return self.image_predictor is not None

    # ------------------------------------------------------------------
    # Image segmentation
    # ------------------------------------------------------------------

    def segment_image(self, image: np.ndarray, x: int, y: int) -> dict:
        """
        Segment from a single foreground point click on an image.
        Returns dict with mask, rle, polygon, bbox, area, score, mask_png_b64.
        """
        if self.image_predictor is None:
            raise RuntimeError("SAM 2.1 image predictor not loaded.")

        with torch.inference_mode():
            self.image_predictor.set_image(image)
            masks, scores, _ = self.image_predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])

        return {
            "mask": best_mask,
            "rle": mask_to_rle(best_mask),
            "polygon": mask_to_polygon(best_mask),
            "bbox": mask_to_bbox(best_mask),
            "area": float(best_mask.sum()),
            "score": best_score,
            "mask_png_b64": mask_to_png_bytes(best_mask),
        }

    # ------------------------------------------------------------------
    # Video segmentation (track across frames)
    # ------------------------------------------------------------------

    def segment_video(
        self,
        frames_dir: str,
        click_frame: int,
        x: int,
        y: int,
    ) -> list[dict]:
        """
        Propagate a single-point mask through all frames in a video.

        Args:
            frames_dir: directory containing sequential JPEG frames
            click_frame: frame index where the user clicked
            x, y: click coordinates in image space

        Returns:
            list of {frame_number, mask, rle, polygon, bbox, area, mask_png_b64}
        """
        if self.video_predictor is None:
            raise RuntimeError("SAM 2.1 video predictor not loaded.")

        with torch.inference_mode():
            state = self.video_predictor.init_state(video_path=frames_dir)

            self.video_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=click_frame,
                obj_id=1,
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

            results = []
            for frame_idx, _obj_ids, mask_logits in self.video_predictor.propagate_in_video(state):
                # mask_logits: (num_objects, 1, H, W)
                mask_np = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                results.append({
                    "frame_number": frame_idx,
                    "mask": mask_np,
                    "rle": mask_to_rle(mask_np),
                    "polygon": mask_to_polygon(mask_np),
                    "bbox": mask_to_bbox(mask_np),
                    "area": float(mask_np.sum()),
                    "mask_png_b64": mask_to_png_bytes(mask_np),
                })

            self.video_predictor.reset_state(state)

        results.sort(key=lambda r: r["frame_number"])
        return results


# ===================================================================
# Mask conversion helpers
# ===================================================================

def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert boolean mask to COCO RLE via pycocotools."""
    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygon(mask: np.ndarray) -> list:
    """Convert boolean mask to polygon contours [x1,y1,x2,y2,...] via OpenCV."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        flat = c.flatten().tolist()
        if len(flat) >= 6:
            polys.append(flat)
    return polys


def mask_to_bbox(mask: np.ndarray) -> list:
    """Bounding box [x, y, w, h] from boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def mask_to_png_bytes(
    mask: np.ndarray,
    color: tuple = (0, 120, 255),
    alpha: int = 128,
) -> str:
    """Semi-transparent RGBA PNG of the mask, base64-encoded."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = alpha

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
