"""
Surgical Annotator — Depth Service
Optional integration for Dense Prediction Transformers (DPT).
Uses Intel's MiDaS_small via PyTorch Hub for lightweight depth estimation.
Loads conditionally based on ENABLE_DEPTH_ESTIMATION.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from app.config import ENABLE_DEPTH_ESTIMATION, DEVICE

logger = logging.getLogger(__name__)


class DepthService:
    def __init__(self):
        self.model = None
        self.transform = None
        self.is_enabled = ENABLE_DEPTH_ESTIMATION
        self._loaded = False

    def load_model(self):
        """Lazy load the MiDaS model to save memory if unused."""
        if not self.is_enabled:
            return

        if self._loaded:
            return

        try:
            logger.info("Loading Optional Depth Estimation Model (MiDaS_small) on %s...", DEVICE)
            # MiDaS_small is fast and lightweight enough for real-time CPU/MPS
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(DEVICE)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform

            self._loaded = True
            logger.info("  ✓ Depth Estimation Model loaded.")
        except Exception as e:
            logger.error("  ✗ Failed to load Depth Estimation Model: %s", e)
            self._loaded = False

    def is_ready(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def estimate_depth(self, image_np: np.ndarray) -> np.ndarray:
        """
        Produce a dense depth map for an image.
        Returns a 2D numpy array normalized between 0-255 (uint8 type)
        where 255 is closest and 0 is furthest.
        """
        if not self._loaded:
            self.load_model()
            if not self._loaded:
                return np.zeros(image_np.shape[:2], dtype=np.uint8)

        # Convert to RGB (OpenCV uses BGR natively, but our app pipeline sends RGB np.array)
        input_batch = self.transform(image_np).to(DEVICE)

        prediction = self.model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to 0-255 for standard visualization/storage
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return (depth_map * 255.0).astype(np.uint8)

    def get_organ_average_depth(self, image_np: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """
        Calculates the average relative depth inside the segmented organ mask.
        Returns a normalized physical float proxy between 0.0 - 1.0 (where 1.0 is closest to camera).
        """
        if not self.is_enabled:
            return None

        try:
            depth_map = self.estimate_depth(image_np)
            # Mask must be boolean
            bool_mask = mask.astype(bool)
            if not bool_mask.any():
                return 0.0

            organ_depths = depth_map[bool_mask]
            # Convert the 0-255 pixel average to a 0.0-1.0 confidence/distance ratio
            avg_depth_pixel = np.mean(organ_depths)
            
            return round(float(avg_depth_pixel / 255.0), 3)
            
        except Exception as e:
            logger.error("Depth extraction failed: %s", e)
            return None
