"""
Surgical Organ Dataset
Loads COCO JSON exports from the annotation app and produces
cropped, mask-overlaid organ images ready for PaliGemma 2 fine-tuning.

The image preprocessing exactly mirrors vlm_service.py so the model sees
identical input at inference time.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)

# Must match vlm_service.py _prepare_masked_image
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_ALPHA = 0.3
CROP_PADDING  = 20
MIN_SAMPLES_WARNING = 50


class SurgicalOrganDataset(Dataset):
    """
    Dataset of cropped organ images derived from COCO JSON exports.

    Each sample is a bbox crop of a surgical image with the segmentation
    mask rendered as a semi-transparent green overlay — identical to the
    preprocessing used by vlm_service.py at inference time.
    """

    def __init__(
        self,
        coco_json_paths: list[str],
        images_dir: str,
        transform=None,
    ):
        self.transform   = transform
        self.samples: list[dict] = []
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

        images_dir = Path(images_dir)

        for json_path in coco_json_paths:
            self._load_coco(Path(json_path), images_dir)

        if not self.samples:
            raise ValueError(
                "No valid samples found. Check that exports/ contains approved "
                "annotations and that images_dir points to uploads/images/."
            )

        # Build label mappings from all observed labels
        all_labels = sorted({s["label_name"] for s in self.samples})
        self.label_to_id = {lbl: i for i, lbl in enumerate(all_labels)}
        self.id_to_label = {i: lbl for lbl, i in self.label_to_id.items()}

        # Log class distribution and warn on thin classes
        counts = Counter(s["label_name"] for s in self.samples)
        logger.info(
            "Dataset: %d samples across %d classes from %d export file(s)",
            len(self.samples), len(self.label_to_id), len(coco_json_paths),
        )
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            flag = " ⚠ LOW" if count < MIN_SAMPLES_WARNING else ""
            logger.info("  %-22s %4d samples%s", label, count, flag)

        thin = [l for l, c in counts.items() if c < MIN_SAMPLES_WARNING]
        if thin:
            logger.warning(
                "Classes with fewer than %d samples (model may underfit): %s",
                MIN_SAMPLES_WARNING, ", ".join(thin),
            )

    # ------------------------------------------------------------------
    def _load_coco(self, json_path: Path, images_dir: Path) -> None:
        with open(json_path) as f:
            coco = json.load(f)

        categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
        img_info   = {img["id"]: img for img in coco.get("images", [])}

        for ann in coco.get("annotations", []):
            info = img_info.get(ann["image_id"])
            if info is None:
                continue

            bbox = ann.get("bbox")  # [x, y, w, h]
            if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                continue

            label_name = categories.get(ann.get("category_id"), "unknown")

            # Resolve actual image path.
            # COCO image entries include "file_path" (stored UUID path) when
            # exported by this app; fall back to searching images_dir by name.
            img_path = self._resolve_image_path(info, images_dir)
            if img_path is None:
                logger.debug("Image not found, skipping: %s", info.get("file_name"))
                continue

            self.samples.append({
                "image_path": str(img_path),
                "bbox":        bbox,
                "segmentation": ann.get("segmentation"),
                "img_height":  info.get("height", 0),
                "img_width":   info.get("width", 0),
                "label_name":  label_name,
                "ann_id":      ann["id"],
            })

    @staticmethod
    def _resolve_image_path(info: dict, images_dir: Path) -> Optional[Path]:
        # Preferred: use the stored disk path recorded at export time
        if info.get("file_path"):
            p = Path(info["file_path"])
            if p.exists():
                return p

        # Fallback: search images_dir by original filename
        file_name = info.get("file_name", "")
        for candidate in [
            images_dir / file_name,
            images_dir / "images" / file_name,
            Path(file_name),
        ]:
            if candidate.exists():
                return candidate

        return None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        image = np.array(Image.open(s["image_path"]).convert("RGB"))
        h, w  = image.shape[:2]

        mask = _decode_mask(s["segmentation"], h, w)

        # Crop with padding (matches vlm_service.py)
        x, y, bw, bh = s["bbox"]
        x1 = max(0, int(x) - CROP_PADDING)
        y1 = max(0, int(y) - CROP_PADDING)
        x2 = min(w, int(x + bw) + CROP_PADDING)
        y2 = min(h, int(y + bh) + CROP_PADDING)

        crop = image[y1:y2, x1:x2].copy()

        # Apply mask overlay (matches vlm_service.py)
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            overlay = crop.copy()
            overlay[mask_crop] = OVERLAY_COLOR
            cv2.addWeighted(overlay, OVERLAY_ALPHA, crop, 1 - OVERLAY_ALPHA, 0, crop)

        pil_image = Image.fromarray(crop)

        if self.transform:
            pil_image = self.transform(pil_image)

        return {
            "image":      pil_image,
            "label":      self.label_to_id[s["label_name"]],
            "label_name": s["label_name"],
        }

    # ------------------------------------------------------------------
    def get_class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights for CrossEntropyLoss.
        Balances training across under-represented organ classes.
        """
        counts  = Counter(s["label_name"] for s in self.samples)
        total   = len(self.samples)
        n_cls   = len(self.label_to_id)
        weights = torch.zeros(n_cls)
        for label, label_id in self.label_to_id.items():
            count = counts.get(label, 1)
            weights[label_id] = total / (n_cls * count)
        return weights

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio:   float = 0.1,
        test_ratio:  float = 0.1,
        seed: int = 42,
    ) -> tuple["SurgicalOrganDataset", "SurgicalOrganDataset", "SurgicalOrganDataset"]:
        """
        Stratified train / val / test split.
        Falls back to random split if any class has too few samples to stratify.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        indices = list(range(len(self.samples)))
        labels  = [s["label_name"] for s in self.samples]

        try:
            train_idx, temp_idx = train_test_split(
                indices,
                test_size=1 - train_ratio,
                stratify=labels,
                random_state=seed,
            )
            temp_labels = [labels[i] for i in temp_idx]
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=1 - val_size,
                stratify=temp_labels,
                random_state=seed,
            )
        except ValueError:
            logger.warning(
                "Stratified split failed (some classes too small). Using random split."
            )
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
# Helpers
# ======================================================================

def _decode_mask(segmentation, h: int, w: int) -> Optional[np.ndarray]:
    if segmentation is None:
        return None
    try:
        if isinstance(segmentation, dict):
            rle = segmentation.copy()
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("utf-8")
            return mask_utils.decode(rle).astype(bool)
        if isinstance(segmentation, list) and segmentation:
            rles = mask_utils.frPyObjects(segmentation, h, w)
            return mask_utils.decode(mask_utils.merge(rles)).astype(bool)
    except Exception as e:
        logger.debug("Mask decode failed: %s", e)
    return None


def collate_fn(batch: list[dict], target_size: tuple[int, int] = (224, 224)) -> dict:
    """
    Collate a batch of dataset items into tensors.
    Resizes variable-size crops to target_size and normalises with
    mean=0.5 std=0.5 (matching SigLIP / PaliGemma 2 preprocessing).
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    pixel_values = []
    labels       = []
    label_names  = []

    for item in batch:
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        pixel_values.append(transform(img))
        labels.append(item["label"])
        label_names.append(item["label_name"])

    return {
        "pixel_values": torch.stack(pixel_values),
        "labels":       torch.tensor(labels, dtype=torch.long),
        "label_names":  label_names,
    }
