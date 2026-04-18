"""
Surgical Annotator — COCO Export Service
Generates COCO JSON from approved annotations (images + video frames).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import ABDOMINAL_ORGANS, EXPORT_DIR
from app.models.database import get_approved_annotations

logger = logging.getLogger(__name__)


class COCOExporter:
    """Export approved annotations to standard COCO JSON."""

    def __init__(self):
        self.categories = []
        self.category_name_to_id = {}
        for idx, organ in enumerate(ABDOMINAL_ORGANS):
            cat_id = idx + 1
            self.categories.append({
                "id": cat_id,
                "name": organ,
                "supercategory": "organ",
            })
            self.category_name_to_id[organ] = cat_id

    async def export(
        self,
        description: str = "Surgical annotation dataset",
        version: str = "1.0",
        filename: Optional[str] = None,
    ) -> dict:
        """
        Build COCO JSON from all approved annotations and save to exports/.
        Supports both standalone images and video frames.
        """
        approved = await get_approved_annotations()

        # --- Images section ---
        image_id_map = {}   # unique key → COCO integer image ID
        coco_images = []
        img_counter = 1

        for ann in approved:
            # Build a unique key for each "image" (standalone image or video frame)
            if ann.get("image_id") and not ann.get("video_id"):
                key = f"img_{ann['image_id']}"
                file_name = ann.get("img_filename", "unknown.jpg")
            else:
                key = f"vid_{ann['video_id']}_f{ann.get('frame_number', 0)}"
                vid_name = ann.get("vid_filename", "video")
                frame_num = ann.get("frame_number", 0)
                file_name = f"{vid_name}_frame_{frame_num:06d}.jpg"

            if key not in image_id_map:
                image_id_map[key] = img_counter
                entry = {
                    "id": img_counter,
                    "file_name": file_name,
                    "width": ann.get("img_width", 0),
                    "height": ann.get("img_height", 0),
                }
                # Include stored filepath so the training pipeline can locate
                # the actual file (which uses a UUID name, not the original name).
                if ann.get("img_filepath"):
                    entry["file_path"] = ann["img_filepath"]
                coco_images.append(entry)
                img_counter += 1

        # --- Annotations section ---
        coco_annotations = []
        ann_counter = 1

        for ann in approved:
            cat_id = self.category_name_to_id.get(ann["label"])
            if cat_id is None:
                continue

            if ann.get("image_id") and not ann.get("video_id"):
                key = f"img_{ann['image_id']}"
            else:
                key = f"vid_{ann['video_id']}_f{ann.get('frame_number', 0)}"

            coco_ann = {
                "id": ann_counter,
                "image_id": image_id_map.get(key, 0),
                "category_id": cat_id,
                "area": ann.get("area", 0),
                "iscrowd": 0,
            }

            # Segmentation
            if ann.get("mask_rle"):
                try:
                    coco_ann["segmentation"] = json.loads(ann["mask_rle"])
                except json.JSONDecodeError:
                    pass
            if "segmentation" not in coco_ann and ann.get("mask_polygon"):
                try:
                    coco_ann["segmentation"] = json.loads(ann["mask_polygon"])
                except json.JSONDecodeError:
                    pass

            # Bbox
            try:
                coco_ann["bbox"] = json.loads(ann["bbox"]) if ann.get("bbox") else [0, 0, 0, 0]
            except json.JSONDecodeError:
                coco_ann["bbox"] = [0, 0, 0, 0]

            # Custom track_id for video annotations (robotics pipeline use)
            if ann.get("track_id"):
                coco_ann["track_id"] = ann["track_id"]

            coco_annotations.append(coco_ann)
            ann_counter += 1

        # --- Assemble ---
        now = datetime.now(timezone.utc).isoformat()
        coco = {
            "info": {
                "description": description,
                "version": version,
                "year": datetime.now().year,
                "contributor": "Surgical Annotator",
                "date_created": now,
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": self.categories,
        }

        # --- Validate structure ---
        self._validate_coco(coco)

        # --- Save ---
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"coco_export_{ts}.json"

        Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
        filepath = Path(EXPORT_DIR) / filename

        with open(filepath, "w") as f:
            json.dump(coco, f, indent=2)

        logger.info(
            "COCO export: %s (%d images, %d annotations)",
            filepath, len(coco_images), len(coco_annotations),
        )

        return {
            "filename": filename,
            "path": str(filepath),
            "stats": {
                "images": len(coco_images),
                "annotations": len(coco_annotations),
                "categories": len(self.categories),
            },
            "coco": coco,
        }

    @staticmethod
    def _validate_coco(coco: dict) -> None:
        """Basic COCO structure validation."""
        required = {"info", "images", "annotations", "categories"}
        missing = required - set(coco.keys())
        if missing:
            raise ValueError(f"COCO missing required keys: {missing}")

        for img in coco["images"]:
            assert "id" in img and "file_name" in img, f"Invalid image entry: {img}"

        for ann in coco["annotations"]:
            assert "id" in ann and "image_id" in ann and "category_id" in ann, \
                f"Invalid annotation entry: {ann}"

        for cat in coco["categories"]:
            assert "id" in cat and "name" in cat, f"Invalid category entry: {cat}"
