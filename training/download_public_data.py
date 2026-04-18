"""
Download public surgical datasets for PaliGemma 2 fine-tuning.

Supported datasets
------------------
  cholec80   - CholecT50/Cholec80 laparoscopic cholecystectomy frames
  roboflow   - Roboflow surgical instrument + organ datasets (public)
  surgisr4k  - SurgIS-R4K high-resolution laparoscopic frames (unlabelled)
  dsad       - Dresden Surgical Anatomy Dataset (DSAD) — colorectal surgery
               with pixel-level segmentation; BEST for colorectal fine-tuning.
  all        - Download all of the above in sequence

Usage:
    python download_public_data.py --dataset dsad --output_dir ./datasets
    python download_public_data.py --dataset all  --output_dir ./datasets
    python download_public_data.py --dataset dsad --output_dir ./datasets --max_samples 500
    python download_public_data.py --list
"""

import argparse
import json
import os
import sys

# -----------------------------------------------------------------------
# Canonical organ list (must match vlm_service.py ABDOMINAL_ORGANS)
# -----------------------------------------------------------------------
ABDOMINAL_ORGANS = [
    "gallbladder",
    "liver",
    "common_bile_duct",
    "cystic_duct",
    "hepatic_artery",
    "small_intestine",
    "colon",
    "stomach",
    "spleen",
    "kidney",
    "pancreas",
    "omentum",
    "peritoneum",
    "blood_vessel",
    "unknown",
]

# -----------------------------------------------------------------------
# Dataset registry
# -----------------------------------------------------------------------
DATASET_INFO: dict[str, dict] = {
    "cholec80": {
        "description": "Cholec80 / CholecT50 laparoscopic cholecystectomy frames",
        "size": "~80 videos / ~90 000 frames",
        "has_labels": True,
        "label_type": "Phase labels + tool presence (no pixel segmentation)",
        "procedure": "Laparoscopic cholecystectomy (gallbladder removal)",
        "license": "CC BY-NC-SA 4.0",
        "citation": "Twinanda et al. (2017)",
        "notes": "Good general laparoscopic domain; tool/phase labels only",
    },
    "roboflow": {
        "description": "Roboflow public surgical segmentation datasets",
        "size": "Varies by dataset version",
        "has_labels": True,
        "label_type": "Bounding-box + instance segmentation annotations",
        "procedure": "Mixed laparoscopic procedures",
        "license": "Varies per dataset (check Roboflow project page)",
        "citation": "Roboflow",
        "notes": "Use --roboflow_api_key to access private or gated datasets",
    },
    "surgisr4k": {
        "description": "SurgIS-R4K high-resolution laparoscopic image dataset",
        "size": "~4 000 images",
        "has_labels": False,
        "label_type": "No semantic labels (image-only)",
        "procedure": "Mixed laparoscopic procedures",
        "license": "Research use only — check dataset page",
        "citation": "SurgIS consortium",
        "notes": "Useful for domain-adaptive pre-training; no organ labels",
    },
    "dsad": {
        "description": "Dresden Surgical Anatomy Dataset — robot-assisted rectal resections",
        "size": "13,195 images",
        "has_labels": True,
        "label_type": "Pixel-level segmentation, 8 organ classes",
        "procedure": "Colorectal surgery (rectal resection)",
        "license": "CC BY 4.0 (see Figshare page)",
        "citation": "Kolbinger et al. (2023)",
        "notes": "BEST dataset for colorectal surgery fine-tuning",
        "download_source": "Kaggle: anindyamajumder/the-dresden-surgical-anatomy-dataset",
        "alternative": (
            "Manual download from https://springernature.figshare.com/articles/dataset/"
            "The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_segmentation_in_"
            "surgical_data_science/21702600"
        ),
        "requires": "kaggle CLI (pip install kaggle) + KAGGLE_USERNAME + KAGGLE_KEY env vars",
    },
}


# ======================================================================
# Downloaders
# ======================================================================

def download_cholec80(output_dir: str, max_samples: int = 2000) -> str:
    """
    Download a sample of Cholec80 / CholecT50 frames from HuggingFace.

    Source: cholecT50 public subsets available on HuggingFace.
    Note: Full Cholec80 requires registration at CAMMA IRCAD.
    This function downloads the freely available CholecT50 subset.

    Args:
        output_dir:  Directory to save downloaded data.
        max_samples: Maximum number of frames to download.

    Returns:
        Path to the output COCO JSON.
    """
    from datasets import load_dataset
    from tqdm import tqdm
    from datetime import datetime
    import numpy as np
    from PIL import Image

    print(f"Downloading Cholec80/CholecT50 sample (max {max_samples} frames)...")

    try:
        dataset = load_dataset(
            "Pedro-Pereira/CholecT50",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Could not load CholecT50 from HuggingFace: {e}")
        print("  For full Cholec80 access, register at https://camma.unistra.fr/datasets/")
        return ""

    images_dir = os.path.join(output_dir, "cholec80", "images")
    os.makedirs(images_dir, exist_ok=True)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    skipped = 0
    annotation_id = 0

    for idx, sample in enumerate(tqdm(dataset, total=max_samples)):
        if idx >= max_samples:
            break
        try:
            image = sample.get("image") or sample.get("img")
            if image is None:
                skipped += 1
                continue

            filename = f"cholec80_{idx:06d}.jpg"
            filepath = os.path.join(images_dir, filename)
            if not os.path.exists(filepath):
                image.convert("RGB").save(filepath, "JPEG", quality=95)

            coco_images.append({
                "id": idx,
                "file_name": filename,
                "width": image.width,
                "height": image.height,
                "source": "cholec80",
            })
            # Phase / tool labels — no pixel segmentation, skip annotations
        except Exception as e:
            print(f"  Skipping frame {idx}: {e}")
            skipped += 1

    coco = _build_coco_json(
        description="Cholec80/CholecT50 laparoscopic cholecystectomy frames",
        source="Pedro-Pereira/CholecT50",
        url="https://huggingface.co/datasets/Pedro-Pereira/CholecT50",
        images=coco_images,
        annotations=coco_annotations,
    )
    coco_path = os.path.join(output_dir, "cholec80", "annotations.json")
    _save_json(coco, coco_path)

    _save_metadata(
        path=os.path.join(output_dir, "cholec80", "metadata.json"),
        dataset="Cholec80/CholecT50",
        source="Pedro-Pereira/CholecT50",
        total_downloaded=len(coco_images),
        total_annotations=len(coco_annotations),
        total_skipped=skipped,
        has_organ_labels=False,
        approved=False,
        procedure="Laparoscopic cholecystectomy",
        organs_covered=[],
        citation="Twinanda et al. (2017)",
        coco_json=coco_path,
    )
    _print_summary("Cholec80", len(coco_images), len(coco_annotations), skipped, coco_path)
    return coco_path


def download_roboflow_surgical(
    output_dir: str,
    max_samples: int = 2000,
    roboflow_api_key: str = "",
) -> str:
    """
    Download a public surgical segmentation dataset from Roboflow Universe.

    Uses the 'surgical-instrument-segmentation' dataset (public, CC BY 4.0).
    Pass roboflow_api_key to access gated or private datasets.

    Args:
        output_dir:        Directory to save downloaded data.
        max_samples:       Maximum samples to download.
        roboflow_api_key:  Optional Roboflow API key.

    Returns:
        Path to the output COCO JSON.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Downloading Roboflow surgical dataset (max {max_samples} samples)...")

    # Attempt to load a known public surgical dataset from HuggingFace mirror
    try:
        dataset = load_dataset(
            "Kooolkia/Instruments_Segmentation-1",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Could not load Roboflow dataset: {e}")
        print("  Visit https://universe.roboflow.com/search?q=surgical+segmentation for options.")
        return ""

    images_dir = os.path.join(output_dir, "roboflow", "images")
    os.makedirs(images_dir, exist_ok=True)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    skipped = 0

    for idx, sample in enumerate(tqdm(dataset, total=max_samples)):
        if idx >= max_samples:
            break
        try:
            image = sample.get("image") or sample.get("img")
            if image is None:
                skipped += 1
                continue

            filename = f"roboflow_{idx:06d}.jpg"
            filepath = os.path.join(images_dir, filename)
            if not os.path.exists(filepath):
                image.convert("RGB").save(filepath, "JPEG", quality=95)

            coco_images.append({
                "id": idx,
                "file_name": filename,
                "width": image.width,
                "height": image.height,
                "source": "roboflow",
            })
        except Exception as e:
            print(f"  Skipping sample {idx}: {e}")
            skipped += 1

    coco = _build_coco_json(
        description="Roboflow surgical instrument segmentation dataset",
        source="Kooolkia/Instruments_Segmentation-1",
        url="https://universe.roboflow.com",
        images=coco_images,
        annotations=coco_annotations,
    )
    coco_path = os.path.join(output_dir, "roboflow", "annotations.json")
    _save_json(coco, coco_path)

    _save_metadata(
        path=os.path.join(output_dir, "roboflow", "metadata.json"),
        dataset="Roboflow Surgical",
        source="Kooolkia/Instruments_Segmentation-1",
        total_downloaded=len(coco_images),
        total_annotations=len(coco_annotations),
        total_skipped=skipped,
        has_organ_labels=False,
        approved=False,
        procedure="Mixed laparoscopic",
        organs_covered=[],
        citation="Roboflow",
        coco_json=coco_path,
    )
    _print_summary("Roboflow", len(coco_images), len(coco_annotations), skipped, coco_path)
    return coco_path


def download_surgisr4k(output_dir: str, max_samples: int = 2000) -> str:
    """
    Download SurgIS-R4K high-resolution laparoscopic frames.

    These are unlabelled frames useful for domain-adaptive pre-training.
    No organ segment annotations are available.

    Args:
        output_dir:  Directory to save downloaded data.
        max_samples: Maximum frames to download.

    Returns:
        Path to the output COCO JSON (images only, no annotations).
    """
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Downloading SurgIS-R4K dataset (max {max_samples} frames)...")

    try:
        dataset = load_dataset(
            "ynuozhang/SurgicalVideoFrames",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Could not load SurgIS-R4K: {e}")
        return ""

    images_dir = os.path.join(output_dir, "surgisr4k", "images")
    os.makedirs(images_dir, exist_ok=True)

    coco_images: list[dict] = []
    skipped = 0

    for idx, sample in enumerate(tqdm(dataset, total=max_samples)):
        if idx >= max_samples:
            break
        try:
            image = sample.get("image") or sample.get("frame")
            if image is None:
                skipped += 1
                continue

            filename = f"surgisr4k_{idx:06d}.jpg"
            filepath = os.path.join(images_dir, filename)
            if not os.path.exists(filepath):
                image.convert("RGB").save(filepath, "JPEG", quality=95)

            coco_images.append({
                "id": idx,
                "file_name": filename,
                "width": image.width,
                "height": image.height,
                "source": "surgisr4k",
            })
        except Exception as e:
            print(f"  Skipping frame {idx}: {e}")
            skipped += 1

    coco = _build_coco_json(
        description="SurgIS-R4K high-resolution laparoscopic frames (unlabelled)",
        source="ynuozhang/SurgicalVideoFrames",
        url="https://huggingface.co/datasets/ynuozhang/SurgicalVideoFrames",
        images=coco_images,
        annotations=[],
    )
    coco_path = os.path.join(output_dir, "surgisr4k", "annotations.json")
    _save_json(coco, coco_path)

    _save_metadata(
        path=os.path.join(output_dir, "surgisr4k", "metadata.json"),
        dataset="SurgIS-R4K",
        source="ynuozhang/SurgicalVideoFrames",
        total_downloaded=len(coco_images),
        total_annotations=0,
        total_skipped=skipped,
        has_organ_labels=False,
        approved=False,
        procedure="Mixed laparoscopic",
        organs_covered=[],
        citation="SurgIS consortium",
        coco_json=coco_path,
    )
    _print_summary("SurgIS-R4K", len(coco_images), 0, skipped, coco_path)
    return coco_path


def download_dsad(output_dir: str, max_samples: int = 2000) -> str:
    """
    Download Dresden Surgical Anatomy Dataset (DSAD) via the Kaggle CLI.

    DSAD is hosted on Kaggle (and Figshare) — NOT on HuggingFace.
    Kaggle dataset: anindyamajumder/the-dresden-surgical-anatomy-dataset
    Figshare page:  https://springernature.figshare.com/articles/dataset/
                    The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_
                    segmentation_in_surgical_data_science/21702600

    Prerequisites:
        pip install kaggle
        Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
        (download kaggle.json from https://www.kaggle.com/settings/account
         and run: export KAGGLE_USERNAME=... KAGGLE_KEY=...)

    On-disk structure after unzip:
        raw/
          colon/
            surgery_01/
              frame_001.png          ← image
              frame_001_mask.png     ← binary segmentation mask
          liver/
            surgery_01/...
          ...

    Args:
        output_dir:  Root directory to save downloaded data.
        max_samples: Maximum images to process across all organs (default 2000).

    Returns:
        Path to the output COCO JSON, or empty string on failure.
    """
    import subprocess
    import zipfile
    import glob
    import numpy as np
    import cv2
    from PIL import Image
    from tqdm import tqdm
    from datetime import datetime

    dsad_dir   = os.path.join(output_dir, "dsad")
    images_dir = os.path.join(dsad_dir, "images")
    raw_dir    = os.path.join(dsad_dir, "raw")
    os.makedirs(images_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Download from Kaggle if not already present
    # ------------------------------------------------------------------
    already_have_raw = os.path.isdir(raw_dir) and any(os.scandir(raw_dir))
    already_have_zip = bool(glob.glob(os.path.join(dsad_dir, "*.zip")))

    if not already_have_raw and not already_have_zip:
        print("Downloading DSAD from Kaggle...")
        print("  Dataset : anindyamajumder/the-dresden-surgical-anatomy-dataset")
        print("  Requires: kaggle CLI  +  KAGGLE_USERNAME / KAGGLE_KEY env vars")
        print("  Get API key: https://www.kaggle.com/settings/account\n")

        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "anindyamajumder/the-dresden-surgical-anatomy-dataset",
                "--path", dsad_dir,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"\n✗ Kaggle download failed:\n{result.stderr.strip()}")
            print("\nManual download options:")
            print("  Kaggle  : https://www.kaggle.com/datasets/anindyamajumder/"
                  "the-dresden-surgical-anatomy-dataset")
            print("  Figshare: https://springernature.figshare.com/articles/dataset/"
                  "The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_"
                  "segmentation_in_surgical_data_science/21702600")
            print(f"  Unzip to: {dsad_dir}/raw/")
            print("  Then re-run this script.")
            return ""

        print(result.stdout.strip())

    # ------------------------------------------------------------------
    # Step 2: Unzip if we have a zip but not the raw dir yet
    # ------------------------------------------------------------------
    if not (os.path.isdir(raw_dir) and any(os.scandir(raw_dir))):
        zip_files = glob.glob(os.path.join(dsad_dir, "*.zip"))
        if not zip_files:
            print("No zip file found in", dsad_dir)
            print("Please download DSAD manually and unzip it to:", raw_dir)
            return ""

        print(f"Unzipping {os.path.basename(zip_files[0])}...")
        os.makedirs(raw_dir, exist_ok=True)
        with zipfile.ZipFile(zip_files[0], "r") as zf:
            zf.extractall(raw_dir)
        print(f"  Extracted to {raw_dir}")

    # ------------------------------------------------------------------
    # Step 3: Walk organ folders and convert to COCO format
    # ------------------------------------------------------------------
    # Map each DSAD folder name → our canonical organ label
    DSAD_ORGAN_MAP: dict[str, str] = {
        "colon":                       "colon",
        "liver":                       "liver",
        "small_intestine":             "small_intestine",
        "spleen":                      "spleen",
        "stomach":                     "stomach",
        "pancreas":                    "pancreas",
        "kidney":                      "kidney",
        "ureter":                      "unknown",
        "abdominal_wall":              "unknown",
        "inferior_mesenteric_artery":  "unknown",
        "intestinal_veins":            "unknown",
        "vesicular_glands":            "unknown",
    }

    # The actual organ folders may sit one level deep inside raw_dir,
    # so find the first level that contains known organ folder names.
    def _find_organ_root(base: str) -> str:
        """Return the directory that directly contains organ sub-folders."""
        for root, dirs, _ in os.walk(base):
            if any(d in DSAD_ORGAN_MAP for d in dirs):
                return root
        return base  # fallback — try base itself

    organ_root = _find_organ_root(raw_dir)
    print(f"Reading organ folders from: {organ_root}")

    coco_images: list[dict]      = []
    coco_annotations: list[dict] = []
    annotation_id                = 0
    total_processed              = 0
    skipped                      = 0

    organ_items = [
        (organ_folder, organ_label)
        for organ_folder, organ_label in DSAD_ORGAN_MAP.items()
        if os.path.isdir(os.path.join(organ_root, organ_folder))
    ]

    if not organ_items:
        print("\n⚠ No recognised organ folders found under:", organ_root)
        print("Expected folders like: colon/, liver/, small_intestine/, ...")
        print("Please check the DSAD zip structure and re-run.")
        return ""

    pbar = tqdm(total=max_samples, desc="Processing DSAD")

    for organ_folder, organ_label in organ_items:
        if total_processed >= max_samples:
            break

        organ_path = os.path.join(organ_root, organ_folder)

        # Iterate surgery sub-directories (surgery_01, surgery_02, …)
        surgery_dirs = sorted(
            d for d in os.listdir(organ_path)
            if os.path.isdir(os.path.join(organ_path, d))
        )

        for surgery_dir in surgery_dirs:
            if total_processed >= max_samples:
                break

            surgery_path = os.path.join(organ_path, surgery_dir)

            # Collect image files (PNG/JPG, excluding mask files)
            all_files = sorted(os.listdir(surgery_path))
            image_files = [
                f for f in all_files
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
                and "mask" not in f.lower()
            ]

            for img_file in image_files:
                if total_processed >= max_samples:
                    break

                img_path = os.path.join(surgery_path, img_file)

                # Convention: frame_001.png → frame_001_mask.png
                stem     = os.path.splitext(img_file)[0]
                ext      = os.path.splitext(img_file)[1]
                mask_file = f"{stem}_mask.png"
                mask_path = os.path.join(surgery_path, mask_file)
                # Also try same extension
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(surgery_path, f"{stem}_mask{ext}")

                try:
                    image    = Image.open(img_path).convert("RGB")
                    img_id   = total_processed
                    filename = (
                        f"dsad_{organ_folder}_{surgery_dir}_{total_processed:06d}.jpg"
                    )
                    out_path = os.path.join(images_dir, filename)

                    if not os.path.exists(out_path):
                        image.save(out_path, "JPEG", quality=95)

                    coco_images.append({
                        "id":       img_id,
                        "file_name": filename,
                        "width":    image.width,
                        "height":   image.height,
                        "source":   "dsad",
                        "organ":    organ_label,
                        "surgery":  surgery_dir,
                    })

                    # --- Mask → COCO polygon ---
                    if os.path.exists(mask_path) and organ_label != "unknown":
                        mask_img    = Image.open(mask_path).convert("L")
                        mask_binary = (np.array(mask_img) > 127).astype(np.uint8)

                        if mask_binary.sum() > 0:
                            contours, _ = cv2.findContours(
                                mask_binary,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )
                            polys: list[list[float]] = []
                            for c in contours:
                                if len(c) >= 3:
                                    poly = c.flatten().tolist()
                                    if len(poly) >= 6:
                                        polys.append(poly)

                            if polys:
                                rows = np.any(mask_binary, axis=1)
                                cols = np.any(mask_binary, axis=0)
                                rmin = int(np.where(rows)[0][0])
                                rmax = int(np.where(rows)[0][-1])
                                cmin = int(np.where(cols)[0][0])
                                cmax = int(np.where(cols)[0][-1])

                                cat_id = (
                                    ABDOMINAL_ORGANS.index(organ_label) + 1
                                    if organ_label in ABDOMINAL_ORGANS
                                    else len(ABDOMINAL_ORGANS)
                                )

                                coco_annotations.append({
                                    "id":           annotation_id,
                                    "image_id":     img_id,
                                    "category_id":  cat_id,
                                    "segmentation": polys,
                                    "area":         float(mask_binary.sum()),
                                    "bbox":         [cmin, rmin,
                                                     cmax - cmin, rmax - rmin],
                                    "iscrowd":      0,
                                    "organ":        organ_label,
                                    "source":       "dsad",
                                    "approved":     True,  # expert pixel-level labels
                                })
                                annotation_id += 1

                    total_processed += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"  Skipping {img_file}: {e}")
                    skipped += 1

    pbar.close()

    # ------------------------------------------------------------------
    # Step 4: Build & save COCO JSON + metadata
    # ------------------------------------------------------------------
    coco = {
        "info": {
            "description":     "Dresden Surgical Anatomy Dataset (DSAD)",
            "source":          "Kaggle: anindyamajumder/the-dresden-surgical-anatomy-dataset",
            "url":             "https://www.kaggle.com/datasets/anindyamajumder/"
                               "the-dresden-surgical-anatomy-dataset",
            "figshare_url":    "https://springernature.figshare.com/articles/dataset/"
                               "The_Dresden_Surgical_Anatomy_Dataset_for_abdominal_organ_"
                               "segmentation_in_surgical_data_science/21702600",
            "version":         "1.0",
            "year":            2023,
            "contributor":     "Kolbinger et al., Dresden University of Technology",
            "date_downloaded": datetime.now().isoformat(),
            "license":         "CC BY 4.0",
            "citation":        "Kolbinger et al. (2023)",
            "procedure":       "Robot-assisted rectal resection (colorectal surgery)",
            "notes":           "Expert pixel-level segmentation, approved=True",
        },
        "licenses": [],
        "images":      coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": i + 1, "name": organ, "supercategory": "organ"}
            for i, organ in enumerate(ABDOMINAL_ORGANS)
        ],
    }

    coco_path = os.path.join(dsad_dir, "annotations.json")
    _save_json(coco, coco_path)

    _save_metadata(
        path=os.path.join(dsad_dir, "metadata.json"),
        dataset="DSAD",
        source="Kaggle: anindyamajumder/the-dresden-surgical-anatomy-dataset",
        total_downloaded=len(coco_images),
        total_annotations=len(coco_annotations),
        total_skipped=skipped,
        has_organ_labels=True,
        approved=True,
        procedure="Robot-assisted rectal resection",
        organs_covered=["colon", "small_intestine", "liver", "spleen",
                        "kidney", "stomach", "pancreas"],
        citation="Kolbinger et al. (2023)",
        coco_json=coco_path,
    )

    _print_summary("DSAD", len(coco_images), len(coco_annotations), skipped, coco_path)
    print("  All annotations marked approved=True (expert labeled)")
    return coco_path


# -----------------------------------------------------------------------
# Dataset registry (must come after function definitions)
# -----------------------------------------------------------------------
AVAILABLE_DATASETS: dict[str, object] = {
    "cholec80":  download_cholec80,
    "roboflow":  download_roboflow_surgical,
    "surgisr4k": download_surgisr4k,
    "dsad":      download_dsad,
    "all":       None,
}


# ======================================================================
# Shared helpers
# ======================================================================

def _build_coco_json(
    description: str,
    source: str,
    url: str,
    images: list[dict],
    annotations: list[dict],
) -> dict:
    from datetime import datetime
    return {
        "info": {
            "description":     description,
            "source":          source,
            "url":             url,
            "version":         "1.0",
            "date_downloaded": datetime.now().isoformat(),
        },
        "licenses":    [],
        "images":      images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": organ, "supercategory": "organ"}
            for i, organ in enumerate(ABDOMINAL_ORGANS)
        ],
    }


def _save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_metadata(
    path: str,
    dataset: str,
    source: str,
    total_downloaded: int,
    total_annotations: int,
    total_skipped: int,
    has_organ_labels: bool,
    approved: bool,
    procedure: str,
    organs_covered: list[str],
    citation: str,
    coco_json: str,
) -> None:
    from datetime import datetime
    meta = {
        "dataset":           dataset,
        "source":            source,
        "downloaded_at":     datetime.now().isoformat(),
        "total_downloaded":  total_downloaded,
        "total_annotations": total_annotations,
        "total_skipped":     total_skipped,
        "has_organ_labels":  has_organ_labels,
        "approved":          approved,
        "procedure":         procedure,
        "organs_covered":    organs_covered,
        "citation":          citation,
        "coco_json":         coco_json,
    }
    _save_json(meta, path)


def _print_summary(
    name: str,
    n_images: int,
    n_annotations: int,
    skipped: int,
    coco_path: str,
) -> None:
    print(f"\n{name} download complete:")
    print(f"  Images:      {n_images}")
    print(f"  Annotations: {n_annotations}")
    print(f"  Skipped:     {skipped}")
    print(f"  COCO JSON:   {coco_path}")


# ======================================================================
# CLI
# ======================================================================

def _list_datasets() -> None:
    print("\nAvailable datasets:\n")
    for name, info in DATASET_INFO.items():
        print(f"  {name}")
        print(f"    Description: {info['description']}")
        print(f"    Size:        {info['size']}")
        print(f"    Has labels:  {info['has_labels']}  ({info['label_type']})")
        print(f"    Procedure:   {info['procedure']}")
        print(f"    License:     {info['license']}")
        print(f"    Citation:    {info['citation']}")
        if info.get("notes"):
            print(f"    Notes:       {info['notes']}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download public surgical datasets for fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        choices=list(AVAILABLE_DATASETS.keys()),
        default="dsad",
        help="Dataset to download (default: dsad)",
    )
    p.add_argument(
        "--output_dir",
        default="./datasets",
        help="Root directory to save all downloaded datasets (default: ./datasets)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum number of images/frames to download per dataset (default: 2000)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit",
    )
    p.add_argument(
        "--roboflow_api_key",
        default="",
        help="Optional Roboflow API key for private datasets",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        _list_datasets()
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "all":
        results = {}
        for name, fn in AVAILABLE_DATASETS.items():
            if fn is None or name == "all":
                continue
            print(f"\n{'=' * 60}")
            print(f"  Downloading: {name}")
            print(f"{'=' * 60}")
            if name == "roboflow":
                results[name] = fn(args.output_dir, args.max_samples, args.roboflow_api_key)
            else:
                results[name] = fn(args.output_dir, args.max_samples)

        print("\n\nAll downloads complete:")
        for name, path in results.items():
            status = f"✓  {path}" if path else "✗  FAILED"
            print(f"  {name:<12} {status}")
    else:
        fn = AVAILABLE_DATASETS[args.dataset]
        if args.dataset == "roboflow":
            fn(args.output_dir, args.max_samples, args.roboflow_api_key)
        else:
            fn(args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
