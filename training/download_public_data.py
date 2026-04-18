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
        "license": "See HuggingFace dataset page",
        "citation": "Kolbinger et al. (2023)",
        "notes": "BEST dataset for colorectal surgery fine-tuning",
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
    Download Dresden Surgical Anatomy Dataset (DSAD).

    Source: https://huggingface.co/datasets/dsad/dsad
    Contains 13,195 laparoscopic images from robot-assisted rectal resections
    with pixel-level segmentation of: colon, small intestine, liver, spleen,
    kidney, ureter, abdominal wall, and vascular structures.

    This is the most relevant public dataset for colorectal surgery organ
    segmentation — directly matches the anatomy visible in colorectal
    laparoscopic procedures.

    Args:
        output_dir:  Directory to save downloaded data.
        max_samples: Maximum images to download (default 2000; DSAD has 13 195).

    Returns:
        Path to the output COCO JSON.
    """
    import numpy as np
    from PIL import Image
    from tqdm import tqdm
    from datetime import datetime
    from datasets import load_dataset

    print(f"Downloading DSAD dataset (max {max_samples} samples)...")

    # Load from HuggingFace — streaming keeps RAM usage flat
    dataset = load_dataset(
        "dsad/dsad",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # DSAD organ label mapping → our ABDOMINAL_ORGANS list
    DSAD_LABEL_MAP: dict[str, str] = {
        "colon": "colon",
        "small_intestine": "small_intestine",
        "small intestine": "small_intestine",
        "liver": "liver",
        "spleen": "spleen",
        "kidney": "kidney",
        "ureter": "unknown",          # not in our list
        "abdominal_wall": "unknown",
        "abdominal wall": "unknown",
        "inferior_mesenteric_artery": "unknown",
        "superior_rectal_artery": "unknown",
    }

    images_dir = os.path.join(output_dir, "dsad", "images")
    masks_dir  = os.path.join(output_dir, "dsad", "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    coco_images: list[dict]       = []
    coco_annotations: list[dict]  = []
    annotation_id                 = 0
    skipped                       = 0

    for idx, sample in enumerate(tqdm(dataset, total=max_samples)):
        if idx >= max_samples:
            break

        try:
            image       = sample.get("image")
            segmentation = sample.get("segmentation") or sample.get("mask")
            label       = sample.get("label") or sample.get("organ")

            if image is None:
                skipped += 1
                continue

            # --- Save image ---
            filename = f"dsad_{idx:06d}.jpg"
            filepath = os.path.join(images_dir, filename)
            if not os.path.exists(filepath):
                image.convert("RGB").save(filepath, "JPEG", quality=95)

            # --- Map DSAD label → our organ vocabulary ---
            organ_label = "unknown"
            if label:
                label_str  = str(label).lower().strip()
                organ_label = DSAD_LABEL_MAP.get(label_str, "unknown")

            # --- Convert segmentation mask → COCO polygon format ---
            if segmentation is not None:
                mask_array = np.array(segmentation)
                if mask_array.ndim == 3:
                    # Some HuggingFace masks are (H, W, 1) — squeeze
                    mask_array = mask_array.squeeze(-1)

                if mask_array.max() > 0:
                    # Bounding box from mask
                    rows = np.any(mask_array, axis=1)
                    cols = np.any(mask_array, axis=0)
                    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
                    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
                    bbox  = [cmin, rmin, cmax - cmin, rmax - rmin]
                    area  = float(mask_array.sum())

                    # Polygon contours via OpenCV
                    import cv2
                    contours, _ = cv2.findContours(
                        mask_array.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    segmentation_poly: list[list[float]] = []
                    for contour in contours:
                        if len(contour) >= 3:
                            poly = contour.flatten().tolist()
                            if len(poly) >= 6:
                                segmentation_poly.append(poly)

                    if segmentation_poly:
                        # Resolve category_id; fall back to last category if unknown
                        if organ_label in ABDOMINAL_ORGANS:
                            cat_id = ABDOMINAL_ORGANS.index(organ_label) + 1
                        else:
                            cat_id = len(ABDOMINAL_ORGANS)

                        coco_annotations.append({
                            "id":            annotation_id,
                            "image_id":      idx,
                            "category_id":   cat_id,
                            "segmentation":  segmentation_poly,
                            "area":          area,
                            "bbox":          bbox,
                            "iscrowd":       0,
                            "organ":         organ_label,
                            "source":        "dsad",
                            "approved":      True,   # expert pixel-level labels
                        })
                        annotation_id += 1

            coco_images.append({
                "id":        idx,
                "file_name": filename,
                "width":     image.width,
                "height":    image.height,
                "source":    "dsad",
            })

        except Exception as e:
            print(f"  Skipping sample {idx}: {e}")
            skipped += 1
            continue

    # --- Build & save COCO JSON ---
    coco = {
        "info": {
            "description":     "Dresden Surgical Anatomy Dataset (DSAD)",
            "source":          "dsad/dsad",
            "url":             "https://huggingface.co/datasets/dsad/dsad",
            "version":         "1.0",
            "year":            2023,
            "contributor":     "Kolbinger et al., Dresden University of Technology",
            "date_downloaded": datetime.now().isoformat(),
            "license":         "See dataset page for license details",
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

    coco_path = os.path.join(output_dir, "dsad", "annotations.json")
    _save_json(coco, coco_path)

    _save_metadata(
        path=os.path.join(output_dir, "dsad", "metadata.json"),
        dataset="DSAD",
        source="dsad/dsad",
        total_downloaded=len(coco_images),
        total_annotations=len(coco_annotations),
        total_skipped=skipped,
        has_organ_labels=True,
        approved=True,
        procedure="Robot-assisted rectal resection",
        organs_covered=["colon", "small_intestine", "liver", "spleen", "kidney"],
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
