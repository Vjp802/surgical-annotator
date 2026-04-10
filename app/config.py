"""
Surgical Annotator — Configuration
"""

import os

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

DEVICE = _detect_device()

# ---------------------------------------------------------------------------
# SAM 2.1
# ---------------------------------------------------------------------------
SAM_IMAGE_MODEL_ID = "facebook/sam2.1-hiera-small"
SAM_VIDEO_MODEL_ID = "facebook/sam2.1-hiera-small"

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"

# ---------------------------------------------------------------------------
# VLM backend selection
# ---------------------------------------------------------------------------
# "ollama"    — local Ollama (default)
# "hf"        — HuggingFace Inference API (auto-selected on HF Spaces)
# "finetuned" — locally exported PaliGemma 2 fine-tune
VLM_BACKEND = os.getenv("VLM_BACKEND", "ollama")
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "./training/production_model")

# ---------------------------------------------------------------------------
# HF Spaces detection
# ---------------------------------------------------------------------------
IS_HF_SPACES = os.getenv("SPACE_ID") is not None
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UPLOAD_IMAGE_DIR = "uploads/images"
UPLOAD_VIDEO_DIR = "uploads/videos"
FRAMES_DIR = "frames"
EXPORT_DIR = "exports"
DB_PATH = "data/annotations.db"

# ---------------------------------------------------------------------------
# Organs
# ---------------------------------------------------------------------------
ABDOMINAL_ORGANS = [
    "liver", "gallbladder", "stomach", "spleen", "pancreas",
    "colon", "small_intestine", "appendix", "kidney", "adrenal_gland",
    "omentum", "mesentery", "diaphragm", "bladder", "uterus",
    "ovary", "peritoneum", "unknown",
]

# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
FRAME_EXTRACTION_FPS = 5
MAX_VIDEO_FRAMES = 500
