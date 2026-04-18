"""
Surgical Annotator — Configuration
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

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

class Settings(BaseSettings):
    # Device
    DEVICE: str = Field(default_factory=_detect_device)
    
    # SAM 2.1
    SAM_IMAGE_MODEL_ID: str = "facebook/sam2.1-hiera-small"
    SAM_VIDEO_MODEL_ID: str = "facebook/sam2.1-hiera-small"
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_GENERAL: str = "gemma4:e4b"
    OLLAMA_MODEL_MEDICAL: str = "medgemma1.5"
    
    # VLM backend selection
    VLM_BACKEND: str = "ollama"
    FINETUNED_MODEL_PATH: str = "./training/production_model"

    @property
    def OLLAMA_MODEL(self) -> str:
        return self.OLLAMA_MODEL_MEDICAL if self.VLM_BACKEND == "medgemma" else self.OLLAMA_MODEL_GENERAL
    
    # Gemini API
    GEMINI_API_KEY: str = ""
    
    # HF Spaces detection
    SPACE_ID: str | None = None
    HF_TOKEN: str = ""
    HF_VLM_MODEL: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    @property
    def IS_HF_SPACES(self) -> bool:
        return self.SPACE_ID is not None
    
    # Paths
    UPLOAD_IMAGE_DIR: str = "uploads/images"
    UPLOAD_VIDEO_DIR: str = "uploads/videos"
    FRAMES_DIR: str = "frames"
    EXPORT_DIR: str = "exports"
    DB_PATH: str = "data/annotations.db"
    
    # Organs
    ABDOMINAL_ORGANS: List[str] = [
        "liver", "gallbladder", "stomach", "spleen", "pancreas",
        "colon", "small_intestine", "appendix", "kidney", "adrenal_gland",
        "omentum", "mesentery", "diaphragm", "bladder", "uterus",
        "ovary", "peritoneum", "unknown",
    ]
    
    # Video
    FRAME_EXTRACTION_FPS: int = 5
    MAX_VIDEO_FRAMES: int = 500
    
    # Experimental (Opt-in)
    ENABLE_DEPTH_ESTIMATION: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Backward compatibility (alias attributes back to module level)
DEVICE = settings.DEVICE
SAM_IMAGE_MODEL_ID = settings.SAM_IMAGE_MODEL_ID
SAM_VIDEO_MODEL_ID = settings.SAM_VIDEO_MODEL_ID
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
VLM_BACKEND = settings.VLM_BACKEND
FINETUNED_MODEL_PATH = settings.FINETUNED_MODEL_PATH
GEMINI_API_KEY = settings.GEMINI_API_KEY
IS_HF_SPACES = settings.IS_HF_SPACES
HF_TOKEN = settings.HF_TOKEN
HF_VLM_MODEL = settings.HF_VLM_MODEL
UPLOAD_IMAGE_DIR = settings.UPLOAD_IMAGE_DIR
UPLOAD_VIDEO_DIR = settings.UPLOAD_VIDEO_DIR
FRAMES_DIR = settings.FRAMES_DIR
EXPORT_DIR = settings.EXPORT_DIR
DB_PATH = settings.DB_PATH
ABDOMINAL_ORGANS = settings.ABDOMINAL_ORGANS
FRAME_EXTRACTION_FPS = settings.FRAME_EXTRACTION_FPS
MAX_VIDEO_FRAMES = settings.MAX_VIDEO_FRAMES
ENABLE_DEPTH_ESTIMATION = settings.ENABLE_DEPTH_ESTIMATION
