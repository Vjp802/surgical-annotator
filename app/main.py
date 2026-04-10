"""
Surgical Annotator — FastAPI Application
Main entry point with HF Spaces auto-detection, dual SAM 2 predictors,
and conditional VLM backend (Ollama local vs HuggingFace cloud).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    DEVICE, IS_HF_SPACES,
    OLLAMA_MODEL, VLM_BACKEND,
    SAM_IMAGE_MODEL_ID, SAM_VIDEO_MODEL_ID,
    UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR, FRAMES_DIR, EXPORT_DIR,
)
from app.models.database import init_db
from app.routers import annotations, export, identify, segment, upload
from app.services.sam_service import SAMService
from app.services.video_service import VideoService
from app.services.coco_export import COCOExporter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Surgical Annotator — Starting up")
    logger.info("=" * 60)

    # 1. Create directories
    for d in [UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR, FRAMES_DIR, EXPORT_DIR, "data"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info("  ✓ Directories ready.")

    # 2. Database
    init_db()
    logger.info("  ✓ Database ready.")

    # 3. SAM 2 (image + video predictors)
    logger.info("Loading SAM 2 models on device '%s'...", DEVICE)
    sam_service = SAMService()
    try:
        sam_service.load_models()
        logger.info("  ✓ SAM 2 loaded (image: %s, video: %s)", SAM_IMAGE_MODEL_ID, SAM_VIDEO_MODEL_ID)
    except Exception as e:
        logger.error("  ✗ SAM 2 load failed: %s", e)
    app.state.sam_service = sam_service

    # 4. Video service
    video_service = VideoService()
    app.state.video_service = video_service
    logger.info("  ✓ Video service ready.")

    # 5. VLM backend selection
    if IS_HF_SPACES:
        logger.info("  HF Spaces detected — using HuggingFace VLM backend.")
        from app.services.vlm_service_hf import VLMService
        _vlm_label = "HuggingFace"
    elif VLM_BACKEND == "finetuned":
        logger.info("  VLM_BACKEND=finetuned — using local fine-tuned PaliGemma 2.")
        from app.services.vlm_service_finetuned import VLMService
        _vlm_label = "finetuned (PaliGemma 2)"
    else:
        logger.info("  VLM_BACKEND=ollama — using local Ollama (%s).", OLLAMA_MODEL)
        from app.services.vlm_service import VLMService
        _vlm_label = f"Ollama ({OLLAMA_MODEL})"

    vlm_service = VLMService()
    if vlm_service.is_available():
        logger.info("  ✓ VLM available (%s)", _vlm_label)
    else:
        logger.warning("  ⚠ VLM not available — organ ID will return 'unknown'.")
    app.state.vlm_service = vlm_service

    # 6. COCO exporter
    app.state.coco_exporter = COCOExporter()
    logger.info("  ✓ COCO exporter ready.")

    logger.info("=" * 60)
    logger.info("  Startup complete — device: %s", DEVICE)
    logger.info("  SAM 2:   %s", "loaded" if sam_service.is_ready() else "FAILED")
    logger.info("  VLM:     %s", "available" if vlm_service.is_available() else "unavailable")
    logger.info("  DB:      ready")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down Surgical Annotator...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Surgical Annotator",
    description="Laparoscopic surgery annotation with SAM 2 image/video segmentation and AI organ identification.",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(upload.router)
app.include_router(segment.router)
app.include_router(identify.router)
app.include_router(annotations.router)
app.include_router(export.router)


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
Path("static").mkdir(parents=True, exist_ok=True)
Path(UPLOAD_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_VIDEO_DIR).mkdir(parents=True, exist_ok=True)
Path(FRAMES_DIR).mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads/images", StaticFiles(directory=UPLOAD_IMAGE_DIR), name="upload_images")
app.mount("/uploads/videos", StaticFiles(directory=UPLOAD_VIDEO_DIR), name="upload_videos")
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = Path("static/index.html")
    if index.exists():
        return HTMLResponse(content=index.read_text())
    return HTMLResponse("<h1>Surgical Annotator</h1><p>Phase 3 pending.</p>")


@app.get("/api/health")
async def health(request: Request):
    sam_ready = request.app.state.sam_service.is_ready()
    vlm_ok = request.app.state.vlm_service.is_available()
    return {
        "status": "ok",
        "device": DEVICE,
        "hf_spaces": IS_HF_SPACES,
        "services": {
            "sam2": {"ready": sam_ready, "image_model": SAM_IMAGE_MODEL_ID, "video_model": SAM_VIDEO_MODEL_ID},
            "vlm": {"available": vlm_ok, "backend": "huggingface" if IS_HF_SPACES else VLM_BACKEND},
            "db": {"ready": True},
        },
    }
