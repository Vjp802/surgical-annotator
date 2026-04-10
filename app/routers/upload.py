"""
Surgical Annotator — Upload Router
Image and video upload, listing, and serving.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from PIL import Image

from app.config import UPLOAD_IMAGE_DIR, UPLOAD_VIDEO_DIR, FRAMES_DIR
from app.models.database import (
    create_image, get_image, list_images,
    create_video, get_video, list_videos,
    create_video_frames_bulk, get_video_frame,
)

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ===================================================================
# Image upload
# ===================================================================

@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload a surgical image."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(400, f"Unsupported image type '{ext}'.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")

    Path(UPLOAD_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = Path(UPLOAD_IMAGE_DIR) / unique_name

    with open(filepath, "wb") as f:
        f.write(content)

    try:
        img = Image.open(filepath)
        width, height = img.size
        img.close()
    except Exception:
        filepath.unlink(missing_ok=True)
        raise HTTPException(400, "Invalid image file.")

    record = create_image(file.filename, str(filepath), width, height)
    record["url"] = f"/uploads/images/{unique_name}"

    return {"status": "success", "image": record}


@router.get("/images")
async def get_images():
    """List all uploaded images."""
    images = list_images()
    for img in images:
        img["url"] = f"/uploads/images/{Path(img['filepath']).name}"
    return {"images": images, "count": len(images)}


@router.get("/images/{image_id}")
async def serve_image(image_id: str):
    """Serve an image file by ID."""
    record = get_image(image_id)
    if not record:
        raise HTTPException(404, "Image not found.")
    fp = Path(record["filepath"])
    if not fp.exists():
        raise HTTPException(404, "Image file missing from disk.")
    return FileResponse(str(fp))


# ===================================================================
# Video upload
# ===================================================================

@router.post("/upload/video")
async def upload_video(file: UploadFile = File(...), request: Request = None):
    """Upload a surgical video, extract frames, create DB entries."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(400, f"Unsupported video type '{ext}'.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")

    Path(UPLOAD_VIDEO_DIR).mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = Path(UPLOAD_VIDEO_DIR) / unique_name

    with open(filepath, "wb") as f:
        f.write(content)

    # Get video metadata
    video_service = request.app.state.video_service
    meta = video_service.get_video_metadata(str(filepath))

    # Create video DB record
    record = create_video(
        filename=file.filename,
        filepath=str(filepath),
        duration_seconds=meta["duration_seconds"],
        fps=meta["fps"],
        total_frames=meta["total_frames"],
        width=meta["width"],
        height=meta["height"],
    )
    video_id = record["id"]

    # Extract frames
    frames_dir = Path(FRAMES_DIR) / video_id
    frames = video_service.extract_frames(str(filepath), str(frames_dir))

    # Bulk-insert frame records
    import uuid as _uuid
    frame_rows = []
    for fr in frames:
        frame_rows.append({
            "id": str(_uuid.uuid4()),
            "video_id": video_id,
            "frame_number": fr["frame_number"],
            "timestamp_ms": fr["timestamp_ms"],
            "filepath": fr["filepath"],
        })

    if frame_rows:
        create_video_frames_bulk(frame_rows)

    return {
        "status": "success",
        "video": record,
        "total_frames": len(frames),
    }


@router.get("/videos")
async def get_videos():
    """List all uploaded videos."""
    videos = list_videos()
    return {"videos": videos, "count": len(videos)}


@router.get("/videos/{video_id}/frame/{frame_number}")
async def serve_video_frame(video_id: str, frame_number: int):
    """Serve a specific extracted video frame as JPEG."""
    frame = get_video_frame(video_id, frame_number)
    if not frame:
        raise HTTPException(404, "Frame not found.")
    fp = Path(frame["filepath"])
    if not fp.exists():
        raise HTTPException(404, "Frame file missing from disk.")
    return FileResponse(str(fp), media_type="image/jpeg")
