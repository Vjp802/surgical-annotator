"""
Surgical Annotator — Segment Router
Image segmentation and video mask propagation via SAM 2.
"""

import uuid

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from PIL import Image

from app.models.database import get_image, get_video
from app.models.schemas import (
    SegmentImageRequest, SegmentImageResponse,
    SegmentVideoRequest, SegmentVideoResponse, FrameMask,
)
from app.config import FRAMES_DIR

router = APIRouter(prefix="/api", tags=["segment"])


@router.post("/segment/image", response_model=SegmentImageResponse)
async def segment_image(req: SegmentImageRequest, request: Request):
    """Segment an organ from a point click on an image."""
    sam = request.app.state.sam_service
    if not sam.is_ready():
        raise HTTPException(503, "SAM 2 model not loaded.")

    record = await get_image(req.image_id)
    if not record:
        raise HTTPException(404, "Image not found.")

    try:
        img = np.array(Image.open(record["filepath"]).convert("RGB"))
    except Exception as e:
        raise HTTPException(500, f"Failed to load image: {e}")

    h, w = img.shape[:2]
    if req.x >= w or req.y >= h:
        raise HTTPException(400, f"Click ({req.x},{req.y}) out of bounds ({w}x{h}).")

    from starlette.concurrency import run_in_threadpool
    try:
        result = await run_in_threadpool(sam.segment_image, img, req.x, req.y)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")

    return SegmentImageResponse(
        mask_png_b64=result["mask_png_b64"],
        mask_rle=result["rle"],
        mask_polygon=result["polygon"],
        bbox=result["bbox"],
        area=result["area"],
        score=result["score"],
    )


@router.post("/segment/video", response_model=SegmentVideoResponse)
async def segment_video(req: SegmentVideoRequest, request: Request):
    """
    Run SAM 2 video segmentation: click on one frame,
    propagate mask through all frames.
    Returns per-frame masks and a track_id.
    """
    sam = request.app.state.sam_service
    if not sam.is_ready():
        raise HTTPException(503, "SAM 2 model not loaded.")

    record = await get_video(req.video_id)
    if not record:
        raise HTTPException(404, "Video not found.")

    from pathlib import Path
    frames_dir = str(Path(FRAMES_DIR) / req.video_id)
    if not Path(frames_dir).exists():
        raise HTTPException(404, "Frames directory not found. Was the video processed?")

    from starlette.concurrency import run_in_threadpool
    try:
        results = await run_in_threadpool(
            sam.segment_video,
            frames_dir=frames_dir,
            click_frame=req.frame_number,
            x=req.x,
            y=req.y,
        )
    except Exception as e:
        raise HTTPException(500, f"Video segmentation failed: {e}")

    track_id = str(uuid.uuid4())

    frame_masks = [
        FrameMask(
            frame_number=r["frame_number"],
            mask_png_b64=r["mask_png_b64"],
            mask_rle=r["rle"],
            mask_polygon=r["polygon"],
            bbox=r["bbox"],
            area=r["area"],
        )
        for r in results
    ]

    return SegmentVideoResponse(track_id=track_id, frame_masks=frame_masks)
