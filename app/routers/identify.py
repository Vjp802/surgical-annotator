"""
Surgical Annotator — Identify Router
Organ identification via VLM for both images and video frames.
"""

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pycocotools import mask as mask_utils

from app.models.database import get_image, get_video_frame
from app.models.schemas import IdentifyRequest, IdentifyResponse

router = APIRouter(prefix="/api", tags=["identify"])


@router.post("/identify", response_model=IdentifyResponse)
async def identify_organ(req: IdentifyRequest, request: Request):
    """
    Identify the organ in a masked region.
    Accepts image_id OR (video_id + frame_number), plus mask_rle.
    """
    vlm = request.app.state.vlm_service
    depth_service = request.app.state.depth_service

    # Load the right image
    if req.image_id:
        record = await get_image(req.image_id)
        if not record:
            raise HTTPException(404, "Image not found.")
        try:
            image_np = np.array(Image.open(record["filepath"]).convert("RGB"))
        except Exception as e:
            raise HTTPException(500, f"Failed to load image: {e}")

    elif req.video_id is not None and req.frame_number is not None:
        frame = await get_video_frame(req.video_id, req.frame_number)
        if not frame:
            raise HTTPException(404, "Video frame not found.")
        try:
            image_np = np.array(Image.open(frame["filepath"]).convert("RGB"))
        except Exception as e:
            raise HTTPException(500, f"Failed to load frame: {e}")

    else:
        raise HTTPException(400, "Provide image_id or (video_id + frame_number).")

    # Decode RLE mask
    try:
        rle = req.mask_rle.copy()
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        mask = mask_utils.decode(rle).astype(bool)
    except Exception as e:
        raise HTTPException(400, f"Invalid mask RLE: {e}")

    from starlette.concurrency import run_in_threadpool
    result = await run_in_threadpool(vlm.identify_organ, image_np, mask)

    # Optional Depth Extraction
    avg_depth = await run_in_threadpool(depth_service.get_organ_average_depth, image_np, mask)

    return IdentifyResponse(
        label=result.get("label", "unknown"),
        confidence=result.get("confidence", 0.0),
        spatial_context=result.get("spatial_context"),
        average_depth=avg_depth
    )
