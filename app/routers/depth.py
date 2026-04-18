"""
Surgical Annotator — Depth Router
Optional endpoint to retrieve dense depth map estimates.
"""

import base64
import io

from fastapi import APIRouter, HTTPException, Request
from PIL import Image

from app.models.database import get_image, get_video_frame

router = APIRouter(prefix="/api", tags=["depth"])


@router.get("/depth/{image_id}")
async def get_depth_map(image_id: str, request: Request):
    """
    Get the dense depth map for an image.
    Returns the depth map as a base64 encoded PNG heatmap.
    """
    depth_service = request.app.state.depth_service
    if not depth_service.is_ready():
        raise HTTPException(503, "Depth Estimation is disabled or failed to load.")

    record = await get_image(image_id)
    if not record:
        raise HTTPException(404, "Image not found.")

    try:
        from numpy import array
        image_np = array(Image.open(record["filepath"]).convert("RGB"))
        
        from starlette.concurrency import run_in_threadpool
        # Returns 2D uint8 depth array
        depth_np = await run_in_threadpool(depth_service.estimate_depth, image_np)
        
        # Encode as grayscale PNG
        buf = io.BytesIO()
        Image.fromarray(depth_np).save(buf, format="PNG")
        buf.seek(0)
        
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"depth_map_b64": b64}
        
    except Exception as e:
        raise HTTPException(500, f"Failed to estimate depth: {e}")
