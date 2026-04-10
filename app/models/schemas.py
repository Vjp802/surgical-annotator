"""
Surgical Annotator — Pydantic Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------

class SegmentImageRequest(BaseModel):
    image_id: str
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class SegmentVideoRequest(BaseModel):
    video_id: str
    frame_number: int = Field(..., ge=0)
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class SegmentImageResponse(BaseModel):
    mask_png_b64: str
    mask_rle: dict
    mask_polygon: list
    bbox: list
    area: float
    score: float


class FrameMask(BaseModel):
    frame_number: int
    mask_png_b64: str
    mask_rle: dict
    mask_polygon: list
    bbox: list
    area: float


class SegmentVideoResponse(BaseModel):
    track_id: str
    frame_masks: list[FrameMask]


# ---------------------------------------------------------------------------
# Identify
# ---------------------------------------------------------------------------

class IdentifyRequest(BaseModel):
    image_id: Optional[str] = None
    video_id: Optional[str] = None
    frame_number: Optional[int] = None
    mask_rle: dict


class IdentifyResponse(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

class AnnotationCreate(BaseModel):
    image_id: Optional[str] = None
    video_id: Optional[str] = None
    frame_number: Optional[int] = None
    label: str = "unknown"
    confidence: float = 0.0
    mask_rle: Optional[dict] = None
    mask_polygon: Optional[list] = None
    bbox: Optional[list] = None
    area: float = 0.0
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    approved: bool = False
    corrected: bool = False
    track_id: Optional[str] = None


class TrackAnnotationFrame(BaseModel):
    frame_number: int
    mask_rle: Optional[dict] = None
    mask_polygon: Optional[list] = None
    bbox: Optional[list] = None
    area: float = 0.0


class AnnotationTrackCreate(BaseModel):
    video_id: str
    track_id: str
    label: str = "unknown"
    confidence: float = 0.0
    click_x: Optional[int] = None
    click_y: Optional[int] = None
    approved: bool = False
    frames: list[TrackAnnotationFrame]


class AnnotationUpdate(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    approved: Optional[bool] = None
    corrected: Optional[bool] = None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    description: str = "Surgical annotation dataset"
    version: str = "1.0"
