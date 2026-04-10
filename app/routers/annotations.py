"""
Surgical Annotator — Annotations Router
CRUD for annotations + bulk track save/delete for video tracking.
"""

import json
import uuid

from fastapi import APIRouter, HTTPException

from app.models.database import (
    create_annotation, create_annotations_bulk,
    get_annotation, get_annotations_for_image, get_annotations_for_video,
    update_annotation, delete_annotation, delete_annotations_by_track,
)
from app.models.schemas import AnnotationCreate, AnnotationTrackCreate, AnnotationUpdate

router = APIRouter(prefix="/api", tags=["annotations"])


# ===================================================================
# Single annotation CRUD
# ===================================================================

@router.post("/annotations")
async def create_new_annotation(data: AnnotationCreate):
    """Create a single annotation (image or single video frame)."""
    record = create_annotation(
        image_id=data.image_id,
        video_id=data.video_id,
        frame_number=data.frame_number,
        label=data.label,
        confidence=data.confidence,
        mask_rle=json.dumps(data.mask_rle) if data.mask_rle else None,
        mask_polygon=json.dumps(data.mask_polygon) if data.mask_polygon else None,
        bbox=json.dumps(data.bbox) if data.bbox else None,
        area=data.area,
        click_x=data.click_x,
        click_y=data.click_y,
        approved=data.approved,
        corrected=data.corrected,
        track_id=data.track_id,
    )
    return {"status": "created", "annotation": record}


# ===================================================================
# Bulk track save (video)
# ===================================================================

@router.post("/annotations/track")
async def save_track_annotations(data: AnnotationTrackCreate):
    """
    Save all frame annotations from a video tracking session in one call.
    Each frame shares the same track_id, label, confidence.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for fr in data.frames:
        rows.append({
            "id": str(uuid.uuid4()),
            "image_id": None,
            "video_id": data.video_id,
            "frame_number": fr.frame_number,
            "label": data.label,
            "confidence": data.confidence,
            "mask_rle": json.dumps(fr.mask_rle) if fr.mask_rle else None,
            "mask_polygon": json.dumps(fr.mask_polygon) if fr.mask_polygon else None,
            "bbox": json.dumps(fr.bbox) if fr.bbox else None,
            "area": fr.area,
            "click_x": data.click_x,
            "click_y": data.click_y,
            "approved": 0,
            "corrected": 0,
            "track_id": data.track_id,
            "created_at": now,
        })

    if rows:
        create_annotations_bulk(rows)

    return {
        "status": "created",
        "track_id": data.track_id,
        "frames_saved": len(rows),
    }


# ===================================================================
# Queries
# ===================================================================

@router.get("/annotations/image/{image_id}")
async def get_image_annotations(image_id: str):
    """All annotations for an image."""
    anns = get_annotations_for_image(image_id)
    return {"annotations": anns, "count": len(anns)}


@router.get("/annotations/video/{video_id}")
async def get_video_annotations(video_id: str):
    """All annotations for a video grouped by track_id."""
    anns = get_annotations_for_video(video_id)

    # Group by track_id
    tracks = {}
    for a in anns:
        tid = a.get("track_id") or "no_track"
        tracks.setdefault(tid, []).append(a)

    return {"annotations": anns, "tracks": tracks, "count": len(anns)}


# ===================================================================
# Update / Delete
# ===================================================================

@router.put("/annotations/{annotation_id}")
async def update_existing_annotation(annotation_id: str, data: AnnotationUpdate):
    """Update label, confidence, approved, or corrected on an annotation."""
    existing = get_annotation(annotation_id)
    if not existing:
        raise HTTPException(404, "Annotation not found.")

    kwargs = {}
    if data.label is not None:
        kwargs["label"] = data.label
    if data.confidence is not None:
        kwargs["confidence"] = data.confidence
    if data.approved is not None:
        kwargs["approved"] = data.approved
    if data.corrected is not None:
        kwargs["corrected"] = data.corrected

    updated = update_annotation(annotation_id, **kwargs)
    return {"status": "updated", "annotation": updated}


@router.delete("/annotations/{annotation_id}")
async def delete_single_annotation(annotation_id: str):
    """Delete a single annotation."""
    if not delete_annotation(annotation_id):
        raise HTTPException(404, "Annotation not found.")
    return {"status": "deleted", "id": annotation_id}


@router.delete("/annotations/track/{track_id}")
async def delete_track_annotations(track_id: str):
    """Delete all annotations in a tracking session."""
    count = delete_annotations_by_track(track_id)
    if count == 0:
        raise HTTPException(404, "No annotations found for this track.")
    return {"status": "deleted", "track_id": track_id, "deleted_count": count}
