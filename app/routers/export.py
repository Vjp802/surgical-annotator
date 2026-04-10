"""
Surgical Annotator — Export Router
COCO JSON generation and download.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import EXPORT_DIR
from app.models.schemas import ExportRequest

router = APIRouter(prefix="/api", tags=["export"])


@router.post("/export")
async def export_coco(data: ExportRequest, request: Request):
    """Generate COCO JSON from approved annotations (images + video)."""
    exporter = request.app.state.coco_exporter
    result = exporter.export(description=data.description, version=data.version)
    return {
        "status": "success",
        "filename": result["filename"],
        "stats": result["stats"],
    }


@router.get("/export/{filename}")
async def download_export(filename: str):
    """Download a generated COCO export file."""
    fp = Path(EXPORT_DIR) / filename
    if not fp.exists():
        raise HTTPException(404, "Export file not found.")
    return FileResponse(str(fp), media_type="application/json", filename=filename)
