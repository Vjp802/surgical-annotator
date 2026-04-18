"""
Surgical Annotator — Database Layer
Async SQLAlchemy with images, videos, video_frames, and annotations tables.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import Integer, String, Float, ForeignKey, Index, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.sql import select, delete, update
from sqlalchemy.sql.expression import text

from app.config import DB_PATH

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 15}
)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, 
    expire_on_commit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

def _uuid() -> str:
    return str(uuid.uuid4())

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

# ===================================================================
# Models
# ===================================================================

class ImageModel(Base):
    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[str] = mapped_column(String, default=_now)

class VideoModel(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    fps: Mapped[Optional[float]] = mapped_column(Float)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    uploaded_at: Mapped[str] = mapped_column(String, default=_now)

class VideoFrameModel(Base):
    __tablename__ = "video_frames"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_ms: Mapped[Optional[float]] = mapped_column(Float)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    
    __table_args__ = (
        Index('idx_video_frames_video', 'video_id', 'frame_number'),
    )

class AnnotationModel(Base):
    __tablename__ = "annotations"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    image_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("images.id", ondelete="CASCADE"))
    video_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("videos.id", ondelete="CASCADE"))
    frame_number: Mapped[Optional[int]] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String, default='unknown')
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    mask_rle: Mapped[Optional[str]] = mapped_column(String)
    mask_polygon: Mapped[Optional[str]] = mapped_column(String)
    bbox: Mapped[Optional[str]] = mapped_column(String)
    area: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    click_x: Mapped[Optional[int]] = mapped_column(Integer)
    click_y: Mapped[Optional[int]] = mapped_column(Integer)
    approved: Mapped[int] = mapped_column(Integer, default=0)
    corrected: Mapped[int] = mapped_column(Integer, default=0)
    track_id: Mapped[Optional[str]] = mapped_column(String)
    average_depth: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[str] = mapped_column(String, default=_now)
    
    __table_args__ = (
        Index('idx_annotations_image', 'image_id'),
        Index('idx_annotations_video', 'video_id'),
        Index('idx_annotations_track', 'track_id'),
    )


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.execute(text("PRAGMA foreign_keys=ON"))
        await conn.run_sync(Base.metadata.create_all)

def row_to_dict(row) -> dict:
    if row is None: return None
    if getattr(row, '_mapping', None) is not None:
        return dict(row._mapping)
    return row

# ===================================================================
# IMAGES CRUD
# ===================================================================

async def create_image(filename: str, filepath: str, width: int, height: int) -> dict:
    async with AsyncSessionLocal() as session:
        obj = ImageModel(filename=filename, filepath=filepath, width=width, height=height)
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

async def get_image(image_id: str) -> Optional[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(ImageModel.__table__).where(ImageModel.id == image_id))
        return row_to_dict(res.first())

async def list_images() -> List[dict]:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(
                ImageModel.__table__,
                func.count(AnnotationModel.id).label("annotation_count")
            )
            .outerjoin(AnnotationModel, AnnotationModel.image_id == ImageModel.id)
            .group_by(ImageModel.id)
            .order_by(ImageModel.uploaded_at.desc())
        )
        res = await session.execute(stmt)
        return [row_to_dict(r) for r in res.all()]

# ===================================================================
# VIDEOS CRUD
# ===================================================================

async def create_video(
    filename: str, filepath: str,
    duration_seconds: float = 0, fps: float = 0,
    total_frames: int = 0, width: int = 0, height: int = 0,
) -> dict:
    async with AsyncSessionLocal() as session:
        obj = VideoModel(
            filename=filename, filepath=filepath, duration_seconds=duration_seconds,
            fps=fps, total_frames=total_frames, width=width, height=height
        )
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

async def get_video(video_id: str) -> Optional[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(VideoModel.__table__).where(VideoModel.id == video_id))
        return row_to_dict(res.first())

async def list_videos() -> List[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(VideoModel.__table__).order_by(VideoModel.uploaded_at.desc()))
        return [row_to_dict(r) for r in res.all()]

# ===================================================================
# VIDEO FRAMES CRUD
# ===================================================================

async def create_video_frame(
    video_id: str, frame_number: int, timestamp_ms: float, filepath: str,
) -> dict:
    async with AsyncSessionLocal() as session:
        obj = VideoFrameModel(
            video_id=video_id, frame_number=frame_number, timestamp_ms=timestamp_ms, filepath=filepath
        )
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

async def create_video_frames_bulk(frames: List[dict]) -> None:
    async with AsyncSessionLocal() as session:
        for f in frames:
            if 'id' not in f:
                f['id'] = _uuid()
        await session.execute(VideoFrameModel.__table__.insert(), frames)
        await session.commit()

async def get_video_frames(video_id: str) -> List[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(VideoFrameModel.__table__).where(VideoFrameModel.video_id == video_id)
            .order_by(VideoFrameModel.frame_number)
        )
        return [row_to_dict(r) for r in res.all()]

async def get_video_frame(video_id: str, frame_number: int) -> Optional[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(VideoFrameModel.__table__)
            .where(VideoFrameModel.video_id == video_id, VideoFrameModel.frame_number == frame_number)
        )
        return row_to_dict(res.first())

# ===================================================================
# ANNOTATIONS CRUD
# ===================================================================

async def create_annotation(
    image_id: Optional[str] = None,
    video_id: Optional[str] = None,
    frame_number: Optional[int] = None,
    label: str = "unknown",
    confidence: float = 0.0,
    mask_rle: Optional[str] = None,
    mask_polygon: Optional[str] = None,
    bbox: Optional[str] = None,
    area: float = 0.0,
    click_x: Optional[int] = None,
    click_y: Optional[int] = None,
    approved: bool = False,
    corrected: bool = False,
    track_id: Optional[str] = None,
    average_depth: Optional[float] = None,
) -> dict:
    async with AsyncSessionLocal() as session:
        obj = AnnotationModel(
            image_id=image_id, video_id=video_id, frame_number=frame_number, label=label,
            confidence=confidence, mask_rle=mask_rle, mask_polygon=mask_polygon, bbox=bbox,
            area=area, click_x=click_x, click_y=click_y,
            approved=int(approved), corrected=int(corrected),
            track_id=track_id, average_depth=average_depth
        )
        session.add(obj)
        await session.commit()
        await session.refresh(obj)
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

async def create_annotations_bulk(annotations: List[dict]) -> None:
    async with AsyncSessionLocal() as session:
        for a in annotations:
            if 'approved' in a: a['approved'] = int(a['approved'])
            if 'corrected' in a: a['corrected'] = int(a['corrected'])
            if 'id' not in a: a['id'] = _uuid()
            if 'created_at' not in a: a['created_at'] = _now()
        await session.execute(AnnotationModel.__table__.insert(), annotations)
        await session.commit()

async def get_annotation(ann_id: str) -> Optional[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(AnnotationModel.__table__).where(AnnotationModel.id == ann_id))
        return row_to_dict(res.first())

async def get_annotations_for_image(image_id: str) -> List[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(AnnotationModel.__table__)
            .where(AnnotationModel.image_id == image_id)
            .order_by(AnnotationModel.created_at)
        )
        return [row_to_dict(r) for r in res.all()]

async def get_annotations_for_video(video_id: str) -> List[dict]:
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(AnnotationModel.__table__)
            .where(AnnotationModel.video_id == video_id)
            .order_by(AnnotationModel.track_id, AnnotationModel.frame_number)
        )
        return [row_to_dict(r) for r in res.all()]

async def update_annotation(ann_id: str, **kwargs) -> Optional[dict]:
    allowed = {
        "label", "confidence", "mask_rle", "mask_polygon",
        "bbox", "area", "approved", "corrected", "average_depth",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return await get_annotation(ann_id)
    
    for key in ("approved", "corrected"):
        if key in updates:
            updates[key] = int(updates[key])

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(AnnotationModel).where(AnnotationModel.id == ann_id).values(**updates)
        )
        await session.commit()
    return await get_annotation(ann_id)

async def delete_annotation(ann_id: str) -> bool:
    async with AsyncSessionLocal() as session:
        res = await session.execute(delete(AnnotationModel).where(AnnotationModel.id == ann_id))
        await session.commit()
        return res.rowcount > 0

async def delete_annotations_by_track(track_id: str) -> int:
    async with AsyncSessionLocal() as session:
        res = await session.execute(delete(AnnotationModel).where(AnnotationModel.track_id == track_id))
        await session.commit()
        return res.rowcount

async def get_approved_annotations() -> List[dict]:
    async with AsyncSessionLocal() as session:
        # Images
        stmt1 = (
            select(
                AnnotationModel.__table__,
                ImageModel.filename.label("img_filename"),
                ImageModel.filepath.label("img_filepath"),
                ImageModel.width.label("img_width"),
                ImageModel.height.label("img_height")
            )
            .join(ImageModel, AnnotationModel.image_id == ImageModel.id)
            .where(AnnotationModel.approved == 1, AnnotationModel.image_id.isnot(None))
            .order_by(AnnotationModel.image_id, AnnotationModel.created_at)
        )
        res1 = await session.execute(stmt1)
        img_anns = [row_to_dict(r) for r in res1.all()]
        
        # Videos
        stmt2 = (
            select(
                AnnotationModel.__table__,
                VideoFrameModel.filepath.label("frame_filepath"),
                VideoModel.width.label("img_width"),
                VideoModel.height.label("img_height"),
                VideoModel.filename.label("vid_filename")
            )
            .join(VideoModel, AnnotationModel.video_id == VideoModel.id)
            .join(VideoFrameModel, (VideoFrameModel.video_id == AnnotationModel.video_id) & (VideoFrameModel.frame_number == AnnotationModel.frame_number))
            .where(AnnotationModel.approved == 1, AnnotationModel.video_id.isnot(None))
            .order_by(AnnotationModel.video_id, AnnotationModel.track_id, AnnotationModel.frame_number)
        )
        res2 = await session.execute(stmt2)
        vid_anns = [row_to_dict(r) for r in res2.all()]
        
        return img_anns + vid_anns
