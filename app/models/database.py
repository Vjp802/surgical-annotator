"""
Surgical Annotator — Database Layer
SQLite with images, videos, video_frames, and annotations tables.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import DB_PATH


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_db() -> sqlite3.Connection:
    return _get_connection()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def init_db() -> None:
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                filepath    TEXT NOT NULL,
                width       INTEGER NOT NULL,
                height      INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS videos (
                id               TEXT PRIMARY KEY,
                filename         TEXT NOT NULL,
                filepath         TEXT NOT NULL,
                duration_seconds REAL,
                fps              REAL,
                total_frames     INTEGER,
                width            INTEGER,
                height           INTEGER,
                uploaded_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS video_frames (
                id           TEXT PRIMARY KEY,
                video_id     TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp_ms REAL,
                filepath     TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_video_frames_video
                ON video_frames(video_id, frame_number);

            CREATE TABLE IF NOT EXISTS annotations (
                id            TEXT PRIMARY KEY,
                image_id      TEXT,
                video_id      TEXT,
                frame_number  INTEGER,
                label         TEXT NOT NULL DEFAULT 'unknown',
                confidence    REAL NOT NULL DEFAULT 0.0,
                mask_rle      TEXT,
                mask_polygon  TEXT,
                bbox          TEXT,
                area          REAL DEFAULT 0.0,
                click_x       INTEGER,
                click_y       INTEGER,
                approved      INTEGER NOT NULL DEFAULT 0,
                corrected     INTEGER NOT NULL DEFAULT 0,
                track_id      TEXT,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_annotations_image
                ON annotations(image_id);
            CREATE INDEX IF NOT EXISTS idx_annotations_video
                ON annotations(video_id);
            CREATE INDEX IF NOT EXISTS idx_annotations_track
                ON annotations(track_id);
        """)
        conn.commit()
    finally:
        conn.close()


# ===================================================================
# IMAGES CRUD
# ===================================================================

def create_image(filename: str, filepath: str, width: int, height: int) -> dict:
    conn = get_db()
    try:
        row = {
            "id": _uuid(), "filename": filename, "filepath": filepath,
            "width": width, "height": height, "uploaded_at": _now(),
        }
        conn.execute(
            "INSERT INTO images (id,filename,filepath,width,height,uploaded_at) "
            "VALUES (:id,:filename,:filepath,:width,:height,:uploaded_at)", row,
        )
        conn.commit()
        return row
    finally:
        conn.close()


def get_image(image_id: str) -> Optional[dict]:
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def list_images() -> list[dict]:
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute("""
            SELECT i.*, COUNT(a.id) AS annotation_count
            FROM images i
            LEFT JOIN annotations a ON a.image_id = i.id
            GROUP BY i.id
            ORDER BY i.uploaded_at DESC
        """).fetchall()]
    finally:
        conn.close()


# ===================================================================
# VIDEOS CRUD
# ===================================================================

def create_video(
    filename: str, filepath: str,
    duration_seconds: float = 0, fps: float = 0,
    total_frames: int = 0, width: int = 0, height: int = 0,
) -> dict:
    conn = get_db()
    try:
        row = {
            "id": _uuid(), "filename": filename, "filepath": filepath,
            "duration_seconds": duration_seconds, "fps": fps,
            "total_frames": total_frames, "width": width, "height": height,
            "uploaded_at": _now(),
        }
        conn.execute(
            "INSERT INTO videos "
            "(id,filename,filepath,duration_seconds,fps,total_frames,width,height,uploaded_at) "
            "VALUES (:id,:filename,:filepath,:duration_seconds,:fps,:total_frames,:width,:height,:uploaded_at)",
            row,
        )
        conn.commit()
        return row
    finally:
        conn.close()


def get_video(video_id: str) -> Optional[dict]:
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def list_videos() -> list[dict]:
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM videos ORDER BY uploaded_at DESC"
        ).fetchall()]
    finally:
        conn.close()


# ===================================================================
# VIDEO FRAMES CRUD
# ===================================================================

def create_video_frame(
    video_id: str, frame_number: int, timestamp_ms: float, filepath: str,
) -> dict:
    conn = get_db()
    try:
        row = {
            "id": _uuid(), "video_id": video_id,
            "frame_number": frame_number, "timestamp_ms": timestamp_ms,
            "filepath": filepath,
        }
        conn.execute(
            "INSERT INTO video_frames (id,video_id,frame_number,timestamp_ms,filepath) "
            "VALUES (:id,:video_id,:frame_number,:timestamp_ms,:filepath)", row,
        )
        conn.commit()
        return row
    finally:
        conn.close()


def create_video_frames_bulk(frames: list[dict]) -> None:
    """Insert many video frame records in a single transaction."""
    conn = get_db()
    try:
        conn.executemany(
            "INSERT INTO video_frames (id,video_id,frame_number,timestamp_ms,filepath) "
            "VALUES (:id,:video_id,:frame_number,:timestamp_ms,:filepath)",
            frames,
        )
        conn.commit()
    finally:
        conn.close()


def get_video_frames(video_id: str) -> list[dict]:
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM video_frames WHERE video_id=? ORDER BY frame_number",
            (video_id,),
        ).fetchall()]
    finally:
        conn.close()


def get_video_frame(video_id: str, frame_number: int) -> Optional[dict]:
    conn = get_db()
    try:
        r = conn.execute(
            "SELECT * FROM video_frames WHERE video_id=? AND frame_number=?",
            (video_id, frame_number),
        ).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


# ===================================================================
# ANNOTATIONS CRUD
# ===================================================================

def create_annotation(
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
) -> dict:
    conn = get_db()
    try:
        row = {
            "id": _uuid(), "image_id": image_id, "video_id": video_id,
            "frame_number": frame_number, "label": label,
            "confidence": confidence, "mask_rle": mask_rle,
            "mask_polygon": mask_polygon, "bbox": bbox, "area": area,
            "click_x": click_x, "click_y": click_y,
            "approved": int(approved), "corrected": int(corrected),
            "track_id": track_id, "created_at": _now(),
        }
        conn.execute(
            "INSERT INTO annotations "
            "(id,image_id,video_id,frame_number,label,confidence,mask_rle,mask_polygon,"
            "bbox,area,click_x,click_y,approved,corrected,track_id,created_at) "
            "VALUES (:id,:image_id,:video_id,:frame_number,:label,:confidence,:mask_rle,"
            ":mask_polygon,:bbox,:area,:click_x,:click_y,:approved,:corrected,:track_id,:created_at)",
            row,
        )
        conn.commit()
        return row
    finally:
        conn.close()


def create_annotations_bulk(annotations: list[dict]) -> None:
    """Insert a batch of annotations in a single transaction (for video tracks)."""
    conn = get_db()
    try:
        conn.executemany(
            "INSERT INTO annotations "
            "(id,image_id,video_id,frame_number,label,confidence,mask_rle,mask_polygon,"
            "bbox,area,click_x,click_y,approved,corrected,track_id,created_at) "
            "VALUES (:id,:image_id,:video_id,:frame_number,:label,:confidence,:mask_rle,"
            ":mask_polygon,:bbox,:area,:click_x,:click_y,:approved,:corrected,:track_id,:created_at)",
            annotations,
        )
        conn.commit()
    finally:
        conn.close()


def get_annotation(ann_id: str) -> Optional[dict]:
    conn = get_db()
    try:
        r = conn.execute("SELECT * FROM annotations WHERE id=?", (ann_id,)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def get_annotations_for_image(image_id: str) -> list[dict]:
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM annotations WHERE image_id=? ORDER BY created_at",
            (image_id,),
        ).fetchall()]
    finally:
        conn.close()


def get_annotations_for_video(video_id: str) -> list[dict]:
    """Get all annotations for a video, ordered by track then frame."""
    conn = get_db()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM annotations WHERE video_id=? "
            "ORDER BY track_id, frame_number",
            (video_id,),
        ).fetchall()]
    finally:
        conn.close()


def update_annotation(ann_id: str, **kwargs) -> Optional[dict]:
    allowed = {
        "label", "confidence", "mask_rle", "mask_polygon",
        "bbox", "area", "approved", "corrected",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return get_annotation(ann_id)

    for key in ("approved", "corrected"):
        if key in updates:
            updates[key] = int(updates[key])

    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [ann_id]

    conn = get_db()
    try:
        conn.execute(f"UPDATE annotations SET {set_clause} WHERE id=?", values)
        conn.commit()
        return get_annotation(ann_id)
    finally:
        conn.close()


def delete_annotation(ann_id: str) -> bool:
    conn = get_db()
    try:
        cur = conn.execute("DELETE FROM annotations WHERE id=?", (ann_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_annotations_by_track(track_id: str) -> int:
    """Delete all annotations sharing a track_id. Returns count deleted."""
    conn = get_db()
    try:
        cur = conn.execute("DELETE FROM annotations WHERE track_id=?", (track_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def get_approved_annotations() -> list[dict]:
    """All approved annotations with image/video metadata for COCO export."""
    conn = get_db()
    try:
        # Image annotations
        img_anns = [dict(r) for r in conn.execute("""
            SELECT a.*, i.filename AS img_filename, i.filepath AS img_filepath,
                   i.width AS img_width, i.height AS img_height
            FROM annotations a
            JOIN images i ON a.image_id = i.id
            WHERE a.approved = 1 AND a.image_id IS NOT NULL
            ORDER BY a.image_id, a.created_at
        """).fetchall()]

        # Video frame annotations
        vid_anns = [dict(r) for r in conn.execute("""
            SELECT a.*, vf.filepath AS frame_filepath,
                   v.width AS img_width, v.height AS img_height,
                   v.filename AS vid_filename
            FROM annotations a
            JOIN videos v ON a.video_id = v.id
            JOIN video_frames vf ON vf.video_id = a.video_id AND vf.frame_number = a.frame_number
            WHERE a.approved = 1 AND a.video_id IS NOT NULL
            ORDER BY a.video_id, a.track_id, a.frame_number
        """).fetchall()]

        return img_anns + vid_anns
    finally:
        conn.close()
