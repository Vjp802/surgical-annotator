"""
Surgical Annotator — Video Service
Frame extraction from surgical videos using ffmpeg/ffprobe.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import FRAME_EXTRACTION_FPS, MAX_VIDEO_FRAMES, FRAMES_DIR

logger = logging.getLogger(__name__)


class VideoService:
    """Handles video frame extraction and metadata via ffmpeg/ffprobe."""

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        fps: int = FRAME_EXTRACTION_FPS,
    ) -> list[dict]:
        """
        Extract frames from a video at the specified FPS using ffmpeg.

        Args:
            video_path: path to the video file
            output_dir: directory to write frame JPEGs into
            fps: frames per second to extract

        Returns:
            list of {frame_number, filepath, timestamp_ms}
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pattern = str(out / "frame_%06d.jpg")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            pattern,
        ]

        logger.info("Extracting frames: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            logger.error("ffmpeg error: %s", result.stderr[-500:] if result.stderr else "")
            raise RuntimeError(f"ffmpeg frame extraction failed (exit {result.returncode})")

        # Enumerate extracted frames
        frame_files = sorted(out.glob("frame_*.jpg"))

        if len(frame_files) > MAX_VIDEO_FRAMES:
            logger.warning(
                "Extracted %d frames, truncating to %d",
                len(frame_files), MAX_VIDEO_FRAMES,
            )
            for f in frame_files[MAX_VIDEO_FRAMES:]:
                f.unlink()
            frame_files = frame_files[:MAX_VIDEO_FRAMES]

        frames = []
        for idx, fp in enumerate(frame_files):
            timestamp_ms = (idx / fps) * 1000.0
            frames.append({
                "frame_number": idx,
                "filepath": str(fp),
                "timestamp_ms": round(timestamp_ms, 2),
            })

        logger.info("Extracted %d frames to %s", len(frames), output_dir)
        return frames

    def get_frame(self, video_id: str, frame_number: int) -> Optional[np.ndarray]:
        """Load a specific extracted frame as a numpy RGB array."""
        frame_path = Path(FRAMES_DIR) / video_id / f"frame_{frame_number + 1:06d}.jpg"
        if not frame_path.exists():
            return None
        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_video_metadata(self, video_path: str) -> dict:
        """
        Use ffprobe to extract video metadata.
        Returns {duration_seconds, fps, width, height, total_frames}.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
        except Exception as e:
            logger.error("ffprobe failed: %s", str(e))
            return {
                "duration_seconds": 0, "fps": 0,
                "width": 0, "height": 0, "total_frames": 0,
            }

        # Find the video stream
        video_stream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                video_stream = s
                break

        if video_stream is None:
            return {
                "duration_seconds": 0, "fps": 0,
                "width": 0, "height": 0, "total_frames": 0,
            }

        # Parse FPS from r_frame_rate (e.g. "30/1")
        fps = 0.0
        r_frame_rate = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 0
        except (ValueError, ZeroDivisionError):
            pass

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        duration = float(data.get("format", {}).get("duration", 0))
        total_frames = int(video_stream.get("nb_frames", 0))
        if total_frames == 0 and fps > 0 and duration > 0:
            total_frames = int(duration * fps)

        return {
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
        }
