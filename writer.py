# STATUS: not executed, pseudocode – requires adaptation
"""Output writer for frames and metadata artifacts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd

from config import OutputConfig
from logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FrameRecord:
    """Metadata for a processed frame."""

    frame_id: str
    video_id: str
    frame_index: int
    timestamp_ms: float
    split: str
    filters_passed: List[str]
    filters_failed: List[str]
    augmentations: List[Dict[str, object]]
    image_path: str
    resolution: tuple[int, int]
    diff_score: float = 0.0


def prepare_output_dirs(config: OutputConfig) -> None:
    """Create train/val dir structure and guard against accidental reuse."""

    root = config.dataset_root
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)


def generate_frame_id(video_id: str, frame_index: int) -> str:
    """Generate stable-ish unique frame identifier."""

    return f"{video_id}_{frame_index:06d}_{uuid.uuid4().hex[:6]}"


def save_frame_image(
    image_bgr,
    frame_id: str,
    split: str,
    output_cfg: OutputConfig,
) -> Path:
    """Persist frame to disk and return relative path."""

    rel = Path(split) / "images" / f"{frame_id}.{output_cfg.frame_extension}"
    full = output_cfg.dataset_root / rel
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if output_cfg.frame_extension == "jpg" else []
    cv2.imwrite(str(full), image_bgr, params)
    return rel


def flush_metadata(records: List[FrameRecord], output_cfg: OutputConfig, metadata_path: Path) -> None:
    """Write metadata to disk in desired format."""

    if not records:
        return
    df = pd.DataFrame([record.__dict__ for record in records])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if output_cfg.metadata_format == "parquet":
        df.to_parquet(metadata_path, engine="pyarrow", index=False)
    elif output_cfg.metadata_format == "csv":
        df.to_csv(metadata_path, index=False)
    else:
        df.to_json(metadata_path, orient="records", lines=True)
