# STATUS: not executed, pseudocode – requires adaptation
"""Output writer for frames and metadata artifacts."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    quality_scores: Dict[str, float]
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


class MetadataWriter:
    """Batching helper for metadata persistence."""

    def __init__(self, output_cfg: OutputConfig, metadata_path: Path):
        self.output_cfg = output_cfg
        self.metadata_path = metadata_path
        self._buffer: List[FrameRecord] = []
        self._pq_writer = None
        self._csv_has_header = False
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            metadata_path.unlink()

    def add(self, record: FrameRecord) -> None:
        self._buffer.append(record)
        if len(self._buffer) >= self.output_cfg.metadata_batch_size:
            self._flush_buffer()

    def close(self) -> None:
        self._flush_buffer()
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame([record.__dict__ for record in self._buffer])
        fmt = self.output_cfg.metadata_format
        if fmt == "parquet":
            self._write_parquet(df)
        elif fmt == "csv":
            df.to_csv(
                self.metadata_path,
                mode="a",
                header=not self._csv_has_header,
                index=False,
            )
            self._csv_has_header = True
        else:  # jsonl
            with self.metadata_path.open("a", encoding="utf-8") as f:
                for record in df.to_dict(orient="records"):
                    f.write(json.dumps(record))
                    f.write("\n")
        self._buffer.clear()

    def _write_parquet(self, df: pd.DataFrame) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._pq_writer is None:
            self._pq_writer = pq.ParquetWriter(self.metadata_path, schema=table.schema)
        self._pq_writer.write_table(table)


def flush_metadata(records: List[FrameRecord], output_cfg: OutputConfig, metadata_path: Path) -> None:
    """Legacy helper for one-shot writing (used in tests)."""

    writer = MetadataWriter(output_cfg, metadata_path)
    for record in records:
        writer.add(record)
    writer.close()
