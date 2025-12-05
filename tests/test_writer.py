# STATUS: not executed, pseudocode – requires adaptation
"""Tests for writer module."""

from pathlib import Path

import numpy as np

from config import OutputConfig
from writer import FrameRecord, flush_metadata, generate_frame_id, prepare_output_dirs, save_frame_image, MetadataWriter


def test_prepare_dirs_and_save_frame(tmp_path):
    output_cfg = OutputConfig(dataset_root=tmp_path)
    prepare_output_dirs(output_cfg)
    frame_id = generate_frame_id("video1", 1)
    image = (np.ones((16, 16, 3)) * 255).astype("uint8")
    rel_path = save_frame_image(image, frame_id, "train", output_cfg)
    assert (tmp_path / rel_path).exists()


def test_flush_metadata_creates_file(tmp_path):
    output_cfg = OutputConfig(dataset_root=tmp_path, metadata_format="jsonl")
    record = FrameRecord(
        frame_id="f1",
        video_id="v1",
        frame_index=0,
        timestamp_ms=0.0,
        split="train",
        filters_passed=["blur"],
        filters_failed=[],
        quality_scores={"blur": 100.0},
        augmentations=[],
        image_path="train/images/f1.jpg",
        resolution=(16, 16),
        diff_score=0.0,
    )
    metadata_path = tmp_path / "meta.jsonl"
    flush_metadata([record], output_cfg, metadata_path)
    assert metadata_path.exists()


def test_parquet_writer_handles_nested_fields(tmp_path):
    output_cfg = OutputConfig(dataset_root=tmp_path, metadata_format="parquet")
    metadata_path = tmp_path / "meta.parquet"
    writer = MetadataWriter(output_cfg, metadata_path)
    writer.add(
        FrameRecord(
            frame_id="f2",
            video_id="v2",
            frame_index=1,
            timestamp_ms=33.3,
            split="val",
            filters_passed=["blur"],
            filters_failed=["brightness"],
            quality_scores={"blur": 10.0, "brightness": 0.1},
            augmentations=[{"name": "Test", "params": {"k": 1}}],
            image_path="val/images/f2.jpg",
            resolution=(32, 32),
            diff_score=0.5,
        )
    )
    writer.close()
    assert metadata_path.exists() and metadata_path.stat().st_size > 0
