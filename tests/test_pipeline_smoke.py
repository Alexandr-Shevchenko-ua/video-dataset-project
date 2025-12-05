# STATUS: not executed, pseudocode – requires adaptation
"""Smoke tests for pipeline orchestration."""

from pathlib import Path

import numpy as np

from config import PipelineConfig
from dataset_loader import ZipItem
from pipeline import run_pipeline
from video_reader import SampledFrame


def test_pipeline_config_initializes(tmp_path, monkeypatch):
    cfg = PipelineConfig()
    cfg.output.dataset_root = tmp_path / "out"
    cfg.temp_extract_dir = tmp_path / "tmp"
    cfg.raw_cache_dir = tmp_path / "cache"
    cfg.metadata_path = cfg.output.dataset_root / "metadata.parquet"
    cfg.max_frames_total = 0

    monkeypatch.setattr("pipeline._iter_zip_items", lambda config: [])
    run_pipeline(cfg)


def test_dry_run_skips_writes_and_cleans_temp(tmp_path, monkeypatch):
    cfg = PipelineConfig()
    cfg.output.dataset_root = tmp_path / "out"
    cfg.temp_extract_dir = tmp_path / "tmp"
    cfg.raw_cache_dir = tmp_path / "cache"
    cfg.metadata_path = cfg.output.dataset_root / "metadata.parquet"
    cfg.dry_run = True
    cfg.max_frames_total = 1

    temp_zip = tmp_path / "stream_zip_0.zip"
    temp_zip.write_bytes(b"data")
    zip_item = ZipItem("z1", "train", size_bytes=4, disk_path=temp_zip, is_temporary=True)

    monkeypatch.setattr("pipeline._iter_zip_items", lambda config: [zip_item])
    monkeypatch.setattr("pipeline.estimate_uncompressed_size", lambda path: 1)
    monkeypatch.setattr("pipeline.safe_extract", lambda *_args, **_kwargs: [tmp_path / "vid.mp4"])

    sample_frame = SampledFrame(frame_index=0, timestamp_ms=0.0, diff_score=0.0, bgr_image=np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr("pipeline.iter_sampled_frames", lambda *_args, **_kwargs: [sample_frame])

    class _Eval:
        passes = True
        scores = {"blur": 1.0}
        failed_filters: list[str] = []

    monkeypatch.setattr("pipeline.evaluate_frame", lambda *_args, **_kwargs: _Eval())
    monkeypatch.setattr("pipeline.build_augmenter", lambda *_args, **_kwargs: (lambda img: (img, [])))

    run_pipeline(cfg)
    assert not cfg.metadata_path.exists()
    assert not temp_zip.exists()
