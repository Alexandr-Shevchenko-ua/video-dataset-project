# STATUS: not executed, pseudocode – requires adaptation
"""Smoke test for pipeline with mocked components."""

from pathlib import Path

import numpy as np

from config import PipelineConfig
from pipeline import run_pipeline


def test_pipeline_config_initializes(tmp_path, monkeypatch):
    """Ensure pipeline can initialize with basic config."""

    cfg = PipelineConfig()
    cfg.output.dataset_root = tmp_path / "out"
    cfg.temp_extract_dir = tmp_path / "tmp"
    cfg.raw_cache_dir = tmp_path / "cache"
    cfg.metadata_path = cfg.output.dataset_root / "metadata.parquet"
    cfg.max_frames_total = 0

    # monkeypatch heavy dependencies to avoid real dataset downloads
    monkeypatch.setattr("pipeline._iter_zip_items", lambda config: [])
    run_pipeline(cfg)
