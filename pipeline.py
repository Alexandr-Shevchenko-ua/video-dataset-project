# STATUS: not executed, pseudocode – requires adaptation
"""High-level orchestration for the dataset generation pipeline."""

from __future__ import annotations

import random
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from augmentations import build_augmenter
from config import PipelineConfig
from dataset_loader import ZipItem, download_selected_zips, stream_zips
from frame_quality import evaluate_frame
from logging_utils import get_logger, setup_logging
from video_reader import iter_sampled_frames
from writer import FrameRecord, MetadataWriter, generate_frame_id, prepare_output_dirs, save_frame_image
from zip_utils import cleanup_temp_dir, estimate_uncompressed_size, safe_extract

LOGGER = get_logger(__name__)


class ProgressUpdate:
    """Lightweight struct emitted to UI for progress reporting."""

    def __init__(self, message: str, zip_idx: int, video_idx: int, frames: int):
        self.message = message
        self.zip_idx = zip_idx
        self.video_idx = video_idx
        self.frames = frames


def _ensure_path_separation(config: PipelineConfig) -> None:
    raw = config.raw_cache_dir.resolve()
    out = config.output.dataset_root.resolve()
    temp = config.temp_extract_dir.resolve()
    if raw in out.parents or out in raw.parents:
        raise ValueError("raw_cache_dir must not overlap dataset_root")
    if temp in out.parents or out in temp.parents:
        raise ValueError("temp_extract_dir must not overlap dataset_root")


def _iter_zip_items(config: PipelineConfig) -> Iterable[ZipItem]:
    if config.disk_budget_gb < 5:
        return stream_zips(config)
    return download_selected_zips(config)


def _assign_video_split(video_id: str, config: PipelineConfig) -> str:
    """Deterministically assign split based on strategy."""

    key = f"{video_id}_{config.output.train_val_seed}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if value < config.output.train_split else "val"


def run_pipeline(config: PipelineConfig, progress_cb: Optional[Callable[[ProgressUpdate], None]] = None) -> Path:
    """Execute full pipeline according to config."""

    _ensure_path_separation(config)
    setup_logging(config.output.dataset_root, config.log_level)
    prepare_output_dirs(config.output)
    random.seed(config.sampling.random_seed)
    np.random.seed(config.sampling.random_seed)

    augmenter = build_augmenter(config.augmentations)
    metadata_writer = MetadataWriter(config.output, config.metadata_path)
    total_frames = 0
    videos_processed = 0
    per_frame_rng = random.Random(config.output.train_val_seed)

    for zip_idx, zip_item in enumerate(_iter_zip_items(config)):
        LOGGER.info("Processing zip %s", zip_item.identifier)
        extract_root = config.temp_extract_dir / zip_item.identifier
        estimate = estimate_uncompressed_size(zip_item.disk_path)
        if estimate > config.disk_budget_gb * (1024**3):
            LOGGER.warning("Skipping %s due to size %s", zip_item.identifier, estimate)
            continue
        extracted_videos = safe_extract(
            zip_item.disk_path,
            extract_root,
            max_bytes=int(config.disk_budget_gb * (1024**3)),
        )
        for video_idx, video_path in enumerate(extracted_videos):
            if config.dataset_selection.max_videos and videos_processed >= config.dataset_selection.max_videos:
                break
            LOGGER.info("Sampling video %s", video_path.name)
            video_split = _assign_video_split(video_path.stem, config)
            prev_frame = None
            videos_processed += 1
            for sampled in iter_sampled_frames(str(video_path), config.sampling):
                split = (
                    video_split
                    if config.output.split_strategy == "per_video"
                    else ("train" if per_frame_rng.random() < config.output.train_split else "val")
                )
                evaluation = evaluate_frame(sampled.bgr_image, prev_frame, config.quality)
                prev_frame = sampled.bgr_image
                if not evaluation.passes and not config.quality.allow_manual_override:
                    continue
                rgb_image = sampled.bgr_image[:, :, ::-1]
                augmented_image, applied = augmenter(rgb_image)
                augmented_bgr = augmented_image[:, :, ::-1]
                if config.dry_run:
                    continue
                frame_id = generate_frame_id(video_path.stem, sampled.frame_index)
                rel_path = save_frame_image(augmented_bgr, frame_id, split, config.output)
                passed_filters = [name for name in evaluation.scores.keys() if name not in evaluation.failed_filters]
                record = FrameRecord(
                    frame_id=frame_id,
                    video_id=video_path.stem,
                    frame_index=sampled.frame_index,
                    timestamp_ms=sampled.timestamp_ms,
                    split=split,
                    filters_passed=passed_filters,
                    filters_failed=evaluation.failed_filters,
                    quality_scores=evaluation.scores,
                    augmentations=[aug.__dict__ for aug in applied],
                    image_path=str(rel_path),
                    resolution=tuple(int(x) for x in augmented_bgr.shape[:2]),
                    diff_score=sampled.diff_score,
                )
                metadata_writer.add(record)
                total_frames += 1
                if progress_cb:
                    progress_cb(ProgressUpdate("frame", zip_idx, video_idx, total_frames))
                if config.max_frames_total and total_frames >= config.max_frames_total:
                    break
            if config.max_frames_total and total_frames >= config.max_frames_total:
                break
        cleanup_temp_dir(extract_root)
        if config.max_frames_total and total_frames >= config.max_frames_total:
            break

    metadata_writer.close()
    LOGGER.info("Pipeline finished: %d frames", total_frames)
    return config.output.dataset_root
