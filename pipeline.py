# STATUS: not executed, pseudocode – requires adaptation
"""High-level orchestration for the dataset generation pipeline."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np

from augmentations import AppliedAugmentation, build_augmenter
from config import PipelineConfig
from dataset_loader import ZipItem, download_selected_zips, stream_zips
from frame_quality import evaluate_frame
from logging_utils import get_logger, setup_logging
from video_reader import SampledFrame, iter_sampled_frames
from writer import FrameRecord, flush_metadata, generate_frame_id, prepare_output_dirs, save_frame_image
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


def run_pipeline(config: PipelineConfig, progress_cb: Optional[Callable[[ProgressUpdate], None]] = None) -> Path:
    """Execute full pipeline according to config."""

    _ensure_path_separation(config)
    setup_logging(config.output.dataset_root, config.log_level)
    prepare_output_dirs(config.output)
    random.seed(config.sampling.random_seed)
    np.random.seed(config.sampling.random_seed)

    augmenter = build_augmenter(config.augmentations)
    metadata_records: List[FrameRecord] = []
    total_frames = 0

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
            if config.dataset_selection.max_videos and video_idx >= config.dataset_selection.max_videos:
                break
            LOGGER.info("Sampling video %s", video_path.name)
            split = "train" if (hash(video_path.name) % 100) < int(config.output.train_split * 100) else "val"
            prev_frame = None
            for sampled in iter_sampled_frames(str(video_path), config.sampling):
                evaluation = evaluate_frame(sampled.bgr_image, prev_frame, config.quality)
                prev_frame = sampled.bgr_image
                if not evaluation.passes and not config.quality.allow_manual_override:
                    continue
                augmented_image, applied = augmenter(sampled.bgr_image[:, :, ::-1])  # convert to RGB if needed
                augmented_bgr = augmented_image[:, :, ::-1]
                frame_id = generate_frame_id(video_path.stem, sampled.frame_index)
                rel_path = save_frame_image(augmented_bgr, frame_id, split, config.output)
                record = FrameRecord(
                    frame_id=frame_id,
                    video_id=video_path.stem,
                    frame_index=sampled.frame_index,
                    timestamp_ms=sampled.timestamp_ms,
                    split=split,
                    filters_passed=[name for name, score in evaluation.scores.items()],
                    filters_failed=[] if evaluation.passes else list(evaluation.scores.keys()),
                    augmentations=[aug.__dict__ for aug in applied],
                    image_path=str(rel_path),
                    resolution=tuple(augmented_bgr.shape[:2]),
                    diff_score=sampled.diff_score,
                )
                metadata_records.append(record)
                total_frames += 1
                if progress_cb:
                    progress_cb(ProgressUpdate("frame", zip_idx, video_idx, total_frames))
                if config.max_frames_total and total_frames >= config.max_frames_total:
                    break
            if config.max_frames_total and total_frames >= config.max_frames_total:
                break
        flush_metadata(metadata_records, config.output, config.metadata_path)
        cleanup_temp_dir(extract_root)
        if config.max_frames_total and total_frames >= config.max_frames_total:
            break

    LOGGER.info("Pipeline finished: %d frames", total_frames)
    return config.output.dataset_root
