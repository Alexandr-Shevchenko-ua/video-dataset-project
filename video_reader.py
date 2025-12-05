# STATUS: not executed, pseudocode – requires adaptation
"""Video reading and frame sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from config import SamplingConfig
from logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SampledFrame:
    """DTO for frames emitted by the sampling iterator."""

    frame_index: int
    timestamp_ms: float
    diff_score: float
    bgr_image: np.ndarray


def _should_emit(
    timestamp_ms: float,
    next_allowed_ms: float,
    diff_score: float,
    config: SamplingConfig,
    frames_emitted: int,
) -> bool:
    """Decide whether current frame should be emitted."""

    if frames_emitted >= config.max_frames_per_video:
        return False
    if config.mode == "fixed_fps":
        return timestamp_ms >= next_allowed_ms
    adaptive_trigger = diff_score >= config.adaptive_diff_threshold
    interval_ok = timestamp_ms >= next_allowed_ms
    return adaptive_trigger or interval_ok


def iter_sampled_frames(video_path: str, config: SamplingConfig) -> Iterator[SampledFrame]:
    """Iterate frames from video according to sampling strategy."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    target_interval = 1000.0 / config.target_fps
    next_allowed = 0.0
    frames_emitted = 0
    prev_gray: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_score = 255.0 if prev_gray is None else float(np.mean(cv2.absdiff(gray, prev_gray)))
        if _should_emit(timestamp_ms, next_allowed, diff_score, config, frames_emitted):
            yield SampledFrame(
                frame_index=frame_idx,
                timestamp_ms=timestamp_ms,
                diff_score=diff_score,
                bgr_image=frame.copy(),
            )
            frames_emitted += 1
            next_allowed = timestamp_ms + max(config.min_frame_interval_ms, target_interval)
        prev_gray = gray
    cap.release()
