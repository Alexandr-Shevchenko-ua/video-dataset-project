# STATUS: not executed, pseudocode – requires adaptation
"""Frame quality scoring and filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from config import QualityFilterConfig


@dataclass
class FrameEvaluation:
    """Result of evaluating a frame."""

    passes: bool
    scores: Dict[str, float]


def _variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_score(gray: np.ndarray) -> float:
    low = np.mean(gray < 5)
    high = np.mean(gray > 250)
    return float(low + high)


def _motion_blur(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(gray, prev_gray)
    sobelx = cv2.Sobel(diff, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(diff, cv2.CV_64F, 0, 1)
    return float(np.mean(np.abs(sobelx)) + np.mean(np.abs(sobely)))


def _compression_blockiness(gray: np.ndarray) -> float:
    h, w = gray.shape
    h_trim = h - (h % 8)
    w_trim = w - (w % 8)
    trimmed = gray[:h_trim, :w_trim]
    blocks = trimmed.reshape(h_trim // 8, 8, w_trim // 8, 8)
    diffs = np.diff(blocks, axis=1).mean() + np.diff(blocks, axis=3).mean()
    return float(abs(diffs))


def evaluate_frame(
    frame_bgr: np.ndarray,
    prev_bgr: Optional[np.ndarray],
    config: QualityFilterConfig,
) -> FrameEvaluation:
    """Compute quality metrics and pass/fail decision."""

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = _variance_of_laplacian(gray)
    brightness = float(np.mean(gray) / 255.0)
    exposure = _exposure_score(gray)
    motion = _motion_blur(
        None if prev_bgr is None else cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY),
        gray,
    )
    compression = _compression_blockiness(gray)

    passes = (
        blur >= config.blur_laplacian_thresh
        and config.brightness_min <= brightness <= config.brightness_max
        and exposure <= config.exposure_saturation_thresh
        and motion <= config.motion_blur_thresh
        and compression <= config.compression_block_thresh
    )
    return FrameEvaluation(
        passes=passes,
        scores={
            "blur": blur,
            "brightness": brightness,
            "exposure": exposure,
            "motion": motion,
            "compression": compression,
        },
    )
