# STATUS: not executed, pseudocode – requires adaptation
"""Unit tests for frame quality metrics."""

import cv2
import numpy as np

from config import QualityFilterConfig
from frame_quality import evaluate_frame


def test_blur_metric_detects_soft_frames():
    cfg = QualityFilterConfig(blur_laplacian_thresh=50.0)
    sharp = np.zeros((64, 64, 3), dtype=np.uint8)
    sharp[:, 32:] = 255
    blurred = cv2.GaussianBlur(sharp, (9, 9), 5)

    result_sharp = evaluate_frame(sharp, None, cfg)
    result_blur = evaluate_frame(blurred, None, cfg)
    assert result_sharp.scores["blur"] > result_blur.scores["blur"]
