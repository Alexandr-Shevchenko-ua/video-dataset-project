# STATUS: not executed, pseudocode – requires adaptation
"""Tests for augmentation pipeline."""

import numpy as np

from augmentations import build_augmenter
from config import AugmentationConfig


def test_augmentations_preserve_shape():
    cfg = AugmentationConfig()
    augmenter = build_augmenter(cfg)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    out, applied = augmenter(img)
    assert out.shape == img.shape
    assert isinstance(applied, list)
