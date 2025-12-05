# STATUS: not executed, pseudocode – requires adaptation
"""Augmentation pipelines for analog camera & radio interference effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np

from config import AugmentationConfig, EffectConfig


@dataclass
class AppliedAugmentation:
    """Metadata describing which augmentation was applied."""

    name: str
    params: Dict[str, float | int | Tuple[float, float]]


class ScanlinesTransform(A.ImageOnlyTransform):
    """Custom transform for scanline artifacts."""

    def __init__(self, config: EffectConfig, always_apply=False, p=0.5):
        super().__init__(p=p if config.enabled else 0.0)
        self.cfg = config

    def apply(self, img: np.ndarray, **params):
        orientation = self.cfg.params.get("orientation", "horizontal")
        spacing_min = int(self.cfg.params.get("spacing_min", 3))
        spacing_max = int(self.cfg.params.get("spacing_max", 15))
        intensity_min = float(self.cfg.params.get("intensity_min", 0.05))
        intensity_max = float(self.cfg.params.get("intensity_max", 0.2))
        spacing = np.random.randint(spacing_min, spacing_max + 1)
        intensity = np.random.uniform(intensity_min, intensity_max)
        mask = np.zeros_like(img, dtype=np.float32)
        if orientation == "horizontal":
            mask[::spacing, :] = intensity
        else:
            mask[:, ::spacing] = intensity
        noisy = img.astype(np.float32) * (1.0 - mask)
        return np.clip(noisy, 0, 255).astype(np.uint8)


class GhostingTransform(A.ImageOnlyTransform):
    """Ghosting effect."""

    def __init__(self, config: EffectConfig, always_apply=False, p=0.5):
        super().__init__(p=p if config.enabled else 0.0)
        self.cfg = config

    def apply(self, img: np.ndarray, **params):
        shift_min = int(self.cfg.params.get("shift_px_min", 2))
        shift_max = int(self.cfg.params.get("shift_px_max", 10))
        alpha_min = float(self.cfg.params.get("alpha_min", 0.2))
        alpha_max = float(self.cfg.params.get("alpha_max", 0.5))
        shift = np.random.randint(shift_min, shift_max + 1)
        alpha = np.random.uniform(alpha_min, alpha_max)
        ghost = np.roll(img, shift, axis=1)
        blended = cv2.addWeighted(img, 1.0, ghost, alpha, 0.0)  # type: ignore[name-defined]
        return blended


class DropoutPatchesTransform(A.ImageOnlyTransform):
    """Random rectangular dropouts."""

    def __init__(self, config: EffectConfig, always_apply=False, p=0.5):
        super().__init__(p=p if config.enabled else 0.0)
        self.cfg = config

    def apply(self, img: np.ndarray, **params):
        h, w = img.shape[:2]
        num_min = int(self.cfg.params.get("num_min", 1))
        num_max = int(self.cfg.params.get("num_max", 4))
        patch_min = int(self.cfg.params.get("patch_min", 10))
        patch_max = int(self.cfg.params.get("patch_max", 40))
        noise_color = self.cfg.params.get("color", "black")
        num = np.random.randint(num_min, num_max + 1)
        augmented = img.copy()
        for _ in range(num):
            ph = np.random.randint(patch_min, patch_max + 1)
            pw = np.random.randint(patch_min, patch_max + 1)
            top = np.random.randint(0, max(1, h - ph))
            left = np.random.randint(0, max(1, w - pw))
            if noise_color == "noise":
                patch = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)
            else:
                patch = np.zeros((ph, pw, 3), dtype=np.uint8)
            augmented[top : top + ph, left : left + pw] = patch
        return augmented


def build_augmenter(config: AugmentationConfig) -> Callable[[np.ndarray], Tuple[np.ndarray, List[AppliedAugmentation]]]:
    """Build albumentations pipeline + hook to collect metadata."""

    if not config.enabled:
        return lambda img: (img, [])

    sigma_min = config.noise.gaussian.params.get("sigma_min", 2.0)
    sigma_max = config.noise.gaussian.params.get("sigma_max", 8.0)
    transforms: List[A.BasicTransform] = [
        A.OneOf(
            [
                A.GaussNoise(p=1.0),
                A.ISONoise(intensity=(0.3, 0.5), color_shift=(0.01, 0.05), p=1.0),
            ],
            p=config.noise.gaussian.prob,
        ),
        A.SaltAndPepper(
            amount=(config.noise.salt_pepper.params.get("amount_min", 0.001), config.noise.salt_pepper.params.get("amount_max", 0.01)),
            p=config.noise.salt_pepper.prob,
        ),
        ScanlinesTransform(config.analog.scanlines, p=config.analog.scanlines.prob),
        GhostingTransform(config.analog.ghosting, p=config.analog.ghosting.prob),
        DropoutPatchesTransform(config.radio.dropouts, p=config.radio.dropouts.prob),
    ]

    pipeline = A.Compose(transforms, p=config.overall_prob)

    def apply(image: np.ndarray) -> Tuple[np.ndarray, List[AppliedAugmentation]]:
        result = pipeline(image=image)
        applied = [
            AppliedAugmentation(name=transform.__class__.__name__, params={})
            for transform in pipeline.transforms
            if np.random.rand() < getattr(transform, "p", 0.0)
        ]
        return result["image"], applied

    return apply
