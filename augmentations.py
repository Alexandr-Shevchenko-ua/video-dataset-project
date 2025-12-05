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


def _prob(effect: EffectConfig) -> float:
    return effect.prob if effect.enabled else 0.0


class ScanlinesTransform(A.ImageOnlyTransform):
    """Custom transform for scanline artifacts."""

    def __init__(self, config: EffectConfig):
        super().__init__(always_apply=False, p=_prob(config))
        self.cfg = config

    def get_params(self) -> Dict[str, float | int | str]:  # pragma: no cover - randomness
        spacing = int(np.random.randint(int(self.cfg.params.get("spacing_min", 3)), int(self.cfg.params.get("spacing_max", 15)) + 1))
        intensity = float(np.random.uniform(float(self.cfg.params.get("intensity_min", 0.05)), float(self.cfg.params.get("intensity_max", 0.2))))
        return {
            "orientation": self.cfg.params.get("orientation", "horizontal"),
            "spacing": spacing,
            "intensity": intensity,
        }

    def apply(self, img: np.ndarray, **params):
        spacing = max(1, int(params.get("spacing", 3)))
        intensity = float(params.get("intensity", 0.1))
        orientation = params.get("orientation", "horizontal")
        mask = np.zeros_like(img, dtype=np.float32)
        if orientation == "horizontal":
            mask[::spacing, :] = intensity
        else:
            mask[:, ::spacing] = intensity
        noisy = img.astype(np.float32) * (1.0 - mask)
        return np.clip(noisy, 0, 255).astype(np.uint8)


class GhostingTransform(A.ImageOnlyTransform):
    """Ghosting effect."""

    def __init__(self, config: EffectConfig):
        super().__init__(always_apply=False, p=_prob(config))
        self.cfg = config

    def get_params(self) -> Dict[str, float | int]:  # pragma: no cover - randomness
        shift = int(
            np.random.randint(
                int(self.cfg.params.get("shift_px_min", 2)),
                int(self.cfg.params.get("shift_px_max", 10)) + 1,
            )
        )
        alpha = float(
            np.random.uniform(
                float(self.cfg.params.get("alpha_min", 0.2)),
                float(self.cfg.params.get("alpha_max", 0.5)),
            )
        )
        return {"shift": shift, "alpha": alpha}

    def apply(self, img: np.ndarray, **params):
        shift = int(params.get("shift", 2))
        alpha = float(params.get("alpha", 0.3))
        ghost = np.roll(img, shift, axis=1)
        return cv2.addWeighted(img, 1.0, ghost, alpha, 0.0)


class DropoutPatchesTransform(A.ImageOnlyTransform):
    """Random rectangular dropouts."""

    def __init__(self, config: EffectConfig):
        super().__init__(always_apply=False, p=_prob(config))
        self.cfg = config

    def get_params(self) -> Dict[str, float | int | str]:  # pragma: no cover - randomness
        num = int(
            np.random.randint(
                int(self.cfg.params.get("num_min", 1)),
                int(self.cfg.params.get("num_max", 4)) + 1,
            )
        )
        patch_min = int(self.cfg.params.get("patch_min", 10))
        patch_max = int(self.cfg.params.get("patch_max", 40))
        return {
            "num": num,
            "patch_min": patch_min,
            "patch_max": patch_max,
            "color": self.cfg.params.get("color", "black"),
        }

    def apply(self, img: np.ndarray, **params):
        h, w = img.shape[:2]
        num = max(0, int(params.get("num", 1)))
        patch_min = max(1, int(params.get("patch_min", 10)))
        patch_max = max(patch_min, int(params.get("patch_max", patch_min)))
        color_mode = params.get("color", "black")
        augmented = img.copy()
        for _ in range(num):
            ph = np.random.randint(patch_min, min(patch_max, h) + 1)
            pw = np.random.randint(patch_min, min(patch_max, w) + 1)
            top = np.random.randint(0, max(1, h - ph + 1))
            left = np.random.randint(0, max(1, w - pw + 1))
            if color_mode == "noise":
                patch = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)
            else:
                patch = np.zeros((ph, pw, 3), dtype=np.uint8)
            augmented[top : top + ph, left : left + pw] = patch
        return augmented


def _append_if_enabled(container: List[A.BasicTransform], effect: EffectConfig, factory: Callable[[], A.BasicTransform]) -> None:
    if effect.enabled and effect.prob > 0.0:
        transform = factory()
        transform.p = effect.prob
        container.append(transform)


def build_augmenter(config: AugmentationConfig) -> Callable[[np.ndarray], Tuple[np.ndarray, List[AppliedAugmentation]]]:
    """Build albumentations pipeline + hook to collect metadata."""

    if not config.enabled or config.overall_prob <= 0.0:
        return lambda img: (img, [])

    transforms: List[A.BasicTransform] = []
    if config.noise.gaussian.enabled and config.noise.gaussian.prob > 0:
        transforms.append(
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.ISONoise(intensity=(0.3, 0.5), color_shift=(0.01, 0.05), p=1.0),
                ],
                p=config.noise.gaussian.prob,
            )
        )
    if config.noise.salt_pepper.enabled and config.noise.salt_pepper.prob > 0:
        transforms.append(
            A.SaltAndPepper(
                amount=(
                    config.noise.salt_pepper.params.get("amount_min", 0.001),
                    config.noise.salt_pepper.params.get("amount_max", 0.01),
                ),
                p=config.noise.salt_pepper.prob,
            )
        )

    _append_if_enabled(transforms, config.analog.scanlines, lambda: ScanlinesTransform(config.analog.scanlines))
    _append_if_enabled(transforms, config.analog.ghosting, lambda: GhostingTransform(config.analog.ghosting))
    _append_if_enabled(transforms, config.radio.dropouts, lambda: DropoutPatchesTransform(config.radio.dropouts))

    if not transforms:
        return lambda img: (img, [])

    pipeline = A.ReplayCompose(transforms, p=config.overall_prob)

    def apply(image: np.ndarray) -> Tuple[np.ndarray, List[AppliedAugmentation]]:
        result = pipeline(image=image)
        replay = result.get("replay", {})
        applied: List[AppliedAugmentation] = []
        for transform in replay.get("transforms", []):
            if not transform.get("applied"):
                continue
            name = transform.get("__class_fullname__", transform.get("transform"))
            if name:
                name = name.split(".")[-1]
            else:  # defensive fallback
                name = "UnknownTransform"
            params = transform.get("params", {})
            applied.append(AppliedAugmentation(name=name, params=params))
        return result["image"], applied

    return apply
