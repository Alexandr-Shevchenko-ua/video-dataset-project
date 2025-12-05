# STATUS: not executed, pseudocode – requires adaptation
"""Configuration models for the UAV video to YOLO dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, PositiveInt, validator


class _BaseModel(BaseModel):
    """Base model that knows how to JSON-encode Path objects."""

    class Config:
        json_encoders = {Path: str}


class EffectConfig(_BaseModel):
    """Generic configuration for a single augmentation effect."""

    enabled: bool = True
    prob: float = Field(0.3, ge=0.0, le=1.0)
    params: Dict[str, Any] = Field(default_factory=dict)


class SamplingConfig(_BaseModel):
    """Frame sampling behaviour."""

    mode: Literal["fixed_fps", "adaptive"] = "fixed_fps"
    target_fps: float = Field(2.0, ge=0.1, le=30.0)
    min_frame_interval_ms: int = Field(150, ge=0)
    adaptive_diff_threshold: float = Field(12.0, ge=0.0, le=255.0)
    scene_change_window: PositiveInt = 5
    max_frames_per_video: PositiveInt = 200
    random_seed: int = 42


class QualityFilterConfig(_BaseModel):
    """Thresholds for filtering low-quality frames."""

    blur_laplacian_thresh: float = Field(80.0, ge=0.0)
    brightness_min: float = Field(0.15, ge=0.0, le=1.0)
    brightness_max: float = Field(0.9, ge=0.0, le=1.0)
    exposure_saturation_thresh: float = Field(0.05, ge=0.0, le=1.0)
    motion_blur_thresh: float = Field(60.0, ge=0.0)
    compression_block_thresh: float = Field(20.0, ge=0.0)
    allow_manual_override: bool = False

    @validator("brightness_max")
    def _validate_brightness(cls, value: float, values: Dict[str, float]) -> float:
        """Ensure max brightness is greater than min brightness."""

        if "brightness_min" in values and value <= values["brightness_min"]:
            raise ValueError("brightness_max must be greater than brightness_min")
        return value


class NoiseAugmentConfig(_BaseModel):
    """Handy wrapper for frequently used augmentation effects."""

    gaussian: EffectConfig = EffectConfig(
        prob=0.4,
        params={"sigma_min": 2.0, "sigma_max": 8.0},
    )
    salt_pepper: EffectConfig = EffectConfig(
        prob=0.2,
        params={"amount_min": 0.001, "amount_max": 0.01},
    )
    speckle: EffectConfig = EffectConfig(
        prob=0.2,
        params={"gain_min": 0.01, "gain_max": 0.05},
    )


class AnalogAugmentConfig(_BaseModel):
    """Config for analog camera simulation."""

    scanlines: EffectConfig = EffectConfig(
        prob=0.3,
        params={
            "orientation": "horizontal",
            "intensity_min": 0.05,
            "intensity_max": 0.2,
            "spacing_min": 3,
            "spacing_max": 15,
        },
    )
    ghosting: EffectConfig = EffectConfig(
        prob=0.25,
        params={"shift_px_min": 2, "shift_px_max": 10, "alpha_min": 0.2, "alpha_max": 0.5},
    )
    soft_focus: EffectConfig = EffectConfig(
        prob=0.3,
        params={"kernel_size_min": 3, "kernel_size_max": 7},
    )
    color_shift: EffectConfig = EffectConfig(
        prob=0.3,
        params={"hue_delta_min": -5, "hue_delta_max": 5, "sat_scale_min": 0.9, "sat_scale_max": 1.1},
    )


class RadioAugmentConfig(_BaseModel):
    """Config for radio interference simulation."""

    jitter: EffectConfig = EffectConfig(
        prob=0.2,
        params={"max_translation_px": 5},
    )
    dropouts: EffectConfig = EffectConfig(
        prob=0.15,
        params={"patch_min": 10, "patch_max": 40, "num_min": 1, "num_max": 4},
    )
    wave_distortion: EffectConfig = EffectConfig(
        prob=0.2,
        params={"amplitude_min": 2, "amplitude_max": 6, "frequency_min": 0.01, "frequency_max": 0.05},
    )


class AugmentationConfig(_BaseModel):
    """Full augmentation configuration."""

    enabled: bool = True
    overall_prob: float = Field(0.7, ge=0.0, le=1.0)
    noise: NoiseAugmentConfig = NoiseAugmentConfig()
    analog: AnalogAugmentConfig = AnalogAugmentConfig()
    radio: RadioAugmentConfig = RadioAugmentConfig()


class OutputConfig(_BaseModel):
    """Where and how frames + metadata are stored."""

    dataset_root: Path = Path("/data/uav_yolo_output")
    train_split: float = Field(0.8, gt=0.0, lt=1.0)
    split_strategy: Literal["per_video", "per_frame"] = "per_video"
    train_val_seed: int = 42
    metadata_format: Literal["parquet", "csv", "jsonl"] = "parquet"
    metadata_batch_size: PositiveInt = 500
    compress_to_zip: bool = False
    overwrite_policy: Literal["fail", "ask", "force"] = "fail"
    cleanup_temp: bool = True
    frame_extension: Literal["jpg", "png"] = "jpg"
    max_parallel_videos: PositiveInt = 1


class DatasetSelectionConfig(_BaseModel):
    """Subsetting the upstream dataset."""

    max_zips: Optional[int] = None
    zip_indices: Optional[List[int]] = None
    max_videos: Optional[int] = None
    split_filter: Optional[str] = None


class PipelineConfig(_BaseModel):
    """Top-level configuration tying everything together."""

    sampling: SamplingConfig = SamplingConfig()
    quality: QualityFilterConfig = QualityFilterConfig()
    augmentations: AugmentationConfig = AugmentationConfig()
    output: OutputConfig = OutputConfig()
    dataset_selection: DatasetSelectionConfig = DatasetSelectionConfig()
    raw_cache_dir: Path = Path("/data/raw_zip_cache")
    temp_extract_dir: Path = Path("/data/tmp_extract")
    metadata_path: Optional[Path] = None
    log_level: Literal["INFO", "DEBUG"] = "INFO"
    dry_run: bool = False
    disk_budget_gb: float = Field(40.0, gt=0.0)
    ram_budget_gb: float = Field(12.0, gt=0.0)
    max_frames_total: Optional[int] = None
    allow_output_cleanup: bool = False
    hf_token_env: Optional[str] = None

    @validator("metadata_path", always=True)
    def _default_metadata(cls, value: Optional[Path], values: Dict[str, object]) -> Path:
        """Default metadata path inside dataset_root."""

        if value is not None:
            return value
        output: OutputConfig = values["output"]  # type: ignore[assignment]
        return output.dataset_root / "metadata.parquet"

    @validator("temp_extract_dir")
    def _ensure_not_under_output(cls, temp_dir: Path, values: Dict[str, object]) -> Path:
        """Ensure temp dir does not overlap output dir."""

        output: OutputConfig = values.get("output")  # type: ignore[assignment]
        if output and temp_dir.resolve().is_relative_to(output.dataset_root.resolve()):
            raise ValueError("temp_extract_dir must not be inside dataset_root")
        return temp_dir

    def to_jsonable(self) -> Dict[str, Any]:
        """Return a nested dict where Path/Model objects are serialized."""

        def convert(value: Any) -> Any:
            from pydantic import BaseModel as _Base

            if isinstance(value, Path):
                return str(value)
            if isinstance(value, _Base):
                return convert(value.dict())
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return convert(self.dict())


def default_config() -> PipelineConfig:
    """Return a ready-to-use default configuration."""

    return PipelineConfig()


def config_to_dict(config: PipelineConfig) -> Dict[str, Any]:
    """Public helper for UI/tests needing serializable configs."""

    return config.to_jsonable()


def load_config(path: Path) -> PipelineConfig:
    """Load config from a JSON or YAML file."""

    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return PipelineConfig(**data)


def save_config(config: PipelineConfig, path: Path) -> None:
    """Persist config to disk."""

    data = config.to_jsonable()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
