# STATUS: not executed, pseudocode – requires adaptation
"""Gradio entrypoint for the UAV video processing pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import gradio as gr

from config import PipelineConfig, config_to_dict, default_config
from pipeline import ProgressUpdate, run_pipeline


def _optional_int(value: Any) -> Any:
    if value in ("", None):
        return None
    return int(value)


def _build_config_from_inputs(inputs: Dict[str, Any]) -> PipelineConfig:
    """Map Gradio form values into PipelineConfig."""

    cfg = default_config().dict()
    cfg["sampling"].update(
        {
            "mode": inputs["sampling_mode"],
            "target_fps": inputs["target_fps"],
            "min_frame_interval_ms": inputs["min_interval"],
            "adaptive_diff_threshold": inputs["adaptive_diff"],
            "max_frames_per_video": int(inputs["max_frames_video"]),
            "random_seed": int(inputs["sampling_seed"]),
        }
    )
    cfg["quality"].update(
        {
            "blur_laplacian_thresh": inputs["blur_thresh"],
            "brightness_min": inputs["brightness_min"],
            "brightness_max": inputs["brightness_max"],
            "exposure_saturation_thresh": inputs["exposure_thresh"],
            "motion_blur_thresh": inputs["motion_thresh"],
            "compression_block_thresh": inputs["compression_thresh"],
        }
    )
    cfg["augmentations"].update(
        {
            "enabled": inputs["augment_enabled"],
            "overall_prob": inputs["augment_prob"],
        }
    )
    cfg["dataset_selection"].update(
        {
            "max_zips": _optional_int(inputs["max_zips"]),
            "max_videos": _optional_int(inputs["max_videos"]),
            "split_filter": inputs["split_filter"] or None,
        }
    )
    cfg["output"].update(
        {
            "dataset_root": inputs["dataset_root"],
            "train_split": inputs["train_split"],
            "split_strategy": inputs["split_strategy"],
            "metadata_format": inputs["metadata_format"],
            "metadata_batch_size": int(inputs["metadata_batch_size"]),
            "compress_to_zip": inputs["compress_output"],
        }
    )
    cfg.update(
        {
            "dry_run": inputs["dry_run"],
            "disk_budget_gb": inputs["disk_budget"],
            "max_frames_total": inputs["max_frames_total"] or None,
            "allow_output_cleanup": inputs["allow_cleanup"],
        }
    )
    return PipelineConfig(**cfg)


def _config_to_inputs(config: PipelineConfig) -> List[Any]:
    """Convert config to UI component defaults."""

    return [
        config.sampling.mode,
        config.sampling.target_fps,
        config.sampling.min_frame_interval_ms,
        config.sampling.adaptive_diff_threshold,
        config.sampling.max_frames_per_video,
        config.sampling.random_seed,
        config.quality.blur_laplacian_thresh,
        config.quality.brightness_min,
        config.quality.brightness_max,
        config.quality.exposure_saturation_thresh,
        config.quality.motion_blur_thresh,
        config.quality.compression_block_thresh,
        config.augmentations.enabled,
        config.augmentations.overall_prob,
        config.dataset_selection.max_zips,
        config.dataset_selection.max_videos,
        config.dataset_selection.split_filter or "",
        str(config.output.dataset_root),
        config.output.train_split,
        config.output.split_strategy,
        config.output.metadata_format,
        config.output.metadata_batch_size,
        config.output.compress_to_zip,
        config.dry_run,
        config.disk_budget_gb,
        config.max_frames_total,
        config.allow_output_cleanup,
    ]


def run_pipeline_ui(inputs: Dict[str, Any], progress=gr.Progress(track_tqdm=False)) -> str:
    """Gradio callback for processing."""

    config = _build_config_from_inputs(inputs)
    log_lines: list[str] = []

    def on_progress(update: ProgressUpdate) -> None:
        message = f"[zip {update.zip_idx}] video {update.video_idx} frames={update.frames}"
        progress(message=message)
        log_lines.append(message)

    run_pipeline(config, progress_cb=on_progress)
    return "\n".join(log_lines) or "Done"


def build_interface() -> gr.Blocks:
    """Construct the Gradio UI."""

    default_cfg = default_config()
    default_cfg_text = json.dumps(config_to_dict(default_cfg), indent=2)
    with gr.Blocks(title="UAV Video to YOLO Frames") as demo:
        gr.Markdown("## UAV Video Dataset Pipeline")
        with gr.Tabs():
            with gr.Tab("Controls"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Sampling")
                        sampling_mode = gr.Dropdown(
                            choices=["fixed_fps", "adaptive"],
                            value=default_cfg.sampling.mode,
                            label="Mode",
                        )
                        sampling_seed = gr.Number(default_cfg.sampling.random_seed, label="Sampling seed", precision=0)
                        target_fps = gr.Slider(0.5, 15.0, value=default_cfg.sampling.target_fps, step=0.5, label="Target FPS")
                        min_interval = gr.Slider(
                            0, 1000, step=50, value=default_cfg.sampling.min_frame_interval_ms, label="Min interval (ms)"
                        )
                        adaptive_diff = gr.Slider(
                            0, 255, step=1, value=default_cfg.sampling.adaptive_diff_threshold, label="Adaptive diff threshold"
                        )
                        max_frames_video = gr.Number(
                            value=default_cfg.sampling.max_frames_per_video,
                            label="Max frames per video",
                            precision=0,
                        )
                    with gr.Column():
                        gr.Markdown("### Quality filters")
                        blur_thresh = gr.Slider(10, 300, value=default_cfg.quality.blur_laplacian_thresh, step=5, label="Blur threshold")
                        brightness_min = gr.Slider(0.0, 1.0, value=default_cfg.quality.brightness_min, step=0.01, label="Brightness min")
                        brightness_max = gr.Slider(0.0, 1.0, value=default_cfg.quality.brightness_max, step=0.01, label="Brightness max")
                        exposure_thresh = gr.Slider(0.0, 0.5, value=default_cfg.quality.exposure_saturation_thresh, step=0.01, label="Exposure threshold")
                        motion_thresh = gr.Slider(0.0, 200.0, value=default_cfg.quality.motion_blur_thresh, step=1.0, label="Motion blur threshold")
                        compression_thresh = gr.Slider(
                            0.0, 100.0, value=default_cfg.quality.compression_block_thresh, step=1.0, label="Compression threshold"
                        )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Augmentations")
                        augment_enabled = gr.Checkbox(value=default_cfg.augmentations.enabled, label="Enable augmentations")
                        augment_prob = gr.Slider(0.0, 1.0, value=default_cfg.augmentations.overall_prob, step=0.05, label="Overall probability")
                    with gr.Column():
                        gr.Markdown("### Dataset selection")
                        max_zips = gr.Number(value=default_cfg.dataset_selection.max_zips, label="Max zips", precision=0)
                        max_videos = gr.Number(value=default_cfg.dataset_selection.max_videos, label="Max videos", precision=0)
                        split_filter = gr.Textbox(value=default_cfg.dataset_selection.split_filter or "", label="HF split filter")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Output & runtime")
                        dataset_root = gr.Textbox(value=str(default_cfg.output.dataset_root), label="Output dataset root")
                        train_split = gr.Slider(0.5, 0.95, value=default_cfg.output.train_split, step=0.01, label="Train split fraction")
                        split_strategy = gr.Dropdown(["per_video", "per_frame"], value=default_cfg.output.split_strategy, label="Split strategy")
                        metadata_format = gr.Radio(["parquet", "csv", "jsonl"], value=default_cfg.output.metadata_format, label="Metadata format")
                        metadata_batch_size = gr.Number(
                            value=default_cfg.output.metadata_batch_size, label="Metadata batch size", precision=0
                        )
                        compress_output = gr.Checkbox(value=default_cfg.output.compress_to_zip, label="Create zip of output")
                    with gr.Column():
                        dry_run = gr.Checkbox(value=default_cfg.dry_run, label="Dry run (no writes)")
                        disk_budget = gr.Slider(1, 50, value=default_cfg.disk_budget_gb, step=1, label="Disk budget (GB)")
                        max_frames_total = gr.Number(value=default_cfg.max_frames_total, label="Max frames total", precision=0)
                        allow_cleanup = gr.Checkbox(value=default_cfg.allow_output_cleanup, label="Allow output cleanup")
                start_btn = gr.Button("Start processing", variant="primary")
                logs = gr.Textbox(label="Logs", lines=15)

            with gr.Tab("Config JSON"):
                config_text = gr.Textbox(
                    value=default_cfg_text,
                    lines=25,
                    label="PipelineConfig JSON",
                )
                to_json_btn = gr.Button("Update preview from controls")
                from_json_btn = gr.Button("Apply JSON to controls")

        input_components = {
            "sampling_mode": sampling_mode,
            "target_fps": target_fps,
            "min_interval": min_interval,
            "adaptive_diff": adaptive_diff,
            "max_frames_video": max_frames_video,
            "sampling_seed": sampling_seed,
            "blur_thresh": blur_thresh,
            "brightness_min": brightness_min,
            "brightness_max": brightness_max,
            "exposure_thresh": exposure_thresh,
            "motion_thresh": motion_thresh,
            "compression_thresh": compression_thresh,
            "augment_enabled": augment_enabled,
            "augment_prob": augment_prob,
            "max_zips": max_zips,
            "max_videos": max_videos,
            "split_filter": split_filter,
            "dataset_root": dataset_root,
            "train_split": train_split,
            "split_strategy": split_strategy,
            "metadata_format": metadata_format,
            "metadata_batch_size": metadata_batch_size,
            "compress_output": compress_output,
            "dry_run": dry_run,
            "disk_budget": disk_budget,
            "max_frames_total": max_frames_total,
            "allow_cleanup": allow_cleanup,
        }

        component_names = list(input_components.keys())
        component_values = list(input_components.values())

        def _values_to_dict(values: List[Any]) -> Dict[str, Any]:
            return dict(zip(component_names, values))

        def _start_pipeline(*values, progress=gr.Progress(track_tqdm=False)):
            inputs = _values_to_dict(list(values))
            return run_pipeline_ui(inputs, progress=progress)

        start_btn.click(_start_pipeline, inputs=component_values, outputs=logs, queue=True)

        def controls_to_json(*values):
            inputs = _values_to_dict(list(values))
            config = _build_config_from_inputs(inputs)
            return json.dumps(config_to_dict(config), indent=2)

        to_json_btn.click(controls_to_json, inputs=component_values, outputs=config_text)

        def json_to_controls(text: str):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as err:  # pragma: no cover - UI only
                raise gr.Error(f"Invalid JSON: {err}")
            cfg = PipelineConfig(**data)
            return _config_to_inputs(cfg)

        from_json_btn.click(
            json_to_controls,
            inputs=config_text,
            outputs=component_values,
        )
    return demo


if __name__ == "__main__":
    build_interface().queue(max_size=2).launch()
