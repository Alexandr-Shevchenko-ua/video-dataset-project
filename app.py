# STATUS: not executed, pseudocode – requires adaptation
"""Gradio entrypoint for the UAV video processing pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from config import PipelineConfig, default_config, load_config, save_config
from logging_utils import GradioLogBridge, get_logger, setup_logging
from pipeline import ProgressUpdate, run_pipeline

LOGGER = get_logger(__name__)


def _build_config_from_inputs(inputs: Dict[str, Any]) -> PipelineConfig:
    """Map Gradio form values into PipelineConfig."""

    cfg_dict = default_config().dict()
    cfg_dict["sampling"]["target_fps"] = inputs["target_fps"]
    cfg_dict["sampling"]["mode"] = inputs["sampling_mode"]
    cfg_dict["quality"]["blur_laplacian_thresh"] = inputs["blur_thresh"]
    cfg_dict["output"]["dataset_root"] = inputs["dataset_root"]
    cfg_dict["dataset_selection"]["max_zips"] = inputs["max_zips"]
    return PipelineConfig(**cfg_dict)


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
    with gr.Blocks(title="UAV Video to YOLO Frames") as demo:
        gr.Markdown("## UAV Video Dataset Pipeline")
        with gr.Row():
            with gr.Column():
                sampling_mode = gr.Dropdown(
                    choices=["fixed_fps", "adaptive"],
                    value=default_cfg.sampling.mode,
                    label="Sampling mode",
                )
                target_fps = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    step=0.5,
                    value=default_cfg.sampling.target_fps,
                    label="Target FPS",
                )
                blur_thresh = gr.Slider(
                    minimum=10,
                    maximum=200,
                    step=5,
                    value=default_cfg.quality.blur_laplacian_thresh,
                    label="Blur threshold",
                )
                max_zips = gr.Number(
                    value=default_cfg.dataset_selection.max_zips,
                    label="Max zips to process",
                )
                dataset_root = gr.Textbox(
                    value=str(default_cfg.output.dataset_root),
                    label="Output dataset root",
                )
                start_btn = gr.Button("Start processing", variant="primary")
            with gr.Column():
                logs = gr.Textbox(label="Logs", lines=20)
        inputs = {
            "sampling_mode": sampling_mode,
            "target_fps": target_fps,
            "blur_thresh": blur_thresh,
            "max_zips": max_zips,
            "dataset_root": dataset_root,
        }
        start_btn.click(run_pipeline_ui, inputs=inputs, outputs=logs, queue=True)
    return demo


if __name__ == "__main__":
    build_interface().queue(max_size=2).launch()
