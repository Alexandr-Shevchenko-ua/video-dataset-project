# STATUS: not executed, pseudocode – requires adaptation
# UAV Video to YOLO Dataset Pipeline

This repository contains a Hugging Face Space-ready Gradio app that:

- downloads zipped UAV video shards from `shevaheyter/uav_raw_videos_zips`;
- safely unpacks `.mp4` clips in batches, sampling informative frames only;
- filters frames by simple but effective quality metrics;
- optionally applies analog-camera & radio-interference augmentations;
- writes YOLO-style directory layouts + detailed metadata (Parquet/CSV/JSONL).

## Project layout

- `app.py` – Gradio UI entrypoint, exposes controls for sampling/filtering/augs/output.
- `config.py` – Pydantic configuration models + load/save helpers.
- `dataset_loader.py` – wrappers around `datasets.load_dataset` with subsetting logic.
- `zip_utils.py` – secure zip extraction with size caps and path sanitisation.
- `video_reader.py` – OpenCV-based frame iterator with fixed/adaptive strategies.
- `frame_quality.py` – brightness/blur/exposure/motion/compression heuristics.
- `augmentations.py` – Albumentations pipeline incl. custom scanlines/ghosting/dropouts.
- `writer.py` – image writer + metadata persistence.
- `pipeline.py` – orchestration tying everything together.
- `logging_utils.py` – consistent logging setup and Gradio bridge.
- `tests/` – pytest-based unit/integration scaffolding.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Hugging Face Space notes

- The app assumes Linux + Python 3.10, with system ffmpeg available (as in HF CPU/GPU images).
- Disk defaults target `/data/...` folders; adjust via UI or `config.yaml`.
- The pipeline is spec-first & safety-first: extraction happens in temp dirs, output cleanup requires explicit confirmation, and train/val dirs never overlap raw cache.

## Testing

```bash
pytest -q
```

Unit tests rely on synthetic images/frames and do not download datasets.
