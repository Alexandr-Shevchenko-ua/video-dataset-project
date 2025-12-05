# STATUS: not executed, pseudocode – requires adaptation
"""Helpers for downloading/streaming HF dataset zip files."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from datasets import Dataset, IterableDataset, load_dataset

from config import DatasetSelectionConfig, PipelineConfig
from logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ZipItem:
    """Information about a single zip entry in the HF dataset."""

    identifier: str
    split: str
    size_bytes: int
    disk_path: Path
    is_temporary: bool = False


def _resolve_indices(total: int, selection: DatasetSelectionConfig, *, seed: int) -> List[int]:
    """Pick dataset indices according to selection rules."""

    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    if selection.zip_indices:
        return [idx for idx in selection.zip_indices if 0 <= idx < total]
    if selection.max_zips is not None:
        return indices[: selection.max_zips]
    return indices


def download_selected_zips(config: PipelineConfig) -> List[ZipItem]:
    """Download the subset of zip files requested in config."""

    selection = config.dataset_selection
    split = selection.split_filter or "train"
    LOGGER.info("Loading dataset split=%s", split)
    dataset = load_dataset(
        "shevaheyter/uav_raw_videos_zips",
        split=split,
        streaming=False,
        cache_dir=str(config.raw_cache_dir),
    )
    assert isinstance(dataset, Dataset), "Expected standard dataset for caching"

    indices = _resolve_indices(len(dataset), selection, seed=config.sampling.random_seed)
    LOGGER.info("Selected %d zip entries out of %d", len(indices), len(dataset))

    items: List[ZipItem] = []
    for idx in indices:
        row = dataset[idx]
        local_path = Path(row["file_path"])
        items.append(
            ZipItem(
                identifier=row.get("id", str(idx)),
                split=split,
                size_bytes=int(row.get("size_bytes", local_path.stat().st_size)),
                disk_path=local_path,
                is_temporary=False,
            )
        )
    return items


def stream_zips(config: PipelineConfig) -> Iterator[ZipItem]:
    """Stream zip items without downloading entire dataset."""

    selection = config.dataset_selection
    split = selection.split_filter or "train"
    config.raw_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "shevaheyter/uav_raw_videos_zips",
        split=split,
        streaming=True,
        cache_dir=str(config.raw_cache_dir),
    )
    assert isinstance(dataset, IterableDataset)
    limit = selection.max_zips or None

    for idx, row in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        temp_path = config.raw_cache_dir / f"stream_zip_{idx}.zip"
        temp_path.write_bytes(row["zip_bytes"])  # type: ignore[index]
        yield ZipItem(
            identifier=row.get("id", str(idx)),
            split=split,
            size_bytes=int(row.get("size_bytes", temp_path.stat().st_size)),
            disk_path=temp_path,
            is_temporary=True,
        )


def cleanup_zip_item(zip_item: ZipItem) -> None:
    """Delete temporary zip artifacts produced by stream_zips."""

    if zip_item.is_temporary and zip_item.disk_path.exists():
        zip_item.disk_path.unlink()
