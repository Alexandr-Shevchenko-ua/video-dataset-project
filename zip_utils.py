# STATUS: not executed, pseudocode – requires adaptation
"""Safe extraction utilities for potentially large zip archives."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Set

from logging_utils import get_logger

LOGGER = get_logger(__name__)


def _sanitize_path(path: Path, root: Path) -> Path:
    """Ensure members cannot escape the intended root directory."""

    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Illegal path traversal attempt: {path}")
    return resolved


def estimate_uncompressed_size(zip_path: Path) -> int:
    """Estimate total size of members inside the archive."""

    with zipfile.ZipFile(zip_path) as zf:
        return sum(info.file_size for info in zf.infolist())


def safe_extract(
    zip_path: Path,
    target_dir: Path,
    *,
    allowed_extensions: Iterable[str] = (".mp4",),
    max_bytes: int | None = None,
) -> list[Path]:
    """Extract selected files from zip with strict checks."""

    allowed: Set[str] = {ext.lower() for ext in allowed_extensions}
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        total_size = sum(info.file_size for info in zf.infolist())
        if max_bytes is not None and total_size > max_bytes:
            raise ValueError(
                f"Archive {zip_path} exceeds allowed size {total_size} > {max_bytes}"
            )

        extracted: list[Path] = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename)
            if name.suffix.lower() not in allowed:
                LOGGER.debug("Skipping %s (ext not allowed)", name)
                continue
            dest_path = _sanitize_path(name, target_dir)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(dest_path)
    return extracted


def cleanup_temp_dir(path: Path) -> None:
    """Remove temporary extraction directory."""

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
