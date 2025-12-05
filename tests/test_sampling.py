# STATUS: not executed, pseudocode – requires adaptation
"""Unit tests for sampling logic."""

from config import SamplingConfig
from video_reader import _should_emit  # type: ignore


def test_should_emit_respects_max_frames():
    """Verify that once max frames reached nothing else is emitted."""

    cfg = SamplingConfig(max_frames_per_video=2)
    assert _should_emit(0, 0, 100, cfg, 0) is True
    assert _should_emit(10, 0, 100, cfg, 2) is False


def test_should_emit_adaptive_diff():
    """Ensure adaptive mode fires when diff over threshold."""

    cfg = SamplingConfig(mode="adaptive", adaptive_diff_threshold=10.0)
    assert _should_emit(0, 999, 15.0, cfg, 0) is True
