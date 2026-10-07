from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.modality_collators import (
    AudioCollator,
    FramesCollator,
    seconds_to_samples,
    single_clip_dataloader,
)


class _StubDecoder:
    """Minimal VideoDecoder stand-in: returns the requested indices as frames."""

    def __init__(self, num_frames: int, seconds: float) -> None:
        self.metadata = SimpleNamespace(
            num_frames=num_frames, end_stream_seconds=seconds
        )

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        return SimpleNamespace(data=list(indices))


@pytest.mark.parametrize(
    ("n_source", "num_frames", "expected"),
    [
        # evenly spaced over the whole clip, not just its head
        (15, 8, [0, 2, 4, 6, 8, 10, 12, 14]),
        (300, 8, [0, 43, 85, 128, 171, 214, 256, 299]),
        # fewer source frames than requested: repeat to reach the count
        (5, 8, [0, 1, 2, 3, 4, 0, 1, 2]),
    ],
)
def test_resample_video_fixed_frames(
    n_source: int, num_frames: int, expected: list[int]
) -> None:
    video = _StubDecoder(n_source, seconds=10.0)
    assert FramesCollator.resample_video(video, num_frames=num_frames) == expected


def test_resample_video_fps_spans_clip_and_respects_cap() -> None:
    video = _StubDecoder(300, seconds=10.0)
    frames = FramesCollator.resample_video(video, fps=2.0)
    assert len(frames) == 20
    assert frames[0] == 0
    assert frames[-1] == 299
    capped = FramesCollator.resample_video(video, fps=2.0, max_frames=8)
    assert capped == [0, 43, 85, 128, 171, 214, 256, 299]


def test_resample_video_without_sampling_keeps_all_frames() -> None:
    video = _StubDecoder(12, seconds=1.0)
    assert FramesCollator.resample_video(video) == list(range(12))


def test_seconds_to_samples() -> None:
    assert seconds_to_samples(None, 16_000) is None
    assert seconds_to_samples(1.5, 16_000) == 24_000
    with pytest.raises(ValueError, match="must be positive"):
        seconds_to_samples(0, 16_000)


def test_single_clip_dataloader(caplog: pytest.LogCaptureFixture) -> None:
    rows = [{"audio": {"array": [float(i)], "sampling_rate": 16_000}} for i in range(5)]
    loader = DataLoader(rows, batch_size=2, collate_fn=_custom_collate_fn)  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO, logger="mteb.models.modality_collators"):
        single = single_clip_dataloader(loader, "a test reason")
        single_clip_dataloader(loader, "a test reason")
    assert (
        caplog.text.count("one clip at a time (batch_size ignored): a test reason") == 1
    )
    assert single.batch_size == 1
    assert len(single) == 5
    batches = list(single)
    assert [len(b["audio"]) for b in batches] == [1] * 5
    assert [b["audio"][0]["array"] for b in batches] == [[float(i)] for i in range(5)]


def test_audio_collator_rejects_non_positive_cap() -> None:
    audio = {"array": [0.0] * 10, "sampling_rate": 16_000}
    with pytest.raises(ValueError, match="must be positive"):
        AudioCollator.resample_audio(audio, target_sampling_rate=16_000, max_samples=0)


def test_truncation_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    audio = {"audio": {"array": np.zeros(16_000 * 3), "sampling_rate": 16_000}}
    with caplog.at_level(logging.INFO, logger="mteb.models.modality_collators"):
        for _ in range(3):
            out = AudioCollator.resample_audio(
                audio, target_sampling_rate=16_000, max_samples=16_000 * 2
            )
    assert out.shape[-1] == 16_000 * 2
    assert caplog.text.count("Truncating audio longer than 2 s") == 1
