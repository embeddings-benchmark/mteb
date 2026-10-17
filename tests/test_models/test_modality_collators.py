from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.modality_collators import (
    AudioCollator,
    FramesCollator,
    VideoCollator,
    sampled_frame_rate,
    seconds_to_samples,
    single_clip_dataloader,
)


class _StubDecoder:
    """Minimal VideoDecoder stand-in: returns the requested indices as frames."""

    def __init__(self, num_frames: int, seconds: float) -> None:
        self.metadata = SimpleNamespace(
            num_frames=num_frames,
            end_stream_seconds=seconds,
            average_fps=num_frames / seconds,
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
        (5, 8, [0, 0, 1, 1, 2, 3, 3, 4]),
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


def test_single_clip_dataloader() -> None:
    rows = [{"audio": {"array": [float(i)], "sampling_rate": 16_000}} for i in range(5)]
    loader = DataLoader(rows, batch_size=2, collate_fn=_custom_collate_fn)  # type: ignore[arg-type]
    single = single_clip_dataloader(loader, "test")
    assert single.batch_size == 1
    assert len(single) == 5
    batches = list(single)
    assert [len(b["audio"]) for b in batches] == [1] * 5
    assert [b["audio"][0]["array"] for b in batches] == [[float(i)] for i in range(5)]


def test_audio_collator_rejects_non_positive_cap() -> None:
    audio = {"array": [0.0] * 10, "sampling_rate": 16_000}
    with pytest.raises(ValueError, match="must be positive"):
        AudioCollator.resample_audio(audio, target_sampling_rate=16_000, max_samples=0)


def test_resample_video_fps_min_frames() -> None:
    video = _StubDecoder(30, seconds=1.2)
    # default: one frame per second, as before
    assert FramesCollator.resample_video(video, fps=1.0) == [0]
    # opt-in floor for processors that need more frames (Qwen3-VL)
    frames = FramesCollator.resample_video(video, fps=1.0, min_frames=4)
    assert frames == [0, 10, 19, 29]


def test_video_collator_orders_modalities() -> None:
    """Prompts built in row order see text, video, audio, whatever the column order."""
    pytest.importorskip("torchaudio")
    row = {
        "id": "a",
        "audio": {"array": [0.0] * 10, "sampling_rate": 16_000},
        "video": _StubDecoder(4, seconds=1.0),
        "text": "caption",
    }
    batch = VideoCollator(target_sampling_rate=16_000, num_frames=2)([row])
    assert list(batch) == ["id", "text", "video", "audio"]


def test_frames_collator_returns_video_timing() -> None:
    """With video_metadata, each video carries its sampled frames' real timing."""
    collator = FramesCollator(fps=1.0, max_frames=4, video_metadata=True)
    video = collator([{"video": _StubDecoder(300, seconds=10.0)}])["video"][0]
    assert video["array"] == [0, 100, 199, 299]
    assert video["video_metadata"] == {
        "fps": 30.0,
        "total_num_frames": 300,
        "duration": 10.0,
        "frames_indices": [0, 100, 199, 299],
    }


def test_sampled_frame_rate_spans_the_source_clip() -> None:
    """A file of sampled frames written at this rate lasts as long as the source."""
    rate = sampled_frame_rate({"fps": 30.0, "frames_indices": [0, 100, 199, 299]})
    assert rate == pytest.approx(3 / (299 / 30), rel=1e-3)


def test_video_collator_skips_missing_modalities() -> None:
    """Mixed-modality datasets can have None for a row's audio or video."""
    pytest.importorskip("torchaudio")
    rows = [
        {"video": _StubDecoder(4, seconds=1.0), "audio": None},
        {"video": None, "audio": {"array": [0.0] * 10, "sampling_rate": 16_000}},
    ]
    batch = VideoCollator(target_sampling_rate=16_000, num_frames=2)(rows)
    assert batch["video"][0] == [0, 3]
    assert batch["audio"][0] is None
    assert batch["video"][1] is None
