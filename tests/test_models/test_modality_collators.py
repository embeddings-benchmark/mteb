from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.models.modality_collators import (
    AudioCollator,
    FramesCollator,
    VideoCollator,
    frame_indices,
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
        # fewer source frames than requested: repeat each frame in place
        (5, 8, [0, 0, 1, 1, 2, 3, 3, 4]),
        (3, 8, [0, 0, 0, 1, 1, 1, 2, 2]),
        (1, 4, [0, 0, 0, 0]),
        (16, 16, list(range(16))),
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


@pytest.mark.parametrize(
    ("n_source", "seconds", "fps", "max_frames", "expected"),
    [
        # under 2 s at 1 fps: 4 frames, not 1 (a Qwen3-VL processor needs >= 2)
        (11, 0.37, 1.0, 64, [0, 3, 7, 10]),
        (30, 1.2, 2.0, 64, [0, 10, 19, 29]),
        # odd counts are rounded down to even
        (146, 6.09, 0.5, 64, [0, 48, 97, 145]),
        (75, 2.5, 2.0, 64, [0, 25, 49, 74]),
        # fewer source frames than the minimum: rounded down to an even count,
        # like qwen-vl-utils
        (3, 0.1, 2.0, 64, [0, 2]),
        (1, 0.04, 2.0, 64, [0, 0]),
    ],
)
def test_resample_video_fps_min_and_even_frames(
    n_source: int,
    seconds: float,
    fps: float,
    max_frames: int,
    expected: list[int],
) -> None:
    video = _StubDecoder(n_source, seconds=seconds)
    frames = FramesCollator.resample_video(video, fps=fps, max_frames=max_frames)
    assert frames == expected


def test_resample_video_fps_odd_count_rounded_down() -> None:
    video = _StubDecoder(1350, seconds=45.0)
    assert len(FramesCollator.resample_video(video, fps=1.0, max_frames=64)) == 44
    assert len(FramesCollator.resample_video(video, fps=1.0, max_frames=33)) == 32


def test_resample_video_fps_without_min_frames() -> None:
    """Re-encoding callers (API models) opt out to keep the clip duration."""
    video = _StubDecoder(11, seconds=0.37)
    frames = FramesCollator.resample_video(video, fps=1.0, min_frames=1, frame_factor=1)
    assert frames == [0]


def test_resample_video_without_sampling_keeps_all_frames() -> None:
    video = _StubDecoder(12, seconds=1.0)
    assert FramesCollator.resample_video(video) == list(range(12))


@pytest.mark.parametrize(
    "collator",
    [
        VideoCollator(target_sampling_rate=16_000, num_frames=2),
        FramesCollator(num_frames=2),
    ],
)
def test_video_collators_keep_column_order(
    collator: VideoCollator | FramesCollator,
) -> None:
    """Chat templates that follow the dict order must see video before audio."""
    row = {
        "id": "a",
        "video": _StubDecoder(4, seconds=1.0),
        "text": "caption",
    }
    if isinstance(collator, VideoCollator):
        pytest.importorskip("torchaudio")
        row["audio"] = {"array": [0.0] * 10, "sampling_rate": 16_000}
    columns = list(row)
    batch = collator([row])
    assert list(batch) == columns
    assert batch["video"] == [[0, 3]]


def test_video_collator_puts_video_before_audio() -> None:
    """Even when the dataset lists audio first, video must come before audio."""
    pytest.importorskip("torchaudio")
    collator = VideoCollator(target_sampling_rate=16_000, num_frames=2)
    row = {
        "id": "a",
        "audio": {"array": [0.0] * 10, "sampling_rate": 16_000},
        "video": _StubDecoder(4, seconds=1.0),
        "text": "caption",
    }
    batch = collator([row])
    assert list(batch) == ["id", "video", "text", "audio"]


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


@pytest.mark.parametrize(
    ("n_source", "expected"),
    [
        # 8 frames from a 25-frame and a 5-frame clip
        (25, [0, 3, 7, 10, 14, 17, 21, 24]),
        (5, [0, 0, 1, 1, 2, 3, 3, 4]),
    ],
)
def test_frame_indices(n_source: int, expected: list[int]) -> None:
    assert frame_indices(n_source, 8) == expected
    video = _StubDecoder(n_source, seconds=1.0)
    assert FramesCollator.resample_video(video, num_frames=8) == expected


def test_resample_video_fps_short_source_frame_counts() -> None:
    """Short sources: frame_factor=1 keeps every frame, frame_factor=2 rounds down
    to an even count like qwen-vl-utils (but never below 2 frames)."""
    for n_source in range(1, 8):
        video = _StubDecoder(n_source, seconds=0.2)
        frames = FramesCollator.resample_video(
            video, fps=2.0, max_frames=64, frame_factor=1
        )
        assert len(frames) >= min(n_source, 4)
        assert set(frames) == set(range(n_source)) or n_source > 4
        frames = FramesCollator.resample_video(
            video, fps=2.0, max_frames=64, frame_factor=2
        )
        assert len(frames) == max(2, min(n_source, 4) // 2 * 2)
    video = _StubDecoder(3, seconds=0.1)
    assert FramesCollator.resample_video(video, fps=2.0, frame_factor=1) == [0, 1, 2]
    assert FramesCollator.resample_video(video, fps=2.0, frame_factor=2) == [0, 2]
