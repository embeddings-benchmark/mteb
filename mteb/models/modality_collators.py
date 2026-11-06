from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mteb._log_once import LogOnce
from mteb.types._encoder_io import AudioInputItem

if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch
    from torch.utils.data import DataLoader
    from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]

    from mteb.types import BatchedInput

logger = logging.getLogger(__name__)
log_once = LogOnce(logger)

# Minimum number of frames in FPS-based sampling; wrappers whose video processor
# needs more raise it (Qwen3-VL fails on 1 frame: `t:1 must be larger than
# temporal_factor:2`).
DEFAULT_MIN_FRAMES = 1

# Order of the modality columns in a collated row. Wrappers that build a prompt in
# row order (e.g. Qwen-Omni based ones) then match the authors' text, image, video,
# audio order, whatever the dataset's column order.
_MODALITY_ORDER = ("text", "image", "video", "audio")


def _order_modalities(row: dict[str, Any]) -> dict[str, Any]:
    other = {k: v for k, v in row.items() if k not in _MODALITY_ORDER}
    return other | {k: row[k] for k in _MODALITY_ORDER if k in row}


def frame_indices(n_source: int, target: int) -> list[int]:
    """Indices of ``target`` evenly spaced frames from a clip of ``n_source`` frames."""
    if n_source < target:
        # repeat each frame in place (0, 0, 1, 1, ...) instead of looping
        # back to the start, as linspace-based upstream samplers do
        return [i * n_source // target for i in range(target)]
    return [int(i) for i in np.linspace(0, n_source - 1, target).round()]


class AudioCollator:
    """Collator for audio data that resamples audio to a target sampling rate and optionally truncates to a maximum number of samples."""

    def __init__(
        self,
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
        min_samples: int | None = None,
    ) -> None:
        """Initialize the collator.

        Args:
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
            min_samples: Zero-pad (silence) shorter audio to this many samples, for models whose first layer needs a minimum input length. If None, no padding is applied.
        """
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples
        self.min_samples = min_samples

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of audio inputs by resampling them to the target sampling rate and optionally truncating to a maximum number of samples."""
        return self.resample_audios(
            inputs,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
            min_samples=self.min_samples,
        )

    @staticmethod
    def resample_audios(
        inputs: list[dict[str, Any]],
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
        min_samples: int | None = None,
    ) -> BatchedInput:
        """Resample a batch of audio inputs to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            inputs: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
            min_samples: Zero-pad shorter audio to this many samples. If None, no padding is applied.
        """
        collated_inputs = []
        for row in inputs:
            audio_array = AudioCollator.resample_audio(
                row,
                target_sampling_rate=target_sampling_rate,
                max_samples=max_samples,
                min_samples=min_samples,
            )
            row["audio"] = AudioInputItem(
                array=audio_array, sampling_rate=target_sampling_rate
            )
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {key: [row[key] for row in collated_inputs] for key in collated_inputs[0]},
        )

    @staticmethod
    def resample_audio(
        audio: dict[str, Any],
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
        min_samples: int | None = None,
    ) -> np.typing.NDArray[np.floating]:
        """Resample an audio input to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            audio: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
            min_samples: Zero-pad shorter audio to this many samples. If None, no padding is applied.
        """
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}.")
        import torch
        import torchaudio

        audio = audio["audio"]
        if audio["sampling_rate"] != target_sampling_rate:
            logger.debug(
                f"Resampling audio from {audio['sampling_rate']} Hz to {target_sampling_rate} Hz."
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio["sampling_rate"],
                new_freq=target_sampling_rate,
            )
            audio_array = resampler(
                torch.from_numpy(np.asarray(audio["array"])).float()
            ).numpy()
        else:
            audio_array = np.asarray(audio["array"])

        # Convert to mono if needed
        if audio_array.ndim > 1 and audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0)

        if max_samples is not None:
            num_samples = audio_array.shape[-1]
            if num_samples > max_samples:
                log_once.info(
                    f"Truncating audio longer than {max_samples / target_sampling_rate:g} s "
                    "(the configured audio cap); only the start of each longer clip is encoded."
                )
                audio_array = audio_array[..., :max_samples]

        if min_samples is not None and audio_array.shape[-1] < min_samples:
            # pad the time axis only, with silence
            pad = [(0, 0)] * (audio_array.ndim - 1) + [
                (0, min_samples - audio_array.shape[-1])
            ]
            audio_array = np.pad(audio_array, pad)

        audio_array = cast(
            "np.ndarray[tuple[Any, ...], np.dtype[np.floating[Any]]]",
            audio_array,
        )
        return audio_array


def check_duration_cap(seconds: float | None) -> float | None:
    """Validate an optional audio length cap (seconds or samples). None means no cap."""
    if seconds is not None and seconds <= 0:
        raise ValueError(f"Audio length cap must be positive, got {seconds}.")
    return seconds


def seconds_to_samples(seconds: float | None, sampling_rate: int) -> int | None:
    """Convert an optional duration cap to samples. None means no cap."""
    seconds = check_duration_cap(seconds)
    if seconds is None:
        return None
    return int(seconds * sampling_rate)


@contextmanager
def fixed_numpy_seed(seed: int = 0) -> Iterator[None]:
    """Seed numpy's global RNG for a block, then restore it.

    Some feature extractors pick random crops of long clips from the global
    RNG (CLAP fusion, LanguageBind audio); seeding per clip makes the
    embeddings reproducible.
    """
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def single_clip_dataloader(
    inputs: DataLoader[BatchedInput],
    reason: str,
) -> DataLoader[BatchedInput]:
    """Rebuild an audio or video DataLoader with batch_size=1.

    For models whose embeddings change with their batch-mates (batched vs single
    cosine below 0.99 on real clips). ``reason`` is logged once.
    ``DataLoader.batch_size`` cannot be changed after creation.
    """
    from torch.utils.data import DataLoader

    log_once.info(f"Encoding one item at a time (batch_size ignored): {reason}.")

    return DataLoader(
        inputs.dataset,
        batch_size=1,
        collate_fn=inputs.collate_fn,
        num_workers=inputs.num_workers,
    )


class FramesCollator:
    """Collator for video data that resamples video frames.

    Supports two sampling modes:
    - FPS-based: downsamples to ``fps`` frames per second, so the number of
      selected frames scales with video duration. Videos already below the
      target rate keep all frames. ``max_frames`` caps the total to prevent
      OOM on very long videos, and ``min_frames`` sets a floor (Qwen3-VL video
      processors need at least 2 frames).
    - Fixed-sample: always selects exactly ``num_frames`` frames uniformly
      across the video, regardless of duration. Clips with fewer frames
      repeat each frame in place.

    When both ``fps`` and ``num_frames`` are None, no frame resampling is
    performed and all frames are returned as-is for the model's processor
    to handle.
    """

    def __init__(
        self,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
    ) -> None:
        """Initialize the collator.

        Args:
            fps: Target frames per second for downsampling. The number of
                frames scales with video duration. Only downsamples; if the
                source video has fewer frames than the target, all frames
                are kept.
            max_frames: Safety cap on the number of frames when using
                FPS-based sampling.
            num_frames: If set, use fixed-sample mode: always select this many
                frames uniformly from the video.
            min_frames: Minimum number of frames in FPS-based sampling.
        """
        if num_frames is not None and fps is not None:
            raise ValueError(
                "Cannot specify both `num_frames` and `fps`. "
                "Use `num_frames` for fixed-sample mode or `fps` for FPS-based mode."
            )
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.min_frames = min_frames

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of video inputs by resampling the video frames according to the specified parameters."""
        collated_inputs = []
        for row in inputs:
            # assign in place (no pop) to keep the dataset's column order
            row["video"] = self.resample_video(
                row["video"],
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
                min_frames=self.min_frames,
            )
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {key: [row[key] for row in collated_inputs] for key in collated_inputs[0]},
        )

    @staticmethod
    def resample_video(
        video: VideoDecoder,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
    ) -> torch.Tensor:
        """Resample a video input to a target number of frames.

        When both ``fps`` and ``num_frames`` are None, all frames are returned
        for the model's processor to handle.

        Args:
            video: A VideoDecoder object containing the video data.
            fps: Target frames per second for downsampling. Only
                downsamples; source videos below this rate keep all frames.
            max_frames: Safety cap when using FPS-based sampling.
            num_frames: If set, select exactly this many frames uniformly
                (fixed-sample mode).
            min_frames: Minimum number of frames in FPS-based sampling.
        """

        def _indices(n_source: int) -> list[int]:
            if num_frames is None and fps is None:
                return list(range(n_source))
            if num_frames is not None:
                target = num_frames
            elif fps is not None:
                duration = video.metadata.end_stream_seconds
                if duration is None:
                    raise RuntimeError(
                        "video has no duration metadata; cannot use FPS-based sampling"
                    )
                target = max(min_frames, int(duration * fps))
                if max_frames is not None:
                    target = min(target, max_frames)
                # only downsample
                target = min(target, n_source)
            else:
                raise RuntimeError("`fps` is not set; cannot use FPS-based sampling")
            return frame_indices(n_source, target)

        # Retry on the actual call: decrement source count when trailing
        # frames fail to decode, either missing ("no more frames") or corrupt
        # ("Invalid data found when processing input").
        n = video.metadata.num_frames
        if n is None:
            raise RuntimeError("video has no frame count metadata")
        original_n = n
        while n > 0:
            try:
                frames: torch.Tensor = video.get_frames_at(_indices(n)).data
                if n < original_n:
                    logger.warning(
                        "Dropped %d undecodable trailing frame(s) from video",
                        original_n - n,
                    )
                return frames
            except RuntimeError as e:
                if not any(
                    msg in str(e) for msg in ("no more frames", "Invalid data found")
                ):
                    raise
                n -= 1
        raise RuntimeError("video has no decodable frames")


class VideoCollator:
    """Collator that handles any combination of video and audio modalities.

    Uses FramesCollator and AudioCollator static methods to process each modality.
    Supports both FPS-based and fixed-sample video frame selection.
    """

    def __init__(
        self,
        *,
        target_sampling_rate: int,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        max_samples: int | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
    ) -> None:
        """Initialize the collator.

        Args:
            target_sampling_rate: The sampling rate to resample audio to.
            fps: Target frames per second for video downsampling.
            max_frames: Safety cap on frames per video for FPS mode.
            num_frames: If set, use fixed-sample mode instead of FPS-based.
            max_samples: Maximum number of audio samples to keep. If None, no truncation.
            min_frames: Minimum number of frames in FPS-based sampling.
        """
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.min_frames = min_frames
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of inputs containing video and/or audio by resampling videos and audios according to the specified parameters."""
        collated_inputs = []
        for row in inputs:
            # rows can lack one modality (None) in mixed-modality datasets
            if row.get("video") is not None:
                row["video"] = FramesCollator.resample_video(
                    row["video"],
                    fps=self.fps,
                    max_frames=self.max_frames,
                    num_frames=self.num_frames,
                    min_frames=self.min_frames,
                )
            if row.get("audio") is not None:
                audio_array = AudioCollator.resample_audio(
                    row,
                    target_sampling_rate=self.target_sampling_rate,
                    max_samples=self.max_samples,
                )
                row["audio"] = AudioInputItem(
                    array=audio_array, sampling_rate=self.target_sampling_rate
                )
            collated_inputs.append(_order_modalities(row))
        return cast(
            "BatchedInput",
            {key: [row[key] for row in collated_inputs] for key in collated_inputs[0]},
        )
