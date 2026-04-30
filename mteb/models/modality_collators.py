from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from mteb.types._encoder_io import AudioInputItem

if TYPE_CHECKING:
    from torchcodec.decoders import VideoDecoder  # type: ignore[import-untyped]

    from mteb.types import BatchedInput

logger = logging.getLogger(__name__)


class AudioCollator:
    """Collator for audio data that resamples audio to a target sampling rate and optionally truncates to a maximum number of samples."""

    def __init__(
        self,
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the collator.

        Args:
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
        """
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of audio inputs by resampling them to the target sampling rate and optionally truncating to a maximum number of samples."""
        return self.resample_audios(
            inputs,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
        )

    @staticmethod
    def resample_audios(
        inputs: list[dict[str, Any]],
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
    ) -> BatchedInput:
        """Resample a batch of audio inputs to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            inputs: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
        """
        collated_inputs = []
        for row in inputs:
            audio_array = AudioCollator.resample_audio(
                row,
                target_sampling_rate=target_sampling_rate,
                max_samples=max_samples,
            )
            row["audio"] = AudioInputItem(
                array=audio_array, sampling_rate=target_sampling_rate
            )
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {
                key: [row[key] for row in collated_inputs]
                for key in collated_inputs[0].keys()
            },
        )

    @staticmethod
    def resample_audio(
        audio: dict[str, Any],
        *,
        target_sampling_rate: int,
        max_samples: int | None = None,
    ) -> np.typing.NDArray[np.floating]:
        """Resample an audio input to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            audio: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
        """
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
            audio_array = torch.from_numpy(audio["array"]).float()
            audio_array = resampler(audio_array)
            audio_array = audio_array.numpy()
        else:
            audio_array = audio["array"]

        # Convert to mono if needed
        if audio_array.ndim > 1 and audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0)

        if max_samples is not None:
            num_samples = audio_array.shape[-1]
            if num_samples > max_samples:
                audio_array = audio_array[..., :max_samples]
        return audio_array


class FramesCollator:
    """Collator for video data that resamples video frames.

    Supports two sampling modes:
    - FPS-based: downsamples to ``fps`` frames per second, so the number of
      selected frames scales with video duration. Videos already below the
      target rate keep all frames. ``max_frames`` caps the total to prevent
      OOM on very long videos.
    - Fixed-sample: always selects exactly ``num_frames`` frames uniformly
      across the video, regardless of duration.

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
        """
        if num_frames is not None and fps is not None:
            raise ValueError(
                "Cannot specify both `num_frames` and `fps`. "
                "Use `num_frames` for fixed-sample mode or `fps` for FPS-based mode."
            )
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of video inputs by resampling the video frames according to the specified parameters."""
        collated_inputs = []
        for row in inputs:
            video = row.pop("video")
            row["video"] = self.resample_video(
                video,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
            )
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {
                key: [row[key] for row in collated_inputs]
                for key in collated_inputs[0].keys()
            },
        )

    @staticmethod
    def resample_video(
        video: VideoDecoder,
        *,
        fps: float | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
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
        """
        num_source_frames = video.metadata.num_frames

        if num_frames is None and fps is None:
            # No resampling: return all frames
            return video.get_frames_at(
                torch.tensor(list(range(num_source_frames)), dtype=torch.long)
            ).data

        if num_frames is not None:
            # Fixed-sample mode: always select exactly num_frames
            target = num_frames
        else:
            # FPS-based mode: scale with duration
            duration = video.metadata.end_stream_seconds
            target = max(1, int(duration * fps))
            if max_frames is not None:
                target = min(target, max_frames)

        frame_step = max(1, num_source_frames // target)
        selected_frames = list(range(0, num_source_frames, frame_step))[:target]
        return video.get_frames_at(torch.tensor(selected_frames, dtype=torch.long)).data


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
    ) -> None:
        """Initialize the collator.

        Args:
            target_sampling_rate: The sampling rate to resample audio to.
            fps: Target frames per second for video downsampling.
            max_frames: Safety cap on frames per video for FPS mode.
            num_frames: If set, use fixed-sample mode instead of FPS-based.
            max_samples: Maximum number of audio samples to keep. If None, no truncation.
        """
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        """Collate a batch of inputs containing video and/or audio by resampling videos and audios according to the specified parameters."""
        collated_inputs = []
        for row in inputs:
            if "video" in row:
                video = row.pop("video")
                row["video"] = FramesCollator.resample_video(
                    video,
                    fps=self.fps,
                    max_frames=self.max_frames,
                    num_frames=self.num_frames,
                )
            if "audio" in row:
                audio_array = AudioCollator.resample_audio(
                    row,
                    target_sampling_rate=self.target_sampling_rate,
                    max_samples=self.max_samples,
                )
                row["audio"] = AudioInputItem(
                    array=audio_array, sampling_rate=self.target_sampling_rate
                )
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {
                key: [row[key] for row in collated_inputs]
                for key in collated_inputs[0].keys()
            },
        )
