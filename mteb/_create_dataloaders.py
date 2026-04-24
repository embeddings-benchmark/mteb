from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from datasets import Dataset, Image
from torch.utils.data import DataLoader, default_collate

from mteb.types import (
    ConversationTurn,
    PromptType,
)
from mteb.types._encoder_io import AudioInputItem

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torchcodec.decoders import VideoDecoder  # type: ignore[import-untyped]

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        BatchedInput,
        Conversation,
    )
    from mteb.types._encoder_io import (
        TextInput,
    )

logger = logging.getLogger(__name__)


def _create_dataloader_from_texts(
    text: list[str],
    batch_size: int = 32,
    num_proc: int | None = None,
    **kwargs: Any,
) -> DataLoader[TextInput]:
    """Create a dataloader from a list of text.

    Args:
        text: A list of text to create a dataloader from.
        batch_size: Batch size for the dataloader.
        num_proc: Number of processes to use.
        kwargs: Not used, present catching extra arguments.

    Returns:
        A dataloader with the text.
    """
    dataset = Dataset.from_dict({"text": text})
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc if num_proc is not None and num_proc > 1 else 0,
    )


def _corpus_to_dict(
    row: dict[str, str],
) -> dict[str, str]:
    text = (
        (row["title"] + " " + row["text"]).strip()
        if "title" in row and len(row["title"]) > 0
        else row["text"].strip()
    )
    new_row = {
        "id": row["id"],
        "text": text,
        "body": row["text"],
    }
    # dataloaders can't handle None
    if "title" in row and row["title"] is not None and len(row["title"]) > 0:
        new_row["title"] = row["title"]
    return new_row


def _combine_queries_with_instruction_text(dataset: Dataset) -> Dataset:
    texts = dataset["text"]
    if "query" in dataset.column_names:
        dataset = dataset.remove_columns(["query"])
    dataset = dataset.add_column("query", texts)
    if "instruction" in dataset.column_names:
        instructions = dataset["instruction"]
        new_texts = [
            t + " " + instr if instr is not None else t
            for t, instr in zip(texts, instructions, strict=True)
        ]
        dataset = dataset.remove_columns(["text"]).add_column("text", new_texts)
    return dataset


def _convert_conv_history_to_query(
    row: dict[str, str | list[str] | Conversation],
) -> dict[str, str | Conversation]:
    """Convert a conversation history to a single query string.

    If row "conversation" is a list of strings, it will be joined with "; " and the role will be set to "user".
    If row "conversation" is a list of dictionaries, it will be converted to a string with the format "role: content; role: content; ...".

    Returns:
        The updated row with the "query" and "text" fields set to the conversation string, and the "conversation" field set to the list of ConversationTurn.
    """
    conversation = row["text"]
    # if it's a list of strings, just join them
    if isinstance(conversation, list) and isinstance(conversation[0], str):
        conversation_ = cast("list[str]", conversation)
        conv_str = "; ".join(conversation_)
        current_conversation = [
            ConversationTurn(role="user", content=message) for message in conversation_
        ]
        warnings.warn(
            "Conversations are a list of strings. Used 'user' role for all turns.",
            category=UserWarning,
        )
    # otherwise, it's a list of dictionaries, which we need to convert to strings
    elif isinstance(conversation, list) and isinstance(conversation[0], dict):
        conv = []
        current_conversation = []
        for i, turn in enumerate(conversation):
            error_msg = (
                "When converting conversations lists of dictionary to string, each turn in the conversation "
                + "must be a dictionary with 'role' and 'content' keys"
            )
            if not isinstance(turn, dict):
                raise ValueError(f"Turn {i} is not a dictionary. " + error_msg)

            # check for keys 'role' and 'content' in the dictionary, if not found, raise an error
            if "role" not in turn:
                raise ValueError("Key 'role' not found in the dictionary. " + error_msg)
            if "content" not in turn:
                raise ValueError(
                    "Key 'content' not found in the dictionary. " + error_msg
                )
            current_conversation.append(
                ConversationTurn(role=turn["role"], content=turn["content"])
            )
            conv.append(f"{turn['role']}: {turn['content']}")
        conv_str = "; ".join(conv)
    else:
        raise ValueError(
            "Conversations must be a list consisting of strings or dictionaries with 'role' and 'content' keys"
        )

    row["query"] = conv_str

    if "instruction" in row:
        conv_str = f"{row['instruction']} {conv_str}"

    row["text"] = conv_str
    row["conversation"] = current_conversation
    return cast("dict[str, str | list[ConversationTurn]]", row)


def _transform_image_to_rgb(
    image: Any, transform: Callable[[Any], Any] | None = None
) -> Any:
    """Convert image to RGB and apply a transformation (e.g. PILToTensor).

    Args:
        image: The input image, either a PIL image or a tensor.
        transform: An optional transformation function to apply to the image.

    Returns:
        The transformed image in RGB format.
    """
    # For PIL images: ensure RGB format.
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")
    # For tensor images with 1 channel: repeat channels.
    elif isinstance(image, torch.Tensor) and image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    # Apply the additional transformation (e.g., conversion to tensor) if provided.
    if transform is not None:
        return transform(image)
    return image


def _convert_images_to_rgb(
    example: dict[str, Any],
    image_col_name: str = "image",
    transform: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    if image_col_name not in example:
        return example
    example[image_col_name] = _transform_image_to_rgb(
        example[image_col_name], transform
    )
    return example


def _prepare_image_dataset(
    dataset: Dataset,
    image_column_name: str | None = None,
    transform: Callable[[Any], Any] | None = None,
    num_proc: int | None = None,
) -> Dataset:
    """Prepare the image dataset by converting images to RGB and applying transformations."""
    if (
        image_column_name
        and image_column_name in dataset.column_names
        and "image" not in dataset.column_names
    ):
        dataset = dataset.rename_column(image_column_name, "image")
    # don't process image if it's already in the correct format
    if isinstance(dataset.features["image"], Image):
        return dataset
    return dataset.map(
        _convert_images_to_rgb,
        fn_kwargs={"image_col_name": "image", "transform": transform},
        desc="Converting images to RGB",
        num_proc=num_proc,
    )


def _custom_collate_fn(batch: list[dict[str, Any]]) -> BatchedInput:
    """Custom collate function for DataLoader.

    - For the "image", "conversation" key, leave the images as a list (to avoid stacking errors).
    - For other keys, use the default collate.

    Args:
        batch: A list of dictionaries to collate.

    Returns:
        A collated dictionary.
    """
    collated = {}
    for key in batch[0]:
        if key in (  # noqa: PLR6201
            "image",  # images can be with different sizes
            "conversation",  # conversations are lists of varying lengths
            "audio",  # audio can have different lengths
            "video",  # video can have different lengths
        ):
            collated[key] = [item[key] for item in batch]
        else:
            if any(item[key] is None for item in batch):
                raise ValueError(f"Found None in batch for key '{key}'")
            collated[key] = default_collate([item[key] for item in batch])
    return cast("BatchedInput", collated)


def _prepare_dataset(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None = None,
    input_column: str | None = None,
    num_proc: int | None = None,
) -> Dataset:
    """Apply all modality-specific transformations to the dataset.

    Args:
        dataset: The dataset to prepare.
        task_metadata: The metadata of the task.
        prompt_type: The type of prompt.
        input_column: The column to use as input. If None, it will use the first column that matches the modality.
        num_proc: Number of processes.

    Returns the transformed Dataset (no DataLoader wrapping).
    """
    modalities = task_metadata.get_modalities(prompt_type)

    if "text" in modalities:
        if prompt_type == PromptType.document:
            dataset = dataset.map(
                _corpus_to_dict,
                desc="Standardizing text corpus format",
                num_proc=num_proc,
            )
        elif prompt_type == PromptType.query:
            if isinstance(dataset["text"][0], list):
                dataset = dataset.map(
                    _convert_conv_history_to_query,
                    desc="Converting conversations to queries",
                    num_proc=num_proc,
                )
            else:
                dataset = _combine_queries_with_instruction_text(dataset)

    if "image" in modalities:
        dataset = _prepare_image_dataset(
            dataset,
            image_column_name=input_column if input_column else "image",
            num_proc=num_proc,
        )
    for modality in ("audio", "video"):
        if modality in modalities:
            if (
                input_column
                and input_column in dataset.column_names
                and modality not in dataset.column_names
            ):
                dataset = dataset.rename_column(input_column, modality)

    # Drop modality columns not needed for this prompt type to avoid
    # None values in the collate function (e.g. text=None in image-only corpus)
    all_modality_columns = {"text", "image", "audio", "video"}
    for col in all_modality_columns - set(modalities):
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    return dataset


def create_dataloader(
    dataset: Dataset,
    *,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None = None,
    input_column: str | Sequence[str] | None = None,
    batch_size: int = 32,
    num_proc: int | None = None,
    **kwargs: Any,
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a dataset.

    If prompt_type is None, it will create a dataloader based on the modalities of the task.
    if prompt_type is provided, it will create a dataloader for the specified prompt type.

    Args:
        dataset: The dataset to create a dataloader from.
        task_metadata: The metadata of the task.
        prompt_type: The type of prompt to create a dataloader for. If None, it will be inferred from the task metadata.
        input_column: The column(s) to use as input. If a string, used for column renaming.
            If a Sequence, columns are assumed to already match modality names. If None, inferred from task metadata.
        batch_size: The batch size for the dataloader.
        num_proc: The number of processes to use for dataset processing.
        **kwargs: Additional arguments to pass to the dataloader creation functions.

    Returns:
        A dataloader for the dataset.
    """
    # Sequence means columns already match modality names, no renaming needed
    _input_column = input_column if isinstance(input_column, str) else None

    if (
        prompt_type is None
        and task_metadata.modalities == ["text"]
        and _input_column is not None
    ):
        return _create_dataloader_from_texts(
            dataset[_input_column],
            batch_size=batch_size,
        )

    prepared = _prepare_dataset(
        dataset,
        task_metadata,
        prompt_type=prompt_type,
        input_column=_input_column,
        num_proc=num_proc,
    )

    return DataLoader(
        prepared,
        batch_size=batch_size,
        collate_fn=_custom_collate_fn,
        num_workers=num_proc if num_proc is not None and num_proc > 1 else 0,
        shuffle=False,
    )


class AudioCollator:
    """Collator for audio data that resamples audio to a target sampling rate and optionally truncates to a maximum number of samples."""

    def __init__(
        self,
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
        return self.resample_audios(
            inputs,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
        )

    @staticmethod
    def resample_audios(
        inputs: list[dict[str, Any]],
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
                target_sampling_rate,
                max_samples,
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
            return video.get_frames_at(list(range(num_source_frames))).data

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
        return video.get_frames_at(selected_frames).data


class VideoCollator:
    """Collator that handles any combination of video and audio modalities.

    Uses FramesCollator and AudioCollator static methods to process each modality.
    Supports both FPS-based and fixed-sample video frame selection.
    """

    def __init__(
        self,
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
                    row, self.target_sampling_rate, self.max_samples
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
