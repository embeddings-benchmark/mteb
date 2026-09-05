from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, cast

import torch
from datasets import Dataset, Image
from torch.utils.data import DataLoader, default_collate

from mteb.types import (
    ConversationTurn,
    PromptType,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
    row: dict[str, str | None],
) -> dict[str, str | None]:
    # An interleaved corpus leaves the text empty for documents that only carry
    # another modality, so both title and body may be absent on a given row. The
    # absence is preserved here — it is what tells the statistics apart from a
    # document whose text is genuinely the empty string — and normalized to ""
    # when the batch is collated for the model.
    body = row["text"]
    title = row.get("title") or ""
    if title:
        text = f"{title} {body}".strip() if body else title
    else:
        text = body.strip() if body else body
    new_row: dict[str, str | None] = {
        "id": row["id"],
        "text": text,
        "body": body,
    }
    # dataloaders can't handle None
    if title:
        new_row["title"] = title
    return new_row


def _combine_queries_with_instruction_text(dataset: Dataset) -> Dataset:
    # An interleaved query set leaves the text empty for queries carrying only
    # another modality, so `text` may be None on a given row.
    texts = [text or "" for text in dataset["text"]]
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
    # an interleaved query set carries no conversation on rows that are
    # image-/audio-/video-only
    if not conversation:
        row["query"] = ""
        row["text"] = ""
        row["conversation"] = []
        return cast("dict[str, str | list[ConversationTurn]]", row)
    # if it's a list of strings, just join them
    if isinstance(conversation, list) and isinstance(conversation[0], str):
        conv_str = "; ".join(conversation)
        current_conversation = [
            ConversationTurn(role="user", content=message) for message in conversation
        ]
        warnings.warn(
            "Conversations are a list of strings. Used 'user' role for all turns.",
            category=UserWarning,
            stacklevel=2,
        )
    # otherwise, it's a list of dictionaries, which we need to convert to strings
    elif isinstance(conversation, list) and isinstance(conversation[0], dict):
        conv = []
        current_conversation = []
        for i, turn in enumerate(conversation):
            error_msg = (
                "When converting conversations lists of dictionary to string, each turn in the conversation "
                "must be a dictionary with 'role' and 'content' keys"
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
        The transformed image in RGB format, or None if the row carries no image.
    """
    # An interleaved dataset carries no image on rows of another modality.
    if image is None:
        return None
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


# Kept as lists rather than stacked, since their entries vary in shape and a row
# of an interleaved dataset may carry no value for them at all (`None`).
_UNCOLLATED_COLUMNS = frozenset(
    {
        "image",  # images can be with different sizes
        "conversation",  # conversations are lists of varying lengths
        "audio",  # audio can have different lengths
        "video",  # video can have different lengths
    }
)
# Text-valued columns that an interleaved dataset may leave empty on rows that
# only carry another modality. A missing value is passed to the model as "",
# which every text encoder can handle, unlike None.
_OPTIONAL_TEXT_COLUMNS = frozenset({"text", "body", "title", "query", "instruction"})


def _custom_collate_fn(batch: list[dict[str, Any]]) -> BatchedInput:
    """Custom collate function for DataLoader.

    - For the "image", "conversation" key, leave the images as a list (to avoid stacking errors).
    - For other keys, use the default collate.

    Missing values are only tolerated for input columns, where they mark a row of
    an interleaved dataset that does not carry that modality. A None anywhere else
    (`id`, labels, scores, …) is a bug in the dataset and still raises.

    Args:
        batch: A list of dictionaries to collate.

    Returns:
        A collated dictionary.
    """
    collated = {}
    for key in batch[0]:
        if key in _UNCOLLATED_COLUMNS:
            collated[key] = [item[key] for item in batch]
        else:
            values = [item[key] for item in batch]
            if any(value is None for value in values):
                if key not in _OPTIONAL_TEXT_COLUMNS:
                    raise ValueError(f"Found None in batch for key '{key}'")
                values = [value if value is not None else "" for value in values]
            collated[key] = default_collate(values)
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
            # an interleaved query set may leave the first rows without text
            first_text = next((text for text in dataset["text"] if text), None)
            if isinstance(first_text, list):
                dataset = dataset.map(
                    _convert_conv_history_to_query,
                    desc="Converting conversations to queries",
                    num_proc=num_proc,
                )
            else:
                dataset = _combine_queries_with_instruction_text(dataset)

    if "image" in modalities:
        image_column_name = "image" if input_column is None else input_column
        if input_column in dataset.column_names:
            dataset = _prepare_image_dataset(
                dataset,
                image_column_name=image_column_name,
                num_proc=num_proc,
            )
    for modality in ("audio", "video"):
        if modality in modalities and (
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
