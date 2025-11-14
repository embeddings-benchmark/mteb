import logging
from collections.abc import Callable
from typing import Any, cast

import torch
from datasets import Dataset, Image
from torch.utils.data import DataLoader, default_collate

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.types import (
    BatchedInput,
    Conversation,
    ConversationTurn,
    PromptType,
    QueryDatasetType,
)
from mteb.types._encoder_io import CorpusInput, ImageInput, QueryInput, TextInput

logger = logging.getLogger(__name__)


def _create_dataloader_from_texts(
    text: list[str],
    batch_size: int = 32,
    **kwargs: dict[str, Any],
) -> DataLoader[TextInput]:
    """Create a dataloader from a list of text.

    Args:
        text: A list of text to create a dataloader from.
        batch_size: Batch size for the dataloader.
        kwargs: Not used, present catching extra arguments.

    Returns:
        A dataloader with the text.
    """
    dataset = Dataset.from_dict({"text": text})
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
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
    # dataloders can't handle None
    if "title" in row and row["title"] is not None and len(row["title"]) > 0:
        new_row["title"] = row["title"]
    return new_row


def _create_dataloader_for_retrieval_corpus(
    dataset: Dataset,
    batch_size: int = 32,
) -> DataLoader[CorpusInput]:
    """Create a dataloader from a corpus.

    Args:
        dataset: Corpus
        batch_size: Batch size for the dataloader.

    Returns:
        A dataloader with the corpus.
    """
    new_ds = dataset.map(_corpus_to_dict, desc="Converting corpus dict")
    return torch.utils.data.DataLoader(
        new_ds,
        batch_size=batch_size,
    )


def _combine_queries_with_instruction_text(row: dict[str, str]) -> dict[str, str]:
    row["query"] = row["text"]

    if "instruction" in row and row["instruction"] is not None:
        row["text"] = row["query"] + " " + row["instruction"]
    else:
        row["text"] = row["query"]
    return row


def _create_text_dataloader_for_queries(
    queries: QueryDatasetType,
    batch_size: int = 32,
) -> DataLoader[QueryInput]:
    """Create a dataloader from a list of queries.

    Args:
        queries: A list of queries.
        batch_size: Batch size for the dataloader.

    Returns:
        A dataloader with the queries.
    """
    queries = queries.map(
        _combine_queries_with_instruction_text,
        desc="Processing queries for dataloading",
    )
    return torch.utils.data.DataLoader(
        queries,
        batch_size=batch_size,
    )


_warned_about_user_role = False


def _convert_conv_history_to_query(
    row: dict[str, list[str] | Conversation],
) -> dict[str, str | Conversation]:
    """Convert a conversation history to a single query string.

    If row "conversation" is a list of strings, it will be joined with "; " and the role will be set to "user".
    If row "conversation" is a list of dictionaries, it will be converted to a string with the format "role: content; role: content; ...".

    Returns:
        The updated row with the "query" and "text" fields set to the conversation string, and the "conversation" field set to the list of ConversationTurn.
    """
    global _warned_about_user_role

    conversation = row["text"]
    # if it's a list of strings, just join them
    if isinstance(conversation, list) and isinstance(conversation[0], str):
        conversation = cast(list[str], conversation)
        conv_str = "; ".join(conversation)
        current_conversation = [
            ConversationTurn(role="user", content=message) for message in conversation
        ]
        if not _warned_about_user_role:
            logger.warning(
                "Conversations are a list of strings. Used 'user' role for all turns."
            )
            _warned_about_user_role = True
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
    return row


def _create_dataloader_for_queries_conversation(
    queries: QueryDatasetType,
    batch_size: int = 32,
) -> DataLoader[QueryInput]:
    """Create a dataloader from a list of queries.

    Args:
        queries: A list of queries.
        batch_size: Batch size for the dataloader.

    Returns:
        A dataloader with the queries.
    """
    return DataLoader(
        queries.map(
            _convert_conv_history_to_query, desc="Converting conversations to queries"
        ),
        collate_fn=_custom_collate_fn,
        batch_size=batch_size,
    )


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
    )


def _custom_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for DataLoader.

    - For the "image", "conversation" key, leave the images as a list (to avoid stacking errors).
    - For other keys, use the default collate.

    Args:
        batch: A list of dictionaries to collate.

    Returns:
        A collated dictionary.
    """
    collated: dict[str, Any] = {}
    for key in batch[0]:
        if key in ("image", "conversation"):
            # Leave the images as a list to avoid stacking errors.
            collated[key] = [item[key] for item in batch]
        else:
            if any(item[key] is None for item in batch):
                raise ValueError(f"Found None in batch for key '{key}'")
            collated[key] = default_collate([item[key] for item in batch])
    return collated


def _create_image_dataloader(
    dataset: Dataset,
    image_column_name: str | None = None,
    batch_size: int = 32,
    transform: Callable[[Any], Any] | None = None,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = _custom_collate_fn,
) -> DataLoader[ImageInput]:
    """Creates a DataLoader with the image dataset prepared using the explicit transformation.

    Args:
        dataset: The dataset containing images.
        image_column_name: The name of the column containing images. If None, defaults to "image".
        batch_size: Batch size for the dataloader.
        transform: A transformation function to apply to each image (e.g., converting to tensor).
        collate_fn: A custom collate function to handle batching.

    Returns:
        A DataLoader with the image dataset.
    """
    dataset = _prepare_image_dataset(
        dataset, image_column_name, transform
    ).select_columns(["image"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )


def _create_text_queries_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
) -> DataLoader[QueryInput]:
    if not isinstance(dataset["text"][0], list):
        return _create_text_dataloader_for_queries(
            dataset,
            batch_size=batch_size,
        )
    return _create_dataloader_for_queries_conversation(
        dataset,
        batch_size=batch_size,
    )


def _create_queries_dataloader(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    input_column: str | None = None,
    batch_size: int = 32,
) -> DataLoader[QueryInput | ImageInput]:
    """Create a dataloader for queries."""
    queries_type = task_metadata.get_modalities(PromptType.query)
    if queries_type == ["text"]:  # text only
        return _create_text_queries_dataloader(
            dataset,
            batch_size=batch_size,
        )
    if "image" in queries_type:  # contains image
        return _create_image_dataloader(
            dataset,
            image_column_name="image",
            batch_size=batch_size,
        )
    raise ValueError(f"Can't handle queries type {queries_type}")


def _create_document_dataloader(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    input_column: str | None = None,
    batch_size: int = 32,
) -> DataLoader[CorpusInput | ImageInput]:
    """Create a dataloader for documents.

    Args:
        dataset: The dataset containing the documents.
        task_metadata: Metadata of the task to determine the document type.
        input_column: The column to use as input. If None, it will use the first column that matches the modality.
        batch_size: Batch size for the dataloader.
    """
    document_type = task_metadata.get_modalities(PromptType.document)
    if document_type == ["text"]:  # text only
        return _create_dataloader_for_retrieval_corpus(
            dataset,
            batch_size=batch_size,
        )
    if "image" in document_type:  # contains image
        return _create_image_dataloader(
            dataset,
            image_column_name="image",
            batch_size=batch_size,
        )
    raise ValueError(f"Can't handle queries type {document_type}")


def create_dataloader(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None = None,
    input_column: str | None = None,
    batch_size: int = 32,
    **kwargs: dict[str, Any],
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a dataset.

    If prompt_type is None, it will create a dataloader based on the modalities of the task.
    if prompt_type is provided, it will create a dataloader for the specified prompt type.

    Args:
        dataset: The dataset to create a dataloader from.
        task_metadata: The metadata of the task.
        prompt_type: The type of prompt to create a dataloader for. If None, it will be inferred from the task metadata.
        input_column: The column to use as input. If None, it will use the first column that matches the modality.
        batch_size: The batch size for the dataloader.
        **kwargs: Additional arguments to pass to the dataloader creation functions.

    Returns:
        A dataloader for the dataset.
    """
    if prompt_type == PromptType.query:
        return _create_queries_dataloader(
            dataset,
            task_metadata,
            batch_size=batch_size,
            input_column=input_column,
        )
    if prompt_type == PromptType.document:
        return _create_document_dataloader(
            dataset,
            task_metadata,
            input_column=input_column,
            batch_size=batch_size,
        )

    if "image" in task_metadata.modalities:
        return _create_image_dataloader(
            dataset,
            image_column_name=input_column,
            batch_size=batch_size,
        )
    if "text" in task_metadata.modalities and input_column is not None:
        return _create_dataloader_from_texts(
            dataset[input_column],
            batch_size=batch_size,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
    )
