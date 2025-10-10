from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
from datasets import Dataset
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


def create_dataloader_from_texts(
    text: list[str],
    batch_size: int = 32,
) -> DataLoader[TextInput]:
    """Create a dataloader from a list of text.

    Args:
        text: A list of text to create a dataloader from.
        batch_size: Batch size for the dataloader.

    Returns:
        A dataloader with the text.
    """
    dataset = Dataset.from_dict({"text": text})
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )


def corpus_to_dict(
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


def create_dataloader_for_retrieval_corpus(
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
    new_ds = dataset.map(corpus_to_dict, desc="Converting corpus dict")
    return torch.utils.data.DataLoader(
        new_ds,
        batch_size=batch_size,
    )


def combine_queries_with_instruction_text(row: dict[str, str]) -> dict[str, str]:
    row["query"] = row["text"]

    if "instruction" in row and row["instruction"] is not None:
        row["text"] = row["query"] + " " + row["instruction"]
    else:
        row["text"] = row["query"]
    return row


def create_text_dataloader_for_queries(
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
        combine_queries_with_instruction_text, desc="Processing queries for dataloading"
    )
    return torch.utils.data.DataLoader(
        queries,
        batch_size=batch_size,
    )


def convert_conv_history_to_query(
    row: dict[str, list[str] | Conversation],
) -> dict[str, str | Conversation]:
    conversation = row["text"]
    # if it's a list of strings, just join them
    if isinstance(conversation, list) and isinstance(conversation[0], str):
        conversation = cast(list[str], conversation)
        conv_str = "; ".join(conversation)
        current_conversation = [
            ConversationTurn(role="user", content=message) for message in conversation
        ]

        logger.warning(
            "Conversations are a list of strings. Used 'user' role for all turns."
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
    return row


def create_dataloader_for_queries_conversation(
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
            convert_conv_history_to_query, desc="Converting conversations to queries"
        ),
        collate_fn=custom_collate_fn,
        batch_size=batch_size,
    )


def transform_image_to_rgb(
    image: Any, transform: Callable[[Any], Any] | None = None
) -> Any:
    """Convert image to RGB and apply a transformation (e.g. PILToTensor)."""
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


def convert_images_to_rgb(
    example: dict[str, Any],
    image_col_name: str = "image",
    transform: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    if image_col_name not in example:
        return example
    example[image_col_name] = transform_image_to_rgb(example[image_col_name], transform)
    return example


def prepare_image_dataset(
    dataset: Dataset,
    image_column_name: str | None = None,
    transform: Callable[[Any], Any] | None = None,
) -> Dataset:
    # If the dataset uses a different column name for images, rename it to "image".
    if (
        image_column_name
        and image_column_name in dataset.column_names
        and "image" not in dataset.column_names
    ):
        dataset = dataset.rename_column(image_column_name, "image")
    # Map the conversion function over the dataset.
    return dataset.map(
        convert_images_to_rgb,
        fn_kwargs={"image_col_name": "image", "transform": transform},
        desc="Converting images to RGB",
    )


def custom_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function that mimics the old pipeline:
    - For the "image", "conversation" key, leave the images as a list (to avoid stacking errors).
    - For other keys, use the default collate.
    """
    collated: dict[str, Any] = {}
    for key in batch[0]:
        if key in ("image", "conversation"):
            # Leave the images as a list to avoid stacking errors.
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = default_collate([item[key] for item in batch])
    return collated


def create_image_dataloader(
    dataset: Dataset,
    image_column_name: str | None = None,
    batch_size: int = 32,
    transform: Callable[[Any], Any] | None = None,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = custom_collate_fn,
) -> DataLoader[ImageInput]:
    """Creates a DataLoader with the image dataset prepared using the explicit transformation.
    This should mirror the behavior of the old code.
    """
    dataset = prepare_image_dataset(dataset, image_column_name, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )


def create_text_queries_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
) -> DataLoader[BatchedInput]:
    if not isinstance(dataset["text"][0], list):
        return create_text_dataloader_for_queries(
            dataset,
            batch_size=batch_size,
        )
    return create_dataloader_for_queries_conversation(
        dataset,
        batch_size=batch_size,
    )


def create_queries_dataloader(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    input_column: str | None = None,
    batch_size: int = 32,
) -> DataLoader[BatchedInput]:
    queries_type, _ = task_metadata.category.split("2")
    if queries_type == "t":  # text only
        return create_text_queries_dataloader(
            dataset,
            batch_size=batch_size,
        )
    if "i" in queries_type:  # contains image
        return create_image_dataloader(
            dataset,
            image_column_name="image",
            batch_size=batch_size,
        )
    raise ValueError(f"Can't handle queries type {queries_type}")


def create_document_dataloader(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    input_column: str | None = None,
    batch_size: int = 32,
) -> DataLoader[BatchedInput]:
    if task_metadata.category is None:
        document_type = "t"
    else:
        _, document_type = task_metadata.category.split("2")
    if document_type == "t":  # text only
        return create_dataloader_for_retrieval_corpus(
            dataset,
            batch_size=batch_size,
        )
    if "i" in document_type:  # contains image
        return create_image_dataloader(
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
    if prompt_type == PromptType.query:
        return create_queries_dataloader(
            dataset,
            task_metadata,
            batch_size=batch_size,
            input_column=input_column,
        )
    if prompt_type == PromptType.document:
        return create_document_dataloader(
            dataset,
            task_metadata,
            input_column=input_column,
            batch_size=batch_size,
        )

    if "image" in task_metadata.modalities:
        return create_image_dataloader(
            dataset,
            image_column_name=input_column,
        )
    if "text" in task_metadata.modalities and input_column is not None:
        return create_dataloader_from_texts(
            dataset[input_column],
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
    )
