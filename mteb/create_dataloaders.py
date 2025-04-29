from __future__ import annotations

import logging
from typing import Any, Callable

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, default_collate

from mteb.types import BatchedInput, Conversation

logger = logging.getLogger(__name__)


def create_dataloader_from_texts(
    text: list[str], **dataloader_kwargs
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a list of text.

    Args:
        text: A list of text to create a dataloader from.
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the text.
    """
    dataset = Dataset.from_dict({"text": text})
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def corpus_to_dict(
    corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
) -> dict[str, list[str | None]]:
    if isinstance(corpus, dict):
        sentences = [
            (corpus["title"][i] + " " + corpus["text"][i]).strip()
            if "title" in corpus
            else corpus["text"][i].strip()
            for i in range(len(corpus["text"]))
        ]
        titles = corpus["title"]
        bodies = corpus["text"]
    elif isinstance(corpus, list) and isinstance(corpus[0], dict):
        sentences = [
            (doc["title"] + " " + doc["text"]).strip()
            if "title" in doc
            else doc["text"].strip()
            for doc in corpus
        ]
        titles = [doc.get("title", "") for doc in corpus]
        bodies = [doc.get("text", "") for doc in corpus]
    else:
        sentences = corpus
        titles = [""] * len(corpus)
        bodies = [""] * len(corpus)
    return {
        "text": sentences,
        "title": titles,
        "body": bodies,
    }


def create_dataloader_for_retrieval_corpus(
    inputs: list[dict[str, str]] | dict[str, list[str]] | list[str], **dataloader_kwargs
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a corpus.

    Args:
        inputs: Corpus
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the corpus.
    """
    dataset = Dataset.from_dict(corpus_to_dict(inputs))
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def create_dataloader_for_queries(
    queries: list[str],
    instructions: list[str] | None = None,
    combine_query_and_instruction: Callable[[str, str], str] | None = None,
    **dataloader_kwargs,
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a list of queries.

    Args:
        queries: A list of queries.
        instructions: A list of instructions. If None, the dataloader will only contain the queries.
        combine_query_and_instruction: A function that combines a query with an instruction. If None, the default function will be used.
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the queries.
    """
    # cross encoder can produce list of None
    any_none_instruction = instructions is None or any(i is None for i in instructions)
    if instructions is None or any_none_instruction:
        dataset = Dataset.from_dict({"text": queries, "query": queries})
    else:
        dataset = Dataset.from_dict(
            {
                "text": [
                    combine_query_and_instruction(q, i)
                    for q, i in zip(queries, instructions)
                ],
                "instruction": instructions,
                "query": queries,
            }
        )
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


def convert_conv_history_to_query(
    conversations: list[list[str | Conversation]],
) -> list[str]:
    conversations_converted = []

    for conversation in conversations:
        # if it's a list of strings, just join them
        if isinstance(conversation[0], str):
            conv_str = "; ".join(conversation)
        # otherwise, it's a list of dictionaries, which we need to convert to strings
        elif isinstance(conversation[0], dict):
            conv = []
            for i, turn in enumerate(conversation):
                error_msg = (
                    "When converting conversations lists of dictionary to string, each turn in the conversation "
                    + "must be a dictionary with 'role' and 'content' keys"
                )
                if not isinstance(turn, dict):
                    raise ValueError(f"Turn {i} is not a dictionary. " + error_msg)

                # check for keys 'role' and 'content' in the dictionary, if not found, raise an error
                if "role" not in turn:
                    raise ValueError(
                        "Key 'role' not found in the dictionary. " + error_msg
                    )
                if "content" not in turn:
                    raise ValueError(
                        "Key 'content' not found in the dictionary. " + error_msg
                    )

                conv.append(f"{turn['role']}: {turn['content']}")
            conv_str = "; ".join(conv)
        else:
            raise ValueError(
                "Conversations must be a list consisting of strings or dictionaries with 'role' and 'content' keys"
            )

        conversations_converted.append(conv_str)

    return conversations_converted


def create_dataloader_for_queries_conversation(
    queries: list[list[str | Conversation]],
    instructions: list[str] | None = None,
    combine_query_and_instruction: Callable[[str, str], str] | None = None,
    **dataloader_kwargs,
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a list of queries.

    Args:
        queries: A list of queries.
        instructions: A list of instructions. If None, the dataloader will only contain the queries.
        combine_query_and_instruction: A function that combines a query with an instruction. If None, the default function will be used.
        dataloader_kwargs: Additional arguments to pass to the dataloader.

    Returns:
        A dataloader with the queries.
    """
    converted_queries = convert_conv_history_to_query(queries)
    if instructions is None:
        dataset = Dataset.from_dict(
            {
                "text": converted_queries,
                "query": converted_queries,
                "conversation": queries,
            }
        )
    else:
        dataset = Dataset.from_dict(
            {
                "text": [
                    combine_query_and_instruction(q, i)
                    for q, i in zip(converted_queries, instructions)
                ],
                "instruction": instructions,
                "query": converted_queries,
                "conversation": queries,
            }
        )
    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)


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
    )


def custom_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function that mimics the old pipeline:
    - For the "image" key, leave the images as a list (to avoid stacking errors).
    - For other keys, use the default collate.
    """
    collated: dict[str, Any] = {}
    for key in batch[0]:
        if key == "image":
            # Leave the images as a list to avoid stacking errors.
            collated["image"] = [item["image"] for item in batch]
        else:
            collated[key] = default_collate([item[key] for item in batch])
    return collated


def create_image_dataloader(
    dataset: Dataset,
    image_column_name: str | None = None,
    batch_size: int = 32,
    transform: Callable[[Any], Any] | None = None,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = custom_collate_fn,
) -> DataLoader[BatchedInput]:
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
