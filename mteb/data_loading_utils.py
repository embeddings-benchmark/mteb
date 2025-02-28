from __future__ import annotations

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.encoder_interface import BatchedInput


def create_dataloader(
    dataset: Dataset, batch_size: int = 32
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from.
        batch_size: The batch size of the dataloader.

    Returns:
        A dataloader with the dataset.
    """
    return torch.utils.data.DataLoader(
        dataset.with_format("torch"), batch_size=batch_size
    )


def create_dataloader_from_texts(
    text: list[str], batch_size: int = 32
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a list of text.

    Args:
        text: A list of text to create a dataloader from.
        batch_size: The batch size of the dataloader.

    Returns:
        A dataloader with the text.
    """
    dataset = Dataset.from_dict({"text": text})
    return create_dataloader(dataset)
