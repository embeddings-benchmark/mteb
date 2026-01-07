"""Simplified version of https://gist.github.com/AlexeyVatolin/ea3adc21aa7a767603ff393b22085adc from https://github.com/embeddings-benchmark/mteb/pull/2900"""

import logging

import datasets
import pandas as pd
from datasets import Dataset, DatasetDict

from mteb import TaskMetadata

logger = logging.getLogger(__name__)


def deduplicate(dataset: Dataset, input_column: str) -> Dataset:
    """Remove duplicate texts, keeping the first occurrence."""
    unique_texts = set()
    indices_to_keep = []
    for i, text in enumerate(dataset[input_column]):
        text = text.strip()
        if text not in unique_texts:
            unique_texts.add(text)
            indices_to_keep.append(i)

    logger.info(
        f"[deduplicate] removed={len(dataset) - len(indices_to_keep)}/{len(dataset)}"
    )
    return dataset.select(indices_to_keep)


def filter_empty(dataset: Dataset, input_column: str) -> Dataset:
    """Filter out empty or whitespace-only examples."""
    before = len(dataset)
    ds = dataset.filter(lambda x: len(x[input_column].strip()) > 0)
    logger.info(f"[filter_empty] removed={before - len(ds)}/{before}")
    return ds


def filter_train_leakage(
    train_dataset: Dataset, test_dataset: Dataset, input_column: str
) -> Dataset:
    """Remove test examples that appear in training."""
    train_texts = set(train_dataset[input_column])
    before = len(test_dataset)
    indices = [
        i
        for i, text in enumerate(test_dataset[input_column])
        if text not in train_texts
    ]
    logger.info(f"[filter_train_leakage] removed={before - len(indices)}/{before}")
    return test_dataset.select(indices)


def filter_unclear_label(
    dataset_dict: DatasetDict, input_column: str, label_column: str
) -> DatasetDict:
    """Remove examples where the same text appears with multiple different labels."""
    normalized: dict[str, set[str | tuple[str, ...]]] = {}
    logger.debug("[filter_controversial] scanning dataset for label conflicts...")

    for split, ds in dataset_dict.items():
        for text, label in zip(ds[input_column], ds[label_column]):
            key = text.strip().lower()
            normalized.setdefault(key, set()).add(
                label if isinstance(label, (str, int, float)) else tuple(label)  # type: ignore[arg-type]
            )

    bad_texts = {t for t, labels in normalized.items() if len(labels) > 1}
    logger.info(f"[filter_controversial] Removing {len(bad_texts)} conflicting texts")

    new_dict = {}
    for split, ds in dataset_dict.items():
        before = len(ds)
        filtered = ds.filter(lambda x: x[input_column].strip().lower() not in bad_texts)
        logger.debug(
            f"[filter_controversial:{split}] removed={before - len(filtered)}/{before}"
        )
        new_dict[split] = filtered

    return DatasetDict(new_dict)


def filter_short(dataset: Dataset, input_column: str, min_words: int = 3) -> Dataset:
    """Filter out texts with fewer than `min_words`."""
    before = len(dataset)
    ds = dataset.filter(lambda x: len(x[input_column].strip().split()) >= min_words)
    logger.debug(f"[filter_short] removed={before - len(ds)}/{before}")
    return ds


def split_train_test(
    ds: DatasetDict,
    metadata: TaskMetadata,
    train_split: str,
    label_column: str,
) -> DatasetDict:
    if train_split in ds and metadata.eval_splits == train_split:
        before = len(ds[train_split])
        logger.info(
            f"[split_train_test] eval_splits == train_split; performing split on {before} examples"
        )
        ds[train_split] = ds[train_split].cast_column(
            label_column,
            datasets.ClassLabel(names=list(set(ds[train_split][label_column]))),
        )
        label_counts = pd.Series(ds[train_split][label_column]).value_counts()
        one_sample_labels = set(label_counts[label_counts == 1].index.tolist())

        if one_sample_labels:
            logger.info(
                f"[split_train_test] Removing {len(one_sample_labels)} labels with only one instance"
            )
            ds[train_split] = ds[train_split].filter(
                lambda x: x[label_column] not in one_sample_labels
            )

        splits = ds[train_split].train_test_split(
            test_size=min(2048, before // 2), seed=42, stratify_by_column=label_column
        )
        ds = DatasetDict({train_split: splits[train_split], "test": splits["test"]})
        metadata.eval_splits = ["test"]
        logger.info(
            f"[split_train_test] Train size={len(ds[train_split])}, Test size={len(ds['test'])}"
        )

    return ds
