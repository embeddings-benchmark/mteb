import logging

from datasets import DatasetDict

from mteb import TaskMetadata
from mteb.abstasks import AbsTaskClassification
from mteb.abstasks._data_filter.filters import (
    deduplicate,
    filter_empty,
    filter_short,
    filter_train_leakage,
    filter_unclear_label,
    split_train_test,
)

logger = logging.getLogger(__name__)


def clean_dataset(
    ds: DatasetDict,
    metadata: TaskMetadata,
    train_split: str,
    input_column: str,
    label_column: str,
    subset: str | None = None,
) -> DatasetDict:
    """Apply the full cleaning pipeline with logging."""
    logger.info("[clean_dataset] Starting dataset cleaning pipeline...")

    transforms = [
        ("filter_empty", filter_empty),
        ("deduplicate", deduplicate),
    ]

    skip_cjk_codes = {"zho", "jpn", "tha", "mya", "cmn"}
    logger.info("[clean_dataset] Applying short-text filter")
    cur_langs = (
        metadata.eval_langs[subset]
        if isinstance(metadata.eval_langs, dict) and subset
        else metadata.eval_langs
    )
    apply_short = not any(lang.split("-")[0] in skip_cjk_codes for lang in cur_langs)
    if apply_short:
        logger.info("[clean_dataset] Applying short-text filter")
        transforms.append(("filter_short", filter_short))

    for split in [train_split, *metadata.eval_splits]:
        if split not in ds:
            logger.warning(f"[clean_dataset] Split '{split}' missing; skipping.")
            continue

        for name, fn in transforms:
            before = len(ds[split])
            ds[split] = fn(ds[split], input_column=input_column)
            logger.info(
                f"[clean_dataset:{split}] {name} removed={before - len(ds[split])}"
            )

    ds = split_train_test(ds, metadata, train_split, label_column)

    for split in metadata.eval_splits:
        if split == train_split:
            continue
        before = len(ds[split])
        ds[split] = filter_train_leakage(ds[train_split], ds[split], input_column)
        logger.info(
            f"[clean_dataset:{split}] leakage_removed={before - len(ds[split])}"
        )

    ds = filter_unclear_label(ds, input_column=input_column, label_column=label_column)

    logger.info("[clean_dataset] Cleaning pipeline complete.")
    return ds


def process_classification(
    task: AbsTaskClassification,
) -> DatasetDict | dict[str, DatasetDict]:
    """Process classification task dataset(s) with cleaning pipeline."""
    if not task.data_loaded:
        task.load_data()
    if isinstance(task.dataset, DatasetDict):
        return clean_dataset(
            task.dataset,
            task.metadata,
            task.train_split,
            task.input_column_name,
            task.label_column_name,
            subset=None,
        )

    if task.dataset is None:
        raise ValueError("Task dataset is None.")

    new_ds = {}
    for subset in task.dataset:
        new_ds[subset] = clean_dataset(
            task.dataset[subset],
            task.metadata,
            task.train_split,
            task.input_column_name,
            task.label_column_name,
            subset=subset,
        )
    return new_ds
