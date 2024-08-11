from __future__ import annotations

import gzip
import os
from typing import Literal

import datasets
import jsonlines
import numpy as np
from tqdm import tqdm

np.random.seed(28042000)

d = datasets.load_dataset("mteb/raw_medrxiv")["train"]


def cluster_stats(labels):
    (unique, counts) = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(u, c)


def get_text(record, type="s2s"):
    if type == "s2s":
        return record["title"]
    elif type == "p2p":
        return record["title"] + " " + record["abstract"]
    raise ValueError


def dedupe_on_id(dataset: datasets.Dataset) -> datasets.Dataset:
    """Creates a new dataset without duplicates in the `"id"` column."""
    seen_ids = set()
    unique_examples = []
    for example in tqdm(dataset):
        id_value = example["id"]
        if id_value not in seen_ids:
            seen_ids.add(id_value)
            unique_examples.append(example)

    return datasets.Dataset.from_list(unique_examples)


def format_dataset_and_export(
    dataset: datasets.Dataset, task_type: Literal["s2s", "p2p"] = "s2s"
) -> None:
    repository = f"medrxiv-clustering-{task_type}"

    formated_dataset = []
    for obs in tqdm(dataset):
        formated_dataset.append(
            {"sentences": get_text(obs, task_type), "labels": obs["category"]}
        )

    # Check for folder
    if not os.path.exists(repository):
        os.makedirs(repository)
    # Export
    with jsonlines.open(f"{repository}/test.jsonl", "w") as f_out:
        f_out.write_all(formated_dataset)
    # Compress
    with open(f"{repository}/test.jsonl", "rb") as f_in:
        with gzip.open(f"{repository}/test.jsonl.gz", "wb") as f_out:
            f_out.writelines(f_in)
    # Remove uncompressed file
    os.remove(f"{repository}/test.jsonl")


if __name__ == "__main__":
    d_unique = dedupe_on_id(d)

    task_type = "s2s"
    format_dataset_and_export(d_unique, task_type=task_type)

    task_type = "p2p"
    format_dataset_and_export(d_unique, task_type=task_type)
