from __future__ import annotations

import gzip
import os

import datasets
import jsonlines
import numpy as np
from tqdm import tqdm

np.random.seed(28042000)

d = datasets.load_dataset("flax-sentence-embeddings/stackexchange_title_body_jsonl")[
    "validation"
]

# d = d.select(range(1000))


def cluster_stats(labels):
    (unique, counts) = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(u, c)


def get_text(record):
    return " ".join(record["texts"])


split_size = 10000
split_number = 5
indices = np.arange(len(d))

splits = []

# Coarse splits 10k
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [get_text(item) for item in subset]
    labels = [item["tags"][0] for item in subset]
    splits.append({"sentences": text, "labels": labels})


split_size = 5000
split_number = 5

# Coarse splits 5k
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [get_text(item) for item in subset]
    labels = [item["tags"][0] for item in subset]
    splits.append({"sentences": text, "labels": labels})

repository = "stackexchange-clustering-p2p"
with jsonlines.open(f"{repository}/test.jsonl", "w") as f_out:
    f_out.write_all(splits)
# Compress
with open(f"{repository}/test.jsonl", "rb") as f_in:
    with gzip.open(f"{repository}/test.jsonl.gz", "wb") as f_out:
        f_out.writelines(f_in)
# Remove uncompressed file
os.remove(f"{repository}/test.jsonl")
