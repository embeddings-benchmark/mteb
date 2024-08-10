from __future__ import annotations

import gzip
import os
import random

import datasets
import jsonlines
import numpy as np

SEED = 42
NUM_SETS = 10
MIN_LABELS = 10
MAX_LABELS = 100
MIN_SAMPLES = 1_000
MAX_SAMPLES = 100_000

np.random.seed(SEED)
random.seed(SEED)

ds = datasets.load_dataset(
    "sentence-transformers/reddit-title-body",
    data_files=["reddit_title_text_2021.jsonl.gz"],
    split="train",
)

unique, counts = np.unique(ds["subreddit"], return_counts=True)
unique_to_count = dict(zip(unique, counts))

# Check top subreddits :)
# sorted(unique_to_count, key=lambda x: unique_to_count[x], reverse=True)[:10]

sets = []
for _ in range(NUM_SETS):
    num_labels = random.randint(MIN_LABELS, MAX_LABELS)
    num_samples = random.randint(MIN_SAMPLES, MAX_SAMPLES)

    print(f"Creating dataset with {num_labels} labels & {num_samples} samples")

    # Weigh by counts to reduce noise from random poorly defined subreddits
    # For 10 labels, ~85K samples; For 100 labels ~850K
    labels = random.choices(
        list(unique_to_count.keys()), weights=unique_to_count.values(), k=num_labels
    )
    sub_ds = ds.filter(lambda x: x["subreddit"] in labels).shuffle()
    if len(sub_ds) < MIN_SAMPLES:
        continue
    # Probability for len(sub_ds) to be smaller than selected samples is <5%
    sub_ds = sub_ds.select(range(min(len(sub_ds), num_samples)))

    text = [f"{x} {y}" for x, y in zip(sub_ds["title"], sub_ds["body"])]
    sets.append({"sentences": text, "labels": sub_ds["subreddit"]})

repo_name = "reddit-clustering-p2p"
with jsonlines.open(f"{repo_name}/test.jsonl", "w") as f_out:
    f_out.write_all(sets)
# Compress
with open(f"{repo_name}/test.jsonl", "rb") as f_in:
    with gzip.open(f"{repo_name}/test.jsonl.gz", "wb") as f_out:
        f_out.writelines(f_in)
# Remove uncompressed file
os.remove(f"{repo_name}/test.jsonl")
