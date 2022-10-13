import gzip
import os

import datasets
import numpy as np
from tqdm import tqdm

import jsonlines


np.random.seed(28042000)

d = datasets.load_dataset("mteb/raw_medrxiv")["train"]
# d = d.select(range(1000))


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


split_size = 5000
split_number = 5
indices = np.arange(len(d))

splits = []

task_type = "s2s"

# Coarse splits 30k
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [get_text(item, task_type) for item in subset]
    labels = [item["category"] for item in subset]
    splits.append({"sentences": text, "labels": labels})

split_size = 2500
split_number = 5
indices = np.arange(len(d))

# Coarse splits 10k
for k in tqdm(range(split_number)):
    np.random.shuffle(indices)
    current_indices = indices[:split_size]
    subset = d.select(current_indices)
    text = [get_text(item, task_type) for item in subset]
    labels = [item["category"] for item in subset]
    splits.append({"sentences": text, "labels": labels})

repository = f"medrxiv-clustering-{task_type}"
with jsonlines.open(f"{repository}/test.jsonl", "w") as f_out:
    f_out.write_all(splits)
# Compress
with open(f"{repository}/test.jsonl", "rb") as f_in:
    with gzip.open(f"{repository}/test.jsonl.gz", "wb") as f_out:
        f_out.writelines(f_in)
# Remove uncompressed file
os.remove(f"{repository}/test.jsonl")
