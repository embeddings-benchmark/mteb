from __future__ import annotations

import json
import random
from collections import Counter

import pandas as pd

df = pd.read_csv("original.csv")

print(df)
"""
for field in ["target", "severe_toxicity", "obscene", "identity_attack", "insult", "threat"]:
    print("\n\n", field)
    num_greater = 0
    for val in df[field]:
        if val >= 0.5:
            num_greater += 1

    print(num_greater, len(df[field]), f"{num_greater/len(df[field])*100:.2f}%")
"""


rows = [
    {
        "text": row["comment_text"].strip(),
        "label": 1 if row["target"] >= 0.5 else 0,
        "label_text": "toxic" if row["target"] >= 0.5 else "not toxic",
    }
    for idx, row in df.iterrows()
]

random.seed(42)
random.shuffle(rows)

num_test = 50000
splits = {"test": rows[0:num_test], "train": rows[num_test:]}

print("Train:", len(splits["train"]))
print("Test:", len(splits["test"]))

num_labels = Counter()

for row in splits["test"]:
    num_labels[row["label"]] += 1
print(num_labels)

for split in ["train", "test"]:
    with open(f"{split}.jsonl", "w") as fOut:
        for row in splits[split]:
            fOut.write(json.dumps(row) + "\n")
