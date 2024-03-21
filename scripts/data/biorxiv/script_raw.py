"""
Fetch data from https://api.biorxiv.org/ and keep useful information
"""

from __future__ import annotations

import gzip

import jsonlines
import requests
from tqdm import tqdm

api = "https://api.biorxiv.org/details/biorxiv/2021-01-01/2022-05-10/"

articles = []
cursor = 0
count = 0
while True:
    if count % 10 == 0:
        print(count)
    try:
        r = requests.get(f"{api}{cursor}")
    except Exception as e:
        print(e)
    if r.status_code == 200:
        data = r.json()
        tmp = data["collection"]
        articles.extend(tmp)
        if len(tmp) == 0:
            break
        cursor += len(tmp)
        count += 1


old_lines = articles
new_lines = []
split = 0

for idx, line in enumerate(tqdm(old_lines)):
    # Write split each 100k lines
    if idx > 0 and idx % 100000 == 0:
        file_name = f"raw_biorxiv/train_{split}"
        with jsonlines.open(f"{file_name}.jsonl", "w") as writer:
            writer.write_all(new_lines)
        with open(f"{file_name}.jsonl", "rb") as f_in:
            with gzip.open(f"{file_name}.jsonl.gz", "wb") as f_out:
                f_out.writelines(f_in)
        new_lines = []
        split += 1

    new_json = {
        "id": line["doi"],
        "title": line["title"],
        "abstract": line["abstract"],
        "category": line["category"],
    }
    new_lines.append(new_json)

# Flush buffer
file_name = f"raw_biorxiv/train_{split}"
with jsonlines.open(f"{file_name}.jsonl", "w") as writer:
    writer.write_all(new_lines)
with open(f"{file_name}.jsonl", "rb") as f_in:
    with gzip.open(f"{file_name}.jsonl.gz", "wb") as f_out:
        f_out.writelines(f_in)
new_lines = []
split += 1
