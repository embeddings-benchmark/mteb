"""Take data from https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download and
only keep useful information
"""

from __future__ import annotations

import gzip
import json

import jsonlines
from tqdm import tqdm

with open("archive/arxiv-metadata-oai-snapshot.json") as file:
    old_lines = file.readlines()
    new_lines = []
    split = 0

    for idx, line in enumerate(tqdm(old_lines)):
        # Write split each 100k lines
        if idx > 0 and idx % 100000 == 0:
            file_name = f"raw_arxiv/train_{split}"
            with jsonlines.open(f"{file_name}.jsonl", "w") as writer:
                writer.write_all(new_lines)
            with open(f"{file_name}.jsonl", "rb") as f_in:
                with gzip.open(f"{file_name}.jsonl.gz", "wb") as f_out:
                    f_out.writelines(f_in)
            new_lines = []
            split += 1

        old_json = json.loads(line)
        new_json = {
            "id": old_json["id"],
            "title": old_json["title"],
            "abstract": old_json["abstract"],
            "categories": old_json["categories"],
        }
        new_lines.append(new_json)

    file_name = f"raw_arxiv/train_{split}"
    with jsonlines.open(f"{file_name}.jsonl", "w") as writer:
        writer.write_all(new_lines)
    with open(f"{file_name}.jsonl", "rb") as f_in:
        with gzip.open(f"{file_name}.jsonl.gz", "wb") as f_out:
            f_out.writelines(f_in)
