from __future__ import annotations

import os

from huggingface_hub import create_repo, upload_file

from datasets import load_dataset

repo_name = "imdb"
create_repo(repo_name, organization="mteb", repo_type="dataset")

id2label = {0: "negative", 1: "positive", -1: "unlabeled"}
raw_dset = load_dataset("imdb")
raw_dset = raw_dset.map(lambda x: {"label_text": id2label[x["label"]]}, num_proc=4)
for split, dset in raw_dset.items():
    save_path = f"{split}.jsonl"
    dset.to_json(save_path)
    upload_file(
        path_or_fileobj=save_path,
        path_in_repo=save_path,
        repo_id="mteb/" + repo_name,
        repo_type="dataset",
    )
    os.system(f"rm {save_path}")
