from __future__ import annotations

import os
import uuid
from typing import Dict

from datasets import load_dataset
from huggingface_hub import create_repo, upload_file


def preprocess_data(example: Dict) -> Dict:
    """Preprocessed the data in a format easier
    to handle for the loading of queries and corpus
    ------
    PARAMS
    example : element in med-qa dataset
    """
    return {
        "query-id": str(uuid.uuid4()),
        "query_text": example["Question"],
        "corpus-id": str(uuid.uuid4()),
        "answer_text": example["Answer"],
    }


repo_name = "mteb/medical_qa"
create_repo(repo_name, repo_type="dataset", token="")


raw_dset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
dset = raw_dset["train"]
trimmed_dataset = dset.select(range(2048))
updated_dataset = trimmed_dataset.map(
    preprocess_data, remove_columns=["Question", "Answer", "qtype"]
)
corpus_ds = updated_dataset.map(
    lambda example: {"_id": example["corpus-id"], "text": example["answer_text"]},
    remove_columns=["query-id", "query_text", "corpus-id", "answer_text"],
)
corpus_ds = corpus_ds.add_column("title", len(corpus_ds) * [""])
default_ds = updated_dataset.map(
    lambda example: example, remove_columns=["answer_text", "query_text"]
)
default_ds = default_ds.add_column("score", len(corpus_ds) * [0])
queries_ds = updated_dataset.map(
    lambda example: {"_id": example["query-id"], "text": example["query_text"]},
    remove_columns=["corpus-id", "answer_text", "query-id", "query_text"],
)
data = {"corpus": corpus_ds, "default": default_ds, "queries": queries_ds}
for splits in ["default", "queries"]:
    save_path = f"{splits}.jsonl"
    data[splits].to_json(save_path)
    upload_file(
        path_or_fileobj=save_path,
        path_in_repo=save_path,
        repo_id=repo_name,
        repo_type="dataset",
    )
    os.system(f"rm {save_path}")
