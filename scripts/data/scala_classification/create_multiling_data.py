from __future__ import annotations

from datasets import load_dataset
from huggingface_hub import create_repo

repo_id = "mteb/multilingual-scala-classification"
create_repo(repo_id, repo_type="dataset")

languages = {
    "Danish": "da",
    "Norwegian_b": "nb",
    "Norwegian_n": "nn",
    "Swedish": "sv",
}

for lang in languages.keys():
    raw_ds = load_dataset(f"mteb/scala_{languages[lang]}_classification")
    raw_ds.push_to_hub(repo_id=repo_id, config_name=lang)
