from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import create_repo

ds = load_dataset("timpal0l/scandisent")
repo_name = "mteb/scandisent"
create_repo(repo_name, repo_type="dataset")

ds1 = {}
df_split = ds["train"].to_polars()
df_grouped = dict(df_split.group_by(["language"]))
for lang in set(df_split["language"].unique()):
    ds1.setdefault(lang, {})
    # Remove lang column and convert back to HF datasets, not strictly necessary but better for compatibility
    ds1[lang] = DatasetDict(
        {
            "train": Dataset.from_polars(df_grouped[(lang,)].drop("language")).select(
                range(7500)
            ),
            "test": Dataset.from_polars(df_grouped[(lang,)].drop("language")).select(
                range(7500, 10000)
            ),
        }
    )
    ds1[lang].push_to_hub(repo_name, config_name=lang)
