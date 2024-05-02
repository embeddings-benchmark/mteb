from __future__ import annotations

from datasets import create_repo, load_dataset

repo_id = "mteb/xnli2.0-multi-pair"
create_repo(repo_id, repo_type="dataset")

_LANGS = {
    "punjabi": ["pan-Guru"],
    "gujrati": ["guj-Gujr"],
    "kannada": ["kan-Knda"],
    "assamese": ["asm-Beng"],
    "bengali": ["ben-Beng"],
    "marathi": ["mar-Deva"],
    "bhojpuri": ["bho-Deva"], 
    "odiya": ["ory-Orya"],
    "sanskrit": ["san-Deva"],
    "tamil": ["tam-Taml"],
}

for lang in _LANGS.keys():
    raw_ds = load_dataset(f"Harsit/xnli2.0_{lang}")
    raw_ds.push_to_hub(repo_id=repo_id, config_name=lang)
