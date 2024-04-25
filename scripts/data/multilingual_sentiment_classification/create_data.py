from __future__ import annotations

from datasets import load_dataset
from huggingface_hub import create_repo

repo_id = "mteb/multilingual-sentiment-classification"
create_repo(repo_id, repo_type="dataset")

id2label = {0: "negative", 1: "positive"}
languages = {
    "Urdu": "urd",
    "Vietnamese": "vie",
    "Algerian": "dza",
    "Thai": "tha",
    "Turkish": "tur",
    "Slovak": "slk",
    "Norwegian": "nor",
    "Spanish": "spa",
    "Russian": "rus",
    "Maltese": "mlt",
    "Korean": "kor",
    "Indonesian": "ind",
    "Hebrew": "heb",
    "Japanese": "jpn",
    "Greek": "ell",
    "German": "deu",
    "English": "eng",
    "Finnish": "fin",
    "Croatian": "hrv",
    "Chinese": "zho",
    "Cantonese": "cmn",
    "Bulgarian": "bul",
    "Basque": "eus",
    "Uyghur": "uig",
    "Bambara": "bam",
    "Polish": "pol",
    "Welsh": "cym",
    "Hindi": "hin",
    "Arabic": "ara",
    "Persian": "fas",
}

for lang in languages.keys():
    raw_ds = load_dataset(f"sepidmnorozy/{lang}_sentiment")

    raw_ds.push_to_hub(repo_id=repo_id, config_name=languages[lang])
