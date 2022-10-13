import os

from datasets import load_dataset

from huggingface_hub import create_repo, upload_file


_LANGUAGES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese",
}
repo_name = "amazon_reviews_multi"
create_repo(repo_name, organization="mteb", repo_type="dataset")

for lang in _LANGUAGES:
    raw_dset = load_dataset("amazon_reviews_multi", lang)
    raw_dset = raw_dset.map(lambda x: {"text": x["review_title"] + "\n\n" + x["review_body"]}, num_proc=4)
    raw_dset = raw_dset.map(lambda x: {"label": x["stars"] - 1}, num_proc=4)
    raw_dset = raw_dset.map(lambda x: {"label_text": str(x["label"])}, num_proc=4)

    raw_dset = raw_dset.rename_column("review_id", "id")
    raw_dset = raw_dset.remove_columns(
        ["product_id", "reviewer_id", "review_body", "review_title", "language", "product_category", "stars"]
    )

    for split, dset in raw_dset.items():
        save_path = f"{lang}/{split}.jsonl"
        dset.to_json(save_path)
        upload_file(
            path_or_fileobj=save_path, path_in_repo=save_path, repo_id="mteb/" + repo_name, repo_type="dataset"
        )
    os.system(f"rm -r {lang}")
