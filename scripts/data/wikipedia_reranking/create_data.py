from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from huggingface_hub import hf_hub_download, HfApi
import io
from collections import defaultdict

languages = ["de"]

def map_corpus_to_query(example, negatives_dict):
    title = example["title"]
    positive = [example["text"]]
    negatives = negatives_dict[title]
    return {"query": example["text"], "positive": positive, "negative": negatives}

for lang in languages:
    ds_queries = load_dataset(f"rasdani/cohere-wikipedia-2023-11-{lang}-queries", split="train")
    ds_corpus = load_dataset(f"rasdani/cohere-wikipedia-2023-11-{lang}-1.5k-articles", split="train")
    ds_corpus = ds_corpus.filter(lambda x: x["score"] != 1)

    negatives_dict = defaultdict(list)
    for row in ds_corpus:
        negatives_dict[row["title"]].append(row["text"])

    ds = ds_queries.map( lambda x: map_corpus_to_query(x, negatives_dict), remove_columns=ds_queries.column_names)

    repo_id = f"ellamind/wikipedia-2023-11-reranking-{lang}"
    ds.push_to_hub(repo_id, config_name="corpus", split="test")

    # Download the README from the repository
    readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")

    with open(readme_path, "r") as f:
        readme_content = f.read()

    readme = """
This dataset is derived from Cohere's wikipedia-2023-11 dataset, which is in turn derived from `wikimedia/wikipedia`.
The dataset is licensed under the Creative Commons CC BY-SA 3.0 license.
    """
    # Prepend the license key to the YAML header and append the custom README
    if "- license: " not in readme_content and readme not in readme_content:
        license = "cc-by-sa-3.0"

        updated_readme = readme_content.replace(
        "---\ndataset_info:",
        "---\nlicense: {license}\ndataset_info:"
        ).format(license=license)
        updated_readme += readme

        api = HfApi()
        readme_bytes = updated_readme.encode("utf-8")
        api.upload_file(path_or_fileobj=readme_bytes, path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
