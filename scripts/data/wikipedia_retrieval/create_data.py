from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from huggingface_hub import hf_hub_download, HfApi
import io
from time import sleep

languages = ["de", "bn", "it", "pt", "nl", "cs", "ro", "bg", "sr", "fi", "fa", "hi", "da", "en"]

def apply_query_id(example, queries_dict):
    title = example["title"]
    corpus_id = example["_id"]
    score = example["score"]
    query_id = queries_dict[title]
    return {"query-id": query_id, "corpus-id": corpus_id, "score": score}

for lang in languages:
    ds_queries = load_dataset(f"rasdani/cohere-wikipedia-2023-11-{lang}-queries", split="train")
    ds_corpus = load_dataset(f"rasdani/cohere-wikipedia-2023-11-{lang}-1.5k-articles", split="train")
    sleep(5) # HF hub rate limit

    queries = ds_queries.map(lambda x: {"_id": "q" + x["_id"], "text": x["query"]}, remove_columns=ds_queries.column_names)
    queries_dict = {row["title"]: "q" + row["_id"] for row in ds_queries}
    corpus = ds_corpus.map(lambda x: {"_id": x["_id"], "title": x["title"], "text": x["text"]}, remove_columns=ds_corpus.column_names)
    qrels = ds_corpus.map(lambda x: apply_query_id(x, queries_dict=queries_dict), remove_columns=ds_corpus.column_names)

    corpus_features = Features({"_id": Value("string"), "title": Value("string"), "text": Value("string")})
    queries_features = Features({"_id": Value("string"), "text": Value("string")})
    qrels_features = Features({"query-id": Value("string"), "corpus-id": Value("string"), "score": Value("float32")})

    corpus = corpus.cast(corpus_features)
    queries = queries.cast(queries_features)
    qrels = qrels.cast(qrels_features)

    ds_dict = DatasetDict({"corpus": corpus, "queries": queries, "qrels": qrels})


    repo_id = f"ellamind/wikipedia-2023-11-retrieval-{lang}"
    corpus.push_to_hub(repo_id, config_name="corpus", split="test")
    queries.push_to_hub(repo_id, config_name="queries", split="test")
    qrels.push_to_hub(repo_id, config_name="default", split="test")

    # Download the README from the repository
    sleep(5)
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
