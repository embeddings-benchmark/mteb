from __future__ import annotations

import os

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm

WRITE_TOK = os.environ["HF_TOKEN"]

eval_split = "test"
data_raw = load_dataset("clip-benchmark/wds_flickr30k")[eval_split]


## t2i
queries_ = {"id": [], "modality": [], "text": []}
corpus_ = {"id": [], "modality": [], "image": []}
relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}

for row in tqdm(data_raw, total=len(data_raw)):
    image = row["jpg"]
    texts = row["txt"].split("\n")
    key = row["__key__"]

    doc_id = f"d_{key}"
    corpus_["id"].append(doc_id)
    corpus_["image"].append(image)
    corpus_["modality"].append("image")

    for i, text in enumerate(texts):
        query_id = f"q_{key}_{i}"
        queries_["id"].append(query_id)
        queries_["text"].append(text)
        queries_["modality"].append("text")

        relevant_docs_["query-id"].append(query_id)
        relevant_docs_["corpus-id"].append(doc_id)
        relevant_docs_["score"].append(1)

corpus = Dataset.from_dict(corpus_)
queries = Dataset.from_dict(queries_)
relevant_docs = Dataset.from_dict(relevant_docs_)

corpus = DatasetDict({"corpus": corpus})
queries = DatasetDict({"test": queries})
relevant_docs = DatasetDict({"test": relevant_docs})


repo_name = "isaacchung/flickr30kt2i"
create_repo(repo_name, repo_type="dataset", token=WRITE_TOK)

corpus.push_to_hub(repo_name, "corpus", token=WRITE_TOK)
queries.push_to_hub(repo_name, "query", token=WRITE_TOK)
relevant_docs.push_to_hub(repo_name, "qrels", token=WRITE_TOK)
