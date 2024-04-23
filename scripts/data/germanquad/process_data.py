"""See clarin-knext/arguana-pl, clarin-knext/arguana-pl-qrels and
beir.datasets.data_loader_hf.HFDataLoader for BEIR format.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, Features, Value, load_dataset

dataset = load_dataset("deepset/germanquad")
dataset.pop("train")

# Deduplicate contexts and map them uniquely one-to-one to ids
context_to_id = {}

corpus_data = {"_id": [], "text": []}
queries_data = {"_id": [], "text": []}
qrels_data = {"query-id": [], "corpus-id": [], "score": []}

for item in dataset["test"]:
    # Check if the context is already in the dictionary
    if item["context"] not in context_to_id:
        context_to_id[item["context"]] = "c" + str(item["id"])
        entry = {"_id": context_to_id[item["context"]], "text": item["context"]}
        corpus_data["_id"].append(entry["_id"])
        corpus_data["text"].append(entry["text"])

for item in dataset["test"]:
    entry = {"_id": "q" + str(item["id"]), "text": item["question"]}
    queries_data["_id"].append(entry["_id"])
    queries_data["text"].append(entry["text"])

# this maps queries to relevant documents
for item in dataset["test"]:
    corpus_id = context_to_id[item["context"]]
    entry = {"query-id": "q" + str(item["id"]), "corpus-id": corpus_id, "score": 1}
    qrels_data["query-id"].append(entry["query-id"])
    qrels_data["corpus-id"].append(entry["corpus-id"])
    qrels_data["score"].append(entry["score"])

corpus_features = Features({"_id": Value("string"), "text": Value("string")})
qrels_features = Features(
    {"query-id": Value("string"), "corpus-id": Value("string"), "score": Value("int32")}
)
corpus_dataset = Dataset.from_dict(corpus_data, features=corpus_features)
queries_dataset = Dataset.from_dict(queries_data, features=corpus_features)
qrels_dataset = Dataset.from_dict(qrels_data, features=qrels_features)

corpus_datadict = DatasetDict({"corpus": corpus_dataset, "queries": queries_dataset})
qrels_datadict = DatasetDict({"test": qrels_dataset})

corpus_datadict.save_to_disk("scripts/data/germanquad/corpus")
qrels_datadict.save_to_disk("scripts/data/germanquad/qrels")
