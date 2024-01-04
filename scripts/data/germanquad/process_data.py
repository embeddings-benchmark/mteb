"""See clarin-knext/arguana-pl, clarin-knext/arguana-pl-qrels and
beir.datasets.data_loader_hf.HFDataLoader for BEIR format."""
import json
import os
from datasets import load_dataset, DatasetDict, Features, Value


dataset = load_dataset("deepset/germanquad")
dataset.pop("train")
# dataset["test"] = dataset["test"].select(range(0, len(dataset["test"]), 13)) # avoid duplicate contexts
# dataset["test"] = dataset["test"].select(range(400)) # for testing

PATH = "scripts/data/germanquad/"

corpus_file_path = PATH + "corpus.jsonl"
queries_file_path = PATH + "queries.jsonl"
os.makedirs(PATH + "qrels/", exist_ok=True)
qrels_file_path = PATH + "qrels/test.tsv"

# Deduplicate contexts and map them uniquely one-to-one to ids
context_to_id = {}

with open(corpus_file_path, 'w') as file:
    for item in dataset["test"]:
        # Check if the context is already in the dictionary
        if item["context"] not in context_to_id:
            context_to_id[item["context"]] = "c" + str(item["id"])
            entry = {
                "_id": context_to_id[item["context"]],
                "text": item["context"]
            }
            line = json.dumps(entry)
            file.write(line + '\n')

with open(queries_file_path, 'w') as file:
    for item in dataset["test"]:
        entry = {
            "_id": "q" + str(item["id"]),
            "text": item["question"]
        }
        line = json.dumps(entry)
        file.write(line + '\n')

# this maps queries to relevant documents
with open(qrels_file_path, 'w') as file:
    # qrels_data = []
    file.write('\t'.join(["query-id", "corpus-id", "score"]) + '\n')
    for item in dataset["test"]:
        # entry = {
        #     "query-id": str(item["id"]),
        #     "corpus-id": str(item["id"]),
        #     "score": 1
        # }
        # qrels_data.append(entry)
        corpus_id = context_to_id[item["context"]]
        file.write('\t'.join(["q"+str(item["id"]), corpus_id, "1"]) + '\n')
