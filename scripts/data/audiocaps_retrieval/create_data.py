import os

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm

WRITE_TOK = os.environ["HF_TOKEN"]

ds = load_dataset("OpenSound/AudioCaps", split="test")

# t2a
queries_ = {"id": [], "modality": [], "text": []}
corpus_ = {"id": [], "modality": [], "audio": []}
relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}

qid = {}
did = {}

for row in tqdm(ds, total=len(ds)):
    audio = row["audio"]
    text = row["caption"]
    query_id = str(row["audiocap_id"])

    if query_id not in qid:
        qid[query_id] = query_id
        queries_["id"].append(query_id)
        queries_["text"].append(text)
        queries_["modality"].append("text")

    doc_id = str(row["youtube_id"])
    if doc_id not in did:
        did[doc_id] = doc_id
        corpus_["id"].append(doc_id)
        corpus_["audio"].append(audio)
        corpus_["modality"].append("audio")

    relevant_docs_["query-id"].append(query_id)
    relevant_docs_["corpus-id"].append(doc_id)
    relevant_docs_["score"].append(1)

corpus = Dataset.from_dict(corpus_)
queries = Dataset.from_dict(queries_)
relevant_docs = Dataset.from_dict(relevant_docs_)

corpus = DatasetDict({"corpus": corpus})
queries = DatasetDict({"test": queries})
relevant_docs = DatasetDict({"test": relevant_docs})


repo_name = "mteb/audiocaps_t2a"
create_repo(repo_name, repo_type="dataset", token=WRITE_TOK)

corpus.push_to_hub(repo_name, "corpus", token=WRITE_TOK)
queries.push_to_hub(repo_name, "query", token=WRITE_TOK)
relevant_docs.push_to_hub(repo_name, "qrels", token=WRITE_TOK)
