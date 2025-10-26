import os

from datasets import Audio, Dataset, DatasetDict, load_dataset
from huggingface_hub import create_repo
from tqdm import tqdm

WRITE_TOK = os.environ["HF_TOKEN"]

ds = load_dataset("CLAPv2/SoundDescs", split="test")

# at2
queries_ = {"id": [], "modality": [], "audio": []}
corpus_ = {"id": [], "modality": [], "text": []}
relevant_docs_ = {"query-id": [], "corpus-id": [], "score": []}

qid = {}
did = {}

for row in tqdm(ds, total=len(ds)):
    text = row["text"]
    row_index = row["index"]
    audio_path = (
        "/Users/isaac/work/audio-retrieval-benchmark/audio_data/audios/"
        + row_index
        + f"/{row_index}.wav"
    )
    query_id = f"q-{row_index}"

    if query_id not in qid:
        qid[query_id] = query_id
        queries_["id"].append(query_id)
        queries_["audio"].append(audio_path)
        queries_["modality"].append("audio")

    doc_id = f"d-{row_index}"
    if doc_id not in did:
        did[doc_id] = doc_id
        corpus_["id"].append(doc_id)
        corpus_["text"].append(text)
        corpus_["modality"].append("text")

    relevant_docs_["query-id"].append(query_id)
    relevant_docs_["corpus-id"].append(doc_id)
    relevant_docs_["score"].append(1)

corpus = Dataset.from_dict(corpus_)
queries = Dataset.from_dict(queries_).cast_column("audio", Audio())
relevant_docs = Dataset.from_dict(relevant_docs_)

corpus = DatasetDict({"corpus": corpus})
queries = DatasetDict({"test": queries})
relevant_docs = DatasetDict({"test": relevant_docs})


repo_name = "mteb/sounddescs_a2t"
create_repo(repo_name, repo_type="dataset", token=WRITE_TOK)

corpus.push_to_hub(repo_name, "corpus", token=WRITE_TOK)
queries.push_to_hub(repo_name, "query", token=WRITE_TOK)
relevant_docs.push_to_hub(repo_name, "qrels", token=WRITE_TOK)
