import datasets
from sentence_transformers import SentenceTransformer, util
import torch
from huggingface_hub import create_repo
from huggingface_hub.utils._errors import HfHubHTTPError

"""
To create a reranking dataset from the initial retrieval dataset, 
we use a model (sentence-transformers/all-MiniLM-L6-v2) to embed the queries and the documents.
We then compute the cosine similarity for each query and document.
For each query we get the topk articles, as we would for a retrieval task.
Each couple query-document is labeled as relevant if it was labeled like so in the retrieval dataset,
or irrelevant if it was not
"""
# Download the documents (corpus)
corpus_raw = datasets.load_dataset("lyon-nlp/mteb-fr-retrieval-syntec-s2p", "documents")
# Download the queries
queries_raw = datasets.load_dataset("lyon-nlp/mteb-fr-retrieval-syntec-s2p", "queries")
# Get the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate document text (title + content)
corpus = corpus_raw.map(lambda x: {"text": x["title"] + "\n\n" + x["content"]})
# Embed documents and queries
corpus = corpus.map(lambda x: {"embeddings": model.encode(x['text'])}, batched=True)
queries = queries_raw.map(lambda x: {"embeddings": model.encode(x["Question"])}, batched=True)

# change document id with index instead of name
doc_name_id_mapping = {doc["id"]: i for i, doc in enumerate(corpus["documents"])}
corpus = corpus.map(lambda x: {"doc_id" : doc_name_id_mapping[x["id"]]})
queries = queries.map(lambda x: {"doc_id": doc_name_id_mapping[x["Article"]]})


# Retrieve best documents by cosine similarity
def retrieve_documents(queries_embs, documents_embs, topk:int=10) -> torch.return_types.topk:
    """Finds the topk documents for each embed query among all the embed documents

    Args:
        queries_embs (_type_): the embedings of all queries of the dataset (dataset["queries"]["embeddings"])
        documents_embs (_type_): the embedings of all coprus of the dataset (dataset["corpus"]["embeddings"])
        topk (int, optional): The amount of top documents to retrieve. Defaults to 5.

    Returns:
        torch.return_types.topk : The topk object, with topk.values being the cosine similarities
            and the topk.indices being the indices of best documents for each queries
    """
    similarities = util.cos_sim(queries_embs, documents_embs)
    tops = torch.topk(similarities, k=topk, axis=1)

    return tops

top_docs = retrieve_documents(queries["queries"]["embeddings"], corpus["documents"]["embeddings"])
queries = queries.map(
    lambda _, i: {"top_cosim_values": top_docs.values[i], "top_cosim_indexes": top_docs.indices[i]},
    with_indices=True
    )

# Remove id in best_indices if it corresponds to ground truth
queries = queries.map(lambda x : {"top_cosim_indexes": [i for i in x["top_cosim_indexes"] if i != x["doc_id"]]})
# Convert document ids to document texts based on the corpus
queries = queries.map(lambda x: {"negative": [corpus["documents"][i]["text"] for i in x["top_cosim_indexes"]]})
queries = queries.map(lambda x: {"positive": [corpus["documents"][x["doc_id"]]["text"]]})

# Format as the MTEB format
queries = queries.rename_column("Question", "query")
dataset = queries.remove_columns(['Article', 'embeddings', 'doc_id', 'top_cosim_values', 'top_cosim_indexes'])
# Rename the key of dataset key as "test"
dataset["test"] = dataset.pop("queries")

# create HF repo
repo_id = "lyon-nlp/mteb-fr-reranking-syntec-s2p"
try:
    create_repo(repo_id, repo_type="dataset")
except HfHubHTTPError as e:
    print("HF repo already exist")

# save dataset as json
dataset.push_to_hub(repo_id)