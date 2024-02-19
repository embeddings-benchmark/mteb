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
corpus_raw = datasets.load_dataset("lyon-nlp/alloprof", "documents")
# Download the queries
queries_raw = datasets.load_dataset("lyon-nlp/alloprof", "queries")
# Get the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate document text (title + content)
corpus = corpus_raw.map(lambda x: {"text": x["title"] + "\n\n" + x["text"]})
# Embed documents and queries
corpus = corpus.map(lambda x: {"embeddings": model.encode(x['text'])}, batched=True)
queries = queries_raw.map(lambda x: {"embeddings": model.encode(x["text"])}, batched=True)

# change document uuid with integer id
doc_name_id_mapping = {doc["uuid"]: i for i, doc in enumerate(corpus["documents"])}
corpus = corpus.map(lambda x: {"uuid" : doc_name_id_mapping[x["uuid"]]})
queries = queries.map(lambda x: {"relevant": [doc_name_id_mapping[r] for r in x["relevant"]]})

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

top_docs_train = retrieve_documents(queries["train"]["embeddings"], corpus["documents"]["embeddings"])
top_docs_test =  retrieve_documents(queries["test"]["embeddings"], corpus["documents"]["embeddings"])
queries["train"] = queries["train"].map(
    lambda _, i: {"top_cosim_values": top_docs_train.values[i], "top_cosim_indexes": top_docs_train.indices[i]},
    with_indices=True
    )
queries["test"] = queries["test"].map(
    lambda _, i: {"top_cosim_values": top_docs_test.values[i], "top_cosim_indexes": top_docs_test.indices[i]},
    with_indices=True
    )

# Remove id in best_indices if it corresponds to ground truth a relevant document
queries = queries.map(lambda x : {"top_cosim_indexes": [i for i in x["top_cosim_indexes"] if i not in x["relevant"]]})
# Convert document ids to document texts based on the corpus
queries = queries.map(lambda x: {"negative": [corpus["documents"][i]["text"] for i in x["top_cosim_indexes"]]})
queries = queries.map(lambda x: {"positive": [corpus["documents"][i]["text"] for i in x["relevant"]]})

# Format as the MTEB format
queries = queries.rename_column("text", "query")
dataset = queries.remove_columns(['embeddings', 'relevant', 'top_cosim_values', 'top_cosim_indexes', 'answer', 'subject', "id"])
# Rename the key of dataset key as "test"
# dataset["test"] = dataset.pop("queries")

# create HF repo
repo_id = "lyon-nlp/mteb-fr-reranking-alloprof-s2p"
try:
    create_repo(repo_id, repo_type="dataset")
except HfHubHTTPError as e:
    print("HF repo already exist")

# save dataset as json
dataset.push_to_hub(repo_id)
