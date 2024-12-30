from __future__ import annotations
import mteb
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from functools import partial
from mteb.model_meta import ModelMeta, sentence_transformers_loader
import random


cde_small_v1_meta = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        name="jxm/cde-small-v1",
        revision="6a8c2f9f0a8184480f2e58f7d1413320b7b6392d",
        model_prompts={
            "query": "search_query: ",
            "passage": "search_document: ",
        }     
    ),

    name="jxm/cde-small-v1",
    revision="6a8c2f9f0a8184480f2e58f7d1413320b7b6392d", 
    release_date="2024-10-01",
    languages=["eng-Latn"],  
    n_parameters=281_000_000,  
    memory_usage=None,
    max_tokens=512, 
    embed_dim=None, 
    license="mit", 
    open_weights=True,  
    public_training_data=None,  
    public_training_code=True,  
    framework=["Sentence Transformers", "PyTorch"],  
    reference="https://huggingface.co/jxm/cde-small-v1",
    similarity_fn_name="cosine", 
    use_instructions=True,  

)

#implement the model
model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
model.prompts = {
    "query": "search_query: ",
    "passage": "search_document: ",# Use 'passage' instead of 'document' for consistency with MTEB
}

corpus_file = "random_strings_cde.txt"

with open(corpus_file, "r") as file:
    random_corpus = [line.strip() for line in file]

minicorpus_size = 512  
assert len(random_corpus) >= minicorpus_size, "Corpus size is smaller than required!"

minicorpus_docs = random.sample(random_corpus, k=minicorpus_size)

print("Generating dataset embeddings...")
dataset_embeddings = model.encode(
    minicorpus_docs,
    prompt_name="passage",
    convert_to_tensor=True
)

print("Dataset embeddings shape:", dataset_embeddings.shape)


tasks=[
        # classification
        "AmazonCounterfactualClassification",
        # clustering
        "RedditClustering",
        # pair classification
        "TwitterSemEval2015",
        # reranking
        "AskUbuntuDupQuestions",
        # retrieval
        "SCIDOCS",
        # semantic textual similarity
        "STS22",
        # summarization
        "SummEval",
    ]


evaluation = MTEB(tasks=tasks)

print("Running MTEB evaluation...")
results = evaluation.run(
    model=model,
    output_folder="results",
    extra_kwargs={"batch_size": 8},
    overwrite_results=True,
)