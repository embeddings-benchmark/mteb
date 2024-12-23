from __future__ import annotations
import mteb

from functools import partial
from mteb.model_meta import ModelMeta, sentence_transformers_loader

# first we create a model meta card
cde_small_v1_meta = ModelMeta(
    loader=partial(
        name="jxm/cde-small-v1",
        revision="6a8c2f9f0a8184480f2e58f7d1413320b7b6392d",
        model_prompts={"query": "search_query: ", "passage": "search_document: "}; 
    ),

    name="jxm/cde-small-v1",
   
    revision="6a8c2f9f0a8184480f2e58f7d1413320b7b6392d", # use latest commit SHA as revision
   
    release_date="2024-10-01",  # First public commit was Oct 30 and on HF it says "As of October 1st, 2024"
   
    languages=["eng-Latn"],  # Assuming English, as it's not explicitly stated
   
    n_parameters=281_000_000,  # Model size is said to be 281 M params
    memory_usage=None,  # Not specified in the search results
    
    max_tokens=512,  # "max_length=512" mentioned in the code examples
    embed_dim=None,  # Not specified in the search results
    
    license=None,  # Not specified in the search results
    open_weights=True,  # Assuming open weights as it's available on Hugging Face
    
    public_training_data=None,  # Not specified in the search results
    public_training_code=True,  # Code link provided: "github.com/jxmorris12/cde"
    
    framework=["Sentence Transformers", "PyTorch"],  # Based on the provided code examples
    
    reference="https://huggingface.co/jxm/cde-small-v1",
    
    similarity_fn_name="cosine",  # Based on the use of cosine similarity in examples
    
    use_instructions=True,  # Uses prefixes for queries and documents
    
    zero_shot_benchmarks=["MTEB"]  # Mentioned in the model description

)





#temporarily commented out below for clarity


# model = mteb.get_model(
#     "jxm/cde-small-v1",
#     trust_remote_code=True,
#     model_prompts={"query": "search_query: ", "passage": "search_document: "},
# )
# tasks = mteb.get_tasks(
#     tasks=[
#         # classification
#         "AmazonCounterfactualClassification",
#         # clustering
#         "RedditClustering",
#         # pair classification
#         "TwitterSemEval2015",
#         # reranking
#         "AskUbuntuDupQuestions",
#         # retrieval
#         "SCIDOCS",
#         #         # sts
#         "STS22",
#         #         # summarization
#         "SummEval",
#     ]
# )
# evaluation = mteb.MTEB(tasks=tasks)
# results = evaluation.run(
#     model,
#     output_folder="results",
#     extra_kwargs={"batch_size": 8},
#     overwrite_results=True,
# )
