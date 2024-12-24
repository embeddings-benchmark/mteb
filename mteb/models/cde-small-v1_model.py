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





#temporarily commented out below for clarity


model = mteb.get_model(
    "jxm/cde-small-v1",
    trust_remote_code=True,
    model_prompts={"query": "search_query: ", "passage": "search_document: "},
)
tasks = mteb.get_tasks(
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
        #         # sts
        "STS22",
        #         # summarization
        "SummEval",
    ]
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(
    model,
    output_folder="results",
    extra_kwargs={"batch_size": 8},
    overwrite_results=True,
)
