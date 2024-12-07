from __future__ import annotations

import mteb

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
