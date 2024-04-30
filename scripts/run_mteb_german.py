"""Example script for benchmarking German Context models."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonReviewsClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
]

TASK_LIST_CLUSTERING = [
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "TenKGnadClusteringP2P",
    "TenKGnadClusteringS2S",
]

TASK_LIST_PAIR_CLASSIFICATION = ["FalseFriendsGermanEnglish", "PawsX"]

TASK_LIST_RERANKING = ["MIRACL"]

TASK_LIST_RETRIEVAL = ["GermanQuAD-Retrieval", "GermanDPR", "XMarketDE", "GerDaLIR"]

TASK_LIST_STS = ["GermanSTSBenchmark", "STS22"]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=TASK_LIST, task_langs=["de"])
evaluation.run(
    model,
    overwrite_results=True,
    output_folder=f"results/de/{model_name.split('/')[-1]}",
)
