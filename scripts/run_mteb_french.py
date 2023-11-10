"""Example script for benchmarking all datasets constituting the MTEB French leaderboard & average scores"""

import logging

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = [
    "AmazonReviewsClassification",
    "MasakhaneClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
]

TASK_LIST_CLUSTERING = [
]

TASK_LIST_PAIR_CLASSIFICATION = [
]

TASK_LIST_RERANKING = [
]

TASK_LIST_RETRIEVAL = [
    "AlloprofRetrieval"
]

TASK_LIST_STS = [
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

model_name = "dangvantuan/sentence-camembert-base"
model = SentenceTransformer(model_name)

logger.info(f"Task list : {TASK_LIST}")
for task in TASK_LIST:
    logger.info(f"Running task: {task}")
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = MTEB(tasks=[task], task_langs=["fr"])  # Remove "fr" for running all languages
    evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=eval_splits)
