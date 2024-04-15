"""Example script for benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = []

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = []

TASK_LIST_RERANKING = []

TASK_LIST_RETRIEVAL = ["Ko-StrategyQA", "Ko-miracl"]

TASK_LIST_STS = []

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)

for task in TASK_LIST:
    logger.info(f"Running task: {task}")
    evaluation = MTEB(
        tasks=[task],
        task_langs=["ko"],
    )
    evaluation.run(model, output_folder=f"results/{model_name}")
