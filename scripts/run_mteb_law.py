"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_RETRIEVAL = [
    "LegalSummarization",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "AILACasedocs",
    "AILAStatutes",
    "LeCaRDv2",
    "LegalQuAD",
    "GerDaLIR",
]

model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)

for task in TASK_LIST_RETRIEVAL_LAW:
    logger.info(f"Running task: {task}")
    eval_splits = ["test"]
    evaluation = MTEB(
        tasks=[task]
    )
    evaluation.run(
        model, output_folder=f"results/{model_name}", eval_splits=eval_splits
    )
