"""Example script for benchmarking all datasets constituting the Retrieval Law leaderboard & average scores"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_RETRIEVAL_LAW = [
    # "LegalSummarization",
    # "LegalBenchConsumerContractsQA",
    # "LegalBenchCorporateLobbying",
    # "AILACasedocs",
    # "AILAStatutes",
    # "LeCaRDv2",
    # "LegalQuAD",
    # "GerDaLIRSmall",
    # "WikipediaRetrievalDE",
    # "WikipediaRerankingDE",
    "WikipediaRerankingBN",
    # "WikipediaRetrievalBN",
    # "SciDocsRR",
]

model_name = "average_word_embeddings_komninos"
# model_name = "deepset/gbert-base"
# model_name = "intfloat/e5-base"
model = SentenceTransformer(model_name)
# model = SentenceTransformer(model_name, device="cpu")

for task in TASK_LIST_RETRIEVAL_LAW:
    logger.info(f"Running task: {task}")
    eval_splits = ["test"]
    evaluation = MTEB(tasks=[task])
    evaluation.run(
        model, output_folder=f"results/{model_name}", eval_splits=eval_splits
    )
