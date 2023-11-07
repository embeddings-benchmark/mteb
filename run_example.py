"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging

from mteb.evaluation.MTEB import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

model_name = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(model_name)

# evaluation = MTEB(task_langs=["es"])
evaluation = MTEB(task_langs=["es"])

evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=["test"])