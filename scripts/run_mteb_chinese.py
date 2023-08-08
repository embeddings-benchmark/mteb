"""Example script for benchmarking all datasets constituting the MTEB Chinese leaderboard & average scores"""

import logging
import functools

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


model_name = "BAAI/bge-large-zh-noinstruct"
model = SentenceTransformer(model_name)
# normalize_embeddings should be true for this model
model.encode = functools.partial(model.encode, normalize_embeddings=True)

evaluation = MTEB(task_langs=["zh"])
evaluation.run(model, output_folder=f"results/zh/{model_name}")
