"""Example script for benchmarking all datasets constituting the MTEB Chinese leaderboard & average scores"""

from __future__ import annotations

import functools
import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


model_name = "BAAI/bge-large-zh-noinstruct"
model = SentenceTransformer(model_name)
# normalize_embeddings should be true for this model
model.encode = functools.partial(model.encode, normalize_embeddings=True)

evaluation = MTEB(task_langs=["zh", "zh-CN"])
evaluation.run(model, output_folder=f"results/zh/{model_name.split('/')[-1]}")
