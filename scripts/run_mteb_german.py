"""Example script for benchmarking German Context Retrieval models."""

import logging
import os

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

tasks = ["GermanQuAD-Retrieval"]

model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=tasks)
evaluation.run(model, overwrite_results=True, output_folder=f"results/de/{model_name.split('/')[-1]}")
