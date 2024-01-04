"""Example script for benchmarking German Context Retrieval models."""

import logging
import os

from mteb import MTEB
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

if os.path.exists("results/de/multilingual-e5-small/GermanQuAD-Retrieval.json"):
    os.remove("results/de/multilingual-e5-small/GermanQuAD-Retrieval.json")

if os.path.exists("results/pl/multilingual-e5-small/GermanQuAD-Retrieval.json"):
    os.remove("results/pl/multilingual-e5-small/GermanQuAD-Retrieval.json")


retrieval_tasks = ["GermanQuAD-Retrieval"]

tasks =  retrieval_tasks

model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=tasks)
evaluation.run(model, output_folder=f"results/pl/{model_name.split('/')[-1]}")
