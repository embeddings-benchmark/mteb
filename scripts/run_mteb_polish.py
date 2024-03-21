"""Example script for benchmarking all datasets constituting the MTEB Polish leaderboard & average scores."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

classification_tasks = [
    "CBD",
    "PolEmo2.0-IN",
    "PolEmo2.0-OUT",
    "AllegroReviews",
    "PAC",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
]

clustering_tasks = ["8TagsClustering"]

pair_classification_tasks = ["SICK-E-PL", "PPC", "CDSC-E", "PSC"]

sts_tasks = ["SICK-R-PL", "CDSC-R", "STS22"]

tasks = classification_tasks + clustering_tasks + pair_classification_tasks + sts_tasks

model_name = "sdadas/st-polish-paraphrase-from-distilroberta"
model = SentenceTransformer(model_name)

evaluation = MTEB(tasks=tasks, task_langs=["pl"])
evaluation.run(model, output_folder=f"results/pl/{model_name.split('/')[-1]}")
