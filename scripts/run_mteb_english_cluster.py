"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging

from mteb import MTEB, get_model

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLUSTERING = [
    # "ArxivClusteringP2P",  # hierarchical
    # "ArxivClusteringS2S",  # hierarchical
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST = [x + ".v2" for x in TASK_LIST_CLUSTERING] + TASK_LIST_CLUSTERING

model = get_model(
    model_name="intfloat/multilingual-e5-base",
    revision="d13f1b27baf31030b7fd040960d60d909913633f",
)

eval_splits = ["test"]
evaluation = MTEB(
    tasks=TASK_LIST_CLUSTERING, task_langs=["en"]
)  # Remove "en" for running all languages
evaluation.run(
    model, output_folder="results", eval_splits=eval_splits, overwrite_results=True
)
