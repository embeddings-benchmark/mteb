"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
from pathlib import Path

from scipy import stats

from mteb import get_model_meta
from mteb.models.e5_models import e5_mult_base, e5_mult_small, e5_mult_large
from mteb.MTEBResults import MTEBResults

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

MODELS = [e5_mult_small, e5_mult_base, e5_mult_large]

for model in MODELS:
    model_name = model.name
    revision = model.revision

    main_scores = []
    main_scores_fast = []
    for task in TASK_LIST:
        model_meta = get_model_meta(model_name=model_name, revision=revision)
        model_path_name = model_meta.model_name_as_path()
        output_path = Path("./results") / model_path_name / revision
        results_path = output_path / f"{task}.json"
        res = MTEBResults.from_disk(path=results_path, load_historic_data=False)
        main_score = res.scores["test"][0]["main_score"]
        if "v2" in res.task_name:
            main_scores_fast.append(main_score)
        else:
            main_scores.append(main_score)

    ## intfloat/multilingual-e5-base = 0.8333
    print(f"{model_name} | {revision}")
    print(stats.spearmanr(main_scores, main_scores_fast).statistic, "\n")
