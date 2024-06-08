"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
from pathlib import Path

from scipy import stats

from mteb import get_model_meta
from mteb.models.e5_models import e5_mult_base, e5_mult_large, e5_mult_small
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

version = "v3-2"
TASK_LIST = [x + f".{version}" for x in TASK_LIST_CLUSTERING] + TASK_LIST_CLUSTERING

MODELS = [
    e5_mult_small,
    e5_mult_base,
    e5_mult_large,
]

for model in MODELS:
    model_name = model.name
    revision = model.revision

    main_scores = []
    main_scores_fast = []
    times = []
    times_fast = []
    for task in TASK_LIST:
        model_meta = get_model_meta(model_name=model_name, revision=revision)
        model_path_name = model_meta.model_name_as_path()
        output_path = Path("./results") / model_path_name / revision
        results_path = output_path / f"{task}.json"
        res = MTEBResults.from_disk(path=results_path, load_historic_data=False)
        main_score = res.scores["test"][0]["main_score"]
        eval_time = res.evaluation_time
        if version in res.task_name:
            main_scores_fast.append(main_score)
            times_fast.append(eval_time)
        else:
            main_scores.append(main_score)
            times.append(eval_time)

    ## Spearman score
    print(f"{model_name} | {revision}")
    print(stats.spearmanr(main_scores, main_scores_fast).statistic)

    ## speed up
    speedup = sum(times) / sum(times_fast)
    print(f"Speedup: {speedup:.2f}x\n")
