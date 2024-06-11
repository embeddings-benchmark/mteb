"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import stats

from mteb import get_model_meta
from mteb.models.e5_models import e5_mult_base, e5_mult_large, e5_mult_small, e5_eng_small, e5_eng_small_v2
from mteb.models.sentence_transformers_models import (
    paraphrase_multilingual_MiniLM_L12_v2, all_MiniLM_L6_v2
)
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

MODELS = [
    paraphrase_multilingual_MiniLM_L12_v2,
    all_MiniLM_L6_v2,
    e5_eng_small,
    e5_eng_small_v2,
    e5_mult_small,
    e5_mult_base,
    e5_mult_large,
]

versions = [
    "v2",
]


def test_significance(dist1, dist2, n_samples=10_000):
    """Test if two distributions are significantly different using a bootstrapping method

    # sanity check:
    # should be significant
    dist1=np.random.normal(1, 1, 10)
    dist2=np.random.normal(4, 1, 10)
    n_samples=10_000
    test_significance(dist1, dist2, n_samples)
    # should not be significant
    test_significance(np.random.normal(1, 1, 10), np.random.normal(1, 1, 10), n_samples=10_000)
    """
    # Compute the difference between the two distributions
    diff = abs(np.mean(dist1) - np.mean(dist2))

    # Concatenate the two distributions
    concat = np.concatenate([dist1, dist2])

    # Initialize the bootstrap samples
    bootstrap_diff = np.zeros(n_samples)

    # Generate the bootstrap samples
    for i in range(n_samples):
        # Generate a bootstrap sample
        bootstrap_sample = np.random.choice(concat, size=len(concat), replace=True)

        # Compute the difference between the two bootstrap samples
        bootstrap_diff[i] = abs(
            np.mean(bootstrap_sample[: len(dist1)])
            - np.mean(bootstrap_sample[len(dist1) :])
        )

    # Compute the p-value
    # I.e. what is the probability of observing a difference as extreme as the one we observed
    # given that the null hypothesis is true (i.e. the two distributions are the same)
    p_value = np.mean(bootstrap_diff >= diff)
    # print(f"The p-value is: {p_value}")

    return p_value


def compute_significant_rank(scores: dict, threshold=0.05):
    """Compute significant rank for models.

    Example:
        ```
        scores = {"model1": np.random.normal(1, 1, 10) # 1 and 2 are similar
                "model2": np.random.normal(1.1, 1, 10)
                "model3": np.random.normal(5, 1, 10) # 3 is much better
        ranks = compute_significant_rank(scores)
        print(ranks)
        # {
        "models": ["model3", "model2", "model1"],
        # "significant rank": [1, 2, 2],
        # "rank": [1, 2, 3],
        # }
        ```
    """
    ranks = {}
    mean_scores = [(m, np.mean(s)) for m, s in scores.items()]
    mean_scores = sorted(mean_scores, key=lambda x: -x[1])  # higher is first

    ranks["models"], _ = zip(*mean_scores)
    ranks["models"] = list(ranks["models"])
    ranks["rank"] = list(range(1, len(ranks["models"]) + 1))

    pairs = [
        (ranks["models"][i], ranks["models"][i + 1])
        for i in range(len(mean_scores) - 1)
    ]

    rank = 1
    ranks["significant_rank"] = [rank]  # first model always get rank 1
    for p1, p2 in pairs:
        # test if the two models are significantly different
        p_value = test_significance(scores[p1], scores[p2])

        if p_value < threshold:
            rank += 1

        ranks["significant_rank"].append(rank)
    return ranks


for version in versions:
    task_list = [(x, x + f".{version}") for x in TASK_LIST_CLUSTERING]

    print(f"\n### {version}")

    print("| Model | Spearman | Significant Spearman | Speedup |")
    print("| --- | --- | --- | --- |")

    for task_pair in task_list:
        scores = {}
        scores_fast = {}
        times = []
        times_fast = []
        for task in task_pair:
            for model in MODELS:
                model_name = model.name
                revision = model.revision
                model_meta = get_model_meta(model_name=model_name, revision=revision)
                model_path_name = model_meta.model_name_as_path()
                output_path = Path("./results") / model_path_name / revision
                results_path = output_path / f"{task}.json"
                res = MTEBResults.from_disk(path=results_path, load_historic_data=False)
                main_score = res.scores["test"][0]["main_score"]
                eval_time = res.evaluation_time

                if version in res.task_name:
                    times_fast.append(eval_time)
                    scores_fast.update(
                        {
                            str(model.name).split("/")[-1]: res.scores["test"][0][
                                "v_measures"
                            ]["Level 0"]
                        }
                    )
                else:
                    times.append(eval_time)
                    scores.update(
                        {
                            str(model.name).split("/")[-1]: res.scores["test"][0][
                                "v_measures"
                            ]
                        }
                    )

        ## Spearman score and speed up
        main_score_sig_rank = compute_significant_rank(scores)
        main_score_fast_sig_rank = compute_significant_rank(scores_fast)

        sig_rank = []
        sig_rank_fast = []
        rank = []
        rank_fast = []

        # ensure they are the same order
        for model in scores:
            sig_rank.append(main_score_sig_rank["significant_rank"][main_score_sig_rank["models"].index(model)])
            sig_rank_fast.append(main_score_fast_sig_rank["significant_rank"][main_score_fast_sig_rank["models"].index(model)])
            rank.append(main_score_sig_rank["rank"][main_score_sig_rank["models"].index(model)])
            rank_fast.append(main_score_fast_sig_rank["rank"][main_score_fast_sig_rank["models"].index(model)])

        spearman = stats.spearmanr(rank, rank_fast).statistic
        sig_spearman = stats.spearmanr(sig_rank, sig_rank_fast).statistic
        speedup = sum(times) / sum(times_fast)
        print(f"| {task_pair[0]:<27} | {spearman:.4f} | {sig_spearman:.4f} | {speedup:.2f}x |")

        # print(f'classic | rank: {main_score_sig_rank["rank"]} | sig_rank: {main_score_sig_rank["significant_rank"]}')
        # print(f'fast    | rank: {main_score_fast_sig_rank["rank"]} | sig_rank: {main_score_fast_sig_rank["significant_rank"]}')
