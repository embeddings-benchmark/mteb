"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from mteb import get_model_meta
from mteb.models.e5_models import (
    e5_eng_small,
    e5_eng_small_v2,
    e5_mult_base,
    e5_mult_large,
    e5_mult_small,
)
from mteb.models.sentence_transformers_models import (
    all_MiniLM_L6_v2,
    paraphrase_multilingual_MiniLM_L12_v2,
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


for version in versions:
    task_list = [(x, x + f".{version}") for x in TASK_LIST_CLUSTERING]

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
                output_path = (
                    Path(__file__).parent / "../results" / model_path_name / revision
                )
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

        mean_scores = {k: np.mean(v) for k, v in scores.items()}

        model_order = list(m for m in sorted(mean_scores, key=lambda x: mean_scores[x]))

        import pandas as pd

        df = pd.DataFrame({k: scores[k] for k in model_order}).T
        df_fast = pd.DataFrame({k: scores_fast[k] for k in model_order}).T
        df["Model"] = df.index
        df_fast["Model"] = df_fast.index

        # plot the scores

        import seaborn as sns

        sns.set_theme(style="whitegrid")
        df_long = df.melt(id_vars="Model", var_name="Measure", value_name="Score")
        df_long["Type"] = "Original"
        df_long_fast = df_fast.melt(
            id_vars="Model", var_name="Measure", value_name="Score"
        )
        df_long_fast["Type"] = "Fast"

        df_long = pd.concat([df_long, df_long_fast])

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))

        # plot the points
        sns.stripplot(
            data=df_long,
            x="Model",
            y="Score",
            hue="Type",
            dodge=True,
            alpha=0.5,
        )
        # title
        plt.title(f"{task_pair[0]}")

        # with xticks rotated
        plt.xticks(rotation=85)
        plt.tight_layout()
        # plt.show()

        # save the plot
        save_path = Path(__file__).parent / f"task_plot_{task_pair[0]}.png"
        plt.savefig(save_path)