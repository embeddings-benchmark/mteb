from __future__ import annotations

import pandas as pd
import pytest

from mteb.benchmarks import MTEB_EN, MTEB_ENG_CLASSIC, Benchmark, MTEB_multilingual
from mteb.leaderboard.table import scores_to_tables
from mteb.leaderboard.utils import filter_models, load_results

# A couple of models to look for.
SHOULD_BE_THERE = [
    "NV-Embed-v2",
    "e5-mistral-7b-instruct",
    "text-embedding-005",
    "voyage-large-2-instruct",
    "cde-small-v2",
    "SONAR",
    "text-embedding-3-small",
    "GritLM-7B",
    "sentence-t5-xxl",
    "M2V_base_glove",
]
BENCHMARKS_TO_CHECK = [
    MTEB_EN,
    MTEB_ENG_CLASSIC,
    MTEB_multilingual,
]

all_results = load_results()


@pytest.mark.parametrize("benchmark", BENCHMARKS_TO_CHECK)
def test_leaderboard_tables(benchmark: Benchmark):
    benchmark_results = benchmark.load_results(
        base_results=all_results
    ).join_revisions()
    scores = benchmark_results.get_scores(format="long")
    all_models = list({entry["model_name"] for entry in scores})
    filtered_models = filter_models(
        all_models,
        benchmark_results.task_names,
        availability=None,
        compatibility=[],
        instructions=None,
        model_size=(None, None),
        zero_shot_setting="off",
    )
    _, per_task_table = scores_to_tables(
        [entry for entry in scores if entry["model_name"] in filtered_models],
    )
    per_task_df = pd.DataFrame(
        per_task_table.value["data"], columns=per_task_table.value["headers"]
    )
    models_in_table = set(per_task_df["Model"])
    for model in SHOULD_BE_THERE:
        assert model in models_in_table
