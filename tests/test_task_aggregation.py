from __future__ import annotations

import mteb
import mteb.task_aggregation as task_aggregation
from mteb.load_results.benchmark_results import BenchmarkResults

# define some test data
bitext1_1 = mteb.TaskResult(
    dataset_revision="test_rev",
    task_name="BornholmBitextMining",
    mteb_version="test_version",
    evaluation_time=1,
    scores={"test": [{"main_score": 1, "hf_subset": "NaN", "languages": ["eng-Latn"]}]},
)

bitext1_2 = mteb.TaskResult(
    dataset_revision="test_rev",
    task_name="BornholmBitextMining",
    mteb_version="test_version",
    evaluation_time=1,
    scores={"test": [{"main_score": 2, "hf_subset": "NaN", "languages": ["eng-Latn"]}]},
)

classification1_1 = mteb.TaskResult(
    dataset_revision="test_rev",
    task_name="Banking77Classification",
    mteb_version="test_version",
    evaluation_time=1,
    scores={"test": [{"main_score": 1, "hf_subset": "NaN", "languages": ["eng-Latn"]}]},
)

classification1_2 = mteb.TaskResult(
    dataset_revision="test_rev",
    task_name="Banking77Classification",
    mteb_version="test_version",
    evaluation_time=1,
    scores={"test": [{"main_score": 2, "hf_subset": "NaN", "languages": ["eng-Latn"]}]},
)

classification2_1 = mteb.TaskResult(
    dataset_revision="test_rev",
    task_name="AfriSentiClassification",
    mteb_version="test_version",
    evaluation_time=1,
    scores={"test": [{"main_score": 1, "hf_subset": "NaN", "languages": ["eng-Latn"]}]},
)

mteb_results = {
    "model1": {
        "rev1": [bitext1_1, classification1_2, classification2_1],
        "rev2": [bitext1_1, classification1_1, classification2_1],
    },
    "model2": {
        "rev1": [bitext1_2, classification1_2, classification2_1],
        "rev2": [bitext1_2, classification1_1, classification2_1],
    },
}
mteb_results = BenchmarkResults.from_legacy_dict(mteb_results)


def test_mean():
    expected = {
        "model1": {
            "rev1": {"mean": 4 / 3},  # (1 + 2 + 1) / 3
            "rev2": {"mean": 1},  # (1 + 1 + 1) / 3
        },
        "model2": {
            "rev1": {"mean": 5 / 3},  # (2 + 2 + 1) / 3
            "rev2": {"mean": 4 / 3},  # (2 + 1 + 1) / 3
        },
    }

    assert task_aggregation.mean(mteb_results) == expected


def test_task_category_weighted_mean():
    expected = {
        "model1": {
            "rev1": {
                "mean (BitextMining)": 1.0,
                "mean (Classification)": 1.5,
                "mean (weighted by task type)": 1.25,  # ( 1/1 + (2 + 1) / 2 ) / 2
            },
            "rev2": {
                "mean (BitextMining)": 1.0,
                "mean (Classification)": 1.0,
                "mean (weighted by task type)": 1.0,  # ( 1/1 + (1 + 1) / 2 ) / 2
            },
        },
        "model2": {
            "rev1": {
                "mean (BitextMining)": 2.0,
                "mean (Classification)": 1.5,
                "mean (weighted by task type)": 1.75,  # ( 2/1 + (2 + 1) / 2 ) / 2
            },
            "rev2": {
                "mean (BitextMining)": 2.0,
                "mean (Classification)": 1.0,
                "mean (weighted by task type)": 1.5,  # ( 2/1 + (1 + 1) / 2 ) / 2
            },
        },
    }

    assert task_aggregation.task_category_weighted_mean(mteb_results) == expected


def test_borda_count_simple():
    mteb_results_simple = BenchmarkResults.from_legacy_dict(
        {
            "model1": {
                "rev1": [bitext1_1],
            },
            "model2": {
                "rev2": [bitext1_2],
            },
        }
    )
    expected = {
        "model1": {
            "rev1": {"borda_count": 0},
        },
        "model2": {
            "rev2": {"borda_count": 1},
        },
    }
    assert task_aggregation.borda_count(mteb_results_simple) == expected


def test_borda_count_simple_with_tie():
    mteb_results_simple_with_tie = {
        "model1": {
            "rev1": [bitext1_1],
            "rev2": [bitext1_1],
        },
        "model2": {
            "rev1": [bitext1_2],
            "rev2": [bitext1_2],
        },
    }
    expected = {
        "model1": {
            "rev1": {"borda_count": 0.5},
            "rev2": {"borda_count": 0.5},
        },
        "model2": {
            "rev1": {"borda_count": 2.5},
            "rev2": {"borda_count": 2.5},
        },
    }
    mteb_results_simple_with_tie = BenchmarkResults.from_legacy_dict(
        mteb_results_simple_with_tie
    )
    assert task_aggregation.borda_count(mteb_results_simple_with_tie) == expected


def test_borda_count_multiple_task_and_ties():
    # task 1: model1/rev1 == model2/rev1 > model1/rev2 == model2/rev2
    # task 2: model1/rev1 == model2/rev1 > model1/rev2 == model2/rev2
    # task 3: model1/rev1 == model2/rev1 == model1/rev2 == model2/rev2
    # given there is 4 candidates the max score is 3 (4 - 1)
    # we use tournament borda count so shared ranks get the average of the ranks they would have gotten

    expected = {
        "model1": {
            "rev1": {"borda_count": 0.5 + 2.5 + (6 / 4)},
            "rev2": {"borda_count": 0.5 + 0.5 + (6 / 4)},
        },
        "model2": {
            "rev1": {"borda_count": 2.5 + 2.5 + (6 / 4)},
            "rev2": {"borda_count": 2.5 + 0.5 + (6 / 4)},
        },
    }

    assert task_aggregation.borda_count(mteb_results) == expected
