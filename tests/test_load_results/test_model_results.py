"""Tests for the ModelResult class"""

from __future__ import annotations

import pandas as pd
import pytest

import mteb
from mteb.load_results import ModelResult
from mteb.load_results.task_results import TaskResult

# TODO: v2 ^ we probably want to refactor such that this import looks like
# from mteb.results import ModelResult, TaskResults


@pytest.fixture
def model_result() -> ModelResult:
    task_result = TaskResult(
        dataset_revision="1.0",
        task_name="BornholmBitextMining",  # just dummy results
        mteb_version="1.0.0",
        evaluation_time=100,
        scores={
            "train": [
                {
                    "main_score": 0.5,
                    "hf_subset": "en-de",
                    "languages": ["eng-Latn", "deu-Latn"],
                },
                {
                    "main_score": 0.6,
                    "hf_subset": "en-fr",
                    "languages": ["eng-Latn", "fra-Latn"],
                },
            ]
        },
    )

    return ModelResult(
        model_name="mock_model",
        model_revision="dummy",
        task_results=[task_result],
    )


def test_indexing(model_result: ModelResult) -> None:
    res = model_result[0]
    assert isinstance(res, TaskResult), (
        "indexing into the list should return a ModelResult"
    )


def test_utility_properties(
    model_result: ModelResult,
) -> None:
    mr = model_result
    assert isinstance(mr.task_names, list) and isinstance(mr.task_names[0], str)
    assert (
        isinstance(mr.languages, list)
        and isinstance(mr.languages[0], str)
        and "eng" in mr.languages  # known from mock data
    )
    assert isinstance(mr.task_types, list) and isinstance(mr.task_types[0], str)
    assert isinstance(mr.domains, list) and isinstance(mr.domains[0], str)


def test_select_tasks(
    model_result: ModelResult,
) -> None:
    tasks = [mteb.get_task("BornholmBitextMining")]
    mr = model_result.select_tasks(tasks=tasks)
    task_names = mr.task_names
    assert isinstance(task_names, list)
    assert len(task_names) == 1
    assert task_names[0] == "BornholmBitextMining"


def test_to_dataframe(
    model_result: ModelResult,  # noqa: F811
) -> None:
    mr = model_result
    required_columns = [
        "model_name",
        "task_name",
        "task_name",
        "score",
        "subset",
        "split",
    ]
    t1 = mr.to_dataframe(aggregation_level="subset", format="long")
    assert isinstance(t1, pd.DataFrame)
    assert all(col in t1.columns for col in required_columns), "Columns are missing"
    assert t1.shape[0] > 0, "Results table is empty"

    t2 = mr.to_dataframe(aggregation_level="split", format="long")
    assert all(
        col in t2.columns for col in required_columns if col not in ["subset"]
    ), "Columns are missing"
    assert "subset" not in t2.columns, "Subset column should not be present"
    assert t1.shape[0] >= t2.shape[0], (
        "Aggregation level 'split' should have more rows than 'subset'"
    )

    t3 = mr.to_dataframe(aggregation_level="task", format="long")
    assert all(
        col in t3.columns for col in required_columns if col not in ["subset", "split"]
    ), "Columns are missing"
    assert "subset" not in t3.columns, "Subset column should not be present"
    assert "split" not in t3.columns, "Split column should not be present"
    assert t2.shape[0] >= t3.shape[0], (
        "Aggregation level 'task' should have more rows than 'split'"
    )

    t4_wide = mr.to_dataframe(aggregation_level="task", format="wide")
    t4_long = mr.to_dataframe(aggregation_level="task", format="long")
    assert isinstance(t4_wide, pd.DataFrame)

    # we know it is only one task
    assert t4_wide[mr.model_name].tolist()[0] == t4_long["score"][0], (
        "The scores in wide and long format should be the same"
    )
