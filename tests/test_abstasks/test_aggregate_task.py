"""Tests for AbsTaskAggregate"""

import json
import logging
import pathlib

import mteb
from mteb.abstasks.aggregated_task import AbsTaskAggregate

logging.basicConfig(level=logging.INFO)


def test_is_aggregate_property_correct():
    tasks = mteb.get_tasks()

    for task in tasks:
        assert task.is_aggregate == isinstance(task, AbsTaskAggregate)


def test_dynamic_aggregation(tmp_path):
    # Verify that whether load_results dynamically aggregates results
    subtasks = [
        "Robust04InstructionRetrieval",
        "News21InstructionRetrieval",
        "Core17InstructionRetrieval",
    ]

    model_name = "test-model"
    revision = "test-revision"
    model_dir = tmp_path / "results" / f"test-org__{model_name}" / revision
    model_dir.mkdir(parents=True, exist_ok=True)

    model_meta = {
        "name": f"test-org/{model_name}",
        "revision": revision,
        "languages": ["eng"],
    }
    with pathlib.Path(model_dir / "model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f)

    for subtask in subtasks:
        result_data = {
            "dataset_revision": "test-dataset-rev",
            "evaluation_time": 1.0,
            "kg_co2_emissions": 0.1,
            "mteb_version": "1.0.0",
            "scores": {
                "test": [
                    {
                        "hf_subset": "default",
                        "languages": ["eng-Latn"],
                        "main_score": 0.8,
                    }
                ]
            },
            "task_name": subtask,
        }
        with pathlib.Path(model_dir / f"{subtask}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(result_data, f)

    cache = mteb.ResultCache(cache_path=tmp_path)
    res = cache.load_results(
        models=[f"test-org/{model_name}"],
        tasks=["FollowIR"],
        include_remote=False,
    )

    assert len(res.model_results) > 0
    mr = res.model_results[0]
    task_names = [tr.task_name for tr in mr.task_results]

    assert "FollowIR" in task_names
    assert "News21InstructionRetrieval" not in task_names
