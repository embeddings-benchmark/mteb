from __future__ import annotations

import math

import mteb
from mteb.results.task_result import TaskResult
from mteb.tasks.aggregated_tasks.eng.xmod_bench import XModBench
from mteb.types import PromptType

_REVISION = "db1d4695d359be83e8fa34575970c6d9c58dbfb4"
_EXPECTED_MODALITIES = {
    "at2t": (["audio", "text"], ["text"]),
    "at2i": (["audio", "text"], ["image"]),
    "at2v": (["audio", "text"], ["video"]),
    "t2a": (["text"], ["audio"]),
    "t2i": (["text"], ["image"]),
    "t2v": (["text"], ["video"]),
    "it2a": (["image", "text"], ["audio"]),
    "vt2a": (["video", "text"], ["audio"]),
    "it2t": (["image", "text"], ["text"]),
    "vt2t": (["video", "text"], ["text"]),
}


def _result(task_name: str, direction: str, score: float) -> TaskResult:
    return TaskResult(
        dataset_revision=_REVISION,
        task_name=task_name,
        mteb_version="2.19.5",
        scores={
            "test": [
                {
                    "hf_subset": direction,
                    "languages": ["eng-Latn", "zho-Hans"],
                    "accuracy": score,
                    "main_score": score,
                }
            ]
        },
        evaluation_time=1.0,
    )


def test_xmodbench_is_registered_as_one_aggregate():
    task = mteb.get_task("XModBench")

    assert isinstance(task, XModBench)
    assert len(task.tasks) == 10
    assert task.metadata.main_score == "accuracy"
    assert set(task.metadata.modalities) == {"audio", "image", "text", "video"}


def test_xmodbench_directions_have_exact_modality_metadata():
    task = XModBench()

    directions = set()
    for child in task.tasks:
        direction = child.metadata.hf_subsets[0]
        directions.add(direction)
        query_modalities, document_modalities = _EXPECTED_MODALITIES[direction]

        assert child.metadata.dataset == {
            "path": "jupyterjazz/XModBench-MTEB",
            "revision": _REVISION,
        }
        assert child.metadata.category == direction
        assert child.metadata.main_score == "accuracy"
        assert child.metadata.get_modalities(PromptType.query) == query_modalities
        assert child.metadata.get_modalities(PromptType.document) == document_modalities

    assert directions == set(_EXPECTED_MODALITIES)


def test_xmodbench_aggregate_is_weighted_by_query_count():
    task = XModBench()
    results = [
        _result(
            child.metadata.name,
            child.metadata.hf_subsets[0],
            float(child.metadata.name == "XModBenchAT2IRetrieval"),
        )
        for child in task.tasks
    ]

    scores = task.task_results_to_scores(results)

    expected = 617 / 5_981
    assert math.isclose(scores["test"]["default"]["accuracy"], expected)
    assert math.isclose(scores["test"]["default"]["main_score"], expected)


def test_xmodbench_aggregate_requires_every_direction():
    task = XModBench()
    child = task.tasks[0]
    result = _result(child.metadata.name, child.metadata.hf_subsets[0], 1.0)

    scores = task.task_results_to_scores([result])

    assert scores["test"]["default"]["accuracy"] is None
    assert scores["test"]["default"]["main_score"] is None
