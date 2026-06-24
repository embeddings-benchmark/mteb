"""Tests for AbsTaskAggregate"""

import logging

import mteb
from mteb.abstasks.aggregated_task import AbsTaskAggregate

logging.basicConfig(level=logging.INFO)


def test_is_aggregate_property_correct():
    tasks = mteb.get_tasks()

    for task in tasks:
        assert task.is_aggregate == isinstance(task, AbsTaskAggregate)


def test_dynamic_aggregation():
    # Verify that load_results dynamically aggregates results
    cache = mteb.ResultCache()
    res = cache.load_results(
        models=["ByteDance-Seed__Seed1.5-Embedding"],
        tasks=["MTEB(eng, v2)"],
    )
    assert len(res.model_results) > 0
    mr = res.model_results[0]
    task_names = [tr.task_name for tr in mr.task_results]

    assert "MTEB(eng, v2)" in task_names
    assert "ArguAna" not in task_names
