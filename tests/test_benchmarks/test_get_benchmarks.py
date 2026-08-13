import logging

import pytest

import mteb
from mteb.benchmarks.benchmark import CustomGrouping

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "name", ["MTEB(eng, v1)", "MTEB(rus, v1)", "MTEB(Scandinavian, v1)"]
)
def test_get_benchmark(name):
    benchmark = mteb.get_benchmark(benchmark_name=name)
    assert isinstance(benchmark, mteb.Benchmark)


@pytest.mark.parametrize(
    "alias, full_name",
    [
        (
            "MTEB(eng, classic)",
            "MTEB(eng, v1)",
        ),
        ("MTEB(rus)", "MTEB(rus, v1)"),
        ("MTEB(Scandinavian)", "MTEB(Scandinavian, v1)"),
    ],
)
def test_benchmark_aliases(alias, full_name):
    benchmark = mteb.get_benchmark(benchmark_name=alias)
    assert benchmark.name == full_name
    assert isinstance(benchmark, mteb.Benchmark)
    assert alias in benchmark.aliases


def test_benchmark_on_leaderboard():
    on_leaderboard = "RTEB(eng, beta)"
    not_on_leaderboard = "MTEB(Multilingual, v1)"
    benchmark = mteb.get_benchmarks(display_on_leaderboard=True)
    names = {b.name for b in benchmark}
    assert on_leaderboard in names
    assert not_on_leaderboard not in names

    benchmark = mteb.get_benchmarks(display_on_leaderboard=False)
    names = {b.name for b in benchmark}
    assert on_leaderboard not in names
    assert not_on_leaderboard in names


def test_lmeb_memory_grouping_covers_all_tasks():
    """LMEB's "Memory Type" CustomGrouping (issue #4898) should partition
    every one of its tasks into exactly one of the four memory-type groups,
    with no task left ungrouped and no group left without a description."""
    benchmark = mteb.get_benchmark("LMEB")
    task_names = {t.metadata.name for t in benchmark.tasks}

    grouping = next(a for a in benchmark.aggregations if isinstance(a, CustomGrouping))
    assert grouping.name == "Memory Type"
    assert set(grouping.task_to_label) == task_names
    assert set(grouping.task_to_label.values()) == {
        "Episodic",
        "Dialogue",
        "Semantic",
        "Procedural",
    }
    for group in grouping.groups:
        assert group.description


def test_bright_document_length_grouping_covers_all_tasks():
    """BRIGHT(v1.1)'s "Document Length" CustomGrouping should partition every
    task into Short or Long by whether its name carries the "Long" variant
    suffix, with no task left ungrouped."""
    benchmark = mteb.get_benchmark("BRIGHT(v1.1)")
    task_names = {t.metadata.name for t in benchmark.tasks}

    grouping = next(a for a in benchmark.aggregations if isinstance(a, CustomGrouping))
    assert grouping.name == "Document Length"
    assert set(grouping.task_to_label) == task_names
    assert set(grouping.task_to_label.values()) == {"Short", "Long"}
    for group in grouping.groups:
        assert group.description
        is_long_group = group.label == "Long"
        for task_name in group.tasks:
            assert ("Long" in task_name) == is_long_group
