import logging

import pytest

import mteb

logging.basicConfig(level=logging.INFO)

LMEB_MEMORY_TASKS = {
    "LMEB-Episodic": {"EPBench", "KnowMeBench"},
    "LMEB-Dialogue": {
        "LoCoMo",
        "LongMemEval",
        "REALTALK",
        "TMD",
        "MemBench",
        "ConvoMem",
    },
    "LMEB-Semantic": {
        "QASPER",
        "NovelQA",
        "PeerQA",
        "CovidQA",
        "ESGReports",
        "LMEBMLDR",
        "LooGLE",
        "LMEB_SciFact",
    },
    "LMEB-Procedural": {
        "Gorilla",
        "ToolBench",
        "ReMe",
        "ProceduralMemBench",
        "MemGovern",
        "DeepPlanning",
    },
}


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


@pytest.mark.parametrize(("name", "expected_tasks"), LMEB_MEMORY_TASKS.items())
def test_lmeb_memory_benchmark(name, expected_tasks):
    benchmark = mteb.get_benchmark(name)

    assert benchmark.name == name
    assert {task.metadata.name for task in benchmark.tasks} == expected_tasks
    assert benchmark.display_on_leaderboard


def test_lmeb_aggregate_contains_all_memory_tasks():
    benchmark = mteb.get_benchmark("LMEB")
    expected_tasks = set().union(*LMEB_MEMORY_TASKS.values())

    assert benchmark.name == "LMEB"
    assert {task.metadata.name for task in benchmark.tasks} == expected_tasks
    assert len(benchmark.tasks) == 22
    assert benchmark.display_on_leaderboard


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
