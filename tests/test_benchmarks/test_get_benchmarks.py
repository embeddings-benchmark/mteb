import logging

import pytest

import mteb

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "name", ["MTEB(eng, v1)", "MTEB(rus, v1)", "MTEB(Scandinavian, v1)"]
)
def test_get_benchmark(name):
    benchmark = mteb.get_benchmark(benchmark_name=name)
    assert isinstance(benchmark, mteb.Benchmark)


@pytest.mark.parametrize(
    ("alias", "full_name"),
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


def test_mteb_code_v1_1_removes_only_code_trans_ocean_dl():
    v1 = mteb.get_benchmark("MTEB(Code, v1)")
    v1_1 = mteb.get_benchmark("MTEB(Code, v1.1)")

    v1_tasks = {task.metadata.name for task in v1.tasks}
    v1_1_tasks = {task.metadata.name for task in v1_1.tasks}

    assert v1_tasks - v1_1_tasks == {"CodeTransOceanDL"}
    assert not v1_1_tasks - v1_tasks
    assert v1.superseded_by == [v1_1.name]
    assert mteb.get_benchmark("MTEB(code)").name == v1.name

    leaderboard_names = {
        benchmark.name for benchmark in mteb.get_benchmarks(display_on_leaderboard=True)
    }
    assert v1.name not in leaderboard_names
    assert v1_1.name in leaderboard_names
