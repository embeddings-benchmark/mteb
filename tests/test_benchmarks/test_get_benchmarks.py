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
    on_leaderboard = "MTEB(Multilingual, v2)"
    not_on_leaderboard = "MTEB(Multilingual, v1)"
    benchmark = mteb.get_benchmarks(display_on_leaderboard=True)
    names = {b.name for b in benchmark}
    assert on_leaderboard in names
    assert not_on_leaderboard not in names

    benchmark = mteb.get_benchmarks(display_on_leaderboard=False)
    names = {b.name for b in benchmark}
    assert on_leaderboard not in names
    assert not_on_leaderboard in names
