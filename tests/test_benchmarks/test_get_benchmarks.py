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


def _custom_grouping(benchmark: mteb.Benchmark, name: str) -> CustomGrouping:
    for agg in benchmark.aggregations:
        if isinstance(agg, CustomGrouping) and agg.name == name:
            return agg
    raise AssertionError(f"{benchmark.name!r} has no CustomGrouping named {name!r}")


def test_lmeb_memory_grouping_covers_all_tasks():
    """Every LMEB task is claimed by exactly one 'Memory Type' group."""
    benchmark = mteb.get_benchmark("LMEB")
    grouping = _custom_grouping(benchmark, "Memory Type")

    claimed = [t.metadata.name for g in grouping.groups for t in g.tasks]
    assert len(claimed) == len(set(claimed)), "a task is claimed by >1 group"
    assert set(claimed) == {t.metadata.name for t in benchmark.tasks}
    assert not grouping.has_scoped_refs


def test_bright_document_length_grouping_covers_all_tasks():
    """Every BRIGHT(v1.1) task is claimed by exactly one 'Document Length' group."""
    benchmark = mteb.get_benchmark("BRIGHT(v1.1)")
    grouping = _custom_grouping(benchmark, "Document Length")

    claimed = [t.metadata.name for g in grouping.groups for t in g.tasks]
    assert len(claimed) == len(set(claimed)), "a task is claimed by >1 group"
    assert set(claimed) == {t.metadata.name for t in benchmark.tasks}
    assert not grouping.has_scoped_refs


def test_long_embed_context_length_grouping_covers_all_splits():
    """Every one of LEMBNeedleRetrieval/LEMBPasskeyRetrieval's 8 eval_splits
    is claimed by exactly one 'Context Length' group (16 claims total)."""
    benchmark = mteb.get_benchmark("LongEmbed")
    grouping = _custom_grouping(benchmark, "Context Length")
    assert grouping.has_scoped_refs

    claimed: set[tuple[str, str]] = set()
    total_claims = 0
    for g in grouping.groups:
        for t in g.tasks:
            assert t.metadata.name in {"LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"}
            for split in t.eval_splits:
                claimed.add((t.metadata.name, split))
                total_claims += 1
    assert total_claims == 16, "expected 8 groups x 2 tasks = 16 claims"
    assert len(claimed) == total_claims, "a (task, split) pair is claimed twice"

    needle_splits = {split for name, split in claimed if name == "LEMBNeedleRetrieval"}
    passkey_splits = {
        split for name, split in claimed if name == "LEMBPasskeyRetrieval"
    }
    expected_splits = {
        "test_256",
        "test_512",
        "test_1024",
        "test_2048",
        "test_4096",
        "test_8192",
        "test_16384",
        "test_32768",
    }
    assert needle_splits == expected_splits
    assert passkey_splits == expected_splits
