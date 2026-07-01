import pytest

pytest.importorskip("gradio", reason="Gradio not installed")

from mteb.leaderboard.app import _resolve_benchmark_name_from_query
from mteb.leaderboard.benchmark_selector import DEFAULT_BENCHMARK_NAME


def test_resolve_benchmark_name_from_query_defaults_when_missing():
    assert _resolve_benchmark_name_from_query(None) == DEFAULT_BENCHMARK_NAME


def test_resolve_benchmark_name_from_query_defaults_when_invalid():
    assert (
        _resolve_benchmark_name_from_query("not-a-real-benchmark")
        == DEFAULT_BENCHMARK_NAME
    )


def test_resolve_benchmark_name_from_query_defaults_when_not_on_leaderboard():
    assert (
        _resolve_benchmark_name_from_query("MTEB(Multilingual, v1)")
        == DEFAULT_BENCHMARK_NAME
    )


def test_resolve_benchmark_name_from_query_keeps_valid_name():
    assert _resolve_benchmark_name_from_query("MTEB(eng, v1)") == "MTEB(eng, v1)"


def test_resolve_benchmark_name_from_query_normalizes_alias():
    assert _resolve_benchmark_name_from_query("MTEB(eng, classic)") == "MTEB(eng, v1)"
