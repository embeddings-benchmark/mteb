"""Tests for the BenchmarkResult class"""

import os
from pathlib import Path

import pandas as pd
import pytest

import mteb
from mteb import Benchmark
from mteb.cache import ResultCache
from mteb.cli.generate_model_card import generate_model_card
from mteb.results import BenchmarkResults, ModelResult


@pytest.fixture
def cache_path() -> Path:
    tests_path = Path(__file__).parent.parent / "mock_mteb_cache"

    os.environ["MTEB_CACHE"] = str(tests_path)
    return tests_path


@pytest.fixture
def benchmark_results(cache_path: Path) -> BenchmarkResults:
    return mteb.load_results(download_latest=False, require_model_meta=False)


def test_indexing(benchmark_results: BenchmarkResults) -> None:
    model_res = benchmark_results.model_results[0]
    assert isinstance(model_res, ModelResult), (
        "indexing into the list should return a ModelResult"
    )


def test_select_models(benchmark_results: BenchmarkResults) -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    bench_res = benchmark_results.select_models([model_name])
    assert isinstance(bench_res, BenchmarkResults)
    assert isinstance(bench_res[0], ModelResult)
    assert len(bench_res.model_results) > 1  # multiple revisions
    assert bench_res[0].model_name == model_name

    # with revision
    model_meta = mteb.get_model_meta(model_name)
    bench_res = benchmark_results.select_models(
        names=[model_name],
        revisions=[model_meta.revision],
    )
    assert bench_res[0].model_name == model_name
    assert bench_res[0].model_revision == model_meta.revision
    assert len(bench_res.model_results) == 1  # only one revision

    # with model_meta
    model_meta = mteb.get_model_meta(model_name)
    bench_res = benchmark_results.select_models(
        names=[model_meta],
    )
    assert bench_res[0].model_name == model_name
    assert bench_res[0].model_revision == model_meta.revision
    assert len(bench_res.model_results) == 1  # only one revision


def test_select_tasks(benchmark_results: BenchmarkResults) -> None:
    tasks = [mteb.get_task("STS12")]
    bench_res = benchmark_results.select_tasks(tasks=tasks)
    task_names = bench_res.task_names
    assert isinstance(task_names, list)
    assert len(task_names) == 1
    assert task_names[0] == "STS12"


def test_join_revisions(benchmark_results: BenchmarkResults) -> None:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    bench_res = benchmark_results.select_models([model_name])

    assert len(bench_res.model_revisions) == 2, (
        "There should only be two revisions for this model in the mock cache"
    )

    bench_res = bench_res.join_revisions()
    assert isinstance(bench_res, BenchmarkResults)
    assert len(bench_res.model_revisions) == 1
    revision = bench_res.model_revisions[0]["revision"]
    assert revision == mteb.get_model_meta(model_name).revision


def test_to_dataframe(
    benchmark_results: BenchmarkResults,
) -> None:
    required_columns = [
        "model_name",
        "task_name",
        "task_name",
        "score",
        "subset",
        "split",
    ]
    t1 = benchmark_results.to_dataframe(aggregation_level="subset", format="long")
    assert isinstance(t1, pd.DataFrame)
    assert all(col in t1.columns for col in required_columns), "Columns are missing"
    assert t1.shape[0] > 0, "Results table is empty"

    t2 = benchmark_results.to_dataframe(aggregation_level="split", format="long")
    assert all(
        col in t2.columns for col in required_columns if col not in ["subset"]
    ), "Columns are missing"
    assert "subset" not in t2.columns, "Subset column should not be present"
    assert t1.shape[0] >= t2.shape[0], (
        "Aggregation level 'split' should have more rows than 'subset'"
    )

    t3 = benchmark_results.to_dataframe(aggregation_level="task", format="long")
    assert all(
        col in t3.columns for col in required_columns if col not in ["subset", "split"]
    ), "Columns are missing"
    assert "subset" not in t3.columns, "Subset column should not be present"
    assert "split" not in t3.columns, "Split column should not be present"
    assert t2.shape[0] >= t3.shape[0], (
        "Aggregation level 'task' should have more rows than 'split'"
    )

    # test no model revisions
    benchmark_res = benchmark_results.join_revisions()
    t1 = benchmark_res.to_dataframe(aggregation_level="subset", format="long")
    assert "model_revision" not in t1.columns, (
        "Model revision column should not be present"
    )
    # Test the wide format
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    task = mteb.get_task("BornholmBitextMining")

    # simplify down to one model and one task
    br = benchmark_res.select_models([model_name]).select_tasks([task])

    t4_wide = br.to_dataframe(aggregation_level="task", format="wide")
    t4_long = br.to_dataframe(aggregation_level="task", format="long")
    assert isinstance(t4_wide, pd.DataFrame)

    # check that the scores are the same for a given model
    assert t4_wide[model_name][0] == t4_long["score"][0], (
        "The scores in wide and long format should be the same"
    )


def test_utility_properties(
    benchmark_results: BenchmarkResults,
) -> None:
    br = benchmark_results
    assert isinstance(br.task_names, list) and isinstance(br.task_names[0], str)
    assert (
        isinstance(br.languages, list)
        and isinstance(br.languages[0], str)
        and "eng" in br.languages
    )
    assert isinstance(br.model_names, list) and isinstance(br.model_names[0], str)
    assert (
        isinstance(br.model_revisions, list)
        and isinstance(br.model_revisions[0], dict)
        and "model_name" in br.model_revisions[0]
        and "revision" in br.model_revisions[0]
    )
    assert isinstance(br.task_types, list) and isinstance(br.task_types[0], str)
    assert isinstance(br.domains, list) and isinstance(br.domains[0], str)


def test_benchmark_results(cache_path: Path) -> None:
    cache = ResultCache(cache_path)
    bench = Benchmark(
        name="MockBenchmark",
        tasks=mteb.get_tasks(
            [
                "NanoSCIDOCSRetrieval",
                "Banking77Classification",
            ],
        ),
    )
    results = cache.load_results(
        tasks=bench,
        models=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "baseline/random-encoder-baseline",
        ],
    )
    df = results.get_benchmark_result()

    assert "Classification" in df.columns
    assert "Retrieval" in df.columns
    assert df.shape[0] == 2
    assert df.at[0, "Mean (Task)"] == pytest.approx(0.616616)


@pytest.fixture
def temp_output(tmp_path) -> Path:
    return tmp_path / "model_card.md"


def test_generate_model_card_with_tasks(cache_path: Path, temp_output: Path) -> None:
    """Test generating model card with tasks."""
    tasks = [mteb.get_task("STS12"), mteb.get_task("Banking77Classification")]

    generate_model_card(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tasks=tasks,
        results_cache=ResultCache(cache_path),
        output_path=temp_output,
    )

    assert temp_output.exists(), "Model card file not created"
    content = temp_output.read_text()
    assert "---" in content, "YAML frontmatter not found"
    assert "model-index" in content, "Model index not found in frontmatter"


def test_generate_model_card_with_benchmarks(
    cache_path: Path, temp_output: Path
) -> None:
    """Test generating model card with multiple benchmarks."""
    benchmarks = [
        Benchmark(
            name="TestBenchmark1",
            tasks=mteb.get_tasks(["STS12"]),
        ),
        Benchmark(
            name="TestBenchmark2",
            tasks=mteb.get_tasks(["Banking77Classification"]),
        ),
    ]

    generate_model_card(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        benchmarks=benchmarks,
        results_cache=ResultCache(cache_path),
        output_path=temp_output,
    )

    assert temp_output.exists(), "Model card file not created"
    content = temp_output.read_text()
    assert "---" in content, "YAML frontmatter not found"


def test_generate_model_card_with_table_and_tasks(
    cache_path: Path, temp_output: Path
) -> None:
    """Test generating model card with results table for tasks."""
    tasks = [mteb.get_task("STS12"), mteb.get_task("Banking77Classification")]

    generate_model_card(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        tasks=tasks,
        results_cache=ResultCache(cache_path),
        output_path=temp_output,
        add_table_to_model_card=True,
    )

    assert temp_output.exists(), "Model card file not created"
    content = temp_output.read_text()

    assert "# MTEB Results" in content or "# MTEB results" in content, (
        "Results table heading not found"
    )
    assert "STS12" in content or "Banking77" in content, "Task names not in content"


def test_generate_model_card_with_table_and_benchmark(
    cache_path: Path, temp_output: Path
) -> None:
    """Test generating model card with results table for benchmarks."""
    benchmark = Benchmark(
        name="TestBenchmark",
        tasks=mteb.get_tasks(["STS12", "Banking77Classification"]),
    )

    generate_model_card(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        benchmarks=[benchmark],
        results_cache=ResultCache(cache_path),
        output_path=temp_output,
        add_table_to_model_card=True,
    )

    assert temp_output.exists(), "Model card file not created"
    content = temp_output.read_text()

    assert "# MTEB Results" in content or "# MTEB results" in content, (
        "Results table heading not found"
    )
    assert len(content.split("|")) > 3, "Table should have multiple columns"
