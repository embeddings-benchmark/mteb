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
            "mteb/baseline-random-encoder",
        ],
    )
    df = results.get_benchmark_result()

    assert "Classification" in df.columns
    assert "Retrieval" in df.columns
    assert df.shape[0] == 2
    assert df.at[0, "Mean (Task)"] == pytest.approx(0.616616)


def test_generate_model_card_with_table_and_benchmarks(
    cache_path: Path, tmp_path: Path
) -> None:
    """Test generating model card with results table for multiple benchmarks and models."""
    test_folder = Path(__file__).parent.parent
    golden_files_path = test_folder / "create_meta"
    output_path = tmp_path / "model_card_benchmark.md"
    golden_file = golden_files_path / "model_card_benchmark_gold.md"

    # Create benchmarks
    benchmarks = [
        Benchmark(
            name="STS_Benchmark",
            tasks=mteb.get_tasks(["STS12", "STS13"]),
        ),
        Benchmark(
            name="Classification_Benchmark",
            tasks=mteb.get_tasks(["Banking77Classification"]),
        ),
    ]

    generate_model_card(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        benchmarks=benchmarks,
        results_cache=ResultCache(cache_path),
        output_path=output_path,
        add_table_to_model_card=True,
        models_to_compare=["mteb/baseline-random-encoder"],
    )

    assert output_path.exists(), "Model card file not created"
    assert output_path.stat().st_size > 0, "Model card file is empty"

    with output_path.open("r") as f:
        output_content = f.read()

    assert golden_file.exists(), f"Golden file not found: {golden_file}"
    with golden_file.open("r") as f:
        golden_content = f.read()

    def extract_table(content: str) -> list[str]:
        """Extract markdown table rows from content."""
        lines = content.split("\n")
        table_rows = []
        in_table = False

        for line in lines:
            if "# MTEB Results" in line or "# MTEB results" in line:
                in_table = True
                continue

            if in_table:
                # Table rows contain pipes
                if "|" in line:
                    # Normalize whitespace in table rows
                    normalized = "|".join(cell.strip() for cell in line.split("|"))
                    table_rows.append(normalized)
                elif line.strip() == "":
                    # Empty line might signal end of table
                    if table_rows:
                        break

        return table_rows

    output_table = extract_table(output_content)
    golden_table = extract_table(golden_content)

    # Validate table exists
    assert len(output_table) > 0, "Output table not found"
    assert len(golden_table) > 0, "Golden table not found"

    # Compare table structure (header and separator)
    assert len(output_table) >= 2, (
        "Output table should have at least header and separator"
    )
    assert len(golden_table) >= 2, (
        "Golden table should have at least header and separator"
    )

    # Compare headers
    assert output_table[0] == golden_table[0], (
        f"Table headers don't match.\n"
        f"Expected: {golden_table[0]}\n"
        f"Got: {output_table[0]}"
    )

    # Compare data rows (skip header and separator)
    output_data_rows = output_table[2:]
    golden_data_rows = golden_table[2:]

    assert len(output_data_rows) == len(golden_data_rows), (
        f"Number of data rows doesn't match.\n"
        f"Expected {len(golden_data_rows)} rows, got {len(output_data_rows)}"
    )

    # Compare each data row
    for i, (output_row, golden_row) in enumerate(
        zip(output_data_rows, golden_data_rows)
    ):
        assert output_row == golden_row, (
            f"Row {i} doesn't match.\nExpected: {golden_row}\nGot: {output_row}"
        )
