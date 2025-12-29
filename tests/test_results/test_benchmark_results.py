"""Tests for the BenchmarkResult class"""

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

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

    # Generate model card with 2 models to compare
    generate_model_card(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        benchmarks=benchmarks,
        results_cache=ResultCache(cache_path),
        output_path=output_path,
        add_table_to_model_card=True,
        models_to_compare=["baseline/random-encoder-baseline"],
    )

    assert output_path.exists(), "Model card file not created"
    assert output_path.stat().st_size > 0, "Model card file is empty"

    # Load generated output
    with output_path.open("r") as f:
        output_content = f.read()

    # Load golden file
    assert golden_file.exists(), f"Golden file not found: {golden_file}"
    with golden_file.open("r") as f:
        golden_content = f.read()

    # Extract frontmatter and readme from output
    yaml_start_sep = "---"
    yaml_end_sep = "\n---\n"

    start_yaml = output_content.index(yaml_start_sep) + len(yaml_start_sep)
    end_yaml = output_content.index(yaml_end_sep, start_yaml)
    output_frontmatter_str = output_content[start_yaml:end_yaml]
    output_frontmatter = yaml.safe_load(output_frontmatter_str)
    output_readme = output_content[end_yaml + len(yaml_end_sep) :]

    # Extract frontmatter and readme from golden file
    start_yaml_gold = golden_content.index(yaml_start_sep) + len(yaml_start_sep)
    end_yaml_gold = golden_content.index(yaml_end_sep, start_yaml_gold)
    golden_frontmatter_str = golden_content[start_yaml_gold:end_yaml_gold]
    golden_frontmatter = yaml.safe_load(golden_frontmatter_str)
    golden_readme = golden_content[end_yaml_gold + len(yaml_end_sep) :]

    # Validate frontmatter structure
    assert output_frontmatter is not None, "Output frontmatter should not be empty"
    assert golden_frontmatter is not None, "Golden frontmatter should not be empty"

    # Validate key frontmatter fields exist
    assert "model-index" in output_frontmatter, "model-index not found in output"
    assert "tags" in output_frontmatter, "tags not found in output"
    assert "mteb" in output_frontmatter.get("tags", []), "mteb tag not found"

    # Validate readme content (check key sections)
    assert "# MTEB Results" in output_readme or "# MTEB results" in output_readme, (
        "Results table heading not found in output"
    )

    # Check that benchmarks are mentioned
    for benchmark in benchmarks:
        assert benchmark.name.lower() in output_readme.lower() or any(
            task.metadata.name in output_readme for task in benchmark.tasks
        ), f"Benchmark {benchmark.name} not found in output readme"

    # Check table structure
    assert "|" in output_readme, "Table format not found in output"
    output_table_rows = [row for row in output_readme.split("\n") if "|" in row]
    assert len(output_table_rows) > 2, (
        "Table should have header, separator, and data rows"
    )

    # Compare readme content (normalize whitespace)
    output_readme_normalized = "\n".join(
        line.rstrip() for line in output_readme.split("\n")
    )
    golden_readme_normalized = "\n".join(
        line.rstrip() for line in golden_readme.split("\n")
    )

    assert output_readme_normalized == golden_readme_normalized, (
        "Output readme content does not match golden file.\n"
        f"Expected:\n{golden_readme_normalized}\n\n"
        f"Got:\n{output_readme_normalized}"
    )

    # Validate frontmatter model-index structure
    output_model_index = output_frontmatter.get("model-index", [])
    golden_model_index = golden_frontmatter.get("model-index", [])

    assert len(output_model_index) > 0, "Output model-index should not be empty"
    assert len(output_model_index) == len(golden_model_index), (
        f"Number of models in model-index should match: "
        f"expected {len(golden_model_index)}, got {len(output_model_index)}"
    )

    # Verify model names match
    output_models = [m["name"] for m in output_model_index]
    golden_models = [m["name"] for m in golden_model_index]
    assert output_models == golden_models, (
        f"Model names should match: expected {golden_models}, got {output_models}"
    )
