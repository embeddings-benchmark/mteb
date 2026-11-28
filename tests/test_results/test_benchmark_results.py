"""Tests for the BenchmarkResult class"""

import os
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from packaging.version import Version

import mteb
from mteb import Benchmark
from mteb.cache import ResultCache
from mteb.cli.generate_model_card import generate_model_card
from mteb.results import BenchmarkResults, ModelResult, TaskResult

_POLARS_TOO_OLD = Version(pl.__version__) < Version("1.40.0")


@pytest.fixture
def cache_path() -> Path:
    tests_path = Path(__file__).parent.parent / "mock_mteb_cache"

    os.environ["MTEB_CACHE"] = str(tests_path)
    return tests_path


@pytest.fixture
def benchmark_results(cache_path: Path) -> BenchmarkResults:
    return mteb.load_results(download_latest=False, require_model_meta=False)


@pytest.mark.core
def test_indexing(benchmark_results: BenchmarkResults) -> None:
    model_res = benchmark_results.model_results[0]
    assert isinstance(model_res, ModelResult), (
        "indexing into the list should return a ModelResult"
    )


@pytest.mark.core
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


@pytest.mark.core
def test_select_tasks(benchmark_results: BenchmarkResults) -> None:
    tasks = [mteb.get_task("STS12")]
    bench_res = benchmark_results.select_tasks(tasks=tasks)
    task_names = bench_res.task_names
    assert isinstance(task_names, list)
    assert len(task_names) == 1
    assert task_names[0] == "STS12"


def test_select_tasks_preserves_model_meta(tmp_path: Path) -> None:
    """`select_tasks` must not drop `model_meta` (and with it, experiment identity).

    Regression test: `select_tasks`/`_filter_tasks` reconstructed a new
    `ModelResult` via `model_construct` without passing `model_meta`
    through, silently dropping any experiment row's `experiment_kwargs`.
    """
    from mteb.mocks.mock_tasks import MockRetrievalTask

    model_name = "mteb/baseline-random-encoder"
    task = MockRetrievalTask()
    cache = mteb.ResultCache(tmp_path)

    model = mteb.get_model(model_name, a="test")
    mteb.evaluate(model, task, cache=cache)

    from mteb.cache import LoadExperimentEnum

    results = cache.load_results(load_experiments=LoadExperimentEnum.MATCH_NAME)
    model_res = results.model_results[0]
    assert model_res.model_meta is not None
    assert model_res.model_meta.experiment_kwargs == {"a": "test"}

    filtered = model_res.select_tasks([task])
    assert filtered.model_meta is not None
    assert filtered.model_meta.experiment_kwargs == {"a": "test"}


@pytest.mark.core
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


@pytest.mark.core
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
        col in t2.columns
        for col in required_columns
        if col not in ["subset"]  # noqa: PLR6201
    ), "Columns are missing"
    assert "subset" not in t2.columns, "Subset column should not be present"
    assert t1.shape[0] >= t2.shape[0], (
        "Aggregation level 'split' should have more rows than 'subset'"
    )

    t3 = benchmark_results.to_dataframe(aggregation_level="task", format="long")
    assert all(
        col in t3.columns
        for col in required_columns
        if col not in ["subset", "split"]  # noqa: PLR6201
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


@pytest.mark.core
def test_utility_properties(
    benchmark_results: BenchmarkResults,
) -> None:
    br = benchmark_results
    assert isinstance(br.task_names, list)
    assert isinstance(br.task_names[0], str)
    assert isinstance(br.languages, list)
    assert isinstance(br.languages[0], str)
    assert "eng" in br.languages
    assert isinstance(br.model_names, list)
    assert isinstance(br.model_names[0], str)
    assert isinstance(br.model_revisions, list)
    assert isinstance(br.model_revisions[0], dict)
    assert "model_name" in br.model_revisions[0]
    assert "revision" in br.model_revisions[0]
    assert isinstance(br.task_types, list)
    assert isinstance(br.task_types[0], str)
    assert isinstance(br.domains, list)
    assert isinstance(br.domains[0], str)


@pytest.mark.core
@pytest.mark.skipif(_POLARS_TOO_OLD, reason="requires polars >= 1.40.0")
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
    assert df.loc[0, "Mean (Task)"] == pytest.approx(0.616616)


@pytest.mark.core
@pytest.mark.skipif(_POLARS_TOO_OLD, reason="requires polars >= 1.40.0")
def test_benchmark_results_model_filters_keep_benchmark(cache_path: Path) -> None:
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
    results = cache.load_results(tasks=bench)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    for filtered in [
        results.select_models([model_name]),
        results._filter_models(model_names=[model_name]),
    ]:
        assert filtered.benchmark is bench
        df = filtered.get_benchmark_result()
        assert df.shape[0] == 1
        assert df.loc[0, "Mean (Task)"] == pytest.approx(0.616616)


@pytest.mark.parametrize(
    ("other_version", "expected_revision"),
    [
        ("1.0.0", "resumed"),  # the range is newer
        ("2.25.0", "other"),  # part of the range is older
    ],
)
def test_join_revisions_compares_version_ranges(
    other_version: str, expected_revision: str
) -> None:
    """A run resumed with a newer mteb stores a version range, which has to be compared like a single version."""

    def _model_result(revision: str, version: str) -> ModelResult:
        task_result = TaskResult(
            dataset_revision="1",
            task_name="NFCorpus",
            mteb_version=version,
            scores={
                "test": [
                    {
                        "hf_subset": "default",
                        "languages": ["eng-Latn"],
                        "main_score": 0.5,
                        "mteb_version": version.rsplit("-", maxsplit=1)[-1],
                    }
                ]
            },
            evaluation_time=1.0,
        )
        return ModelResult(
            model_name="org/model-not-in-registry",
            model_revision=revision,
            task_results=[task_result],
        )

    results = BenchmarkResults(
        model_results=[
            _model_result("other", other_version),
            _model_result("resumed", "2.20.0-2.27.0"),
        ]
    )

    joined = results.join_revisions()

    assert [m.model_revision for m in joined.model_results] == [expected_revision]


def test_get_aggregated_scores_without_benchmark_joins_model_parts() -> None:
    """Parts of a model kept from two revisions are combined, and an experiment stays apart."""

    def _model_result(
        revision: str, task_name: str, score: float, experiment_name: str | None = None
    ) -> ModelResult:
        task_result = TaskResult(
            dataset_revision="1",
            task_name=task_name,
            mteb_version="2.27.0",
            scores={
                "test": [
                    {
                        "hf_subset": "default",
                        "languages": ["eng-Latn"],
                        "main_score": score,
                    }
                ]
            },
            evaluation_time=1.0,
        )
        return ModelResult(
            model_name="org/model-not-in-registry",
            model_revision=revision,
            experiment_name=experiment_name,
            task_results=[task_result],
        )

    results = BenchmarkResults(
        model_results=[
            _model_result("first", "NFCorpus", 0.2),
            _model_result("second", "Banking77Classification", 0.8),
            _model_result("first", "NFCorpus", 0.6, experiment_name="normalize_False"),
        ]
    )

    scores = results.get_aggregated_scores()

    assert scores == {
        "org/model-not-in-registry": {
            "Mean(Task)": pytest.approx(0.5),
            "Mean(TaskType)": pytest.approx(0.5),
        },
        "org/model-not-in-registry (normalize_False)": {
            "Mean(Task)": pytest.approx(0.6),
            "Mean(TaskType)": pytest.approx(0.6),
        },
    }


@pytest.mark.skipif(_POLARS_TOO_OLD, reason="requires polars >= 1.40.0")
def test_join_revisions_keeps_benchmark(cache_path: Path) -> None:
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
    results = cache.load_results(tasks=bench)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    joined = results.select_models([model_name]).join_revisions()
    assert joined.benchmark is bench
    df = joined.get_benchmark_result()
    assert df.shape[0] == 1
    assert df.loc[0, "Mean (Task)"] == pytest.approx(0.616616)

    # the early return for results without any task results keeps it as well
    empty = results.select_models(["no-such-model"]).join_revisions()
    assert empty.benchmark is bench


@pytest.mark.skipif(_POLARS_TOO_OLD, reason="requires polars >= 1.40.0")
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

    with output_path.open("r", encoding="utf-8") as f:
        output_content = f.read()

    assert golden_file.exists(), f"Golden file not found: {golden_file}"
    with golden_file.open("r", encoding="utf-8") as f:
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
                elif not line.strip():
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
        zip(output_data_rows, golden_data_rows, strict=True)
    ):
        assert output_row == golden_row, (
            f"Row {i} doesn't match.\nExpected: {golden_row}\nGot: {output_row}"
        )


def test_aggregate_scores_preserve_colliding_experiment_labels(
    benchmark_results: BenchmarkResults,
):
    base = benchmark_results.select_models(
        ["mteb/baseline-random-encoder"]
    ).model_results[0]
    meta = mteb.get_model_meta(base.model_name)
    runs = []
    for kwargs in ({"a": "x, b=y"}, {"a": "x", "b": "y"}):
        experiment = meta.model_copy(update={"experiment_kwargs": kwargs})
        runs.append(
            base.model_copy(
                update={
                    "model_meta": experiment,
                    "experiment_name": experiment.experiment_name,
                }
            )
        )
    scores = BenchmarkResults(model_results=runs).get_aggregated_scores()
    assert len(scores) == 2
    assert f"{base.model_name} (a=x, b=y)" in scores


@pytest.mark.parametrize("include_model_revision", [False, True])
def test_dataframe_exports_each_experiment(include_model_revision):
    model = "mteb/baseline-random-encoder"
    runs = []
    for experiment, score in [(None, 0.2), ("named", 0.8)]:
        result = TaskResult(
            dataset_revision="1",
            task_name="SciFact",
            mteb_version="2.27.0",
            scores={
                "test": [
                    {
                        "hf_subset": "default",
                        "main_score": score,
                        "languages": ["eng-Latn"],
                    }
                ]
            },
            evaluation_time=1.0,
        )
        runs.append(
            ModelResult(
                model_name=model,
                model_revision="rev",
                experiment_name=experiment,
                task_results=[result],
            )
        )
    results = BenchmarkResults(model_results=runs)
    long = results.to_dataframe(
        include_model_revision=include_model_revision, format="long"
    )
    assert dict(zip(long["experiment_id"], long["score"], strict=True)) == {
        "": 0.2,
        "named": 0.8,
    }
    wide = results.to_dataframe(include_model_revision=include_model_revision)
    key = (model, "rev") if include_model_revision else (model,)
    assert wide[(*key, "")].item() == 0.2
    assert wide[(*key, "named")].item() == 0.8


@pytest.mark.parametrize("include_model_revision", [False, True])
def test_to_dataframe_nulls_partly_evaluated_tasks(
    benchmark_results: BenchmarkResults, include_model_revision: bool
):
    # In the mock cache all-MiniLM-L6-v2 has no test/catalan score for this task.
    partial = "sentence-transformers/all-MiniLM-L6-v2"
    full = "mteb/baseline-random-encoder"
    task = mteb.get_task("CataloniaTweetClassification")
    results = (
        benchmark_results.select_models([partial, full])
        .join_revisions()
        .select_tasks([task])
    )

    def scores(
        level: str, by: tuple[str, ...] = (), aggregation_fn=None
    ) -> dict[tuple[str, ...], float]:
        df = results.to_dataframe(
            aggregation_level=level,
            aggregation_fn=aggregation_fn,
            include_model_revision=include_model_revision,
            format="long",
        )
        rows = df[["model_name", *by, "score"]].itertuples(index=False)
        return {tuple(keys): score for *keys, score in rows}

    task_scores = scores("task")
    assert pd.isna(task_scores[(partial,)])
    assert task_scores[(full,)] == pytest.approx(0.3373515)

    split_scores = scores("split", by=("split",))
    assert pd.isna(split_scores[(partial, "test")])
    assert split_scores[(partial, "validation")] == pytest.approx(0.444954)

    language_scores = scores("language", by=("language",))
    assert pd.isna(language_scores[(partial, "cat-Latn")])
    assert pd.isna(language_scores[(partial, "spa-Latn")])
    assert language_scores[(full, "cat-Latn")] == pytest.approx(0.339179)

    # A custom aggregation still gets every score, missing ones included.
    custom = scores("task", aggregation_fn=lambda task_scores: task_scores.mean())
    assert custom[(partial,)] == pytest.approx(0.451613)
