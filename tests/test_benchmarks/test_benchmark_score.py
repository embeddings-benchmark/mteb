import numpy as np

import mteb
from mteb import ResultCache
from mteb.benchmarks._benchmark_metrics import (
    _compute_mean_public_private,
    _compute_mean_subset,
    _compute_mean_task,
    _compute_mean_task_type,
    _compute_task_types,
)
from mteb.benchmarks.benchmark import Benchmark, BenchmarkAggregation

MODELS_SCORES = {
    "Mean(Task)": {
        "mteb/baseline-random-encoder": 0.005604,
        "sentence-transformers/all-MiniLM-L6-v2": 0.59396733,
    },
    "Mean(TaskType)": {
        "mteb/baseline-random-encoder": 0.007336,
        "sentence-transformers/all-MiniLM-L6-v2": 0.645581,
    },
}


def _make_benchmark(extra_tasks: list[str] | None = None):
    tasks = mteb.get_tasks(
        [
            "NanoSCIDOCSRetrieval",
            "Banking77Classification",
            "NanoArguAnaRetrieval",
        ]
    )
    if extra_tasks:
        tasks += mteb.get_tasks(extra_tasks)
    return Benchmark(name="mock_benchmark", tasks=tasks)


def test_benchmark_get_score(mock_mteb_cache: ResultCache):
    """get_score returns Mean(Task), Mean(TaskType), and Borda Rank for each model."""
    mock_benchmark = _make_benchmark()
    mock_results = mock_mteb_cache.load_results()

    scores = mock_benchmark.get_score(mock_results)

    for model_name, expected in MODELS_SCORES["Mean(Task)"].items():
        assert model_name in scores, f"{model_name} missing from scores"
        assert np.allclose(scores[model_name]["Mean(Task)"], expected)

    for model_name, expected in MODELS_SCORES["Mean(TaskType)"].items():
        assert np.allclose(scores[model_name]["Mean(TaskType)"], expected)


def test_benchmark_get_score_missing_tasks(mock_mteb_cache: ResultCache):
    """get_score returns None for models with missing task results."""
    mock_benchmark = _make_benchmark(extra_tasks=["BelebeleRetrieval"])
    mock_results = mock_mteb_cache.load_results()

    scores = mock_benchmark.get_score(mock_results)

    for model_name in MODELS_SCORES["Mean(Task)"]:
        assert scores[model_name]["Mean(Task)"] is None
        assert scores[model_name]["Mean(TaskType)"] is None


def test_compute_mean_task(mock_mteb_cache: ResultCache):
    """_compute_mean_task returns the correct scalar mean."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark()

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    assert np.allclose(
        _compute_mean_task(task_results),
        MODELS_SCORES["Mean(Task)"][mock_model_name],
    )


def test_compute_mean_task_type(mock_mteb_cache: ResultCache):
    """_compute_mean_task_type returns the correct mean of task-type means."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark()

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    assert np.allclose(
        _compute_mean_task_type(task_results),
        MODELS_SCORES["Mean(TaskType)"][mock_model_name],
    )


def test_compute_mean_task_missing(mock_mteb_cache: ResultCache):
    """_compute_mean_task returns None when any task score is missing."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark(extra_tasks=["BelebeleRetrieval"])

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    assert _compute_mean_task(task_results) is None
    assert _compute_mean_task_type(task_results) is None


def test_compute_task_types(mock_mteb_cache: ResultCache):
    """_compute_task_types returns one entry per task type."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark()

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    per_type = _compute_task_types(task_results)

    assert per_type is not None
    assert set(per_type) == {"Retrieval", "Classification"}


def test_compute_mean_subset(mock_mteb_cache: ResultCache):
    """_compute_mean_subset matches the polars subset-weighted Mean (Subset).

    Builds a MEAN_SUBSET benchmark on mock tasks, runs both paths, and asserts
    the Python helper agrees with the polars summary's ``Mean (Subset)``
    column (and with :meth:`BenchmarkAggregation.aggregate`) for every model
    on the summary.
    """

    mock_results = mock_mteb_cache.load_results()
    tasks = mteb.get_tasks(
        ["NanoSCIDOCSRetrieval", "Banking77Classification", "NanoArguAnaRetrieval"]
    )
    bench = Benchmark(
        name="mock_subset",
        tasks=tasks,
        aggregations=(BenchmarkAggregation.MEAN_SUBSET,),
    )

    pl_df = mock_results.select_tasks(bench.tasks)._to_results_df(bench.tasks)
    summary_pl = bench._create_summary_table(pl_df).df
    summary_by_model = {row["Model"]: row for row in summary_pl.iter_rows(named=True)}

    checked = 0
    for model_result in mock_results.model_results:
        if model_result.model_name not in summary_by_model:
            continue
        task_results = model_result.select_tasks(bench.tasks).task_results
        if len(task_results) < len(bench.tasks):
            continue

        helper_out = _compute_mean_subset(task_results)
        agg_out = BenchmarkAggregation.MEAN_SUBSET.aggregate(task_results)
        assert helper_out == agg_out, "aggregate() must defer to _compute_mean_subset"

        polars_value = summary_by_model[model_result.model_name].get("Mean (Subset)")
        python_value = helper_out["Mean(Subset)"]
        assert polars_value is not None and python_value is not None
        assert np.isclose(polars_value, python_value), (
            f"{model_result.model_name}: polars={polars_value:.6f} "
            f"vs python={python_value:.6f}"
        )
        checked += 1

    assert checked > 0, "test never matched a model; fixture or registry change?"


def test_compute_mean_public_private(mock_mteb_cache: ResultCache):
    """_compute_mean_public_private returns Mean(Public) and Mean(Private) keys."""
    mock_model_name = "mteb/baseline-random-encoder"
    mock_benchmark = _make_benchmark()

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    means = _compute_mean_public_private(task_results)

    assert set(means) == {"Mean(Public)", "Mean(Private)"}
    # Mock tasks used in the benchmark are all public.
    assert means["Mean(Private)"] is None


def _summary_columns(benchmark: Benchmark, mock_results) -> set[str]:
    pl_df = mock_results.select_tasks(benchmark.tasks)._to_results_df(benchmark.tasks)
    return set(benchmark._create_summary_table(pl_df).df.columns)


def test_summary_table_aggregations_drive_columns(mock_mteb_cache: ResultCache):
    """_create_summary_table surfaces the mean columns named in self.aggregations."""
    tasks = mteb.get_tasks(
        ["NanoSCIDOCSRetrieval", "Banking77Classification", "NanoArguAnaRetrieval"]
    )
    mock_results = mock_mteb_cache.load_results()

    default_cols = _summary_columns(
        Benchmark(name="mock_default", tasks=tasks), mock_results
    )
    assert {"Mean (Task)", "Mean (TaskType)", "Retrieval", "Classification"} <= (
        default_cols
    )

    mean_task_cols = _summary_columns(
        Benchmark(
            name="mock_mean_task",
            tasks=tasks,
            aggregations=(BenchmarkAggregation.MEAN_TASK,),
        ),
        mock_results,
    )
    assert "Mean (Task)" in mean_task_cols
    assert "Mean (TaskType)" not in mean_task_cols
    assert "Retrieval" not in mean_task_cols
    assert "Classification" not in mean_task_cols

    task_type_only_cols = _summary_columns(
        Benchmark(
            name="mock_task_type_only",
            tasks=tasks,
            aggregations=(
                BenchmarkAggregation.MEAN_TASK_TYPE,
                BenchmarkAggregation.TASK_TYPES,
            ),
        ),
        mock_results,
    )
    assert "Mean (Task)" not in task_type_only_cols
    assert "Mean (TaskType)" in task_type_only_cols
    assert {"Retrieval", "Classification"} <= task_type_only_cols


def test_get_score_matches_summary_table_means(mock_mteb_cache: ResultCache):
    """get_score and the summary table return the same numbers for the same data.

    The two paths use the same ``BenchmarkAggregation.summary_columns`` /
    ``get_score_keys`` mapping for column / key names — this test asserts
    they also compute equal numbers, so drift between the Python dispatcher
    in :meth:`BenchmarkAggregation.aggregate` and the polars pipeline in
    :func:`_create_summary_table` gets caught here.
    """
    tasks = mteb.get_tasks(
        ["NanoSCIDOCSRetrieval", "Banking77Classification", "NanoArguAnaRetrieval"]
    )
    bench = Benchmark(name="mock_parity", tasks=tasks)
    mock_results = mock_mteb_cache.load_results()

    get_score_out = bench.get_score(mock_results)
    pl_df = mock_results.select_tasks(bench.tasks)._to_results_df(bench.tasks)
    summary_pl = bench._create_summary_table(pl_df).df

    summary_by_model = {row["Model"]: row for row in summary_pl.iter_rows(named=True)}

    # Pair (get_score key, summary column) for each aggregation surfaced.
    parity_pairs: list[tuple[str, str]] = []
    for agg in bench.aggregations:
        parity_pairs.extend(zip(agg.get_score_keys, agg.summary_columns))
    # TASK_TYPES is dynamic — match observed type names directly.
    if BenchmarkAggregation.TASK_TYPES in bench.aggregations:
        type_names = {t.metadata.type for t in bench.tasks}
        parity_pairs.extend((t, t) for t in type_names)

    checked = 0
    for model_name, scores in get_score_out.items():
        if model_name not in summary_by_model:
            # Summary builder inner-joins against MODEL_REGISTRY — random or
            # synthetic test models drop out, so skip them here.
            continue
        srow = summary_by_model[model_name]
        for gs_key, summary_col in parity_pairs:
            gs = scores.get(gs_key)
            sm = srow.get(summary_col)
            if gs is None and sm is None:
                continue
            assert gs is not None and sm is not None, (
                f"{model_name}: get_score[{gs_key}]={gs!r} vs summary[{summary_col}]={sm!r}"
            )
            assert np.isclose(gs, sm), (
                f"{model_name}: get_score[{gs_key}]={gs:.6f} vs summary[{summary_col}]={sm:.6f}"
            )
            checked += 1
    assert checked > 0, (
        "Parity test never matched a model — fixture or registry change?"
    )


def test_benchmark_get_score_aggregations_drive_keys(mock_mteb_cache: ResultCache):
    """Returned per-model score keys follow self.aggregations."""
    tasks = mteb.get_tasks(
        ["NanoSCIDOCSRetrieval", "Banking77Classification", "NanoArguAnaRetrieval"]
    )
    mock_results = mock_mteb_cache.load_results()
    model_name = next(iter(MODELS_SCORES["Mean(Task)"]))

    only_mean_task = Benchmark(
        name="mock_mean_task",
        tasks=tasks,
        aggregations=(BenchmarkAggregation.MEAN_TASK,),
    )
    only_task_types = Benchmark(
        name="mock_task_types",
        tasks=tasks,
        aggregations=(BenchmarkAggregation.TASK_TYPES,),
    )
    public_private = Benchmark(
        name="mock_public_private",
        tasks=tasks,
        aggregations=(BenchmarkAggregation.PUBLIC_PRIVATE,),
    )

    mean_task_keys = set(only_mean_task.get_score(mock_results)[model_name])
    assert mean_task_keys == {"Mean(Task)", "Rank"}

    task_types_keys = set(only_task_types.get_score(mock_results)[model_name])
    assert task_types_keys == {"Retrieval", "Classification", "Rank"}

    pp_keys = set(public_private.get_score(mock_results)[model_name])
    assert pp_keys == {"Mean(Public)", "Mean(Private)", "Rank"}
