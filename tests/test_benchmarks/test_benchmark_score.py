import numpy as np
import polars as pl
import pytest
from datasets import Dataset

import mteb
from mteb import ResultCache
from mteb.benchmarks._benchmark_metrics import (
    _compute_custom_group_means,
    _compute_mean_public_private,
    _compute_mean_subset,
    _compute_mean_task,
    _compute_mean_task_type,
    _compute_task_types,
)
from mteb.benchmarks.benchmark import (
    Benchmark,
    BenchmarkAggregation,
    CustomGroup,
    CustomGrouping,
)


def _datasets_supports_dictionary_type() -> bool:
    """True if the installed ``datasets`` can convert Polars Categorical columns.

    ``_to_results_df`` goes through ``Dataset.from_polars`` for categorical
    columns (model_name, task_name, …); older ``datasets`` releases raise
    ``NotImplementedError`` for ``pa.DictionaryType``. Probe once at import.
    """
    try:
        Dataset.from_polars(pl.DataFrame({"x": ["a"]}, schema={"x": pl.Categorical}))
    except NotImplementedError:
        return False
    return True


_skip_if_datasets_too_old = pytest.mark.skipif(
    not _datasets_supports_dictionary_type(),
    reason=(
        "installed `datasets` cannot convert Polars Categorical columns "
        "(pa.DictionaryType -> Features.from_arrow_schema NotImplementedError); "
        "skip the _to_results_df-based parity tests on lowest-pin CI"
    ),
)

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


def _make_custom_groupings() -> tuple[CustomGrouping, CustomGrouping]:
    """Two CustomGrouping dimensions over the mock task set, deliberately
    sharing a group label ("Other") to exercise dimension namespacing."""
    dim_a = CustomGrouping(
        name="DimA",
        groups=(
            CustomGroup(
                label="G1", tasks=["NanoSCIDOCSRetrieval", "Banking77Classification"]
            ),
            CustomGroup(label="Other", tasks=["NanoArguAnaRetrieval"]),
        ),
    )
    dim_b = CustomGrouping(
        name="DimB",
        groups=(
            CustomGroup(label="G2", tasks=["NanoArguAnaRetrieval"]),
            CustomGroup(label="Other", tasks=["Banking77Classification"]),
        ),
    )
    return dim_a, dim_b


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


@_skip_if_datasets_too_old
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


@_skip_if_datasets_too_old
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


@_skip_if_datasets_too_old
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


def test_compute_custom_group_means(mock_mteb_cache: ResultCache):
    """_compute_custom_group_means returns namespaced 'dimension::label' keys,
    and two dimensions sharing a group label ("Other") don't collide."""
    mock_model_name = "mteb/baseline-random-encoder"
    dim_a, dim_b = _make_custom_groupings()
    mock_benchmark = _make_benchmark()

    model_result = mock_mteb_cache.load_results(models=[mock_model_name]).model_results[
        0
    ]
    task_results = model_result.select_tasks(mock_benchmark.tasks).task_results

    out_a = _compute_custom_group_means(task_results, dim_a)
    out_b = _compute_custom_group_means(task_results, dim_b)

    assert set(out_a) == {"DimA::G1", "DimA::Other"}
    assert set(out_b) == {"DimB::G2", "DimB::Other"}
    assert all(v is not None for v in {**out_a, **out_b}.values())
    # Different task sets behind each "Other" -> different values, and
    # namespacing keeps both present at once without overwriting each other.
    assert out_a["DimA::Other"] != out_b["DimB::Other"]


def test_benchmark_get_score_custom_groups(mock_mteb_cache: ResultCache):
    """get_score surfaces CustomGrouping keys alongside the built-in aggregations."""
    dim_a, dim_b = _make_custom_groupings()
    mock_benchmark = _make_benchmark()
    mock_benchmark = Benchmark(
        name=mock_benchmark.name,
        tasks=mock_benchmark.tasks,
        aggregations=(BenchmarkAggregation.MEAN_TASK, dim_a, dim_b),
    )
    mock_results = mock_mteb_cache.load_results()
    model_name = next(iter(MODELS_SCORES["Mean(Task)"]))

    scores = mock_benchmark.get_score(mock_results)[model_name]

    assert set(scores) == {
        "Mean(Task)",
        "DimA::G1",
        "DimA::Other",
        "DimB::G2",
        "DimB::Other",
        "Rank",
    }
    assert np.allclose(scores["Mean(Task)"], MODELS_SCORES["Mean(Task)"][model_name])


@_skip_if_datasets_too_old
def test_summary_table_custom_groups_columns(mock_mteb_cache: ResultCache):
    """_create_summary_table namespaces CustomGrouping columns per dimension
    and exposes them via SummaryTable.custom_group_cols, without polluting
    the TASK_TYPES columns the aggregators.py type_cols catch-all reads."""
    dim_a, dim_b = _make_custom_groupings()
    mock_benchmark = _make_benchmark()
    bench = Benchmark(
        name="mock_custom_groups",
        tasks=mock_benchmark.tasks,
        aggregations=(
            BenchmarkAggregation.MEAN_TASK,
            BenchmarkAggregation.TASK_TYPES,
            dim_a,
            dim_b,
        ),
    )
    mock_results = mock_mteb_cache.load_results()

    pl_df = mock_results.select_tasks(bench.tasks)._to_results_df(bench.tasks)
    summary = bench._create_summary_table(pl_df)

    # Column order within a dimension follows first-occurrence order in the
    # pivoted task frame (same precedent as TASK_TYPES / _get_means_per_types
    # — not the declaration order of `grouping.groups`), so compare as sets.
    assert {dim: set(cols) for dim, cols in summary.custom_group_cols.items()} == {
        "DimA": {"__cg__DimA::G1", "__cg__DimA::Other"},
        "DimB": {"__cg__DimB::G2", "__cg__DimB::Other"},
    }
    cols = set(summary.df.columns)
    assert {
        "__cg__DimA::G1",
        "__cg__DimA::Other",
        "__cg__DimB::G2",
        "__cg__DimB::Other",
    } <= cols
    # TASK_TYPES columns are unaffected by the custom-group columns sharing
    # the frame — no collision, no cross-contamination.
    assert {"Retrieval", "Classification"} <= cols


@_skip_if_datasets_too_old
def test_get_score_matches_summary_table_custom_groups(mock_mteb_cache: ResultCache):
    """get_score() and the polars summary table agree on CustomGrouping values,
    mirroring test_get_score_matches_summary_table_means for the built-ins."""
    dim_a, dim_b = _make_custom_groupings()
    mock_benchmark = _make_benchmark()
    bench = Benchmark(
        name="mock_custom_groups_parity",
        tasks=mock_benchmark.tasks,
        aggregations=(BenchmarkAggregation.MEAN_TASK, dim_a, dim_b),
    )
    mock_results = mock_mteb_cache.load_results()

    get_score_out = bench.get_score(mock_results)
    pl_df = mock_results.select_tasks(bench.tasks)._to_results_df(bench.tasks)
    summary_pl = bench._create_summary_table(pl_df).df
    summary_by_model = {row["Model"]: row for row in summary_pl.iter_rows(named=True)}

    parity_pairs = [
        ("DimA::G1", "__cg__DimA::G1"),
        ("DimA::Other", "__cg__DimA::Other"),
        ("DimB::G2", "__cg__DimB::G2"),
        ("DimB::Other", "__cg__DimB::Other"),
    ]

    checked = 0
    for model_name, scores in get_score_out.items():
        if model_name not in summary_by_model:
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
