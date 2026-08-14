import numpy as np
import pytest

import mteb
from mteb import ResultCache
from mteb.benchmarks._benchmark_metrics import (
    _compute_custom_group_means,
    _compute_mean_public_private,
    _compute_mean_subset,
    _compute_mean_task,
    _compute_mean_task_type,
    _compute_task_types,
    _recompute_lenient_custom_groups,
    _recompute_lenient_means,
)
from mteb.benchmarks.benchmark import (
    Benchmark,
    BenchmarkAggregation,
    CustomGroup,
    CustomGrouping,
)
from tests.conftest import _skip_if_datasets_too_old

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
                label="G1",
                tasks=mteb.get_tasks(
                    ["NanoSCIDOCSRetrieval", "Banking77Classification"]
                ),
            ),
            CustomGroup(label="Other", tasks=mteb.get_tasks(["NanoArguAnaRetrieval"])),
        ),
    )
    dim_b = CustomGrouping(
        name="DimB",
        groups=(
            CustomGroup(label="G2", tasks=mteb.get_tasks(["NanoArguAnaRetrieval"])),
            CustomGroup(
                label="Other", tasks=mteb.get_tasks(["Banking77Classification"])
            ),
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


def test_benchmark_get_score_nulls_partial_split_and_subset_coverage(
    mock_mteb_cache: ResultCache,
):
    """`Benchmark.get_score()` nulls Mean(Task)/Mean(TaskType)/Mean(Subset) etc.
    for a model that skipped a split, or a subset within a split (issue #5101).
    """
    full = "mteb/baseline-random-encoder"
    partial = "sentence-transformers/all-MiniLM-L6-v2"
    tasks = mteb.get_tasks(
        [
            "PoemSentimentClassification.v2",  # partial model missing the "test" split
            "CataloniaTweetClassification",  # partial model missing "catalan"/"test"
            "Banking77Classification",  # fully covered by both models
        ]
    )
    bench = Benchmark(
        name="mock_partial_coverage",
        tasks=tasks,
        aggregations=(
            BenchmarkAggregation.MEAN_TASK,
            BenchmarkAggregation.MEAN_TASK_TYPE,
            BenchmarkAggregation.TASK_TYPES,
            BenchmarkAggregation.MEAN_SUBSET,
        ),
    )
    scores = bench.get_score(mock_mteb_cache.load_results())

    full_scores = scores[full]
    assert full_scores["Mean(Task)"] is not None
    assert full_scores["Mean(TaskType)"] is not None
    assert full_scores["Classification"] is not None
    assert full_scores["Mean(Subset)"] is not None

    partial_scores = scores[partial]
    assert partial_scores["Mean(Task)"] is None
    assert partial_scores["Mean(TaskType)"] is None
    assert partial_scores["Classification"] is None
    assert partial_scores["Mean(Subset)"] is None


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


def _assert_score_parity(
    model_name: str,
    get_score_row: dict[str, float | None],
    summary_row: dict,
    parity_pairs: list[tuple[str, str]],
    *,
    expect_none: bool,
) -> None:
    """Assert `get_score_row` and `summary_row` agree on every (key, column) pair.

    ``expect_none=True`` requires both sides to be `None`; `False` requires
    both to be equal, non-null numbers. Either way, a mismatch (one side
    null, the other not) fails loudly instead of silently passing.
    """
    for gs_key, summary_col in parity_pairs:
        gs, sm = get_score_row.get(gs_key), summary_row.get(summary_col)
        if expect_none:
            assert gs is None, (
                f"{model_name} get_score[{gs_key}] should be None, got {gs!r}"
            )
            assert sm is None, (
                f"{model_name} summary[{summary_col}] should be None, got {sm!r}"
            )
            continue
        assert gs is not None and sm is not None, (
            f"{model_name} {gs_key}/{summary_col}: get_score={gs!r} vs summary={sm!r}"
        )
        assert np.isclose(gs, sm), (
            f"{model_name} {gs_key}/{summary_col}: get_score={gs:.6f} vs summary={sm:.6f}"
        )


@_skip_if_datasets_too_old
def test_get_score_matches_summary_table_on_partial_split_coverage(
    mock_mteb_cache: ResultCache,
):
    """get_score() and the polars summary table null the same keys on partial coverage."""
    full = "mteb/baseline-random-encoder"
    partial = "sentence-transformers/all-MiniLM-L6-v2"
    tasks = mteb.get_tasks(
        [
            "PoemSentimentClassification.v2",
            "CataloniaTweetClassification",
            "Banking77Classification",
        ]
    )
    bench = Benchmark(
        name="mock_partial_parity",
        tasks=tasks,
        aggregations=(
            BenchmarkAggregation.MEAN_TASK,
            BenchmarkAggregation.MEAN_TASK_TYPE,
            BenchmarkAggregation.TASK_TYPES,
            BenchmarkAggregation.MEAN_SUBSET,
        ),
    )
    mock_results = mock_mteb_cache.load_results()

    get_score_out = bench.get_score(mock_results)
    pl_df = mock_results.select_tasks(bench.tasks)._to_results_df(bench.tasks)
    summary_by_model = {
        row["Model"]: row
        for row in bench._create_summary_table(pl_df).df.iter_rows(named=True)
    }
    assert full in get_score_out and full in summary_by_model
    assert partial in get_score_out and partial in summary_by_model

    # All three tasks are Classification, so there's exactly one dynamic
    # TASK_TYPES column/key to pair up alongside the fixed aggregation keys.
    parity_pairs: list[tuple[str, str]] = [("Classification", "Classification")]
    for agg in bench.aggregations:
        parity_pairs.extend(zip(agg.get_score_keys, agg.summary_columns))

    _assert_score_parity(
        full,
        get_score_out[full],
        summary_by_model[full],
        parity_pairs,
        expect_none=False,
    )
    _assert_score_parity(
        partial,
        get_score_out[partial],
        summary_by_model[partial],
        parity_pairs,
        expect_none=True,
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


def test_recompute_lenient_custom_groups_averages_present_tasks_only():
    """Mirrors _recompute_lenient_means: only tasks the model actually has a
    score for (post language-filter) contribute to each group's mean."""
    scores_by_task = {"t1": 0.2, "t2": 0.6, "t3": 0.8}
    custom_group_task_to_label = {
        "DimA": {"t1": "G1", "t2": "G1", "t3": "Other"},
        "DimB": {"t2": "G2", "t3": "Other"},
    }

    out = _recompute_lenient_custom_groups(scores_by_task, custom_group_task_to_label)

    assert out == {
        "DimA": {"G1": (0.2 + 0.6) / 2, "Other": 0.8},
        "DimB": {"G2": 0.6, "Other": 0.8},
    }


def test_recompute_lenient_custom_groups_ignores_tasks_outside_the_visible_set():
    """A task missing from scores_by_task (filtered out / not run) is simply
    absent from its group's bucket, not treated as a zero."""
    scores_by_task = {"t1": 1.0}
    custom_group_task_to_label = {"DimA": {"t1": "G1", "t2": "G1"}}

    out = _recompute_lenient_custom_groups(scores_by_task, custom_group_task_to_label)

    assert out == {"DimA": {"G1": 1.0}}


def test_recompute_lenient_custom_groups_empty_mapping_is_a_no_op():
    """Benchmarks with no CustomGrouping declared pass an empty mapping —
    the function must return {} rather than raise."""
    assert _recompute_lenient_custom_groups({"t1": 0.5}, {}) == {}


def test_recompute_lenient_custom_groups_matches_recompute_lenient_means_semantics():
    """Same 'average only what's present' policy as the task-type recompute,
    just keyed by CustomGrouping label instead of task type."""
    scores_by_task = {"t1": 0.4, "t2": 0.9}
    task_to_type = {"t1": "Retrieval", "t2": "Retrieval"}
    custom_group_task_to_label = {"DimA": {"t1": "G1", "t2": "G1"}}

    _, mean_task, _ = _recompute_lenient_means(scores_by_task, task_to_type)
    out = _recompute_lenient_custom_groups(scores_by_task, custom_group_task_to_label)

    assert out["DimA"]["G1"] == mean_task


# --- Scoped (subset-/split-narrowed) CustomGroup.tasks entries -------------
#
# NOTE (pre-existing, not introduced here): get_score() nulls a whole
# dimension when any entry is missing a score; the polars path only nulls
# the affected group. So the parity test below only checks a fully-covered
# model -- a partial-coverage model would legitimately disagree between paths.


def _make_scoped_custom_grouping() -> CustomGrouping:
    """A subset-scoped 'Language' dimension over the same
    CataloniaTweetClassification/Banking77Classification mock fixture
    `test_benchmark_get_score_nulls_partial_split_and_subset_coverage` uses."""
    catalan = mteb.get_task("CataloniaTweetClassification", hf_subsets=["catalan"])
    spanish = mteb.get_task("CataloniaTweetClassification", hf_subsets=["spanish"])
    banking = mteb.get_task("Banking77Classification")
    return CustomGrouping(
        name="Language",
        groups=(
            CustomGroup(label="Catalan", tasks=[catalan]),
            CustomGroup(label="Mixed", tasks=[spanish, banking]),
        ),
    )


def test_custom_grouping_post_init_scope_detection():
    """A whole-task ref lands in task_to_label; subset-, split-, and
    both-scoped refs don't -- and has_scoped_refs flips accordingly."""
    whole = mteb.get_task("Banking77Classification")
    subset_scoped = mteb.get_task(
        "CataloniaTweetClassification", hf_subsets=["catalan"]
    )
    split_scoped = mteb.get_task(
        "PoemSentimentClassification.v2", eval_splits=["validation"]
    )
    both_scoped = mteb.get_task(
        "CataloniaTweetClassification", hf_subsets=["spanish"], eval_splits=["test"]
    )

    grouping = CustomGrouping(
        name="ScopeDetect",
        groups=(
            CustomGroup(label="Whole", tasks=[whole]),
            CustomGroup(label="SubsetScoped", tasks=[subset_scoped]),
            CustomGroup(label="SplitScoped", tasks=[split_scoped]),
            CustomGroup(label="BothScoped", tasks=[both_scoped]),
        ),
    )

    assert grouping.task_to_label == {"Banking77Classification": "Whole"}
    assert grouping.has_scoped_refs


def test_custom_grouping_rejects_whole_task_and_scope_conflict():
    """A task claimed whole in one group can't also be claimed (even
    partially) by another group in the same dimension."""
    whole = mteb.get_task("CataloniaTweetClassification")
    scoped = mteb.get_task("CataloniaTweetClassification", hf_subsets=["catalan"])

    with pytest.raises(ValueError, match="assigned to multiple groups"):
        CustomGrouping(
            name="Conflict",
            groups=(
                CustomGroup(label="A", tasks=[whole]),
                CustomGroup(label="B", tasks=[scoped]),
            ),
        )


def test_custom_grouping_rejects_duplicate_scope_claim():
    """Two groups can't both claim the same (subset, split) cell of a task."""
    first = mteb.get_task("CataloniaTweetClassification", hf_subsets=["catalan"])
    second = mteb.get_task(
        "CataloniaTweetClassification", hf_subsets=["catalan"], eval_splits=["test"]
    )

    with pytest.raises(ValueError, match="assigned to multiple groups"):
        CustomGrouping(
            name="Conflict",
            groups=(
                CustomGroup(label="A", tasks=[first]),
                CustomGroup(label="B", tasks=[second]),
            ),
        )


def test_custom_grouping_allows_different_scopes_of_same_task_in_different_groups():
    """Two groups *may* each claim a different subset/split scope of the
    same task -- the whole point of scoped refs (e.g. LongEmbed's per-length
    groups)."""
    catalan = mteb.get_task("CataloniaTweetClassification", hf_subsets=["catalan"])
    spanish = mteb.get_task("CataloniaTweetClassification", hf_subsets=["spanish"])

    grouping = CustomGrouping(
        name="Language",
        groups=(
            CustomGroup(label="Catalan", tasks=[catalan]),
            CustomGroup(label="Spanish", tasks=[spanish]),
        ),
    )
    assert grouping.has_scoped_refs
    assert grouping.task_to_label == {}


def test_compute_custom_group_means_subset_scoped(mock_mteb_cache: ResultCache):
    """A subset-scoped entry contributes _get_score_fast(subsets=...), not
    the whole task's score."""
    model_name = "mteb/baseline-random-encoder"
    grouping = _make_scoped_custom_grouping()
    tasks = mteb.get_tasks(["CataloniaTweetClassification", "Banking77Classification"])
    model_result = mock_mteb_cache.load_results(models=[model_name]).model_results[0]
    task_results = model_result.select_tasks(tasks).task_results
    by_name = {tr.task.metadata.name: tr for tr in task_results}

    out = _compute_custom_group_means(task_results, grouping)

    catalan_expected = by_name["CataloniaTweetClassification"]._get_score_fast(
        subsets=["catalan"]
    )
    spanish_expected = by_name["CataloniaTweetClassification"]._get_score_fast(
        subsets=["spanish"]
    )
    banking_expected = by_name["Banking77Classification"].get_score()

    assert out["Language::Catalan"] == pytest.approx(catalan_expected)
    assert out["Language::Mixed"] == pytest.approx(
        (spanish_expected + banking_expected) / 2
    )


def test_compute_custom_group_means_split_scoped(mock_mteb_cache: ResultCache):
    """A split-scoped entry contributes _get_score_fast(splits=...), not the
    whole task's score -- the LongEmbed shape."""
    model_name = "mteb/baseline-random-encoder"
    poem_val = mteb.get_task(
        "PoemSentimentClassification.v2", eval_splits=["validation"]
    )
    poem_test = mteb.get_task("PoemSentimentClassification.v2", eval_splits=["test"])
    banking = mteb.get_task("Banking77Classification")
    grouping = CustomGrouping(
        name="SplitDim",
        groups=(
            CustomGroup(label="ValOnly", tasks=[poem_val]),
            CustomGroup(label="Mixed", tasks=[poem_test, banking]),
        ),
    )
    tasks = mteb.get_tasks(
        ["PoemSentimentClassification.v2", "Banking77Classification"]
    )
    model_result = mock_mteb_cache.load_results(models=[model_name]).model_results[0]
    task_results = model_result.select_tasks(tasks).task_results
    by_name = {tr.task.metadata.name: tr for tr in task_results}

    out = _compute_custom_group_means(task_results, grouping)

    val_expected = by_name["PoemSentimentClassification.v2"]._get_score_fast(
        splits=["validation"]
    )
    test_expected = by_name["PoemSentimentClassification.v2"]._get_score_fast(
        splits=["test"]
    )
    banking_expected = by_name["Banking77Classification"].get_score()

    assert out["SplitDim::ValOnly"] == pytest.approx(val_expected)
    assert out["SplitDim::Mixed"] == pytest.approx(
        (test_expected + banking_expected) / 2
    )


def _weight_regression_expected_values(
    by_name: dict, banking_expected: float
) -> tuple[float, float]:
    """(correct evenly-weighted value, wrong per-cell-weighted value) for
    test_compute_custom_group_means_weights_one_entry_per_ref_not_per_cell."""
    catalonia = by_name["CataloniaTweetClassification"]
    multi_cell_expected = catalonia._get_score_fast(
        subsets=["catalan", "spanish"], splits=["test"]
    )
    catalan_test = catalonia._get_score_fast(subsets=["catalan"], splits=["test"])
    spanish_test = catalonia._get_score_fast(subsets=["spanish"], splits=["test"])
    correct = (multi_cell_expected + banking_expected) / 2
    wrong = (catalan_test + spanish_test + banking_expected) / 3
    return correct, wrong


def test_compute_custom_group_means_weights_one_entry_per_ref_not_per_cell(
    mock_mteb_cache: ResultCache,
):
    """A 2-cell scoped entry (2 subsets x 1 split) and a 1-task entry in the
    same group must weight 50/50, not 2:1 by underlying cell count."""
    model_name = "mteb/baseline-random-encoder"
    multi_cell = mteb.get_task(
        "CataloniaTweetClassification",
        hf_subsets=["catalan", "spanish"],
        eval_splits=["test"],
    )
    banking = mteb.get_task("Banking77Classification")
    grouping = CustomGrouping(
        name="WeightDim",
        groups=(CustomGroup(label="G", tasks=[multi_cell, banking]),),
    )
    tasks = mteb.get_tasks(["CataloniaTweetClassification", "Banking77Classification"])
    model_result = mock_mteb_cache.load_results(models=[model_name]).model_results[0]
    task_results = model_result.select_tasks(tasks).task_results
    by_name = {tr.task.metadata.name: tr for tr in task_results}

    out = _compute_custom_group_means(task_results, grouping)
    correct, wrong = _weight_regression_expected_values(
        by_name, by_name["Banking77Classification"].get_score()
    )

    assert out["WeightDim::G"] == pytest.approx(correct)
    assert out["WeightDim::G"] != pytest.approx(wrong)


def test_compute_custom_group_means_nulls_on_missing_scope_coverage(
    mock_mteb_cache: ResultCache,
):
    """A model missing the subset a scoped entry needs nulls every group in
    the dimension (dimension-global policy, unchanged from whole-task)."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # missing catalan/test
    grouping = _make_scoped_custom_grouping()
    tasks = mteb.get_tasks(["CataloniaTweetClassification", "Banking77Classification"])
    model_result = mock_mteb_cache.load_results(models=[model_name]).model_results[0]
    task_results = model_result.select_tasks(tasks).task_results

    out = _compute_custom_group_means(task_results, grouping)

    assert out == {"Language::Catalan": None, "Language::Mixed": None}


@_skip_if_datasets_too_old
def test_get_score_matches_summary_table_custom_groups_scoped(
    mock_mteb_cache: ResultCache,
):
    """get_score() and the polars summary table agree on a scoped
    CustomGrouping's values for a fully-covered model (see the module note
    on partial models)."""
    grouping = _make_scoped_custom_grouping()
    tasks = mteb.get_tasks(["CataloniaTweetClassification", "Banking77Classification"])
    bench = Benchmark(
        name="mock_custom_groups_scoped_parity",
        tasks=tasks,
        aggregations=(BenchmarkAggregation.MEAN_TASK, grouping),
    )
    mock_results = mock_mteb_cache.load_results()

    get_score_out = bench.get_score(mock_results)
    pl_df = mock_results.select_tasks(bench.tasks)._to_results_df(bench.tasks)
    summary_pl = bench._create_summary_table(pl_df).df
    summary_by_model = {row["Model"]: row for row in summary_pl.iter_rows(named=True)}

    model_name = "mteb/baseline-random-encoder"  # fully covers both subsets
    scores = get_score_out[model_name]
    srow = summary_by_model[model_name]
    for gs_key, summary_col in [
        ("Language::Catalan", "__cg__Language::Catalan"),
        ("Language::Mixed", "__cg__Language::Mixed"),
    ]:
        assert scores[gs_key] is not None and srow[summary_col] is not None
        assert np.isclose(scores[gs_key], srow[summary_col])
