"""Leaderboard performance benchmark suite.

Benchmarks the *current* (polars-based) leaderboard hot paths. The leaderboard
stores one ``pl.DataFrame`` per benchmark in ``all_benchmark_results`` and filters
it in ``update_tables`` with a polars mask, then builds the styled tables.

Scenarios
---------
  startup_parquet   Load the processed parquet cache (the fast startup path)
  startup_build     [--build] Load BenchmarkResults (JSON) + build the polars dict
                    + save the parquet cache (the slow path the cache replaces)
  table_build       summary / per_task / per_language build for a benchmark (cold+warm)
  filter_tasks      task-name filter (update_tables mask) + table build
  filter_models     model-name filter + table build
  filter_language   language filter on a multilingual benchmark + table build (cold+warm)
  filter_combo      language + model filter + table build
  model_filter      _filter_models: allow_all vs zero-shot modes (cold+warm) — exercises
                    the per-model get_training_datasets() cache
  bench_switch      build tables for a different (multilingual) benchmark

cold = first call (caches empty), warm = mean of subsequent calls.

Usage
-----
  python scripts/bench_leaderboard.py            # parquet-based scenarios (fast)
  python scripts/bench_leaderboard.py --build    # also measure the slow build path
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import polars as pl

import mteb
from mteb.benchmarks._create_table import _training_datasets_cached
from mteb.cache import ResultCache
from mteb.leaderboard.app import (
    MAX_MODEL_SIZE,
    MODEL_TYPE_CHOICES,
    _benchmark_full_languages,
    _filter_models,
)
from mteb.leaderboard.table import (
    apply_per_language_styling_from_benchmark,
    apply_per_task_styling_from_benchmark,
    apply_summary_styling_from_benchmark,
)
from mteb.results.benchmark_results import BenchmarkResults

DEFAULT = "MTEB(eng, v2)"
ALT = "MTEB(Multilingual, v2)"

# ── helpers ──────────────────────────────────────────────────────────────────


def hdr(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def row(label: str, cold_ms: float, warm_ms: float | None = None) -> None:
    if warm_ms is None:
        print(f"  {label:<46}  cold={cold_ms:>8.1f}ms")
    else:
        print(f"  {label:<46}  cold={cold_ms:>8.1f}ms  warm={warm_ms:>8.1f}ms")


def time_fn(fn, n: int = 3) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def cold_warm(fn, n: int = 3) -> tuple[float, float]:
    ts = time_fn(fn, n)
    return ts[0], statistics.mean(ts[1:]) if len(ts) > 1 else ts[0]


def filter_df(
    pl_df: pl.DataFrame,
    benchmark_name: str,
    model_names: set[str],
    task_names: set[str],
    languages: list[str] | None,
) -> pl.DataFrame:
    """Replicate the polars filter inside ``update_tables`` (model + task + language)."""
    mask = pl.col("model_name").is_in(model_names) & pl.col("task_name").is_in(
        task_names
    )
    if languages is not None and not set(languages).issuperset(
        _benchmark_full_languages(benchmark_name)
    ):
        lang_set = set(languages)
        mask &= (
            pl.col("language")
            .list.eval(pl.element().str.split("-").list.first().is_in(lang_set))
            .list.any()
        )
    return pl_df.lazy().filter(mask).collect(engine="streaming")


def build_tables(benchmark, pl_df: pl.DataFrame) -> None:
    apply_summary_styling_from_benchmark(benchmark, pl_df)
    apply_per_task_styling_from_benchmark(benchmark, pl_df)
    apply_per_language_styling_from_benchmark(benchmark, pl_df)


# ── data load ──────────────────────────────────────────────────────────────────


def load_all_benchmark_results(
    cache: ResultCache, benchmarks, *, measure_build: bool
) -> dict[str, pl.DataFrame]:
    """Return the per-benchmark polars dict, measuring whichever startup path is used."""
    parquet_path = cache.leaderboard_parquet_path

    if measure_build or not parquet_path.exists():
        hdr(
            "Scenario: startup_build (BenchmarkResults JSON -> per-benchmark frames -> parquet)"
        )
        t0 = time.perf_counter()
        all_results = cache._load_from_cache(rebuild=False)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  Load BenchmarkResults (JSON cache):     {load_ms:>8.0f}ms")

        t0 = time.perf_counter()
        per_benchmark = {
            b.name: all_results._to_results_df(b.tasks) for b in benchmarks
        }
        build_ms = (time.perf_counter() - t0) * 1000
        print(
            f"  Build {len(benchmarks)} per-benchmark frames:        {build_ms:>8.0f}ms"
        )

        t0 = time.perf_counter()
        BenchmarkResults.save_leaderboard_cache(per_benchmark, parquet_path)
        save_ms = (time.perf_counter() - t0) * 1000
        print(f"  Save parquet:                           {save_ms:>8.0f}ms")
        print(
            f"  Build-path total:                       {load_ms + build_ms + save_ms:>8.0f}ms"
        )
        return per_benchmark

    hdr("Scenario: startup_parquet (load per-benchmark leaderboard cache)")
    cold, warm = cold_warm(
        lambda: BenchmarkResults.load_leaderboard_cache(parquet_path), n=3
    )
    row(f"load {parquet_path.name}", cold, warm)
    loaded = BenchmarkResults.load_leaderboard_cache(parquet_path)
    return {b.name: loaded.get(b.name, pl.DataFrame()) for b in benchmarks}


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Measure the slow build path (load 2GB JSON + to_polars + save parquet) "
        "instead of loading the parquet cache.",
    )
    args = parser.parse_args()

    cache = ResultCache()
    benchmarks = sorted(
        mteb.get_benchmarks(display_on_leaderboard=True), key=lambda x: x.name
    )
    all_br = load_all_benchmark_results(cache, benchmarks, measure_build=args.build)

    # Resolve benchmarks (fall back to whatever is available)
    def pick(name: str) -> str:
        if name in all_br and not all_br[name].is_empty():
            return name
        return next(n for n, df in all_br.items() if not df.is_empty())

    default_name, alt_name = pick(DEFAULT), pick(ALT)
    benchmark = mteb.get_benchmark(default_name)
    benchmark_alt = mteb.get_benchmark(alt_name)
    df = all_br[default_name]
    df_alt = all_br[alt_name]

    all_models = df["model_name"].unique().to_list()
    all_tasks = df["task_name"].unique().to_list()
    all_models_set, all_tasks_set = set(all_models), set(all_tasks)
    half_models = set(all_models[: len(all_models) // 2])
    twenty_models = set(all_models[:20])
    half_tasks = set(all_tasks[: len(all_tasks) // 2])

    print(
        f"\n  Benchmark:     {default_name!r}  |  {len(all_models)} models  |  {len(all_tasks)} tasks"
    )
    print(
        f"  Alt benchmark: {alt_name!r}  |  {len(df_alt['model_name'].unique())} models  "
        f"|  {len(df_alt['task_name'].unique())} tasks"
    )

    # ── table build (all filters wide open) ──────────────────────────────────
    hdr(f"Scenario: table_build  ({default_name})")
    for label, fn in [
        ("summary", lambda: apply_summary_styling_from_benchmark(benchmark, df)),
        ("per_task", lambda: apply_per_task_styling_from_benchmark(benchmark, df)),
        (
            "per_language",
            lambda: apply_per_language_styling_from_benchmark(benchmark, df),
        ),
    ]:
        row(label, *cold_warm(fn))

    # ── filter_tasks / filter_models ─────────────────────────────────────────
    hdr(f"Scenario: filter_tasks + build  ({default_name})")
    for label, tasks in [
        ("all tasks", all_tasks_set),
        ("half tasks", half_tasks),
        ("10 tasks", set(all_tasks[:10])),
    ]:
        row(
            label,
            *cold_warm(
                lambda t=tasks: build_tables(
                    benchmark, filter_df(df, default_name, all_models_set, t, None)
                )
            ),
        )

    hdr(f"Scenario: filter_models + build  ({default_name})")
    for label, models in [
        ("all models", all_models_set),
        ("half models", half_models),
        ("20 models", twenty_models),
    ]:
        row(
            label,
            *cold_warm(
                lambda m=models: build_tables(
                    benchmark, filter_df(df, default_name, m, all_tasks_set, None)
                )
            ),
        )

    # ── language filtering (on the multilingual benchmark) ───────────────────
    alt_models = set(df_alt["model_name"].unique().to_list())
    alt_tasks = set(df_alt["task_name"].unique().to_list())
    alt_langs = sorted(_benchmark_full_languages(alt_name))
    some_langs = alt_langs[: max(1, len(alt_langs) // 4)]

    hdr(f"Scenario: filter_language + build  ({alt_name})")
    for label, langs in [
        (f"{len(some_langs)} langs (1/4)", some_langs),
        ("eng only", ["eng"] if "eng" in alt_langs else alt_langs[:1]),
    ]:
        row(
            label,
            *cold_warm(
                lambda la=langs: build_tables(
                    benchmark_alt,
                    filter_df(df_alt, alt_name, alt_models, alt_tasks, la),
                )
            ),
        )

    hdr(f"Scenario: filter_language + model + build  ({alt_name})")
    alt_20 = set(list(alt_models)[:20])
    row(
        "eng-only + 20 models",
        *cold_warm(
            lambda: build_tables(
                benchmark_alt,
                filter_df(
                    df_alt,
                    alt_name,
                    alt_20,
                    alt_tasks,
                    ["eng"] if "eng" in alt_langs else alt_langs[:1],
                ),
            )
        ),
    )

    # ── model filtering (_filter_models) ─────────────────────────────────────
    hdr(f"Scenario: model_filter (_filter_models, {alt_name} tasks)")
    alt_task_list = sorted(alt_tasks)
    alt_model_list = sorted(alt_models)
    for zs in ("allow_all", "remove_unknown", "only_zero_shot"):

        def _run(zs=zs):
            _filter_models(
                alt_model_list,
                alt_task_list,
                None,
                [],
                None,
                MAX_MODEL_SIZE,
                zs,
                MODEL_TYPE_CHOICES,
            )

        _training_datasets_cached.cache_clear()
        cold = time_fn(_run, n=1)[0]
        warm = statistics.mean(time_fn(_run, n=2))
        row(zs, cold, warm)

    # ── benchmark switch ─────────────────────────────────────────────────────
    hdr("Scenario: benchmark switch (build tables for alt benchmark)")
    row(
        f"switch to {alt_name!r}",
        *cold_warm(
            lambda: build_tables(
                benchmark_alt, filter_df(df_alt, alt_name, alt_models, alt_tasks, None)
            )
        ),
    )


if __name__ == "__main__":
    main()
