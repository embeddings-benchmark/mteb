"""
Compare leaderboard performance: current PR (BenchmarkResults caching) vs baseline.

Measures:
  1. Startup: building all_benchmark_results + pre-warming dfs
  2. Filter-callback: simulated UI update_tables for varying filter combos
"""
from __future__ import annotations

import time
import warnings
import logging
import statistics
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import mteb
from mteb.results.benchmark_results import BenchmarkResults
from mteb.leaderboard.app import _filter_benchmark_results_for_tables
from mteb.benchmarks._create_table import (
    _create_summary_table_from_benchmark_results,
    _create_per_task_table_from_benchmark_results,
    _create_per_language_table_from_benchmark_results,
)

CACHE_PATH = Path.home() / ".cache/mteb/leaderboard/__cached_results.json"


def _simulate_table_build(benchmark, br):
    _create_summary_table_from_benchmark_results(br)
    _create_per_task_table_from_benchmark_results(br)
    lv = benchmark.language_view if benchmark.language_view else "all"
    _create_per_language_table_from_benchmark_results(br, lv)


def _clear_br_cache(br):
    """Wipe PrivateAttr caches added by the PR so the call is equivalent to baseline."""
    if hasattr(br, "_df_cache"):
        br._df_cache.clear()
    if hasattr(br, "_parent_results"):
        object.__setattr__(br, "_parent_results", None)
    if hasattr(br, "_filter_model_names"):
        object.__setattr__(br, "_filter_model_names", None)
    if hasattr(br, "_filter_task_names"):
        object.__setattr__(br, "_filter_task_names", None)


def timeit(fn, n=3):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


# ── load data ────────────────────────────────────────────────────────────────

print("Loading benchmark results from local JSON cache...")
t0 = time.perf_counter()
all_results = BenchmarkResults.from_disk(CACHE_PATH)
load_time = time.perf_counter() - t0
print(f"  Loaded in {load_time:.2f}s\n")

benchmarks = sorted(
    mteb.get_benchmarks(display_on_leaderboard=True), key=lambda x: x.name
)

print(f"Building all_benchmark_results for {len(benchmarks)} benchmarks...")
t0 = time.perf_counter()
all_benchmark_results = {
    b.name: all_results.select_tasks(b.tasks).join_revisions()
    for b in benchmarks
}
build_time = time.perf_counter() - t0
print(f"  Built in {build_time:.2f}s\n")

# Pick a representative large benchmark for the test
BENCH_NAME = "MTEB(eng, v2)"
if BENCH_NAME not in all_benchmark_results:
    BENCH_NAME = list(all_benchmark_results.keys())[0]

benchmark = next(b for b in benchmarks if b.name == BENCH_NAME)
br_full = all_benchmark_results[BENCH_NAME]

all_models = [mr.model_name for mr in br_full.model_results]
all_tasks  = [t.metadata.name for t in benchmark.tasks]
print(f"Benchmark: {BENCH_NAME!r}  |  {len(all_models)} models  |  {len(all_tasks)} tasks\n")

# Filter combos to test (simulate real UI interactions)
n_m, n_t = len(all_models), len(all_tasks)
filters = [
    ("all models, all tasks",    set(all_models),              set(all_tasks),               []),
    ("half models, all tasks",   set(all_models[:n_m//2]),     set(all_tasks),               []),
    ("all models, half tasks",   set(all_models),              set(all_tasks[:n_t//2]),       []),
    ("half models, half tasks",  set(all_models[:n_m//2]),     set(all_tasks[:n_t//2]),       []),
    ("20 models, 20 tasks",      set(all_models[:20]),         set(all_tasks[:20]),           []),
]

# ── BASELINE: fresh to_dataframe each call (no parent-link caching) ──────────

print("=" * 65)
print("BASELINE  (no pre-warming, clear cache before each call)")
print("=" * 65)

baseline_times = {}
for label, model_names, task_names, languages in filters:
    def run_baseline(mn=model_names, tn=task_names, la=languages):
        fbr = _filter_benchmark_results_for_tables(br_full, tn, mn, la)
        _clear_br_cache(fbr)  # discard any parent-link hints set by _filter_benchmark_results_for_tables
        _simulate_table_build(benchmark, fbr)

    times = timeit(run_baseline, n=3)
    baseline_times[label] = times
    print(f"  {label:40s}  {statistics.mean(times)*1000:7.0f} ms  {[f'{t*1000:.0f}' for t in times]}")

# ── PR BRANCH: pre-warm at startup, parent-link cache active ─────────────────

print()
print("=" * 65)
print("PR BRANCH  (startup pre-warm + parent-link df cache)")
print("=" * 65)

print("  Pre-warming df cache for all benchmarks...")
t0 = time.perf_counter()
for name, br in all_benchmark_results.items():
    try:
        br.to_dataframe(format="long", aggregation_level="task")
        if hasattr(br, "_df_cache"):
            br._df_cache.pop(("__pre_agg__", False), None)
    except Exception:
        pass
warm_time = time.perf_counter() - t0
print(f"  Pre-warm complete in {warm_time:.2f}s\n")

pr_times = {}
for label, model_names, task_names, languages in filters:
    def run_pr(mn=model_names, tn=task_names, la=languages):
        fbr = _filter_benchmark_results_for_tables(br_full, tn, mn, la)
        _simulate_table_build(benchmark, fbr)

    times = timeit(run_pr, n=3)
    pr_times[label] = times
    print(f"  {label:40s}  {statistics.mean(times)*1000:7.0f} ms  {[f'{t*1000:.0f}' for t in times]}")

# ── Summary ──────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Filter combo':<40}  {'Baseline':>10}  {'PR':>8}  {'Speedup':>7}")
print("-" * 70)
for label, _, _, _ in filters:
    b = statistics.mean(baseline_times[label]) * 1000
    p = statistics.mean(pr_times[label]) * 1000
    speedup = b / p if p > 0 else float("inf")
    print(f"{label:<40}  {b:>10.0f}  {p:>8.0f}  {speedup:>6.1f}x")

print()
print(f"Startup: load cache {load_time:.1f}s + build benchmarks {build_time:.1f}s "
      f"+ pre-warm {warm_time:.1f}s = {load_time+build_time+warm_time:.1f}s total")
print(f"(without pre-warm: {load_time:.1f}s + {build_time:.1f}s = {load_time+build_time:.1f}s)")
