"""Quick breakdown: how much time is to_dataframe vs table building."""
from __future__ import annotations
import time, warnings, logging, statistics
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import mteb
from mteb.cache import ResultCache
from mteb.leaderboard.app import _filter_benchmark_results_for_tables
from mteb.benchmarks._create_table import (
    _create_summary_table_from_benchmark_results,
    _create_per_task_table_from_benchmark_results,
    _create_per_language_table_from_benchmark_results,
)

# Use the already-loaded results from a quick direct JSON parse trick
# (call _load_from_cache but it should find local cache and be fast now)
print("Loading (should use local cache, ~30-60s)...")
t0 = time.perf_counter()
cache = ResultCache()
all_results = cache._load_from_cache()
print(f"  Loaded in {time.perf_counter()-t0:.1f}s\n")

benchmarks = sorted(mteb.get_benchmarks(display_on_leaderboard=True), key=lambda x: x.name)
all_br = {b.name: all_results.select_tasks(b.tasks).join_revisions() for b in benchmarks}

BENCH_NAME = "MTEB(eng, v2)"
benchmark = next(b for b in benchmarks if b.name == BENCH_NAME)
br_full = all_br[BENCH_NAME]
all_models = [mr.model_name for mr in br_full.model_results]
all_tasks  = [t.metadata.name for t in benchmark.tasks]
print(f"Benchmark: {BENCH_NAME}  |  {len(all_models)} models  |  {len(all_tasks)} tasks\n")

# Pre-warm
for name, br in all_br.items():
    try:
        br.to_dataframe(format="long", aggregation_level="task")
        br._df_cache.pop(("__pre_agg__", False), None)
    except Exception:
        pass

ms, ts = set(all_models), set(all_tasks)
fbr = _filter_benchmark_results_for_tables(br_full, ts, ms, [])
lv = benchmark.language_view or "all"

print("Time breakdown (PR branch, all models, all tasks):")
n = 3
for label, fn in [
    ("to_dataframe(task, long)",     lambda: fbr.to_dataframe(format="long", aggregation_level="task")),
    ("to_dataframe(language, long)", lambda: fbr.to_dataframe(format="long", aggregation_level="language")),
    ("_create_summary_table",        lambda: _create_summary_table_from_benchmark_results(fbr)),
    ("_create_per_task_table",       lambda: _create_per_task_table_from_benchmark_results(fbr)),
    ("_create_per_language_table",   lambda: _create_per_language_table_from_benchmark_results(fbr, lv)),
]:
    # Cold run (clear cache)
    fbr._df_cache.clear()
    t0 = time.perf_counter()
    fn()
    cold = (time.perf_counter() - t0) * 1000
    # Warm run (cache populated)
    t0 = time.perf_counter()
    fn()
    warm = (time.perf_counter() - t0) * 1000
    print(f"  {label:<35}  cold={cold:6.0f}ms  warm={warm:5.0f}ms")
