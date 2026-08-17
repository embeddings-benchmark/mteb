---
title: "Rank stability"
icon: lucide/list-ordered
---

# Measuring rank stability

A benchmark aggregate is a mean over the tasks that happen to be in the suite, and
a ranking built from it can move when that suite does. Nothing in a score table
shows how much of the ordering would survive a different draw of tasks. This page
shows how to measure that with a task bootstrap: resample the tasks, one draw
shared by every model so comparisons stay paired, rank models by Borda (mean rank
across tasks) on each resample, and read off the range of ranks each model
occupies.

```python
import mteb
import numpy as np
import pandas as pd

benchmark = mteb.get_benchmark("MTEB(eng, v2)")
cache = mteb.ResultCache()
cache.download_from_remote()  # fetch the public results once; cached afterwards
results = cache.load_results(tasks=benchmark, only_main_score=True)

wide = results.to_dataframe(format="wide")  # one row per task, one column per model
wide = wide.set_index("task_name").select_dtypes("number")
wide = wide.dropna(axis="columns")  # (1)!

n_boot, rng = 1000, np.random.default_rng(0)
scores = wide.to_numpy(dtype=float)
n_tasks = scores.shape[0]

def borda_ranks(sample: np.ndarray) -> np.ndarray:
    """Rank models (1 = best) by mean per-task rank."""
    per_task = pd.DataFrame(sample).rank(axis=1, ascending=False)
    return per_task.mean(axis=0).rank(method="min").to_numpy()

boot = np.stack([
    borda_ranks(scores[rng.integers(0, n_tasks, n_tasks)])  # (2)!
    for _ in range(n_boot)
])

summary = pd.DataFrame({
    "rank": borda_ranks(scores).astype(int),
    "rank_low": np.percentile(boot, 2.5, axis=0).astype(int),
    "rank_high": np.percentile(boot, 97.5, axis=0).astype(int),
    "p_first": (boot == 1).mean(axis=0),
}, index=wide.columns).sort_values("rank")
```

1. A model missing a task has no score to resample; either drop those models, as
   here, or impute before ranking and say so.
2. One index draw applied to every model — the resample is *paired*. Resampling
   each model's tasks independently would break the cross-model correlation and
   overstate the uncertainty.

`summary` has one row per model: its rank on the full suite, the interval of
ranks it occupies across resamples, and how often it comes out first. On the
results available at the time of writing (41 tasks, 122 models with a complete
task set):

```text
                                   rank  rank_low  rank_high  p_first
ByteDance-Seed/Seed1.5-Embedding      1         1          5    0.444
annamodels/LGAI-Embedding-Preview     2         1          6    0.162
Qwen/Qwen3-Embedding-8B               3         1          7    0.103
Bytedance/Seed1.6-embedding           4         1          7    0.175
Kingsoft-LLM/QZhou-Embedding          5         1         10    0.123
Qwen/Qwen3-Embedding-4B               6         3          9    0.000
google/gemini-embedding-001           7         4         12    0.000
```

Two things to read from it:

- **The top group.** Every model whose `rank_low` is 1 is a defensible first
  place under a different draw of tasks — five models here, and the table's #1
  wins under half of the resamples. Reporting that group, rather than a lone
  #1, is the honest headline.
- **Adjacent rows with overlapping intervals** are orderings the suite does not
  support: 120 of the 121 adjacent pairs overlap in this run. The table prints
  the models in *some* order, but the bootstrap says most of that order is
  composition, not signal — distant models still separate cleanly.

What this measures is the sensitivity of the ranking to benchmark composition.
It is not the sampling error inside each task score — that lives at the task
level and would need per-task sample sizes.

## Showing it on a leaderboard

Two ways to surface this, differing in where the bootstrap runs. Precompute the
intervals alongside the aggregate for the default task set — cheap, but the
interval goes stale the moment a user filters tasks, since ranks are recomputed
on the filtered subset. Or run the loop above in the frontend on whatever task
subset is active: it is ~15 lines on a small matrix, so recomputing on filter
change is feasible, and the interval then always describes the table actually
shown. See the discussion in
[#5208](https://github.com/embeddings-benchmark/mteb/issues/5208).
