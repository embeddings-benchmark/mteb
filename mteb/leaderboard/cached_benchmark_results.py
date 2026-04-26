from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from mteb.results.benchmark_results import BenchmarkResults

if TYPE_CHECKING:
    import pandas as pd


class CachedBenchmarkResults(BenchmarkResults):
    """BenchmarkResults with a per-instance pre-agg DataFrame cache.

    Keeps the caching concern in the leaderboard layer so the core
    BenchmarkResults stays free of cache state.

    Because all mutation methods (_filter_tasks, select_models, select_tasks)
    use ``type(self).model_construct(...)``, filtered children automatically
    inherit this subclass and get their own empty cache.
    """

    _pre_agg_df_cache: dict[bool, pd.DataFrame | None] = PrivateAttr(
        default_factory=dict
    )

    def _build_pre_agg_df(self, include_model_revision: bool) -> pd.DataFrame | None:
        if include_model_revision in self._pre_agg_df_cache:
            return self._pre_agg_df_cache[include_model_revision]
        result = super()._build_pre_agg_df(include_model_revision)
        self._pre_agg_df_cache[include_model_revision] = result
        return result
