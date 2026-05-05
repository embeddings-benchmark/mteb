from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from mteb.results.benchmark_results import BenchmarkResults

if TYPE_CHECKING:
    import pandas as pd


class CachedBenchmarkResults(BenchmarkResults):
    """BenchmarkResults with per-instance caches for the leaderboard.

    Keeps all cache state out of the core BenchmarkResults so it stays clean.

    Because all mutation methods (_filter_tasks, select_models, select_tasks)
    use ``type(self).model_construct(...)``, filtered children automatically
    inherit this subclass and get their own empty caches.
    """

    _pre_agg_df_cache: dict[bool, pd.DataFrame | None] = PrivateAttr(
        default_factory=dict
    )
    _props_cache: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    # --- pre-agg DataFrame cache ---

    def _build_pre_agg_df(self, include_model_revision: bool) -> pd.DataFrame | None:
        if include_model_revision in self._pre_agg_df_cache:
            return self._pre_agg_df_cache[include_model_revision]
        result = super()._build_pre_agg_df(include_model_revision)
        self._pre_agg_df_cache[include_model_revision] = result
        return result

    # --- property caches (each iterates all model_results; cache once per instance) ---

    @property
    def languages(self) -> list[str]:  # noqa: D102
        if "languages" not in self._props_cache:
            self._props_cache["languages"] = super().languages
        return self._props_cache["languages"]

    @property
    def domains(self) -> list[str]:  # noqa: D102
        if "domains" not in self._props_cache:
            self._props_cache["domains"] = super().domains
        return self._props_cache["domains"]

    @property
    def task_types(self) -> list[str]:  # noqa: D102
        if "task_types" not in self._props_cache:
            self._props_cache["task_types"] = super().task_types
        return self._props_cache["task_types"]

    @property
    def task_names(self) -> list[str]:  # noqa: D102
        if "task_names" not in self._props_cache:
            self._props_cache["task_names"] = super().task_names
        return self._props_cache["task_names"]

    @property
    def modalities(self) -> list[str]:  # noqa: D102
        if "modalities" not in self._props_cache:
            self._props_cache["modalities"] = super().modalities
        return self._props_cache["modalities"]
