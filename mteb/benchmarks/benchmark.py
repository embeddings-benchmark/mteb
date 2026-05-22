from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import huggingface_hub
import pandas as pd
import yaml
from huggingface_hub import DatasetCard, DatasetCardData

from mteb._hf_integration.eval_model import HFEvalMeta, HFEvalTaskConfig
from mteb._hf_integration.hf_hub_utils import _get_file_on_hub
from mteb.abstasks.abstask import AbsTask
from mteb.types import StrURL

from ._benchmark_metrics import (
    LeaderboardMetrics,
    _compute_mean_task,
    _compute_mean_task_type,
)

if TYPE_CHECKING:
    from mteb.abstasks.aggregated_task import AbsTaskAggregate
    from mteb.results import BenchmarkResults, ModelResult

logger = logging.getLogger(__name__)


@lru_cache
def _get_benchmarks_on_leaderboard() -> set[str]:
    from mteb.benchmarks._leaderboard_menu import (
        GP_BENCHMARK_ENTRIES,
        R_BENCHMARK_ENTRIES,
        MenuEntry,
    )

    entries = GP_BENCHMARK_ENTRIES + R_BENCHMARK_ENTRIES

    def __extract_benchmarks(
        entries: Sequence[Benchmark | MenuEntry],
    ) -> list[Benchmark]:
        benchmarks = []
        for entry in entries:
            if isinstance(entry, Benchmark):
                benchmarks.append(entry)
            else:
                benchmarks.extend(__extract_benchmarks(entry.benchmarks))
        return benchmarks

    names = {benchmark.name for benchmark in __extract_benchmarks(entries)}

    return names


@dataclass
class Benchmark:
    """A benchmark object intended to run a certain benchmark within MTEB.

    Args:
        name: The name of the benchmark
        aliases: Alternative names for the benchmark
        tasks: The tasks within the benchmark.
        description: A description of the benchmark, should include its intended goal and potentially a description of its construction
        reference: A link reference, to a source containing additional information typically to a paper, leaderboard or github.
        citation: A bibtex citation
        contacts: The people to contact in case of a problem in the benchmark, preferably a GitHub handle.

    Examples:
        >>> Benchmark(
        ...     name="MTEB(custom)",
        ...     tasks=mteb.get_tasks(
        ...         tasks=["AmazonCounterfactualClassification", "AmazonPolarityClassification"],
        ...         languages=["eng"],
        ...     ),
        ...     description="A custom benchmark"
        ... )
    """

    name: str
    tasks: Sequence[AbsTask]
    aliases: Sequence[str] = field(default_factory=tuple)
    description: str | None = None
    reference: StrURL | None = None
    citation: str | None = None
    contacts: list[str] | None = None
    icon: str | None = None
    display_name: str | None = None
    language_view: list[str] | Literal["all"] = field(default_factory=list)
    benchmark_hf_repo: str | None = None

    @property
    def display_on_leaderboard(self) -> bool:
        """Whether the benchmark should be displayed on the leaderboard."""
        benchmarks_on_leaderboard = _get_benchmarks_on_leaderboard()
        return self.name in benchmarks_on_leaderboard

    def __iter__(self) -> Iterator[AbsTask]:
        return iter(self.tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> AbsTask:
        return self.tasks[index]

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create summary table. Called by the leaderboard app.

        Returns:
            A pandas DataFrame representing the summary results.
        """
        from mteb.benchmarks._create_table import _add_model_metadata

        scores = self.get_score(benchmark_results)
        if not scores:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table = pd.DataFrame.from_dict(scores, orient="index")
        rank_cols = {LeaderboardMetrics.rank_borda}
        score_cols = [c for c in joint_table.columns if c not in rank_cols]
        joint_table = joint_table[joint_table[score_cols].notna().any(axis=1)]
        if joint_table.empty:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table.index.name = "model_name"
        joint_table = joint_table.sort_values(
            LeaderboardMetrics.rank_borda, ascending=True
        )
        joint_table = joint_table.reset_index()

        _task_names_key = tuple(sorted(t.metadata.name for t in self.tasks))
        joint_table = _add_model_metadata(
            joint_table, _task_names_key, include_zero_shot=True
        )
        joint_table.insert(
            0,
            LeaderboardMetrics.rank_borda,
            joint_table.pop(LeaderboardMetrics.rank_borda),
        )
        return joint_table

    def _create_per_task_table(  # noqa: PLR6301
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create per-task table. Called by the leaderboard app.

        Returns:
            A pandas DataFrame representing the per-task results.
        """
        from mteb.benchmarks._create_table import (
            _create_per_task_table_from_benchmark_results,
        )

        return _create_per_task_table_from_benchmark_results(benchmark_results)

    def _create_per_language_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create per-language table. Called by the leaderboard app.

        Returns:
            A pandas DataFrame representing the per-language results.
        """
        from mteb.benchmarks._create_table import (
            _create_per_language_table_from_benchmark_results,
        )

        if self.language_view == "all" or len(self.language_view) > 0:
            return _create_per_language_table_from_benchmark_results(
                benchmark_results, self.language_view
            )
        else:
            no_results_frame = pd.DataFrame(
                {
                    "No results": [
                        "The per-language table is not available for this benchmark."
                    ]
                }
            )
            return no_results_frame

    def push_collection_to_hub(
        self,
        hf_username: str,
        collection_name: str | None = None,
    ) -> None:
        """Push the benchmark collection to Hugging Face Hub.

        Args:
            hf_username: Hugging Face username or organization name
            collection_name: Name for the collection on Hugging Face Hub. If not provided, the benchmark name will be used.
        """
        collections = huggingface_hub.list_collections(owner=hf_username)
        collection_name = collection_name or self.name
        existing_collection = None
        for collection in collections:
            if collection.title == collection_name:
                existing_collection = collection
                break

        if existing_collection is None:
            description = self.description
            if description and len(description) > 150:
                description = description[:147] + "..."
            collection = huggingface_hub.create_collection(
                title=collection_name,
                namespace=hf_username,
                # hf collections have a 150 character limit for description, so we truncate it if it's too long
                description=description if description else None,
            )
        else:
            # list collections would output only 4 items
            collection = huggingface_hub.get_collection(
                collection_slug=existing_collection.slug
            )

        existing_items = {item.item_id for item in collection.items}

        for task in self.tasks:
            tasks = (
                cast("AbsTaskAggregate", task).tasks if task.is_aggregate else [task]
            )
            for benchmark_task in tasks:
                task_path = benchmark_task.metadata.dataset["path"]
                if task_path in existing_items:
                    continue
                huggingface_hub.add_collection_item(
                    collection_slug=collection.slug,
                    item_id=task_path,
                    item_type="dataset",
                )
                existing_items.add(task_path)

    def __repr__(self) -> str:
        n_tasks = len(self.tasks)
        max_len = 50
        desc = self.description if self.description else ""
        desc = f"'{desc[:max_len]}..." if len(desc) > max_len else f"'{desc}'"
        return f"{self.__class__.__name__}(name='{self.name}', description={desc}, tasks=[...] (#{n_tasks}), ...)"

    def _generate_benchmark_card(self) -> DatasetCard:
        """Generate a README/dataset card for this benchmark."""
        template_path = Path(__file__).parent / "benchmark_card_template.md"

        task_rows = [
            {
                "name": task.metadata.name,
                "reference": task.metadata.reference,
                "simplified_type": task.metadata.simplified_task_type,
                "description": task.metadata.description or "",
            }
            for task in self.tasks
        ]

        return cast(
            "DatasetCard",
            DatasetCard.from_template(
                card_data=DatasetCardData(tags=["mteb", "benchmark"]),
                template_path=str(template_path),
                benchmark_name=self.name,
                benchmark_description=self.description,
                tasks=task_rows,
                citation=self.citation,
            ),
        )

    def push_benchmark_card_to_hub(
        self,
        *,
        create_pr: bool = False,
    ) -> None:
        """Push a README benchmark card to the HuggingFace Hub dataset repo."""
        if self.benchmark_hf_repo is None:
            raise ValueError(
                "`benchmark_hf_repo` must be set to push a benchmark card to the hub."
            )

        if not huggingface_hub.repo_exists(self.benchmark_hf_repo, repo_type="dataset"):
            huggingface_hub.create_repo(
                self.benchmark_hf_repo,
                repo_type="dataset",
            )

        card = self._generate_benchmark_card()
        card.push_to_hub(
            self.benchmark_hf_repo,
            repo_type="dataset",
            commit_message="Add benchmark card",
            create_pr=create_pr,
        )

    def push_eval_to_hub(
        self,
        *,
        create_pr: bool = False,
    ) -> None:
        """Push `eval.yaml` to the HuggingFace Hub

        Args:
            create_pr: Whether to create the PR
        """
        eval_file_name = "eval.yaml"

        if self.benchmark_hf_repo is None:
            raise ValueError(
                "`benchmark_hf_repo` must be set to push eval config to the hub."
            )

        existing_eval_path = _get_file_on_hub(
            repo_id=self.benchmark_hf_repo,
            file_name=eval_file_name,
            repo_type="dataset",
        )

        # handle multiple tasks in one repo (e.g. BRIGHT)
        existing_eval = None
        if existing_eval_path is not None:
            with Path(existing_eval_path).open(encoding="utf-8") as f:
                existing_eval_dict = yaml.safe_load(f)
            if existing_eval_dict is not None:
                existing_eval = HFEvalMeta.model_validate(existing_eval_dict)

        benchmark_config = self._to_hf_eval_config()
        benchmark_config = (
            benchmark_config.merge(existing_eval) if existing_eval else benchmark_config
        )

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as tmp_file:
            tmp_file.write(benchmark_config.to_yaml())
            tmp_file.flush()

            huggingface_hub.upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo=eval_file_name,
                repo_id=self.benchmark_hf_repo,
                repo_type="dataset",
                commit_message="Add eval config",
                create_pr=create_pr,
            )

    def _to_hf_eval_config(self) -> HFEvalMeta:
        return HFEvalMeta(
            name=self.name,
            description=self.description,
            tasks=[
                HFEvalTaskConfig(
                    id=self.name,
                    config=None,
                    split=None,
                )
            ],
        )

    def _get_model_score(
        self,
        model_result: ModelResult,
    ) -> dict[str, float | None]:
        """Compute aggregated scores for a single model."""
        from collections import defaultdict

        import numpy as np

        from mteb.benchmarks._create_table import _split_on_capital

        filtered = model_result.select_tasks(self.tasks).task_results
        if len(filtered) < len(self.tasks):
            raise ValueError(
                "Some scores of benchmark are missing. Please, run model on full benchmark tasks"
            )

        type_to_scores: dict[str, list[float]] = defaultdict(list)
        type_has_none: set[str] = set()
        for tr in filtered:
            score = tr.get_score()
            task_type = tr.task.metadata.type
            if score is None or np.isnan(score):
                type_has_none.add(task_type)
            else:
                type_to_scores[task_type].append(score)

        type_means: dict[str, float | None] = {
            _split_on_capital(task_type): (
                None
                if task_type in type_has_none
                else sum(scores) / len(scores)
                if scores
                else None
            )
            for task_type, scores in type_to_scores.items()
        }

        return {
            LeaderboardMetrics.mean_task: _compute_mean_task(filtered),
            LeaderboardMetrics.mean_task_type: _compute_mean_task_type(filtered),
            **type_means,
        }

    def get_score(
        self,
        results: BenchmarkResults,
        *,
        raise_error: bool = False,
    ) -> dict[str, dict[str, float | None]]:
        """Get aggregated scores for all models in *results*.

        The benchmark class controls how scores are aggregated — subclasses may
        override this method to customise the returned metrics.

        Args:
            results: A `BenchmarkResults` object containing the model
                results to score.
            raise_error: Weather to raise an error on missing results.

        Returns:
            A dict mapping each model name to a dict with the keys:

            - ``LeaderboardMetrics.mean_task``: mean score across all benchmark tasks.
            - ``LeaderboardMetrics.mean_task_type``: mean of per-task-type means.
            - per-task-type means (e.g. ``"Retrieval"``, ``"Classification"``).
            - ``LeaderboardMetrics.rank_borda``: Borda count rank (1 = best). Each model earns
                ``n - rank`` points per task; points are summed and the model
                with the highest total is ranked 1. Matches the leaderboard.
        """
        from mteb.benchmarks._create_table import _get_borda_rank

        bench_results = results.join_revisions()
        scores: dict[str, dict[str, float | None]] = {}
        per_task_rows: dict[str, dict[str, float | None]] = {}

        for model_result in bench_results:
            per_task_rows[model_result.model_name] = {}
            filtered = model_result.select_tasks(self.tasks).task_results
            try:
                scores[model_result.model_name] = self._get_model_score(model_result)
            except ValueError:
                if raise_error:
                    raise
                logger.warning(
                    "Some task results are missing. Filling results with None"
                )
                scores[model_result.model_name] = {
                    LeaderboardMetrics.mean_task: None,
                    LeaderboardMetrics.mean_task_type: None,
                }
                continue

            per_task_rows[model_result.model_name] = {
                tr.task_name: tr.get_score() for tr in filtered
            }

        if per_task_rows:
            per_task_df = pd.DataFrame.from_dict(per_task_rows, orient="index").reindex(
                list(per_task_rows.keys())
            )
            if per_task_df.shape[1] > 0:
                borda_ranks = _get_borda_rank(per_task_df)
                for name, rank in borda_ranks.items():
                    scores[name][LeaderboardMetrics.rank_borda] = int(rank)  # type: ignore[index]
            else:
                for name, model_scores in scores.items():
                    model_scores[LeaderboardMetrics.rank_borda] = None
        else:
            for name, model_scores in scores.items():
                model_scores[LeaderboardMetrics.rank_borda] = None

        return scores


class RtebBenchmark(Benchmark):
    """Wrapper for RTEB benchmark."""

    def get_score(
        self,
        results: BenchmarkResults,
        *,
        raise_error: bool = False,
    ) -> dict[str, dict[str, float | None]]:
        """Get scores for RTEB: mean of public tasks only, Borda rank on public tasks."""
        from mteb.benchmarks._create_table import _get_borda_rank

        bench_results = results.join_revisions()
        data = results.to_dataframe(format="long")
        if "is_public" in data.columns:
            public_task_names = set(
                data.loc[data["is_public"] == True, "task_name"].unique()  # noqa: E712
            )
        else:
            public_task_names = set(results._filter_tasks(is_public=True).task_names)

        scores: dict[str, dict[str, float | None]] = {}
        per_task_rows: dict[str, dict[str, float | None]] = {}

        for model_result in bench_results:
            filtered = model_result.select_tasks(self.tasks).task_results
            public_filtered = [
                tr for tr in filtered if tr.task_name in public_task_names
            ]
            try:
                if len(filtered) < len(self.tasks):
                    raise ValueError(
                        "Some scores of benchmark are missing. Please, run model on full benchmark tasks"
                    )
                scores[model_result.model_name] = {
                    LeaderboardMetrics.mean_task: _compute_mean_task(public_filtered),
                }
            except ValueError:
                if raise_error:
                    raise
                logger.warning(
                    "Some task results are missing. Filling results with None"
                )
                scores[model_result.model_name] = {LeaderboardMetrics.mean_task: None}
                continue

            per_task_rows[model_result.model_name] = {
                tr.task_name: tr.get_score() for tr in public_filtered
            }

        if per_task_rows:
            per_task_df = pd.DataFrame.from_dict(per_task_rows, orient="index").reindex(
                list(per_task_rows.keys())
            )
            if per_task_df.shape[1] > 0:
                borda_ranks = _get_borda_rank(per_task_df)
                for name, rank in borda_ranks.items():
                    scores[name][LeaderboardMetrics.rank_borda] = int(rank)
            else:
                for name, model_scores in scores.items():
                    model_scores[LeaderboardMetrics.rank_borda] = None
        else:
            for name, model_scores in scores.items():
                model_scores[LeaderboardMetrics.rank_borda] = None

        return scores

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        from mteb.benchmarks._create_table import _add_model_metadata

        scores = self.get_score(benchmark_results)
        if not scores:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table = pd.DataFrame.from_dict(scores, orient="index")
        joint_table = joint_table[
            joint_table[[LeaderboardMetrics.mean_task]].notna().any(axis=1)
        ]
        if joint_table.empty:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table.index.name = "model_name"
        joint_table = joint_table.sort_values(
            LeaderboardMetrics.rank_borda, ascending=True
        )
        joint_table = joint_table.reset_index()

        joint_table = _add_model_metadata(joint_table, include_zero_shot=False)
        joint_table.insert(
            0,
            LeaderboardMetrics.rank_borda,
            joint_table.pop(LeaderboardMetrics.rank_borda),
        )
        return joint_table


class HUMEBenchmark(Benchmark):
    """Wrapper for HUME benchmark."""

    def get_score(
        self,
        results: BenchmarkResults,
        *,
        raise_error: bool = False,
    ) -> dict[str, dict[str, float | None]]:
        """Get scores for HUME: subset-weighted mean, Borda rank on per-subset matrix."""
        from mteb.benchmarks._create_table import (
            _get_borda_rank,
            _get_means_per_types,
            _split_on_capital,
        )

        results = results.select_tasks(self.tasks)
        data = results.to_dataframe(format="long")
        if data.empty:
            return {}

        per_task = data.pivot(index="model_name", columns="task_name", values="score")
        to_remove = per_task.isna().all(axis="columns")
        per_task = per_task.drop(per_task[to_remove].index)
        if per_task.empty:
            return {}

        mean_per_type = _get_means_per_types(per_task)
        mean_per_type_wide = mean_per_type.pivot(
            index="model_name", columns="task_type", values="score"
        )
        mean_per_type_wide.columns = [
            _split_on_capital(col) for col in mean_per_type_wide.columns
        ]

        detailed_data = results.to_dataframe(aggregation_level="subset", format="long")
        overall_subset_mean = detailed_data.groupby("model_name")["score"].mean()
        per_subset = detailed_data.pivot(
            index="model_name", columns=["task_name", "subset"], values="score"
        )
        borda_ranks = _get_borda_rank(per_subset)

        scores: dict[str, dict[str, float | None]] = {}
        for model_name in per_task.index:
            subset_mean = overall_subset_mean.get(model_name)
            model_scores: dict[str, float | None] = {
                LeaderboardMetrics.mean_subset: None
                if subset_mean is None or pd.isna(subset_mean)
                else float(subset_mean),
            }
            for col in mean_per_type_wide.columns:
                val = (
                    mean_per_type_wide.loc[model_name, col]
                    if model_name in mean_per_type_wide.index
                    else None
                )
                model_scores[col] = None if val is None or pd.isna(val) else float(val)
            model_scores[LeaderboardMetrics.rank_borda] = (
                int(borda_ranks[model_name])
                if model_name in borda_ranks.index
                else None
            )
            scores[model_name] = model_scores

        return scores

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        from mteb.benchmarks._create_table import _add_model_metadata

        scores = self.get_score(benchmark_results)
        if not scores:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table = pd.DataFrame.from_dict(scores, orient="index")
        rank_cols = {LeaderboardMetrics.rank_borda}
        score_cols = [c for c in joint_table.columns if c not in rank_cols]
        joint_table = joint_table[joint_table[score_cols].notna().any(axis=1)]
        if joint_table.empty:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table.index.name = "model_name"
        joint_table = joint_table.sort_values(
            LeaderboardMetrics.mean_subset, ascending=False
        )
        joint_table = joint_table.reset_index()

        _task_names_key = tuple(sorted(t.metadata.name for t in self.tasks))
        joint_table = _add_model_metadata(
            joint_table, _task_names_key, include_zero_shot=True
        )
        joint_table.insert(
            0,
            LeaderboardMetrics.rank_borda,
            joint_table.pop(LeaderboardMetrics.rank_borda),
        )
        return joint_table


class MIEBBenchmark(Benchmark):
    """Wrapper for MIEB benchmark."""

    def get_score(
        self,
        results: BenchmarkResults,
        *,
        raise_error: bool = False,
    ) -> dict[str, dict[str, float | None]]:
        """Get scores for MIEB: typed mean as 'Mean (Task)', Borda rank, sequential rank."""
        from mteb.benchmarks._create_table import (
            _get_borda_rank,
            _get_means_per_types,
            _split_on_capital,
        )

        results = results.select_tasks(self.tasks)
        data = results.to_dataframe(format="long")
        if data.empty:
            return {}

        per_task = data.pivot(index="model_name", columns="task_name", values="score")
        to_remove = per_task.isna().all(axis="columns")
        per_task = per_task.drop(per_task[to_remove].index)
        if per_task.empty:
            return {}

        mean_per_type = _get_means_per_types(per_task)
        mean_per_type_wide = mean_per_type.pivot(
            index="model_name", columns="task_type", values="score"
        )
        mean_per_type_wide.columns = [
            _split_on_capital(col) for col in mean_per_type_wide.columns
        ]
        if "Any Any Multilingual Retrieval" in mean_per_type_wide.columns:
            mean_per_type_wide = mean_per_type_wide.rename(
                columns={"Any Any Multilingual Retrieval": "Multilingual Retrieval"}
            )
        if "Any Any Retrieval" in mean_per_type_wide.columns:
            mean_per_type_wide = mean_per_type_wide.rename(
                columns={"Any Any Retrieval": "Retrieval"}
            )

        typed_mean = mean_per_type_wide.mean(skipna=False, axis=1)
        borda_ranks = _get_borda_rank(per_task)
        sorted_models = typed_mean.sort_values(ascending=False).index.tolist()
        sequential_rank = {model: i + 1 for i, model in enumerate(sorted_models)}

        scores: dict[str, dict[str, float | None]] = {}
        for model_name in per_task.index:
            tm = typed_mean.get(model_name)
            model_scores: dict[str, float | None] = {
                LeaderboardMetrics.mean_task: None
                if tm is None or pd.isna(tm)
                else float(tm),
            }
            for col in mean_per_type_wide.columns:
                val = (
                    mean_per_type_wide.loc[model_name, col]
                    if model_name in mean_per_type_wide.index
                    else None
                )
                model_scores[col] = None if val is None or pd.isna(val) else float(val)
            model_scores[LeaderboardMetrics.rank_borda] = (
                int(borda_ranks[model_name])
                if model_name in borda_ranks.index
                else None
            )
            model_scores[LeaderboardMetrics.rank] = sequential_rank.get(model_name)
            scores[model_name] = model_scores

        return scores

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        from mteb.benchmarks._create_table import _add_model_metadata

        scores = self.get_score(benchmark_results)
        if not scores:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table = pd.DataFrame.from_dict(scores, orient="index")
        rank_cols = {LeaderboardMetrics.rank_borda, LeaderboardMetrics.rank}
        score_cols = [c for c in joint_table.columns if c not in rank_cols]
        joint_table = joint_table[joint_table[score_cols].notna().any(axis=1)]
        if joint_table.empty:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table.index.name = "model_name"
        joint_table = joint_table.sort_values(LeaderboardMetrics.rank, ascending=True)
        joint_table = joint_table.reset_index()

        _task_names_key = tuple(sorted(t.metadata.name for t in self.tasks))
        joint_table = _add_model_metadata(
            joint_table, _task_names_key, include_zero_shot=True
        )
        joint_table.insert(
            0,
            LeaderboardMetrics.rank_borda,
            joint_table.pop(LeaderboardMetrics.rank_borda),
        )
        joint_table.insert(
            0, LeaderboardMetrics.rank, joint_table.pop(LeaderboardMetrics.rank)
        )
        return joint_table


class VidoreBenchmark(Benchmark):
    """Wrapper for Vidore3 benchmark."""

    def get_score(
        self,
        results: BenchmarkResults,
        *,
        raise_error: bool = False,
    ) -> dict[str, dict[str, float | None]]:
        """Get scores for Vidore: public/private means, task-type mean as 'Mean (Task)', sequential rank."""
        from mteb.benchmarks._create_table import (
            _get_means_per_types,
            _split_on_capital,
        )

        results = results.select_tasks(self.tasks)
        data = results.to_dataframe(format="long")
        if data.empty:
            return {}

        if "is_public" in data.columns:
            public_task_names = list(
                data.loc[data["is_public"] == True, "task_name"].unique()  # noqa: E712
            )
            private_task_names = list(
                data.loc[data["is_public"] == False, "task_name"].unique()  # noqa: E712
            )
        else:
            public_task_names = results._filter_tasks(is_public=True).task_names
            private_task_names = results._filter_tasks(is_public=False).task_names

        per_task = data.pivot(index="model_name", columns="task_name", values="score")
        to_remove = per_task.isna().all(axis="columns")
        per_task = per_task.drop(per_task[to_remove].index)
        if per_task.empty:
            return {}

        mean_per_type = _get_means_per_types(per_task)
        mean_per_type_wide = mean_per_type.pivot(
            index="model_name", columns="task_type", values="score"
        )
        mean_per_type_wide.columns = [
            _split_on_capital(col) for col in mean_per_type_wide.columns
        ]
        task_type_col = mean_per_type_wide.columns[0]  # "Document Understanding"

        public_cols = [c for c in per_task.columns if c in public_task_names]
        private_cols = [c for c in per_task.columns if c in private_task_names]
        public_mean = (
            per_task[public_cols].mean(skipna=False, axis=1)
            if public_cols
            else pd.Series(None, index=per_task.index, dtype=float)
        )
        private_mean = (
            per_task[private_cols].mean(skipna=False, axis=1)
            if private_cols
            else pd.Series(None, index=per_task.index, dtype=float)
        )

        sort_df = pd.DataFrame(
            {
                task_type_col: mean_per_type_wide[task_type_col],
                "Mean (Public)": public_mean,
                "Mean (Private)": private_mean,
            }
        )
        sort_df = sort_df.sort_values(
            [task_type_col, "Mean (Public)", "Mean (Private)"], ascending=False
        )

        def _to_float(v: object) -> float | None:
            return None if v is None or pd.isna(v) else float(v)  # type: ignore[arg-type]

        scores: dict[str, dict[str, float | None]] = {}
        for i, (model_name, row) in enumerate(sort_df.iterrows()):
            scores[model_name] = {
                LeaderboardMetrics.mean_task: _to_float(row[task_type_col]),
                "Mean (Public)": _to_float(row["Mean (Public)"]),
                "Mean (Private)": _to_float(row["Mean (Private)"]),
                LeaderboardMetrics.rank_mean_task: i + 1,
            }

        return scores

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        from mteb.benchmarks._create_table import _add_model_metadata

        scores = self.get_score(benchmark_results)
        if not scores:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table = pd.DataFrame.from_dict(scores, orient="index")
        rank_cols = {LeaderboardMetrics.rank_mean_task}
        score_cols = [c for c in joint_table.columns if c not in rank_cols]
        joint_table = joint_table[joint_table[score_cols].notna().any(axis=1)]
        if joint_table.empty:
            return pd.DataFrame({"No results": ["You can try relaxing your criteria"]})

        joint_table.index.name = "model_name"
        joint_table = joint_table.sort_values(
            LeaderboardMetrics.rank_mean_task, ascending=True
        )
        joint_table = joint_table.reset_index()

        joint_table = _add_model_metadata(joint_table, include_zero_shot=False)
        joint_table.insert(
            0,
            LeaderboardMetrics.rank_mean_task,
            joint_table.pop(LeaderboardMetrics.rank_mean_task),
        )
        return joint_table
