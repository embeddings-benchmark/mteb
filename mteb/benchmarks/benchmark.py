from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import huggingface_hub
import pandas as pd
import polars as pl
import yaml
from huggingface_hub import DatasetCard, DatasetCardData

from mteb._helpful_enum import HelpfulStrEnum
from mteb._hf_integration.eval_model import HFEvalMeta, HFEvalTaskConfig
from mteb._hf_integration.hf_hub_utils import _get_file_on_hub
from mteb.abstasks.abstask import AbsTask
from mteb.types import StrURL

if TYPE_CHECKING:
    from mteb.abstasks.aggregated_task import AbsTaskAggregate
    from mteb.benchmarks._create_table import SummaryTable
    from mteb.results import BenchmarkResults, ModelResult
    from mteb.results.task_result import TaskResult

logger = logging.getLogger(__name__)


class BenchmarkAggregation(HelpfulStrEnum):
    """Aggregation columns a benchmark can surface in its leaderboard summary.

    Inherits from ``str`` so the values serialise as plain strings in JSON
    and round-trip cleanly through pydantic without a custom encoder.

    Each value also knows how to compute its own Python-side scores via
    [aggregate][mteb.benchmarks.benchmark.BenchmarkAggregation.aggregate] —
    the same dispatch the leaderboard's polars summary builder follows
    internally, so both paths stay in lockstep.
    """

    MEAN_TASK = "mean_task"
    """Show the ``Mean (Task)`` aggregate column."""
    MEAN_TASK_TYPE = "mean_task_type"
    """Show the ``Mean (TaskType)`` aggregate column."""
    TASK_TYPES = "task_types"
    """Show one column per per-task-type mean."""
    PUBLIC_PRIVATE = "public_private"
    """Show ``Mean (Public)`` and ``Mean (Private)`` (split-aware benchmarks)."""
    MEAN_SUBSET = "mean_subset"
    """Show ``Mean (Subset)``: subset-weighted mean used by HUME-style benchmarks."""

    @property
    def summary_columns(self) -> tuple[str, ...]:
        """Polars summary column names this aggregation produces (with spaces).

        Returns `()` for
        [TASK_TYPES][mteb.benchmarks.benchmark.BenchmarkAggregation.TASK_TYPES]
        — per-type columns are dynamic (one per task type observed in the
        benchmark) and are surfaced separately as `type_cols` on the summary.
        """
        return _AGGREGATION_SUMMARY_COLUMNS.get(self, ())

    @property
    def get_score_keys(self) -> tuple[str, ...]:
        """[`Benchmark.get_score`][mteb.benchmarks.benchmark.Benchmark.get_score] dict keys this aggregation produces (no spaces).

        Same as
        [summary_columns][mteb.benchmarks.benchmark.BenchmarkAggregation.summary_columns]
        with the space before the opening parenthesis removed — keeps the
        get_score contract (`"Mean(Task)"`) in sync with the polars summary
        column names (`"Mean (Task)"`). Returns `()` for
        [TASK_TYPES][mteb.benchmarks.benchmark.BenchmarkAggregation.TASK_TYPES]
        (dynamic per-type keys).
        """
        return tuple(c.replace(" (", "(") for c in self.summary_columns)

    def aggregate(self, task_results: list[TaskResult]) -> dict[str, float | None]:
        """Compute this aggregation's per-model scores in Python.

        Mirrors the polars branch that
        [_create_summary_table][mteb.benchmarks._create_table._create_summary_table]
        would take for this enum value, so
        [Benchmark.get_score][mteb.benchmarks.benchmark.Benchmark.get_score]
        and the leaderboard summary stay numerically consistent on the same
        data. Returned keys are the
        [get_score_keys][mteb.benchmarks.benchmark.BenchmarkAggregation.get_score_keys]
        for this aggregation.

        Args:
            task_results: All
                [TaskResult][mteb.results.task_result.TaskResult] objects for
                one model on the benchmark's task set.

        Returns:
            dict: Score keys → scalar (or `None` when any input score is
                missing/NaN). Keys come from
                [get_score_keys][mteb.benchmarks.benchmark.BenchmarkAggregation.get_score_keys]
                — except
                [TASK_TYPES][mteb.benchmarks.benchmark.BenchmarkAggregation.TASK_TYPES],
                which emits one key per task type seen in `task_results`.
        """
        from mteb.benchmarks._benchmark_metrics import (
            _compute_mean_public_private,
            _compute_mean_subset,
            _compute_mean_task,
            _compute_mean_task_type,
            _task_types_or_nulls,
        )

        match self:
            case BenchmarkAggregation.MEAN_TASK:
                return {self.get_score_keys[0]: _compute_mean_task(task_results)}
            case BenchmarkAggregation.MEAN_TASK_TYPE:
                return {self.get_score_keys[0]: _compute_mean_task_type(task_results)}
            case BenchmarkAggregation.TASK_TYPES:
                return _task_types_or_nulls(task_results)
            case BenchmarkAggregation.PUBLIC_PRIVATE:
                return _compute_mean_public_private(task_results)
            case BenchmarkAggregation.MEAN_SUBSET:
                return _compute_mean_subset(task_results)
            case _:
                # Surfaces immediately if a new ``BenchmarkAggregation`` member
                # is added without a matching Python compute path here.
                raise NotImplementedError(
                    f"BenchmarkAggregation.{self.name} has no Python-side "
                    f"compute registered in {type(self).__name__}.aggregate"
                )


# Single source of truth for which polars summary columns each aggregation
# produces. The ``get_score_keys`` (no-space form) is derived from this.
# Display order in the summary's ``mean_cols`` follows enum declaration
# order: MEAN_TASK → MEAN_TASK_TYPE → PUBLIC_PRIVATE → MEAN_SUBSET.
_AGGREGATION_SUMMARY_COLUMNS: dict[BenchmarkAggregation, tuple[str, ...]] = {
    BenchmarkAggregation.MEAN_TASK: ("Mean (Task)",),
    BenchmarkAggregation.MEAN_TASK_TYPE: ("Mean (TaskType)",),
    BenchmarkAggregation.PUBLIC_PRIVATE: ("Mean (Public)", "Mean (Private)"),
    BenchmarkAggregation.MEAN_SUBSET: ("Mean (Subset)",),
}


# Priority order for picking the summary's ``primary_metric_col`` — the
# column the rank is built on top of. First aggregation in this tuple that's
# present in ``aggregations`` wins. Aggregations not in this tuple
# (``TASK_TYPES``) never serve as the primary.
_PRIMARY_METRIC_PRIORITY: tuple[BenchmarkAggregation, ...] = (
    BenchmarkAggregation.MEAN_TASK,
    BenchmarkAggregation.MEAN_SUBSET,
    BenchmarkAggregation.MEAN_TASK_TYPE,
    BenchmarkAggregation.PUBLIC_PRIVATE,
)


@lru_cache
def _get_benchmarks_on_leaderboard() -> set[str]:
    from mteb.benchmarks._leaderboard_menu import (
        HOME_BENCHMARK_ENTRIES,
        MenuEntry,
    )

    entries = HOME_BENCHMARK_ENTRIES

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
        superseded_by: Benchmark name with newer version of benchmark
        aggregations: Which aggregations to use in on leaderboard
        summary_sort_column: The column to sort benchmarks by on leaderboard

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
    superseded_by: Sequence[str] | None = None
    # Api aggregation functions
    aggregations: Sequence[BenchmarkAggregation] = (
        BenchmarkAggregation.MEAN_TASK,
        BenchmarkAggregation.MEAN_TASK_TYPE,
        BenchmarkAggregation.TASK_TYPES,
    )
    # Whether the leaderboard summary table surfaces the Zero-shot column.
    # Off for benchmarks where model training-data annotations don't cover
    # the task set (e.g. ViDoRe), so every row would otherwise render as a
    # misleading 100%. The API echoes this on ``BenchmarkSummarySchema`` and
    # the frontend hides the column when False.
    show_zero_shot: bool = True
    # Sort column(s) for the leaderboard summary. ``None`` keeps the default
    # ``Rank (Borda)`` sort; a string or tuple of strings sorts by those
    # columns descending and adds a 1-indexed ``summary_rank_column`` rank.
    summary_sort_column: ClassVar[str | Sequence[str] | None] = None
    # Name of the 1-indexed rank column added when ``summary_sort_column`` is
    # set. ``None`` falls back to ``"Rank"`` (Borda stays as a trailing col).
    summary_rank_column: ClassVar[str | None] = None

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

    def _build_per_task_pivot(  # noqa: PLR6301
        self, pl_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, list[str]] | None:
        """Compute the standard (model × task) wide pivot once.

        Callers building both summary and per-task tables from the same long
        frame can pass the result to both via the ``pivot`` kwarg to halve
        polars CPU on the pivot step. Subclasses whose summary builder needs
        an is_public-aware pivot still benefit because their per-task table
        builder reuses this one. ``None`` when the input frame is empty.
        """
        from mteb.benchmarks._create_table import _build_per_task_pivot

        return _build_per_task_pivot(pl_df)

    def _create_summary_table(self, pl_df: pl.DataFrame) -> SummaryTable:
        """Create summary table from a long polars pre-agg frame.

        Thin wrapper around
        [_create_summary_table][mteb.benchmarks._create_table._create_summary_table]
        that forwards
        [aggregations][mteb.benchmarks.benchmark.Benchmark.aggregations],
        [summary_sort_column][mteb.benchmarks.benchmark.Benchmark.summary_sort_column],
        and
        [summary_rank_column][mteb.benchmarks.benchmark.Benchmark.summary_rank_column].
        Called by the leaderboard app.
        """
        from mteb.benchmarks._create_table import _create_summary_table

        return _create_summary_table(
            pl_df,
            aggregations=self.aggregations,
            sort_by=self.summary_sort_column,
            rank_column_name=self.summary_rank_column,
        )

    def _create_per_task_table(  # noqa: PLR6301
        self,
        pl_df: pl.DataFrame,
        *,
        pivot: tuple[pl.DataFrame, list[str]] | None = None,
    ) -> pl.DataFrame:
        """Create per-task table from a long polars pre-agg frame. Called by the leaderboard app."""
        from mteb.benchmarks._create_table import (
            _create_per_task_table_from_benchmark_results,
        )

        return _create_per_task_table_from_benchmark_results(pl_df, pivot=pivot)

    def _create_per_language_table(self, pl_df: pl.DataFrame) -> pl.DataFrame:
        """Create per-language table from a long polars pre-agg frame. Called by the leaderboard app."""
        from mteb.benchmarks._create_table import (
            _create_per_language_table_from_benchmark_results,
        )

        if self.language_view == "all" or len(self.language_view) > 0:
            return _create_per_language_table_from_benchmark_results(
                pl_df, self.language_view
            )
        return pl.DataFrame(
            {
                "No results": [
                    "The per-language table is not available for this benchmark."
                ]
            }
        )

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
        """Compute aggregated scores for a single model.

        Drives the per-aggregation compute via
        [BenchmarkAggregation.aggregate][mteb.benchmarks.benchmark.BenchmarkAggregation.aggregate],
        so the keys returned match what
        [_create_summary_table][mteb.benchmarks.benchmark.Benchmark._create_summary_table]
        would surface for the same `self.aggregations` set.

        Args:
            model_result: The model whose results to aggregate.

        Returns:
            dict: Score keys produced by `self.aggregations` mapped to their
                values. Keys include `"Mean(Task)"`, `"Mean(TaskType)"`,
                per-type means, `"Mean(Public)"`/`"Mean(Private)"`, and
                `"Mean(Subset)"` depending on the aggregation set.

        Raises:
            ValueError: If the model is missing results for some benchmark tasks.
        """
        filtered = model_result.select_tasks(self.tasks).task_results
        if len(filtered) < len(self.tasks):
            raise ValueError(
                "Some scores of benchmark are missing. Please, run model on full benchmark tasks"
            )

        scores: dict[str, float | None] = {}
        for aggregation in self.aggregations:
            scores.update(aggregation.aggregate(filtered))
        return scores

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
            A dict mapping each model name to a dict whose keys are
            determined by
            [aggregations][mteb.benchmarks.benchmark.Benchmark.aggregations].
            Possible keys include:

            - `"Mean(Task)"`: mean score across all benchmark tasks (when
                [MEAN_TASK][mteb.benchmarks.benchmark.BenchmarkAggregation.MEAN_TASK]
                is enabled).
            - `"Mean(TaskType)"`: mean of per-task-type means (when
                [MEAN_TASK_TYPE][mteb.benchmarks.benchmark.BenchmarkAggregation.MEAN_TASK_TYPE]
                is enabled).
            - per-task-type means keyed by raw type name (e.g. `"Retrieval"`)
                when
                [TASK_TYPES][mteb.benchmarks.benchmark.BenchmarkAggregation.TASK_TYPES]
                is enabled.
            - `"Mean(Public)"` / `"Mean(Private)"` when
                [PUBLIC_PRIVATE][mteb.benchmarks.benchmark.BenchmarkAggregation.PUBLIC_PRIVATE]
                is enabled.
            - `"Rank"`: Borda count rank (1 = best). Each model earns
                `n - rank` points per task; points are summed and the model
                with the highest total is ranked 1. Matches the leaderboard.
                Always present.
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
                    t.metadata.name: None for t in self.tasks
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
                per_task_pl = pl.from_pandas(
                    per_task_df.reset_index(names="model_name")
                )
                task_cols = list(per_task_df.columns)
                borda_list = (
                    per_task_pl.select(_get_borda_rank(task_cols)).to_series().to_list()
                )
                for name, rank in zip(per_task_df.index, borda_list):
                    scores[name]["Rank"] = int(rank)
            else:
                for name, model_scores in scores.items():
                    model_scores["Rank"] = None
        else:
            for name, model_scores in scores.items():
                model_scores["Rank"] = None

        return scores


@dataclass
class RtebBenchmark(Benchmark):
    """Wrapper for RTEB benchmark.

    issue 3902: private RTEB tasks are temporarily hidden from the
    leaderboard summary. The override filters `is_public=True` before
    delegating to
    [Benchmark._create_summary_table][mteb.benchmarks.benchmark.Benchmark._create_summary_table],
    so `Mean (Task)` and `Rank (Borda)` are both computed from public tasks
    only.
    """

    aggregations: Sequence[BenchmarkAggregation] = (BenchmarkAggregation.MEAN_TASK,)
    # RTEB task names aren't tracked in model ``training_datasets`` lists,
    # so the computed zero-shot percentage is 100 % for every row. Hide the
    # column rather than render a misleading uniform value.
    show_zero_shot: bool = False

    def _create_summary_table(self, pl_df: pl.DataFrame) -> SummaryTable:
        if "is_public" in pl_df.columns:
            pl_df = pl_df.filter(pl.col("is_public"))
        return super()._create_summary_table(pl_df)


@dataclass
class HUMEBenchmark(Benchmark):
    """Wrapper for HUME benchmark.

    Summary uses
    [MEAN_SUBSET][mteb.benchmarks.benchmark.BenchmarkAggregation.MEAN_SUBSET]
    so each task-language subset is weighted equally —
    [Benchmark._create_summary_table][mteb.benchmarks.benchmark.Benchmark._create_summary_table]
    routes to the subset-weighted builder.
    """

    aggregations: Sequence[BenchmarkAggregation] = (
        BenchmarkAggregation.MEAN_SUBSET,
        BenchmarkAggregation.TASK_TYPES,
    )


@dataclass
class MIEBBenchmark(Benchmark):
    """Wrapper for MIEB benchmark."""

    aggregations: Sequence[BenchmarkAggregation] = (
        BenchmarkAggregation.MEAN_TASK_TYPE,
        BenchmarkAggregation.TASK_TYPES,
    )
    # Rank rows by the per-type mean column rather than Borda count. Honoured
    # by ``Benchmark._create_summary_table`` (passed as ``sort_by``).
    summary_sort_column: ClassVar[str] = "Mean (TaskType)"


@dataclass
class VidoreBenchmark(Benchmark):
    """Wrapper for Vidore3 benchmark.

    Summary shows `Mean (Task)`, `Mean (Public)`, `Mean (Private)` and
    ranks 1-indexed by `Mean (Task)` (tie-broken by public then private
    mean) —
    [Benchmark._create_summary_table][mteb.benchmarks.benchmark.Benchmark._create_summary_table]
    honours these via
    [summary_sort_column][mteb.benchmarks.benchmark.Benchmark.summary_sort_column]
    +
    [summary_rank_column][mteb.benchmarks.benchmark.Benchmark.summary_rank_column].
    """

    aggregations: Sequence[BenchmarkAggregation] = (
        BenchmarkAggregation.MEAN_TASK,
        BenchmarkAggregation.PUBLIC_PRIVATE,
    )
    # ViDoRe task names aren't tracked in model ``training_datasets`` lists,
    # so the computed zero-shot percentage is 100 % for every row. Hide the
    # column rather than render a misleading uniform value.
    show_zero_shot: bool = False
    summary_sort_column: ClassVar[Sequence[str]] = (
        "Mean (Task)",
        "Mean (Public)",
        "Mean (Private)",
    )
    summary_rank_column: ClassVar[str] = "Rank (Mean Task)"
