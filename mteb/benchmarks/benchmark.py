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
import polars as pl
import yaml
from huggingface_hub import DatasetCard, DatasetCardData

from mteb._hf_integration.eval_model import HFEvalMeta, HFEvalTaskConfig
from mteb._hf_integration.hf_hub_utils import _get_file_on_hub
from mteb.abstasks.abstask import AbsTask
from mteb.types import StrURL

from ._benchmark_metrics import (
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

    def _create_summary_table(self, pl_df: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR6301
        """Create summary table from a long polars pre-agg frame. Called by the leaderboard app."""
        from mteb.benchmarks._create_table import (
            _create_summary_table_from_benchmark_results,
        )

        return _create_summary_table_from_benchmark_results(pl_df)

    def _create_per_task_table(self, pl_df: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR6301
        """Create per-task table from a long polars pre-agg frame. Called by the leaderboard app."""
        from mteb.benchmarks._create_table import (
            _create_per_task_table_from_benchmark_results,
        )

        return _create_per_task_table_from_benchmark_results(pl_df)

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
        """Compute aggregated scores for a single model."""
        filtered = model_result.select_tasks(self.tasks).task_results
        if len(filtered) < len(self.tasks):
            raise ValueError(
                "Some scores of benchmark are missing. Please, run model on full benchmark tasks"
            )
        return {
            "Mean(Task)": _compute_mean_task(filtered),
            "Mean(TaskType)": _compute_mean_task_type(filtered),
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

            - ``"Mean(Task)"``: mean score across all benchmark tasks.
            - ``"Mean(TaskType)"``: mean of per-task-type means.
            - ``"Rank"``: Borda count rank (1 = best). Each model earns
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
                borda_list = _get_borda_rank(per_task_pl, task_cols).to_list()
                for name, rank in zip(per_task_df.index, borda_list):
                    scores[name]["Rank"] = int(rank)  # type: ignore[index]
            else:
                for name, model_scores in scores.items():
                    model_scores["Rank"] = None
        else:
            for name, model_scores in scores.items():
                model_scores["Rank"] = None

        return scores


class RtebBenchmark(Benchmark):
    """Wrapper for RTEB benchmark."""

    def _create_summary_table(self, pl_df: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR6301
        from mteb.benchmarks._create_table import (
            _create_summary_table_mean_public_private,
        )

        joint_table = _create_summary_table_mean_public_private(
            pl_df, exclude_private_from_borda=True
        )
        if "No results" in joint_table.columns:
            return joint_table
        # issue 3902: temporary remove the private column from RTEB summary table
        joint_table = joint_table.drop("Mean (Private)", strict=False)
        # For RTEB: all tasks are Retrieval type, so Retrieval column = Mean (Task)
        # but due to 3902, if Private column existed, Mean (Task) was the mean of Public and Private so instead we drop Mean (Task) and rename Mean (Public) to Mean (Task)
        if "Retrieval" in joint_table.columns:
            joint_table = joint_table.rename({"Retrieval": "Mean (Task)"})
        return joint_table.drop("Mean (Task)", strict=False).rename(
            {"Mean (Public)": "Mean (Task)"}
        )


class HUMEBenchmark(Benchmark):
    """Wrapper for HUME benchmark."""

    def _create_summary_table(self, pl_df: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR6301
        from mteb.benchmarks._create_table import _create_summary_table_mean_subset

        return _create_summary_table_mean_subset(pl_df)


class MIEBBenchmark(Benchmark):
    """Wrapper for MIEB benchmark."""

    def _create_summary_table(self, pl_df: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR6301
        from mteb.benchmarks._create_table import _create_summary_table_mean_task_type

        return _create_summary_table_mean_task_type(
            pl_df, mean_column_name="Mean (Task)"
        )


class VidoreBenchmark(Benchmark):
    """Wrapper for Vidore3 benchmark."""

    def _create_vidore_summary_table(  # noqa: PLR6301
        self, pl_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Create the Vidore summary frame fully in polars.

        Args:
            pl_df: Long polars frame with ``model_name``, ``task_name``, ``score``,
                and ``is_public`` columns.

        Returns:
            Polars frame with one row per model, ready for further renames.
        """
        from mteb.benchmarks._create_table import (
            _attach_model_metadata,
            _get_means_per_types,
            _no_results_frame,
            _skipna_false_mean,
            _split_on_capital,
        )
        from mteb.get_tasks import get_task

        if pl_df.is_empty() or "model_name" not in pl_df.columns:
            return _no_results_frame()

        per_task_long = pl_df.group_by(["model_name", "task_name"]).agg(
            pl.col("score").mean(),
            pl.col("is_public").first(),
        )
        public_tasks = (
            per_task_long.filter(pl.col("is_public"))
            .get_column("task_name")
            .unique()
            .to_list()
        )
        private_tasks = (
            per_task_long.filter(~pl.col("is_public"))
            .get_column("task_name")
            .unique()
            .to_list()
        )
        per_task = per_task_long.pivot(
            on="task_name", index="model_name", values="score"
        )
        task_cols = [c for c in per_task.columns if c != "model_name"]
        if not task_cols:
            return _no_results_frame()
        per_task = per_task.filter(
            pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
        )
        if per_task.is_empty():
            return _no_results_frame()

        mean_per_type, type_cols = _get_means_per_types(per_task, task_cols)
        # Vidore tasks share a single task type — sort primarily by it, then by means.
        primary_type_col = _split_on_capital(get_task(task_cols[0]).metadata.type)

        public_present = [c for c in public_tasks if c in task_cols]
        private_present = [c for c in private_tasks if c in task_cols]

        joint_table = mean_per_type.join(
            per_task.select(
                "model_name",
                (
                    _skipna_false_mean(public_present).alias("Mean (Public)")
                    if public_present
                    else pl.lit(None).cast(pl.Float64).alias("Mean (Public)")
                ),
                (
                    _skipna_false_mean(private_present).alias("Mean (Private)")
                    if private_present
                    else pl.lit(None).cast(pl.Float64).alias("Mean (Private)")
                ),
            ),
            on="model_name",
            how="left",
        ).sort(
            [primary_type_col, "Mean (Public)", "Mean (Private)"],
            descending=True,
            nulls_last=True,
        )

        joint_table = _attach_model_metadata(joint_table).with_columns(
            (pl.int_range(0, pl.len()) + 1).cast(pl.Int64).alias("Rank (Mean Task)")
        )

        final_cols = [
            "Rank (Mean Task)",
            "Model",
            "Active Parameters (B)",
            "Total Parameters (B)",
            "Embedding Dimensions",
            "Max Tokens",
            "Mean (Public)",
            "Mean (Private)",
            *type_cols,
            "Release Date",
        ]
        return joint_table.select([c for c in final_cols if c in joint_table.columns])

    def _create_summary_table(self, pl_df: pl.DataFrame) -> pl.DataFrame:
        joint_table = self._create_vidore_summary_table(pl_df)
        # For ViDoRe (V1, V2, V3): all tasks are Document Understanding type, so
        # Document Understanding column == Mean (Task).
        if "Document Understanding" in joint_table.columns:
            joint_table = joint_table.rename({"Document Understanding": "Mean (Task)"})
        return joint_table
