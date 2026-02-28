from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import huggingface_hub
import pandas as pd

from mteb.abstasks.abstask import AbsTask
from mteb.types import StrURL

if TYPE_CHECKING:
    from mteb.abstasks.aggregated_task import AbsTaskAggregate
    from mteb.results import BenchmarkResults


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
    display_on_leaderboard: bool = True
    icon: str | None = None
    display_name: str | None = None
    language_view: list[str] | Literal["all"] = field(default_factory=list)

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
        from mteb.benchmarks._create_table import (
            _create_summary_table_from_benchmark_results,
        )

        return _create_summary_table_from_benchmark_results(benchmark_results)

    def _create_per_task_table(
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


class RtebBenchmark(Benchmark):
    """Wrapper for RTEB benchmark."""

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        from mteb.benchmarks._create_table import (
            _create_summary_table_mean_public_private,
        )

        joint_table = _create_summary_table_mean_public_private(
            benchmark_results, exclude_private_from_borda=True
        )
        # issue 3902: temporary remove the private column from RTEB summary table
        if "Mean (Private)" in joint_table.columns:
            joint_table = joint_table.drop(columns=["Mean (Private)"])
        # For RTEB: all tasks are Retrieval type, so Retrieval column = Mean (Task)
        # but due to 3902, if Private column existed, Mean (Task) was the mean of Public and Private so instead we drop Mean (Task) and rename Mean (Public) to Mean (Task)
        joint_table = joint_table.rename(columns={"Retrieval": "Mean (Task)"})
        if "Mean (Task)" in joint_table.columns:
            joint_table = joint_table.drop(columns=["Mean (Task)"])
        joint_table = joint_table.rename(columns={"Mean (Public)": "Mean (Task)"})

        return joint_table


class HUMEBenchmark(Benchmark):
    """Wrapper for HUME benchmark."""

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        from mteb.benchmarks._create_table import _create_summary_table_mean_subset

        return _create_summary_table_mean_subset(benchmark_results)


class MIEBBenchmark(Benchmark):
    """Wrapper for MIEB benchmark."""

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        from mteb.benchmarks._create_table import _create_summary_table_mean_task_type

        return _create_summary_table_mean_task_type(
            benchmark_results, mean_column_name="Mean (Task)"
        )


class VidoreBenchmark(Benchmark):
    """Wrapper for Vidore3 benchmark."""

    def _create_vidore_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        """Create summary table from BenchmarkResults.

        Returns a DataFrame with one row per model containing summary statistics
        and task type averages. Customized for Vidore benchmark.

        Args:
            benchmark_results: BenchmarkResults object containing model results

        Returns:
            DataFrame with model summaries, ready for styling in the leaderboard
        """
        import mteb
        from mteb.benchmarks._create_table import (
            _format_max_tokens,
            _format_n_parameters,
            _get_means_per_types,
            _split_on_capital,
        )
        from mteb.get_tasks import get_task

        data = benchmark_results.to_dataframe(format="long")

        if data.empty:
            no_results_frame = pd.DataFrame(
                {"No results": ["You can try relaxing your criteria"]}
            )
            return no_results_frame
        public_task_name = benchmark_results._filter_tasks(is_public=True).task_names
        private_task_name = benchmark_results._filter_tasks(is_public=False).task_names
        # Convert to DataFrame and pivot
        per_task = data.pivot(index="model_name", columns="task_name", values="score")

        # Remove models with no scores
        to_remove = per_task.isna().all(axis="columns")
        if to_remove.all():
            no_results_frame = pd.DataFrame(
                {"No results": ["You can try relaxing your criteria"]}
            )
            return no_results_frame

        models_to_remove = list(per_task[to_remove].index)
        per_task = per_task.drop(models_to_remove, axis=0)

        # Calculate means by task type
        mean_per_type = _get_means_per_types(per_task)
        mean_per_type = mean_per_type.pivot(
            index="model_name", columns="task_type", values="score"
        )
        mean_per_type.columns = [
            _split_on_capital(column) for column in mean_per_type.columns
        ]

        # Calculate overall means
        public_mean = per_task[public_task_name].mean(skipna=False, axis=1)
        private_mean = per_task[private_task_name].mean(skipna=False, axis=1)

        # Build joint table
        joint_table = mean_per_type.copy()
        joint_table.insert(1, "mean(public)", public_mean)
        joint_table.insert(2, "mean(private)", private_mean)
        task_type = get_task(
            per_task.columns[0]
        ).metadata.type  # "DocumentUnderstanding"
        joint_table = joint_table.sort_values(
            [_split_on_capital(task_type), "mean(public)", "mean(private)"],
            ascending=False,
        )

        joint_table = joint_table.reset_index()

        # Add model metadata
        model_metas = joint_table["model_name"].map(mteb.get_model_meta)
        joint_table = joint_table[model_metas.notna()]
        joint_table["model_link"] = model_metas.map(lambda m: m.reference)

        # Insert model metadata columns
        joint_table.insert(
            1,
            "Max Tokens",
            model_metas.map(lambda m: _format_max_tokens(m.max_tokens)),
        )
        joint_table.insert(
            1,
            "Embedding Dimensions",
            model_metas.map(lambda m: int(m.embed_dim) if m.embed_dim else None),
        )
        joint_table.insert(
            1,
            "Number of Parameters (B)",
            model_metas.map(lambda m: _format_n_parameters(m.n_parameters)),
        )
        joint_table.insert(
            1,
            "Memory Usage (MB)",
            model_metas.map(
                lambda m: int(m.memory_usage_mb) if m.memory_usage_mb else None
            ),
        )

        # Clean up model names (remove HF organization)
        joint_table["model_name"] = joint_table["model_name"].map(
            lambda name: name.split("/")[-1]
        )

        # Add markdown links to model names
        name_w_link = (
            "[" + joint_table["model_name"] + "](" + joint_table["model_link"] + ")"
        )
        joint_table["model_name"] = joint_table["model_name"].mask(
            joint_table["model_link"].notna(), name_w_link
        )
        joint_table = joint_table.drop(columns=["model_link"])

        # Rename columns
        rename_dict = {
            "model_name": "Model",
            "mean(public)": "Mean (Public)",
            "mean(private)": "Mean (Private)",
        }

        joint_table = joint_table.rename(columns=rename_dict)

        # Add Rank column
        joint_table.insert(
            0, "Rank (Mean Task)", [i + 1 for i in range(len(joint_table))]
        )

        return joint_table

    def _create_summary_table(
        self, benchmark_results: BenchmarkResults
    ) -> pd.DataFrame:
        joint_table = self._create_vidore_summary_table(benchmark_results)
        # For ViDoRe (V1, V2, V3): all tasks are Document Understanding type, so Document Understanding column = Mean (Task)
        joint_table = joint_table.rename(
            columns={"Document Understanding": "Mean (Task)"}
        )
        return joint_table
