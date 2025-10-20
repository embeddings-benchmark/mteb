from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from itertools import chain
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

from mteb.abstasks.task_metadata import TaskCategory, TaskType
from mteb.models.get_model_meta import (
    _model_meta_from_cross_encoder,
    _model_meta_from_sentence_transformers,
)

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

import datasets

import mteb
from mteb.abstasks import AbsTask
from mteb.benchmarks import Benchmark
from mteb.models import (
    CrossEncoderWrapper,
    EncoderProtocol,
    ModelMeta,
    MTEBModels,
    SentenceTransformerEncoderWrapper,
)
from mteb.results import TaskResult
from mteb.types import ScoresDict

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


class MTEB:
    """Deprecated. Massive Text Embedding Benchmark (MTEB) evaluation pipeline."""

    _tasks: Iterable[str | AbsTask] | None
    tasks: list[AbsTask]

    @deprecated(
        "MTEB is deprecated and will be removed in future versions. "
        "Please use the `mteb.evaluate` function instead."
    )
    def __init__(
        self,
        tasks: Iterable[AbsTask | Benchmark],
        *,
        err_logs_path: str = "error_logs.txt",
    ) -> None:
        """Create an Evaluation pipeline, based on the provided tasks.

        Args:
            tasks: List of tasks or benchmarks to be evaluated, e.g. tasks returned by
                `mteb.get_tasks(["task1","task2"]) or `mteb.get_benchmark("MTEB(eng, classic)").
            err_logs_path: Path to save error logs.
        """
        from mteb.benchmarks import Benchmark

        self.tasks = list(tasks)
        if len(self.tasks) > 0 and isinstance(self.tasks[0], Benchmark):
            self.benchmarks = tasks
            self.tasks = list(chain.from_iterable(self.tasks))

        self.err_logs_path = Path(err_logs_path)
        self.last_evaluated_splits = {}

    @property
    def available_tasks(self) -> list[str]:
        """List of available task names."""
        return [x.metadata.name for x in self.tasks]

    @property
    def available_task_types(self) -> list[TaskType]:
        """List of available task types."""
        return sorted({x.metadata.type for x in self.tasks})

    @property
    def available_task_categories(self) -> set[TaskCategory]:
        """Set of available task categories."""
        return {x.metadata.category for x in self.tasks}

    def _display_tasks(self, task_list: Iterable[AbsTask], name: str | None = None):
        from rich.console import Console

        # disable logging for other ranks
        if int(os.getenv("RANK", 0)) != 0:
            return

        console = Console()
        if name:
            console.rule(f"[bold]{name}\n", style="grey15")
        for task_type in self.available_task_types:  # iterate through sorted task_types
            current_type_tasks = list(
                filter(lambda x: x.metadata.type == task_type, task_list)
            )
            if len(current_type_tasks) == 0:
                continue
            else:
                console.print(f"[bold]{task_type}[/]")
                for (
                    task
                ) in current_type_tasks:  # will be sorted as input to this function
                    prefix = "    - "
                    name = f"{task.metadata.name}"
                    category = f", [italic grey39]{task.metadata.category}[/]"
                    multilingual = (
                        f", [italic red]multilingual {len(task.hf_subsets)} / {len(task.metadata.eval_langs)} Subsets[/]"
                        if task.metadata.is_multilingual
                        else ""
                    )
                    console.print(f"{prefix}{name}{category}{multilingual}")
                console.print("\n")

    def mteb_benchmarks(self):
        """Get all benchmarks available in the MTEB."""
        from mteb.get_tasks import MTEBTasks

        # get all the MTEB specific benchmarks:
        sorted_mteb_benchmarks = sorted(
            self.benchmarks, key=lambda obj: obj.name.lower()
        )

        mteb_b, remaining_b = [], []
        for b in sorted_mteb_benchmarks:
            if "MTEB" in b.name:
                mteb_b.append(b)
            else:
                remaining_b.append(b)

        # place mteb first, then remaining
        sorted_mteb_benchmarks = mteb_b + remaining_b

        # task ordering within each benchmark should be alphabetical
        for st in sorted_mteb_benchmarks:
            st.tasks = MTEBTasks(
                sorted(st.tasks, key=lambda obj: obj.metadata.name.lower())
            )

        for benchmark in sorted_mteb_benchmarks:
            name = benchmark.name
            self._display_tasks(benchmark.tasks, name=name)

    @classmethod
    def mteb_tasks(cls):
        """Get all tasks available in the MTEB."""
        tasks = mteb.get_tasks()
        instance = cls(tasks)
        instance._display_tasks(tasks, name="MTEB tasks")

    def print_selected_tasks(self):
        """Print the selected tasks."""
        self._display_tasks(self.tasks, name="Selected tasks")

    def load_tasks_data(self):
        """Load datasets for the selected tasks."""
        logger.info(f"\n\n## Loading datasets for {len(self.tasks)} tasks")
        for task in self.tasks:
            logger.info(f"\n# Loading dataset for {task.metadata.name}")
            task.load_data()

    @staticmethod
    def _run_eval(
        task: AbsTask,
        model: MTEBModels,
        split: str,
        subsets_to_run: list[str] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ):
        tick = time()
        results = task.evaluate(
            model,
            split,
            subsets_to_run=subsets_to_run,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        tock = time()
        return results, tick, tock

    @staticmethod
    def _get_missing_splits(
        existing_results: TaskResult | None, task_eval_splits: list[str]
    ) -> list[str]:
        if existing_results is None:
            return task_eval_splits

        missing_splits = []
        for split in task_eval_splits:
            if split not in existing_results.scores:
                missing_splits.append(split)
            elif not existing_results.scores[
                split
            ]:  # Check if the split has any scores
                missing_splits.append(split)

        return missing_splits

    @staticmethod
    def _merge_results(
        existing_results: TaskResult, new_results: TaskResult
    ) -> TaskResult:
        merged_scores = existing_results.scores.copy()

        for split, scores in new_results.scores.items():
            if split in merged_scores:
                merged_scores[split] = MTEB._merge_split_scores(
                    merged_scores[split], scores
                )
            else:
                merged_scores[split] = scores

        existing_kg_co2_emissions = (
            existing_results.kg_co2_emissions
            if existing_results.kg_co2_emissions
            else 0
        )
        new_kg_co2_emissions = (
            new_results.kg_co2_emissions if new_results.kg_co2_emissions else 0
        )
        merged_kg_co2_emissions = None
        if existing_kg_co2_emissions and new_kg_co2_emissions:
            merged_kg_co2_emissions = existing_kg_co2_emissions + new_kg_co2_emissions
        merged_results = TaskResult(
            dataset_revision=new_results.dataset_revision,
            task_name=new_results.task_name,
            mteb_version=new_results.mteb_version,
            scores=merged_scores,
            evaluation_time=existing_results.evaluation_time
            + new_results.evaluation_time,
            kg_co2_emissions=merged_kg_co2_emissions,
        )

        return merged_results

    @staticmethod
    def _merge_split_scores(
        existing_scores: list[ScoresDict], new_scores: list[ScoresDict]
    ) -> list[ScoresDict]:
        merged = {score["hf_subset"]: score for score in existing_scores}
        for score in new_scores:
            merged[score["hf_subset"]] = score
        return list(merged.values())

    def run(
        self,
        model: MTEBModels | CrossEncoder | SentenceTransformer,
        verbosity: int = 1,
        output_folder: str | None = "results",
        eval_splits: list[str] | None = None,
        eval_subsets: list[str] | None = None,
        overwrite_results: bool = False,
        raise_error: bool = True,
        co2_tracker: bool = False,
        encode_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[TaskResult]:
        """Run the evaluation pipeline on the selected tasks.

        Args:
            model: Model to be used for evaluation
            verbosity: Verbosity level. Default is 1.
                0: Only shows a progress bar for tasks being processed.
                1: Shows a progress bar and prints task scores.
                2: Prints detailed output, including messages about loading datasets and task scores.
                3: Prints comprehensive logs for debugging, including all data loading and evaluation details.
            output_folder: Folder where the results will be saved. Default to 'results'. Where it will save the results in the format:
                `{output_folder}/{model_name}/{model_revision}/{task_name}.json`.
            eval_splits: List of splits to evaluate on. If None, the splits are taken from the task metadata.
            eval_subsets: List of subsets to evaluate on. If None, the subsets are taken from the task metadata.
            overwrite_results: Whether to overwrite existing results.
            raise_error: Whether to raise an error if an exception occurs during evaluation.
            co2_tracker: Whether to enable or disable CO2 emissions tracker using codecarbon.
            encode_kwargs: Additional keyword arguments to be passed to the model.encode method.
            kwargs: Additional arguments to be passed to `_run_eval` method and task.load_data.

        Returns:
            A list of TaskResult objects, one for each task evaluated.
        """
        # update logging to account for different levels of Verbosity (similar to the command line)
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if encode_kwargs is None:
            encode_kwargs = {}

        encode_kwargs["batch_size"] = encode_kwargs.get("batch_size", 32)

        if verbosity == 0:
            datasets.logging.set_verbosity(logging.CRITICAL)  # 40
            datasets.logging.disable_progress_bar()  # Disable progress bar
        elif verbosity == 1:
            datasets.logging.set_verbosity(logging.WARNING)
            datasets.logging.disable_progress_bar()  # Disable progress bar
        elif verbosity == 2:
            datasets.logging.set_verbosity(logging.INFO)
        elif verbosity == 3:
            datasets.logging.set_verbosity(logging.DEBUG)

        meta = self.create_model_meta(model)
        output_path = self._create_output_folder(meta, output_folder)

        if isinstance(model, SentenceTransformer):
            model = SentenceTransformerEncoderWrapper(model)
        elif isinstance(model, CrossEncoder):
            model = CrossEncoderWrapper(model)

        # Disable co2_tracker for API models
        if "API" in meta.framework:
            co2_tracker = False

        if output_path:
            self._save_model_metadata(meta, output_path)

        # Run selected tasks
        logger.info(f"\n\n## Evaluating {len(self.tasks)} tasks:")

        if verbosity > 0:
            self.print_selected_tasks()

        evaluation_results = []
        original_tasks = deepcopy(
            self.tasks
        )  # save them in case we re-use the object (e.g. for reranking)

        # To evaluate missing splits, we keep track of the task name and the corresponding splits.
        self.last_evaluated_splits = {}

        while len(self.tasks) > 0:
            task = self.tasks[0]
            logger.info(
                f"\n\n********************** Evaluating {task.metadata.name} **********************"
            )

            if task.is_aggregate:
                self_ = MTEB(tasks=task.metadata.tasks)
                task_results = self_.run(
                    model,
                    verbosity=verbosity - 1,
                    output_folder=output_folder,
                    eval_splits=eval_splits,
                    eval_subsets=eval_subsets,
                    overwrite_results=overwrite_results,
                    raise_error=raise_error,
                    co2_tracker=co2_tracker,
                    encode_kwargs=encode_kwargs,
                    **kwargs,
                )
                new_results = task.combine_task_results(task_results)
                evaluation_results.append(new_results)

                if output_path:
                    save_path = output_path / f"{task.metadata.name}.json"
                    new_results.to_disk(save_path)
                del self.tasks[0]
                continue

            # NOTE: skip evaluation if the model does not support all of the task's modalities.
            # If the model covers more than the task's modalities, evaluation will still be run.
            sorted_task_modalities = sorted(task.metadata.modalities)
            if meta.modalities is not None and any(
                m not in meta.modalities for m in sorted_task_modalities
            ):
                logger.info(
                    f"{meta.name} only supports {meta.modalities}, but the task modalities are {sorted_task_modalities}."
                )
                del self.tasks[0]  # empty memory
                continue

            task_eval_splits = (
                eval_splits if eval_splits is not None else task.eval_splits
            )
            task_subsets = task.hf_subsets

            existing_results = None
            save_path = None
            final_splits_to_run = task_eval_splits
            missing_evaluations = self._get_missing_evaluations(
                existing_results,
                task_eval_splits,
                task_subsets,
                eval_subsets,
            )

            if output_path:
                kwargs["output_folder"] = output_folder  # needed for retrieval tasks
                save_path = output_path / f"{task.metadata.name}.json"
                if save_path.exists():
                    existing_results = TaskResult.from_disk(save_path)

                    # Unified call to get missing splits and subsets
                    missing_evaluations = self._get_missing_evaluations(
                        existing_results,
                        task_eval_splits,
                        task_subsets,
                        eval_subsets,
                    )

                    if overwrite_results:
                        final_splits_to_run = task_eval_splits
                    else:
                        # Determine final splits to run
                        final_splits_to_run = []
                        # We need to run any split that is fully missing or has missing subsets
                        for sp, info in missing_evaluations.items():
                            if info["whole_split_missing"] or info["missing_subsets"]:
                                final_splits_to_run.append(sp)

                    if not overwrite_results and len(final_splits_to_run) == 0:
                        logger.info(
                            f"{task.metadata.name} results already exists. Loading results from disk."
                            f" Set overwrite_results=True to overwrite or `--overwrite`."
                        )
                        evaluation_results.append(existing_results)
                        del self.tasks[0]  # empty memory
                        continue

            # If no splits need to be run and results exist, skip
            if not final_splits_to_run:
                if existing_results is not None:
                    evaluation_results.append(existing_results)
                else:
                    logger.info(
                        f"No splits to evaluate for {task.metadata.name}. Skipping evaluation."
                    )
                self.last_evaluated_splits[task.metadata.name] = []
                del self.tasks[0]
                continue

            try:
                task.check_if_dataset_is_superseded()
                task.load_data()

                task_results = {}
                evaluation_time = 0
                kg_co2_emissions: int | None = 0 if co2_tracker else None

                self.last_evaluated_splits[task.metadata.name] = []

                for split in final_splits_to_run:
                    info = missing_evaluations[split]

                    # Determine subsets to run for this split
                    # If the whole split is missing, run all required subsets
                    # If only some subsets are missing, run only those
                    subsets_to_run = (
                        info["missing_subsets"]
                        if not overwrite_results
                        else (eval_subsets or task_subsets)
                    )

                    if (
                        info["whole_split_missing"] or overwrite_results
                    ) and task_subsets is None:
                        subsets_to_run = ["default"]

                    if co2_tracker:
                        try:
                            from codecarbon import EmissionsTracker
                        except ImportError:
                            raise ImportError(
                                "codecarbon is not installed. Please install it using `pip install 'mteb[codecarbon]'` to track CO₂ emissions."
                            )
                        logger.warning(
                            "Evaluating multiple MTEB runs simultaneously will produce incorrect CO₂ results"
                        )
                        with EmissionsTracker(
                            save_to_file=False,
                            save_to_api=False,
                            logging_logger=logger,
                            allow_multiple_runs=True,
                        ) as tracker:
                            results, tick, tock = self._run_eval(
                                task,
                                model,
                                split,
                                encode_kwargs=encode_kwargs,
                                subsets_to_run=subsets_to_run,
                                **kwargs,
                            )

                        kg_co2_emissions += (
                            tracker.final_emissions
                        )  # expressed as kilograms of CO₂-equivalents
                    else:
                        results, tick, tock = self._run_eval(
                            task,
                            model,
                            split,
                            subsets_to_run=subsets_to_run,
                            encode_kwargs=encode_kwargs,
                            **kwargs,
                        )

                    logger.info(
                        f"Evaluation for {task.metadata.name} on {split} took {tock - tick:.2f} seconds"
                    )
                    evaluation_time += tock - tick

                    task_results[split] = results
                    if verbosity >= 1:
                        logger.info(f"Scores: {task_results[split]}")

                    self.last_evaluated_splits[task.metadata.name].append(split)

                # Create new TaskResult
                new_results = TaskResult.from_task_results(
                    task,
                    task_results,
                    evaluation_time=evaluation_time,
                    kg_co2_emissions=kg_co2_emissions,
                )

                # Merge with existing if needed
                if output_path and save_path.exists():
                    existing_results = TaskResult.from_disk(save_path)
                if existing_results:
                    merged_results = self._merge_results(existing_results, new_results)
                else:
                    merged_results = new_results

                if output_path:
                    merged_results.to_disk(save_path)

                evaluation_results.append(merged_results)

            except Exception as e:
                logger.error(f"Error while evaluating {task.metadata.name}: {e}")
                if raise_error:
                    raise e
                logger.error(
                    f"Please check all the error logs at: {self.err_logs_path}"
                )
                with self.err_logs_path.open("a") as f_out:
                    f_out.write(f"{datetime.now()} >>> {task.metadata.name}\n")
                    f_out.write(traceback.format_exc())
                    f_out.write("\n\n")

            # empty memory
            del self.tasks[0]

        self.tasks = original_tasks
        return evaluation_results

    @staticmethod
    def create_model_meta(model: MTEBModels) -> ModelMeta:
        """Create a ModelMeta object for the given model."""
        if hasattr(model, "mteb_model_meta") and model.mteb_model_meta is not None:
            meta = model.mteb_model_meta  # type: ignore
        else:
            meta = MTEB._get_model_meta(model)

        # create a copy of the meta to avoid modifying the original object
        meta = deepcopy(meta)
        meta.revision = meta.revision or "no_revision_available"
        meta.name = meta.name or "no_model_name_available"

        return meta

    def _create_output_folder(
        self, model_meta: ModelMeta, output_folder: str | None
    ) -> Path | None:
        """Create output folder for the results.

        Args:
            model_meta: ModelMeta object of the model being evaluated.
            output_folder: Base output folder. If None, no folder is created and results are not saved.

        Returns:
            Path to the created output folder or None if output_folder is None.
        """
        if output_folder is None:
            return None

        model_revision: str = model_meta.revision  # type: ignore
        model_path_name = model_meta.model_name_as_path()

        output_path = Path(output_folder) / model_path_name / model_revision
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _save_model_metadata(model_meta: ModelMeta, output_folder: Path) -> None:
        save_path = output_folder / "model_meta.json"

        with save_path.open("w") as f:
            json.dump(model_meta.to_dict(), f, default=str)

    def _get_last_evaluated_splits(self) -> dict[str, list[str]]:
        """Last evaluated splits.

        Returns:
             a dictionary of tasks and their evaluated splits from the most recent run.
             Tasks with empty lists indicate that results already existed and no splits were evaluated.
        """
        return deepcopy(
            {task: list(splits) for task, splits in self.last_evaluated_splits.items()}
        )

    @staticmethod
    def _get_missing_evaluations(
        existing_results: TaskResult | None,
        task_eval_splits: list[str],
        task_eval_langs: list[str],
        eval_subsets: list[str] | None,
    ) -> dict[str, dict[str, Any]]:
        """Return a dictionary for each split, indicating if the whole split is missing and which subsets are missing."""
        missing_evaluations = {
            split: {"whole_split_missing": False, "missing_subsets": []}
            for split in task_eval_splits
        }

        # Determine subsets to consider if multilingual
        if eval_subsets is None:
            # If no eval_langs specified, consider all subsets
            subsets_to_consider = task_eval_langs
        else:
            subsets_to_consider = [
                subset for subset in task_eval_langs if subset in eval_subsets
            ]

        # If no existing results, all splits and subsets are missing
        if existing_results is None:
            for split in task_eval_splits:
                missing_evaluations[split]["whole_split_missing"] = True
                missing_evaluations[split]["missing_subsets"] = list(
                    subsets_to_consider
                )
            return missing_evaluations

        # If we have existing results, check which splits and subsets are missing
        for split in task_eval_splits:
            if split not in existing_results.scores:
                # Whole split missing
                missing_evaluations[split]["whole_split_missing"] = True
                missing_evaluations[split]["missing_subsets"] = list(
                    subsets_to_consider
                )
            else:
                # Some subsets may be missing
                existing_subsets = {
                    score_dict["hf_subset"]
                    for score_dict in existing_results.scores[split]
                }
                missing_subsets = [
                    s for s in subsets_to_consider if s not in existing_subsets
                ]
                if missing_subsets:
                    missing_evaluations[split]["missing_subsets"] = missing_subsets

        return missing_evaluations

    @staticmethod
    def _get_model_meta(model: EncoderProtocol) -> ModelMeta:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if isinstance(model, CrossEncoder):
            meta = _model_meta_from_cross_encoder(model)
        elif isinstance(model, SentenceTransformer):
            meta = _model_meta_from_sentence_transformers(model)
        else:
            meta = ModelMeta(
                loader=None,
                name=None,
                revision=None,
                release_date=None,
                languages=None,
                framework=[],
                similarity_fn_name=None,
                n_parameters=None,
                memory_usage_mb=None,
                max_tokens=None,
                embed_dim=None,
                license=None,
                open_weights=None,
                public_training_code=None,
                public_training_data=None,
                use_instructions=None,
                training_datasets=None,
            )
        return meta
