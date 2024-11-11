from __future__ import annotations

import json
import logging
import os
import traceback
from collections.abc import Iterable
from copy import copy
from datetime import datetime
from itertools import chain
from pathlib import Path
from time import time
from typing import Any

import datasets
from sentence_transformers import SentenceTransformer

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models import model_meta_from_sentence_transformers

from ..abstasks import *
from ..abstasks import AbsTask
from ..load_results.task_results import TaskResult
from ..models.sentence_transformer_wrapper import SentenceTransformerWrapper
from ..models.wrapper import Wrapper
from ..tasks import *
from . import LangMapping

logger = logging.getLogger(__name__)


class MTEB:
    def __init__(
        self,
        tasks: Iterable[str | AbsTask] | None = None,
        *,
        task_types: list[str] | None = None,
        task_categories: list[str] | None = None,
        task_langs: list[str] | None = None,
        version=None,
        err_logs_path: str = "error_logs.txt",
        **kwargs,
    ):
        """Create an Evaluation pipeline, based on the provided tasks.

        Args:
            tasks: List of tasks to be evaluated.
            task_types: Will be deprecated we recommend that you use `mteb.get_tasks()` to filter tasks. List of task types (Clustering, Retrieval..) to be
                evaluated. If None, all tasks will be evaluated
            task_categories: Will be deprecated we recommend that you use `mteb.get_tasks()` to filter tasks. List of task categories (s2s, p2p..) to be
                evaluated. If None, all tasks will be evaluated
            task_langs: Will be deprecated we recommend that you use `mteb.get_tasks()` to filter tasks. List of languages to be evaluated. if None, all
                languages will be evaluated. ["eng-Latn", "deu_Latn"] will evaluate on all tasks with these languages.
            version: Will be deprecated. Version of the benchmark to use. If None, latest is used
            err_logs_path: Path to save error logs.
            kwargs: Additional arguments to be passed to the tasks
        """
        from mteb.benchmarks import Benchmark

        self.deprecation_warning(
            task_types, task_categories, task_langs, tasks, version
        )

        if tasks is not None:
            self._tasks = tasks
            if isinstance(tasks[0], Benchmark):
                self.benchmarks = tasks
                self._tasks = list(chain.from_iterable(tasks))
            assert (
                task_types is None and task_categories is None
            ), "Cannot specify both `tasks` and `task_types`/`task_categories`"
        else:
            self._task_types = task_types
            self._task_categories = task_categories
            self._tasks = None

        self._task_langs = task_langs if task_langs is not None else []
        if isinstance(self._task_langs, str):
            self._task_langs = [self._task_langs]

        self._extend_lang_code()
        self._extend_lang_pairs()  # add all possible pairs

        self._version = version
        self.err_logs_path = err_logs_path

        self.select_tasks(**kwargs)

    def deprecation_warning(
        self, task_types, task_categories, task_langs, tasks, version
    ):
        if task_types is not None:
            logger.warning(
                "The `task_types` argument is deprecated and will be removed in the next release. "
                + "Please use `tasks = mteb.get_tasks(... task_types = [...])` to filter tasks instead."
            )
        if task_categories is not None:
            logger.warning(
                "The `task_categories` argument is deprecated and will be removed in the next release. "
                + "Please use `tasks = mteb.get_tasks(... categories = [...])` to filter tasks instead."
            )
        if task_langs is not None:
            logger.warning(
                "The `task_langs` argument is deprecated and will be removed in the next release. "
                + "Please use `tasks = mteb.get_tasks(... languages = [...])` to filter tasks instead. "
                + "Note that this uses 3 letter language codes (ISO 639-3)."
            )
        if version is not None:
            logger.warning(
                "The `version` argument is deprecated and will be removed in the next release."
            )
        task_contains_strings = any(isinstance(x, str) for x in tasks or [])
        if task_contains_strings:
            logger.warning(
                "Passing task names as strings is deprecated and will be removed in the next release. "
                + "Please use `tasks = mteb.get_tasks(tasks=[...])` method to get tasks instead."
            )

    @property
    def available_tasks(self):
        return [x.metadata_dict["name"] for x in self.tasks_cls]

    @property
    def available_task_types(self):
        # sort the task types
        return sorted({x.metadata_dict["type"] for x in self.tasks_cls})

    @property
    def available_task_categories(self):
        return {x.metadata_dict["category"] for x in self.tasks_cls}

    def _extend_lang_code(self):
        # add all possible language codes
        for lang in set(self._task_langs):
            if lang in LangMapping.LANG_MAPPING:
                self._task_langs += LangMapping.LANG_MAPPING[lang]

    def _extend_lang_pairs(self):
        # add all possible language pairs
        langs = set(self._task_langs)
        for x in langs:
            if "-" not in x:
                for y in langs:
                    if "-" not in y:
                        pair = f"{x}-{y}"
                        if pair not in langs:
                            self._task_langs.append(pair)
        return

    def _display_tasks(self, task_list, name=None):
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
                        if task.is_multilingual
                        else ""
                    )
                    console.print(f"{prefix}{name}{category}{multilingual}")
                console.print("\n")

    def mteb_benchmarks(self):
        """Get all benchmarks available in the MTEB."""
        from mteb.overview import MTEBTasks

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
        instance = cls()
        instance._display_tasks(instance.tasks_cls, name="MTEB tasks")

    def print_selected_tasks(self):
        """Print the selected tasks."""
        self._display_tasks(self.tasks, name="Selected tasks")

    def select_tasks(self, **kwargs):
        """Select the tasks to be evaluated."""
        # Get all existing tasks
        tasks_categories_cls = list(AbsTask.__subclasses__())
        self.tasks_cls = [
            cls(hf_subsets=self._task_langs, **kwargs)
            for cat_cls in tasks_categories_cls
            for cls in cat_cls.__subclasses__()
            if cat_cls.__name__.startswith("AbsTask")
        ]

        # If `task_list` is specified, select list of tasks
        if self._tasks is not None:
            self.tasks = list(
                filter(
                    lambda x: (x.metadata_dict["name"] in self._tasks), self.tasks_cls
                )
            )
            if len(self.tasks) != len(self._tasks):
                tasks_known = {x.metadata_dict["name"] for x in self.tasks_cls}
                tasks_unknown = {
                    x for x in self._tasks if isinstance(x, str)
                } - tasks_known
                if tasks_unknown:
                    unknown_str, known_str = (
                        ",".join(sorted(tasks_unknown)),
                        ",".join(sorted(tasks_known)),
                    )
                    logger.warning(
                        f"WARNING: Unknown tasks: {unknown_str}. Known tasks: {known_str}."
                    )
            # add task if subclass of mteb.tasks
            self.tasks.extend([x for x in self._tasks if isinstance(x, AbsTask)])
            return

        # Otherwise use filters to select tasks
        filtered_tasks = filter(
            lambda x: (self._task_types is None)
            or (x.metadata_dict["type"] in self._task_types),
            self.tasks_cls,
        )
        filtered_tasks = filter(
            lambda x: (self._task_categories is None)
            or (x.metadata_dict["category"] in self._task_categories),
            filtered_tasks,
        )
        filtered_tasks = filter(
            lambda x: (self._version is None)
            or (x.metadata_dict["version"] >= self._version),
            filtered_tasks,
        )
        # keep only tasks with at least one language in the filter
        filtered_tasks = filter(
            lambda x: (not (self._task_langs))
            or (len(set(x.metadata_dict["eval_langs"]) & set(self._task_langs)) > 0),
            filtered_tasks,
        )

        # Get final list of tasks
        self.tasks = list(filtered_tasks)

    def load_tasks_data(self):
        """Load datasets for the selected tasks."""
        logger.info(f"\n\n## Loading datasets for {len(self.tasks)} tasks")
        for task in self.tasks:
            logger.info(f"\n# Loading dataset for {task.metadata_dict['name']}")
            task.load_data()

    @staticmethod
    def _run_eval(
        task: AbsTask,
        model: Encoder,
        split,
        output_folder,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ):
        tick = time()
        results = task.evaluate(
            model,
            split,
            output_folder=output_folder,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        tock = time()
        return results, tick, tock

    def run(
        self,
        model: SentenceTransformer | Encoder,
        verbosity: int = 1,
        output_folder: str | None = "results",
        eval_splits=None,
        overwrite_results: bool = False,
        raise_error: bool = True,
        co2_tracker: bool = False,
        encode_kwargs: dict[str, Any] = {},
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
            overwrite_results: Whether to overwrite existing results.
            raise_error: Whether to raise an error if an exception occurs during evaluation.
            co2_tracker: Whether to enable or disable CO2 emissions tracker using codecarbon.
            encode_kwargs: Additional keyword arguments to be passed to the model.encode method.
            kwargs: Additional arguments to be passed to `_run_eval` method and task.load_data.

        Returns:
            A list of TaskResult objects, one for each task evaluated.
        """
        if "batch_size" in kwargs:
            logger.warning(
                "The `batch_size` argument is deprecated and will be removed in the next release. "
                + "Please use `encode_kwargs = {'batch_size': ...}` to set the batch size instead."
            )
            encode_kwargs["batch_size"] = kwargs["batch_size"]

        # update logging to account for different levels of Verbosity (similar to the command line)

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
        output_path = self.create_output_folder(meta, output_folder)
        if not isinstance(model, Wrapper):
            model = SentenceTransformerWrapper(model)

        if output_path:
            self._save_model_metadata(meta, output_path)

        # Run selected tasks
        logger.info(f"\n\n## Evaluating {len(self.tasks)} tasks:")

        if verbosity > 0:
            self.print_selected_tasks()

        evaluation_results = []
        original_tasks = (
            self.tasks.copy()
        )  # save them in case we re-use the object (e.g. for reranking)
        while len(self.tasks) > 0:
            task = self.tasks[0]
            logger.info(
                f"\n\n********************** Evaluating {task.metadata.name} **********************"
            )

            # skip evaluation if results folder exists and overwrite_results is False
            if output_path:
                save_path = output_path / f"{task.metadata.name}{task.save_suffix}.json"
                if save_path.exists() and not overwrite_results:
                    logger.info(
                        f"{task.metadata.name} results already exists. Loading results from disk. Set overwrite_results=True to overwrite."
                    )
                    mteb_results = TaskResult.from_disk(save_path)
                    evaluation_results.append(mteb_results)
                    del self.tasks[0]  # empty memory
                    continue
            try:
                task_eval_splits = (
                    eval_splits if eval_splits is not None else task.eval_splits
                )

                # load data
                logger.info(f"Loading dataset for {task.metadata_dict['name']}")
                task.check_if_dataset_is_superseeded()
                task.load_data(eval_splits=task_eval_splits, **kwargs)

                # run evaluation
                task_results = {}
                evaluation_time = 0
                kg_co2_emissions: int | None = 0 if co2_tracker else None
                for split in task_eval_splits:
                    if co2_tracker:
                        try:
                            from codecarbon import EmissionsTracker
                        except ImportError:
                            raise ImportError(
                                "To use the CO2 emissions tracker, please install codecarbon using 'pip install codecarbon'"
                            )

                        with EmissionsTracker(
                            save_to_file=False, save_to_api=False, logging_logger=logger
                        ) as tracker:
                            results, tick, tock = self._run_eval(
                                task,
                                model,
                                split,
                                output_folder,
                                encode_kwargs=encode_kwargs,
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
                            output_folder,
                            encode_kwargs=encode_kwargs,
                            **kwargs,
                        )

                    logger.info(
                        f"Evaluation for {task.metadata_dict['name']} on {split} took {tock - tick:.2f} seconds"
                    )
                    evaluation_time += tock - tick

                    task_results[split] = results
                    if verbosity >= 1:
                        logger.info(f"Scores: {results}")

                mteb_task_result = TaskResult.from_task_results(
                    task,
                    task_results,
                    evaluation_time=evaluation_time,
                    kg_co2_emissions=kg_co2_emissions,
                )

                # save results
                if output_path:
                    with open(save_path, "w") as f_out:
                        json.dump(
                            mteb_task_result.to_dict(), f_out, indent=2, sort_keys=True
                        )

                evaluation_results.append(mteb_task_result)

            except Exception as e:
                logger.error(
                    f"Error while evaluating {task.metadata_dict['name']}: {e}"
                )
                if raise_error:
                    raise e
                logger.error(
                    f"Please check all the error logs at: {self.err_logs_path}"
                )
                with open(self.err_logs_path, "a") as f_out:
                    f_out.write(f"{datetime.now()} >>> {task.metadata_dict['name']}\n")
                    f_out.write(traceback.format_exc())
                    f_out.write("\n\n")

            # empty memory
            del self.tasks[0]

        # restore original tasks
        self.tasks = original_tasks
        return evaluation_results

    @staticmethod
    def create_model_meta(model: Encoder) -> ModelMeta:
        if hasattr(model, "mteb_model_meta"):
            meta = model.mteb_model_meta  # type: ignore
        else:
            try:
                meta = model_meta_from_sentence_transformers(model)  # type: ignore
            except AttributeError:
                logger.warning(
                    "Could not find model metadata. Please set the model.mteb_model_meta attribute or if you are using "
                    + "SentenceTransformers, please upgrade to version 3.0.0 to ensure that the model.mteb_model_meta "
                    + "attribute is available."
                )
                meta = ModelMeta(
                    name=None,
                    revision=None,
                    release_date=None,
                    languages=None,
                )

        # create a copy of the meta to avoid modifying the original object
        meta = copy(meta)
        meta.revision = meta.revision or "no_revision_available"
        meta.name = meta.name or "no_model_name_available"

        return meta

    def create_output_folder(
        self, model_meta: ModelMeta, output_folder: str | None
    ) -> Path | None:
        """Create output folder for the results."""
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
            json.dump(model_meta.to_dict(), f)
