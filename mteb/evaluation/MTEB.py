from __future__ import annotations

import json
import logging
import os
import pathlib
import traceback
from datetime import datetime
from importlib.metadata import version
from time import time
from typing import List, Union

import datasets

from ..abstasks import *
from ..abstasks import AbsTask, LangMapping
from ..tasks import *

logger = logging.getLogger(__name__)


class MTEB:
    def __init__(
        self,
        task_types: List[str] | None = None,
        task_categories: List[str] | None = None,
        task_langs: List[str] | None = None,
        tasks: List[Union[str, AbsTask]] | None = None,
        version=None,
        err_logs_path="error_logs.txt",
        **kwargs,
    ):
        """Create an Evaluation pipeline. The tasks selected
        depends on the parameters. One can specify the tasks types
        they want to evaluate (e.g. Clustering, Retrieval, etc.)
        the categories of tasks they want (e.g. Sentence2Sentence,
        Sentence2Paragraph, etc.) and the version of the benchmark.
        The selected tasks will be the tasks satisfying conditions
        from the 3 arguments. Alternatively, one can specify a list
        of tasks to be evaluated with the `tasks` argument. If
        `tasks` is specified, we filter tasks based on `task_langs`

        Args:
            task_types: List of task types (Clustering, Retrieval..) to be evaluated. If None, all tasks will be evaluated
            task_categories: List of task categories (s2s, p2p..) to be evaluated. If None, all tasks will be evaluated
            task_langs: List of languages to be evaluated. if None, all languages will be evaluated. ["eng-Latn", "deu_Latn"] will evaluate on all tasks
                with these languages.
            tasks: List of tasks to be evaluated. If specified, we filter tasks based on `task_langs` only
            version: Version of the benchmark to use. If None, latest is used
            err_logs_path: Path to save error logs
            kwargs: Additional arguments to be passed to the tasks
        """
        if tasks is not None:
            self._tasks = tasks
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

    @property
    def available_tasks(self):
        return [x.metadata_dict["name"] for x in self.tasks_cls]

    @property
    def available_task_types(self):
        return set([x.metadata_dict["type"] for x in self.tasks_cls])

    @property
    def available_task_categories(self):
        return set([x.metadata_dict["category"] for x in self.tasks_cls])

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
        for task_type in self.available_task_types:
            current_type_tasks = list(
                filter(lambda x: x.metadata_dict["type"] == task_type, task_list)
            )
            if len(current_type_tasks) == 0:
                continue
            else:
                console.print(f"[bold]{task_type}[/]")
                for task in current_type_tasks:
                    prefix = "    - "
                    name = f"{task.metadata_dict['name']}"
                    category = f", [italic grey39]{task.metadata_dict['category']}[/]"
                    multilingual = (
                        f", [italic red]multilingual {len(task.langs)} / {len(task.metadata_dict['eval_langs'])} langs[/]"
                        if task.is_multilingual
                        else ""
                    )
                    crosslingual = (
                        f", [italic cyan]crosslingual {len(task.langs)} / {len(task.metadata_dict['eval_langs'])} pairs[/]"
                        if task.is_crosslingual
                        else ""
                    )
                    console.print(
                        f"{prefix}{name}{category}{multilingual}{crosslingual}"
                    )
                console.print("\n")

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
        tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
        self.tasks_cls = [
            cls(langs=self._task_langs, **kwargs)
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
                tasks_known = set([x.metadata_dict["name"] for x in self.tasks_cls])
                tasks_unknown = (
                    set(x for x in self._tasks if isinstance(x, str)) - tasks_known
                )
                if tasks_unknown:
                    unknown_str, known_str = (
                        ",".join(sorted(list(tasks_unknown))),
                        ",".join(sorted(list(tasks_known))),
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

    def run(
        self,
        model,
        verbosity=1,
        output_folder="results/result",
        eval_splits=None,
        overwrite_results=False,
        raise_error: bool = True,
        **kwargs,
    ):
        """Run the evaluation pipeline on the selected tasks.

        Parameters
        ----------
        model:
            Model to be used for evaluation
        verbosity: int
            Verbosity level. Default is 1.
            0: print tasks tqdm progress bar
            1: print tasks tqdm progress bar and scores
            2: print everything (including datasets loading)
        output_folder: str
            Folder where the results will be saved
        raise_error: bool
            Whether to raise an error if an exception occurs during evaluation.
        :return: Returns a dictionary of task names and corresponding metrics results.
        """
        # Set logging
        if verbosity < 2:
            datasets.logging.set_verbosity(40)
            datasets.logging.disable_progress_bar()

        # Create output folder
        if output_folder is not None:
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Run selected tasks
        logger.info(f"\n\n## Evaluating {len(self.tasks)} tasks:")
        self.print_selected_tasks()
        evaluation_results = {}
        while len(self.tasks) > 0:
            task = self.tasks[0]
            logger.info(
                f"\n\n********************** Evaluating {task.metadata_dict['name']} **********************"
            )

            # skip evaluation if results folder exists and overwrite_results is False
            if output_folder is not None:
                save_path = os.path.join(
                    output_folder,
                    f"{task.metadata_dict['name']}{task.save_suffix}.json",
                )
                if os.path.exists(save_path) and overwrite_results is False:
                    logger.warning(
                        f"WARNING: {task.metadata_dict['name']} results already exists. Skipping."
                    )
                    del self.tasks[0]
                    continue

            try:
                task_eval_splits = (
                    eval_splits
                    if eval_splits is not None
                    else task.metadata_dict.get("eval_splits", [])
                )

                # load data
                logger.info(f"Loading dataset for {task.metadata_dict['name']}")
                task.load_data(eval_splits=task_eval_splits, **kwargs)

                # run evaluation
                task_results = {
                    "mteb_version": version("mteb"),  # noqa: F405
                    "dataset_revision": task.metadata_dict["dataset"].get(
                        "revision", None
                    ),
                    "mteb_dataset_name": task.metadata_dict["name"],
                }
                for split in task_eval_splits:
                    tick = time()
                    results = task.evaluate(
                        model, split, output_folder=output_folder, **kwargs
                    )
                    tock = time()
                    logger.info(
                        f"Evaluation for {task.metadata_dict['name']} on {split} took {tock - tick:.2f} seconds"
                    )
                    results["evaluation_time"] = round(tock - tick, 2)
                    task_results[split] = results
                    if verbosity >= 1:
                        logger.info(f"Scores: {results}")

                # save results
                if output_folder is not None:
                    with open(save_path, "w") as f_out:
                        json.dump(task_results, f_out, indent=2, sort_keys=True)

                evaluation_results[task.metadata_dict["name"]] = task_results

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

        return evaluation_results
