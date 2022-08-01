import json
import logging
import os
import pathlib
import traceback
from datetime import datetime
from time import time

import datasets
from tqdm import trange

from rich.console import Console

from ..abstasks import *
from ..tasks import *


logger = logging.getLogger(__name__)


class MTEB:
    def __init__(
        self,
        task_types=None,
        task_categories=None,
        tasks=None,
        task_langs=None,
        version=None,
        err_logs_path=None,
        **kwargs
    ):
        """
        Create an Evaluation pipeline. The tasks selected
        depends on the parameters. One can specify the tasks types
        they want to evaluate (e.g. Clustering, Retrieval, etc.)
        the categories of tasks they want (e.g. Sentence2Sentence,
        Sentence2Paragraph, etc.) and the version of the benchmark.
        The selected tasks will be the tasks satisfying conditions
        from the 3 arguments. Alternatively, one can specify a list
        of tasks to be evaluated with the `tasks` argument. If
        `tasks` is specified, the other arguments are ignored.

        Parameters
        ----------
        task_types: list of str / None
            List of task types (Clustering, Retrieval..) to be evaluated. If None, all tasks will be evaluated
        task_categories: list of str / None
            List of task categories (s2s, p2p..) to be evaluated. If None, all tasks will be evaluated
        version: int / None
            Version of the benchmark to use. If None, latest is used
        tasks: list of AbsTask / None
            List of tasks to be evaluated. If specified, the other arguments are ignored.
        """
        self._task_types = task_types
        self._task_categories = task_categories
        self._version = version
        self._task_langs = task_langs if task_langs is not None else []
        if type(self._task_langs) is str:
            self._task_langs = [self._task_langs]
        self._task_langs.extend(
            [f"{x}-{y}" for x in self._task_langs for y in self._task_langs]
        )  # add all possible pairs

        self._tasks = tasks

        self.err_logs_path = err_logs_path if err_logs_path is not None else "error_logs.txt"

        self.select_tasks(**kwargs)

    @property
    def available_tasks(self):
        return [x.description["name"] for x in self.tasks_cls]

    @property
    def available_task_types(self):
        return set([x.description["type"] for x in self.tasks_cls])

    @property
    def available_task_categories(self):
        return set([x.description["category"] for x in self.tasks_cls])

    def _display_tasks(self, task_list, name=None):
        console = Console()
        if name:
            console.rule(f"[bold]{name}\n", style="grey15")
        for task_type in self.available_task_types:
            current_type_tasks = list(filter(lambda x: x.description["type"] == task_type, task_list))
            if len(current_type_tasks) == 0:
                continue
            else:
                console.print(f"[bold]{task_type}[/]")
                for task in current_type_tasks:
                    prefix = f"    - "
                    name = f"{task.description['name']}"
                    category = f", [italic grey39]{task.description['category']}[/]"
                    multilingual = (
                        f", [italic red]multilingual {len(task.description['eval_langs'])} langs[/]"
                        if task.is_multilingual
                        else ""
                    )
                    crosslingual = (
                        f", [italic cyan]crosslingual {len(task.description['eval_langs'])} pairs[/]"
                        if task.is_crosslingual
                        else ""
                    )
                    beir = f", [italic yellow]beir[/]" if task.description.get("beir_name", False) else ""
                    console.print(f"{prefix}{name}{beir}{category}{multilingual}{crosslingual}")
                console.print("\n")

    @classmethod
    def mteb_tasks(cls):
        """
        Get all tasks available in the MTEB.
        """
        instance = cls()
        instance._display_tasks(instance.tasks_cls, name="MTEB tasks")

    def print_selected_tasks(self):
        """ Print the selected tasks. """
        self._display_tasks(self.tasks, name="Selected tasks")

    def select_tasks(self, **kwargs):
        """
        Select the tasks to be evaluated.
        """
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
            self.tasks = list(filter(lambda x: (x.description["name"] in self._tasks), self.tasks_cls))
            # add task if subclass of mteb.tasks
            self.tasks.extend([x for x in self._tasks if isinstance(x, AbsTask)])
            return

        # Otherwise use filters to select tasks
        filtered_tasks = filter(
            lambda x: (self._task_types is None) or (x.description["type"] in self._task_types), self.tasks_cls
        )
        filtered_tasks = filter(
            lambda x: (self._task_categories is None) or (x.description["category"] in self._task_categories),
            filtered_tasks,
        )
        filtered_tasks = filter(
            lambda x: (self._version is None) or (x.description["version"] >= self._version), filtered_tasks
        )
        # keep only tasks with at least one language in the filter
        filtered_tasks = filter(
            lambda x: (self._task_langs is None)
            or (len(set(x.description["eval_langs"]) & set(self._task_langs)) > 0),
            filtered_tasks,
        )

        # Get final list of tasks
        self.tasks = list(filtered_tasks)

    def load_tasks_data(self):
        """
        Load datasets for the selected tasks.
        """
        logger.info(f"\n\n## Loading datasets for {len(self.tasks)} tasks")
        for task in self.tasks:
            logger.info(f"\n# Loading dataset for {task.description['name']}")
            task.load_data()

    def run(self, model, verbosity=1, output_folder="results/result", eval_splits=None, **kwargs):
        """
        Run the evaluation pipeline on the selected tasks.

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
        while len(self.tasks) > 0:
            task = self.tasks[0]
            logger.info(f"\n\n********************** Evaluating {task.description['name']} **********************")

            # skip evaluation if results folder exists
            if output_folder is not None:
                save_path = os.path.join(output_folder, f"{task.description['name']}{task.save_suffix}.json")
                if os.path.exists(save_path):
                    logger.warn(f"WARNING: {task.description['name']} results already exists. Skipping.")
                    del self.tasks[0]
                    continue

            try:
                task_eval_splits = eval_splits if eval_splits is not None else task.description.get("eval_splits", [])

                # load data
                logger.info(f"Loading dataset for {task.description['name']}")
                task.load_data(eval_splits=task_eval_splits)

                # run evaluation
                task_results = {}
                for split in task_eval_splits:
                    tick = time()
                    results = task.evaluate(model, split, **kwargs)
                    tock = time()
                    logger.info(f"Evaluation for {task.description['name']} on {split} took {tock - tick:.2f} seconds")
                    results["evaluation_time"] = round(tock - tick, 2)
                    task_results[split] = results
                    if verbosity >= 1:
                        logger.info(f"Scores: {results}")

                # save results
                if output_folder is not None:
                    with open(save_path, "w") as f_out:
                        json.dump(task_results, f_out, indent=2, sort_keys=True)

            except Exception as e:
                logger.error(f"Error while evaluating {task.description['name']}: {e}")
                logger.error(f"Please check all the error logs at: {self.err_logs_path}")
                with open(self.err_logs_path, "a") as f_out:
                    f_out.write(f"{datetime.now()} >>> {task.description['name']}\n")
                    f_out.write(traceback.format_exc())
                    f_out.write("\n\n")

            # empty memory
            del self.tasks[0]
