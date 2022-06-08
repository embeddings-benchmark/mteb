from click import style
from ..abstasks import *
from ..tasks import *
import pathlib
import os
import json
import logging
import datasets
from datetime import datetime
from rich.console import Console


class MTEB:
    def __init__(self, task_types=None, task_categories=None, version=None, tasks=None, **kwargs):
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

        self._tasks = tasks

        self.select_tasks(**kwargs)
        # self.load_tasks_data()

    @property
    def available_tasks(self):
        return [x.description["name"] for x in self.tasks_cls]

    @property
    def available_task_types(self):
        return set([x.description["type"] for x in self.tasks_cls])

    @property
    def available_task_categories(self):
        return set([x.description["category"] for x in self.tasks_cls])

    def display_tasks(self, task_list, name=None):
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
                    multilingual = f", [italic red]multilingual {len(task.description['eval_langs'])} langs[/]" if task.is_multilingual else ""
                    crosslingual = f", [italic cyan]crosslingual {len(task.description['eval_langs'])} pairs[/]" if task.is_crosslingual else ""
                    beir = f", [italic yellow]beir[/]" if task.description.get('beir_name', False) else ""
                    console.print(f"{prefix}{name}{beir}{category}{multilingual}{crosslingual}")
                console.print("\n")


    def mteb_tasks(self):
        self.display_tasks(self.tasks_cls, name="MTEB tasks")

    def selected_tasks(self):
        self.display_tasks(self.tasks, name="Selected tasks")

    def select_tasks(self, **kwargs):
        """
        Select the tasks to be evaluated.
        """
        # Get all existing tasks
        tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
        self.tasks_cls = [
            cls(**kwargs)
            for cat_cls in tasks_categories_cls
            for cls in cat_cls.__subclasses__()
            if cat_cls.__name__.startswith("AbsTask")
        ]

        # If `task_list` is specified, select list of tasks
        if self._tasks is not None:
            filter_task_list = lambda x: (x.description["name"] in self._tasks)
            self.tasks = list(filter(filter_task_list, self.tasks_cls))
            # add task if subclass of mteb.tasks
            self.tasks.extend([x for x in self._tasks if isinstance(x, AbsTask)])
            return

        # Otherwise use filters to select tasks
        filter_task_type = lambda x: (self._task_types is None) or (x.description["type"] in self._task_types)
        filter_task_category = lambda x: (self._task_categories is None) or (
            x.description["category"] in self._task_categories
        )
        filter_version = lambda x: (self._version is None) or (x.description["version"] >= self._version)

        # Filter tasks
        filtered_tasks = filter(filter_task_type, self.tasks_cls)
        filtered_tasks = filter(filter_task_category, filtered_tasks)
        filtered_tasks = filter(filter_version, filtered_tasks)

        # Get final list of tasks
        self.tasks = list(filtered_tasks)

    def load_tasks_data(self):
        """
        Load datasets for the selected tasks.
        """
        print(f"\n\n## Loading datasets for {len(self.tasks)} tasks")
        for task in self.tasks:
            print(f"\n# Loading dataset for {task.description['name']}")
            task.load_data()

    def run(self, model, verbosity=1.0, output_folder="results/result", **kwargs):
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
        print(f"\n\n## Evaluating {len(self.tasks)} tasks: {self.selected_tasks}")
        for task in self.tasks:
            if output_folder is not None:
                save_path = os.path.join(output_folder, f"{task.description['name']}{task.save_suffix}.json")
                if os.path.exists(save_path):
                    print(f"WARNING: {task.description['name']} results already exists. Skipping.")
                    continue
            task_results = {}
            for split in task.description["eval_splits"]:
                results = task.evaluate(model, split, **kwargs)
                task_results[split] = results
                if verbosity >= 1:
                    print(f"Scores: {results}")
            if output_folder is not None:
                with open(save_path, "w") as f_out:
                    json.dump(task_results, f_out, indent=2, sort_keys=True)
