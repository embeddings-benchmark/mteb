from ..abstasks import *
from ..tasks import *
import pathlib
import os
import json
import logging
import datasets
from datetime import datetime


class MTEB:
    def __init__(self, task_types=None, task_categories=None, version=None, task_list=None):
        """
        Create an Evaluation pipeline. The tasks selected
        depends on the parameters. One can specify the tasks types
        they want to evaluate (e.g. Clustering, Retrieval, etc.)
        the categories of tasks they want (e.g. Sentence2Sentence,
        Sentence2Paragraph, etc.) and the version of the benchmark.
        The selected tasks will be the tasks satisfying conditions
        from the 3 arguments. Alternatively, one can specify a list
        of tasks to be evaluated with the `task_list` argument. If
        `task_list` is specified, the other arguments are ignored.

        Parameters
        ----------
        task_types: list of str / None
            List of task types (Clustering, Retrieval..) to be evaluated. If None, all tasks will be evaluated
        task_categories: list of str / None
            List of task categories (s2s, p2p..) to be evaluated. If None, all tasks will be evaluated
        version: int / None
            Version of the benchmark to use. If None, latest is used
        task_list: list of AbsTask / None
            List of tasks to be evaluated. If specified, the other arguments are ignored.
        """
        self._task_types = task_types
        self._task_categories = task_categories
        self._version = version

        self._task_list = task_list

        self.select_tasks()

    @property
    def available_tasks(self):
        return [x.description["name"] for x in self.tasks_cls]

    @property
    def available_task_types(self):
        return set([x.description["type"] for x in self.tasks_cls])

    @property
    def available_task_categories(self):
        return set([x.description["category"] for x in self.tasks_cls])

    @property
    def selected_tasks(self):
        return [x.description["name"] for x in self.tasks]

    def select_tasks(self):
        """
        Select the tasks to be evaluated.
        """
        # Get all existing tasks
        tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
        self.tasks_cls = [cls() for cat_cls in tasks_categories_cls for cls in cat_cls.__subclasses__()]

        # If `task_list` is specified, select list of tasks
        if self._task_list is not None:
            filter_task_list = lambda x: (x.description["name"] in self._task_list)
            self.tasks = list(filter(filter_task_list, self.tasks_cls))
            # add task if subclass of mteb.tasks
            self.tasks.extend([x for x in self._task_list if isinstance(x, AbsTask)])
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

    def run(self, model, verbosity=1.0, output_folder="results/result"):
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
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Run selected tasks
        for task in self.tasks:
            if os.path.exists(os.path.join(output_folder, f"{task.description['name']}.json")):
                print(f"WARNING: {task.description['name']} already exists. Skipping.")
                continue
            task_results = {}
            for split in task.description["available_splits"]:
                task_results[split] = {}
                for lang in task.description["available_langs"]:
                    print(f"\nTask: {task.description['name']}, split: {split}, language: {lang}. Running...")
                    results = task.evaluate(model, split)
                    if task.description["main_score"] in results:
                        results["main_score"] = results[task.description["main_score"]]
                    else:
                        print(f"WARNING: main score {task.description['main_score']} not found in results {results.keys()}")
                    task_results[split][lang] = results
                    if verbosity >= 1:
                        print(f"Scores: {results}")
            with open(os.path.join(output_folder, f"{task.description['name']}.json"), "w") as f_out:
                json.dump(task_results, f_out, indent=2, sort_keys=True)
