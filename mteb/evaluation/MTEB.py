from ..abstasks import *
from ..tasks import *
import random
import numpy as np

class MTEB():
    def __init__(self, tasks_types=None, tasks_categories=None, version=None, tasks_list=None):
        """
        Create an Evaluation pipeline. The tasks selected
        depends on the parameters. One can specify the tasks types
        they want to evaluate (e.g. Clustering, Retrieval, etc.)
        the categories of tasks they want (e.g. Sentence2Sentence, 
        Sentence2Paragraph, etc.) and the version of the benchmark.
        The selected tasks will be the tasks satisfying conditions
        from the 3 arguments. Alternatively, one can specify a list
        of tasks to be evaluated with the tasks_list argument. If
        tasks_list is specified, the other arguments are ignored.

        Parameters
        ----------
        tasks_types: list of str / None
            List of  types to be evaluated. If None, all tasks will be evaluated
        tasks_categories: list of str / None
            List of task types to be evaluated. If None, all tasks will be evaluated
        version: int / None
            Version of the benchmark to use. If None, latest is used
        tasks_list: list of AbsTask / None
            List of tasks to be evaluated. If specified, the other arguments are ignored.
        """
        self._tasks_types = tasks_types
        self._tasks_categories = tasks_categories
        self._version = version

        self._tasks_list = tasks_list

        self.select_tasks()
        print([x.description['name'] for x in self.tasks])


    def select_tasks(self):
        """
        Select the tasks to be evaluated.
        """
        # Get all existing tasks
        tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
        tasks_cls = [cls() for cat_cls in tasks_categories_cls for cls in cat_cls.__subclasses__()]

        # If tasks_list is specified, select list of tasks
        if self._tasks_list is not None:
            filter_task_list = lambda x: (x.description["name"] in self._tasks_list)
            self.tasks = list(filter(filter_task_list, tasks_cls))
            return

        # Otherwise use filters to select tasks
        filter_task_type = lambda x: (self._tasks_types is None) or (x.description["type"] in self._tasks_types)
        filter_task_category = lambda x: (self._tasks_categories is None) or (x.description["category"] in self._tasks_categories)
        filter_version = lambda x: (self._version is None) or (x.description["version"] >= self._version)

        # Filter tasks
        tasks_cls = filter(filter_task_type, tasks_cls)
        tasks_cls = filter(filter_task_category, tasks_cls)
        tasks_cls = filter(filter_version, tasks_cls)

        # Get final list of tasks
        self.tasks = list(tasks_cls)

    def run(self, model):
        for task in self.tasks:
            for split in task.description['available_splits']:
                print(task.description['name'], split)
                task.load_data()
                results = task.evaluate(model, split)
                print(results)
