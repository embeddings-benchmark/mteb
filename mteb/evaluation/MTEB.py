from ..abstasks import *
from ..tasks import *
import random
import numpy as np

class MTEB():
    def __init__(self, tasks_types=None, tasks_categories=None, version=None):
        """
        Create an Evaluation pipeline. The tasks selected
        depends on the parameters. One can specify the tasks types
        they want to evaluate (e.g. Clustering, Retrieval, etc.)
        the categories of tasks they want (e.g. Sentence2Sentence, 
        Sentence2Paragraph, etc.) and the version of the benchmark.
        The selected tasks will be the tasks satisfying conditions
        from the 3 arguments

        Parameters
        ----------
        tasks_types: list of str / None
            List of  types to be evaluated. If None, all tasks will be evaluated
        tasks_categories: list of str / None
            List of task types to be evaluated. If None, all tasks will be evaluated
        version: int / None
            Version of the benchmark to use. If None, latest is used
        """
        self._tasks_types = tasks_types
        self._tasks_types = tasks_categories
        self._version = version

        self.select_tasks()


    def select_tasks(self):
        """
        Select the tasks to be evaluated.
        """
        tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
        tasks_cls = [cls for cat_cls in tasks_categories_cls for cls in cat_cls.__subclasses__()]

        # Define filter functions
        filter_task_type = lambda x: (self._tasks_types is None) or (x.description["type"] in self.tasks_types)
        filter_task_category = lambda x: (self._tasks_types is None) or (x.description["category"] in self.tasks_categories)
        filter_version = lambda x: (self._version is None) or (x.description["version"] >= self.version)

        # Filter tasks
        tasks_cls = filter(filter_task_type, tasks_cls)
        tasks_cls = filter(filter_task_category, tasks_cls)
        tasks_cls = filter(filter_version, tasks_cls)
        filtered_tasks_cls = list(tasks_cls)

        # Create tasks
        self.tasks = [cls() for cls in filtered_tasks_cls]

    def run(self, model, seed=28042000):
        random.seed(seed)
        np.random.seed(seed)
        for task in self.tasks:
            for split in task.description['available_splits']:
                print(task.description['name'], split)
                task.load_data()
                results = task.evaluate(model, split)
                print(results)
