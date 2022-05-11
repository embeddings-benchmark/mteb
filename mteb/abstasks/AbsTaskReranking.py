from .AbsTask import AbsTask
from ..evaluation.evaluators import RerankingEvaluator
import datasets
import numpy as np


class AbsTaskReranking(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """

    def __init__(self, **kwargs):
        super(AbsTaskReranking, self).__init__(**kwargs)
        self.dataset = None
        self.data_loaded = False

    def load_data(self):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(self.description["hf_hub_name"])
        self.data_loaded = True

    def evaluate(self, model, split="test"):
        if not self.data_loaded:
            self.load_data()

        data_split = self.dataset[split]

        evaluator = RerankingEvaluator(data_split)
        scores = evaluator(model)

        return dict(scores)
