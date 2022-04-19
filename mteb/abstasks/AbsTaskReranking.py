from .AbsTask import AbsTask
import datasets
from sentence_transformers import evaluation
import numpy as np
import logging
from collections import defaultdict

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

        self.dataset = datasets.load_dataset(self.description['hf_hub_name'])
        self.data_loaded = True

    def evaluate(self, model, split='test'):
        if not self.data_loaded:
            self.load_data()

        data_split = self.dataset[split]

        rr_evaluator = evaluation.RerankingEvaluator(data_split, show_progress_bar=False)
        scores = rr_evaluator.compute_metrices(model)

        return dict(scores)