from .AbsTask import AbsTask
import datasets
from sentence_transformers import evaluation
import numpy as np
import logging

class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """
    def __init__(self, **kwargs):
        super(AbsTaskRetrieval, self).__init__(**kwargs)        
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

        split = self.dataset[split][0]

        print()
        corpus = dict(zip(split['corpus']['ids'] , split['corpus']['sentences'])) #qid => query
        queries = dict(zip(split['queries']['ids'] , split['queries']['sentences'])) #cid => doc
        relevant_docs = dict(zip(split['relevant_docs']['ids'] , split['relevant_docs']['relevant_docs'])) #qid => Set[cid]

        #Convert lists to sets
        for doc_id in relevant_docs:
            relevant_docs[doc_id] = set(relevant_docs[doc_id])

        logging.getLogger("sentence_transformers.evaluation.InformationRetrievalEvaluator").setLevel(logging.WARN)
        evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs)
        scores = evaluator.compute_metrices(model)
        return scores
