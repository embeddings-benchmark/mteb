import logging
from typing import Dict, List

import datasets
import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from ..evaluation.evaluators import RetrievalEvaluator
from .AbsTask import AbsTask


class BeIRModel:
    """
    BeIR requires to have an encode_queries and encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into
    BeIR format model
    """

    def __init__(self, model, **kwargs):
        self.model = model

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        return self.model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        sentences = [(doc["title"] + "\n" + doc["text"]).strip() for doc in corpus]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)


class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]

        model = DRES(BeIRModel(model), batch_size=kwargs.get("batch_size", 16), **kwargs)
        retriever = EvaluateRetrieval(model, score_function="cos_sim")  # or "dot" for dot-product
        results = retriever.retrieve(corpus, queries)

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values)

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        }

        return scores
