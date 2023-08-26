import logging
from time import time
from typing import Dict, List

import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings
import os

from .AbsTask import AbsTask

logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]
DRPES_METHODS = [
    "start_multi_process_pool",
    "stop_multi_process_pool",
    "encode_queries",
    "encode_corpus",
    "encode_corpus_parallel",
]


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

    @staticmethod
    def is_dres_compatible(model, is_parallel=True):
        methods = DRPES_METHODS if is_parallel else DRES_METHODS
        for method in methods:
            op = getattr(model, method, None)
            if not (callable(op)):
                return False
        return True

    def evaluate(
        self,
        model,
        split="test",
        batch_size=128,
        corpus_chunk_size=None,
        target_devices=None,
        score_function="cos_sim",
        **kwargs
    ):
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]

        try:
            if os.getenv("RANK", None) is None:
                raise ImportError("DenseRetrievalParallelExactSearch only works in distributed mode")
            
            if self.description["beir_name"].startswith("cqadupstack"):
                raise ImportError("CQADupstack is incompatible with latest BEIR")
            from beir.retrieval.search.dense import (
                DenseRetrievalParallelExactSearch as DRPES,
            )

            model = model if self.is_dres_compatible(model, is_parallel=True) else DRESModel(model)

            model = DRPES(
                model,
                batch_size=batch_size,
                target_devices=target_devices,
                corpus_chunk_size=corpus_chunk_size,
                **kwargs,
            )
        except ImportError:
            if target_devices is not None:
                logger.warning(
                    "DenseRetrievalParallelExactSearch could not be imported from beir. Using DenseRetrievalExactSearch instead."
                )
                logger.warning("The parameter target_devices is ignored.")

            from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

            model = model if self.is_dres_compatible(model, is_parallel=False) else DRESModel(model)

            model = DRES(
                model,
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
                **kwargs,
            )

        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        logger.info("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values, ignore_identical_ids=kwargs.get("ignore_identical_ids", True))
        mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }

        return scores


class DRESModel:
    """
    Dense Retrieval Exact Search (DRES) in BeIR requires an encode_queries & encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
    """

    def __init__(self, model, sep=" ", **kwargs):
        self.model = model
        self.sep = sep
        self.use_sbert_model = isinstance(model, SentenceTransformer)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        if self.use_sbert_model:
            if isinstance(self.model._first_module(), Transformer):
                logger.info(f"Queries will be truncated to {self.model.get_max_seq_length()} tokens.")
            elif isinstance(self.model._first_module(), WordEmbeddings):
                logger.warning(
                    "Queries will not be truncated. This could lead to memory issues. In that case please lower the batch_size."
                )
        return self.model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)