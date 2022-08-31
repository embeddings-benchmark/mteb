import logging
from time import time
from typing import Dict, List

import torch.multiprocessing as mp

from sentence_transformers import SentenceTransformer

from .AbsTask import AbsTask


logger = logging.getLogger(__name__)


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

        if True:
            from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES

            model = DRPES(
                BeIRModel(model),
                batch_size=batch_size,
                target_devices=target_devices,
                corpus_chunk_size=corpus_chunk_size,
                **kwargs,
            )
        else:
            if target_devices is not None:
                logger.warning(
                    "DenseRetrievalParallelExactSearch could not be imported from beir. Using DenseRetrievalExactSearch instead."
                )
                logger.warning("The parameter target_devices is ignored.")

            from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

            model = DRES(
                BeIRModel(model),
                batch_size=batch_size,
                corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
                **kwargs,
            )

        retriever = EvaluateRetrieval(model, score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values)

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        }

        return scores


class BeIRModel:
    """
    BeIR requires to have an encode_queries and encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into
    BeIR format model
    """

    def __init__(self, model, sep=" ", **kwargs):
        self.model = model
        self.sep = sep

    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(
                target=SentenceTransformer._encode_multi_process_worker,
                args=(process_id, device_name, self.model, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        output_queue = pool["output"]
        [output_queue.get() for _ in range(len(pool["processes"]))]
        return self.model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
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

    def encode_corpus_parallel(
        self, corpus: List[Dict[str, str]], pool: Dict[str, object], batch_size: int, chunk_id: int, **kwargs
    ):
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

        if chunk_id is not None and chunk_id >= len(pool["processes"]):
            output_queue = pool["output"]
            output_queue.get()

        input_queue = pool["input"]
        input_queue.put([chunk_id, batch_size, sentences])
