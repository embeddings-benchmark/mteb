import logging
from typing import Dict, List

import torch

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES

from .AbsTask import AbsTask
import math
import logging

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

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]

        model = DRPES(BeIRModel(model), batch_size=kwargs.get("batch_size", 16), **kwargs)
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


class BeIRModel:
    """
    BeIR requires to have an encode_queries and encode_corpus method.
    This class converts a MTEB model (with just an .encode method) into
    BeIR format model
    """

    def __init__(self, model, **kwargs):
        self.model = model
    
    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        return self.model.start_multi_process_pool(target_devices=target_devices)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        return self.model.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        sentences = [(doc["title"] + "\n" + doc["text"]).strip() for doc in corpus]
        return self.model.encode(sentences, batch_size=batch_size, **kwargs)

    def encode_corpus_parallel(
        self, corpus: List[Dict[str, str]], pool: Dict[str, object], batch_size: int, chunk_size: int, **kwargs
    ):
        sentences = [(doc["title"] + "\n" + doc["text"]).strip() for doc in corpus]
        return self.encode_multi_process(sentences, pool, batch_size=batch_size, chunk_size=chunk_size)

    @staticmethod
    def encode_multi_process(sentences: List[str], pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None):
        """
        (taken from UKPLab/sentence-transformers/sentence_transformers/SentenceTransformer.py)
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        Note: Updated the output of the processes to only return the similarity scores of the top k sentences.

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """
        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.info(f"Chunk data into {math.ceil(len(sentences)/chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        cos_scores_top_k_values = torch.cat([result[1] for result in results_list], dim=1)  # (num_queries, (top_k + 1) * num_sentences / chunk_size) = (num_queries, top_k * num_batches)
        cos_scores_top_k_idx = torch.cat([result[2] for result in results_list], dim=1)
        return cos_scores_top_k_values, cos_scores_top_k_idx
        