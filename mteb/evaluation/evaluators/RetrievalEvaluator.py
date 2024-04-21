from __future__ import annotations

import heapq
import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import pytrec_eval
import torch
import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings

from .Evaluator import Evaluator
from .utils import cos_sim, dot_score, hole, mrr, recall_cap, top_k_accuracy

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search.py#L12
class DenseRetrievalExactSearch:
    def __init__(
        self,
        model,
        batch_size: int = 128,
        corpus_chunk_size: int = 50000,
        previous_results: str = None,
        **kwargs,
    ):
        # Model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {
            "cos_sim": "Cosine Similarity",
            "dot": "Dot Product",
        }
        self.corpus_chunk_size = corpus_chunk_size
        self.previous_results = previous_results
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = defaultdict(list)
        self.results = {}

        if self.previous_results is not None:
            self.previous_results = self.load_results_file()

        if isinstance(self.model, CrossEncoder):
            # load the predict instance from the CrossEncoder
            # custom functions can be used by extending the DenseRetrievalExactSearch class
            self.predict = self.model.predict

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        score_function: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                "score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(
                    score_function
                )
            )

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor,
            **kwargs,
        )

        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info(
            "Scoring Function: {} ({})".format(
                self.score_function_desc[score_function], score_function
            )
        )

        itr = range(0, len(corpus), self.corpus_chunk_size)

        result_heaps = {
            qid: [] for qid in query_ids
        }  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            if (
                self.save_corpus_embeddings
                and "qid" in kwargs
                and len(self.corpus_embeddings[kwargs["qid"]])
            ):
                sub_corpus_embeddings = torch.tensor(
                    self.corpus_embeddings[kwargs["qid"]][batch_num]
                )
            else:
                # Encode chunk of corpus
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_tensor=self.convert_to_tensor,
                )
                if self.save_corpus_embeddings and "qid" in kwargs:
                    self.corpus_embeddings[kwargs["qid"]].append(sub_corpus_embeddings)

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](
                query_embeddings, sub_corpus_embeddings
            )
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(
                    top_k + 1,
                    len(cos_scores[1]) if len(cos_scores) > 1 else len(cos_scores[-1]),
                ),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(
                                result_heaps[query_id], (score, corpus_id)
                            )

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

    def load_results_file(self):
        # load the first stage results from file in format {qid: {doc_id: score}}
        with open(self.previous_results, "r") as f:
            previous_results = json.load(f)
        assert isinstance(previous_results, dict)
        assert isinstance(previous_results[list(previous_results.keys())[0]], dict)
        return previous_results

    def search_cross_encoder(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        instructions: Dict[str, str] | None = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        This function provides support for reranker (or cross-encoder) models that encoder query and document at the same time (typically with attention).
            Some notable examples include MonoBERT, MonoT5, RankLlama, etc.
            Note: you must provide the path to the results to rerank to the __init__ function as `previous_results`
        """
        pairs = []  # create the pairs for reranking
        for qid in queries.keys():
            q_results = self.previous_results[qid]
            # take the top-k only
            q_results_sorted = {
                k: v
                for k, v in sorted(
                    q_results.items(), key=lambda item: item[1], reverse=True
                )
            }
            top_n = [k for k, v in list(q_results_sorted.items())[:top_k]]
            query = queries[qid]
            for doc_id in top_n:
                corpus_item = (
                    corpus[doc_id].get("title", "") + " " + corpus[doc_id]["text"]
                ).strip()
                pairs.append(
                    (
                        query,
                        corpus_item,
                        instructions[query] if instructions is not None else None,
                        qid,
                        doc_id,
                    )
                )

        logger.info(f"Reranking the top {top_k} in batches... This might take a while!")
        itr = range(0, len(pairs), self.batch_size)

        results = {qid: {} for qid in queries.keys()}
        for batch_num, corpus_start_idx in enumerate(
            tqdm.tqdm(itr, leave=False, disable=not self.show_progress_bar)
        ):
            corpus_end_idx = min(corpus_start_idx + self.batch_size, len(pairs))
            cur_batch = pairs[corpus_start_idx:corpus_end_idx]

            (
                queries_in_pair,
                corpus_in_pair,
                instructions_in_pair,
                query_ids,
                corpus_ids,
            ) = zip(*cur_batch)

            assert (
                len(queries_in_pair) == len(corpus_in_pair) == len(instructions_in_pair)
            )

            if isinstance(self.model, CrossEncoder):
                # can't take instructions, so add them here
                queries_in_pair = [
                    f"{q} {i}".strip()
                    for i, q in zip(instructions_in_pair, queries_in_pair)
                ]
                scores = self.model.predict(list(zip(queries_in_pair, corpus_in_pair)))
            else:
                # may use the instructions in a unique way, so give them also
                scores = self.model.predict(
                    list(zip(queries_in_pair, corpus_in_pair, instructions_in_pair))
                )

            for i, score in enumerate(scores):
                results[query_ids[i]][corpus_ids[i]] = float(score)

        return results

    def predict(self, queries, passages, **kwargs):
        raise NotImplementedError(
            "You must implement a predict method for your reranker model"
        )


class DRESModel:
    """
    Dense Retrieval Exact Search (DRES) requires an encode_queries & encode_corpus method.
    This class converts a model with just an .encode method into DRES format.
    """

    def __init__(self, model, **kwargs):
        self.model = model
        self.use_sbert_model = isinstance(model, SentenceTransformer)
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = {}

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        if self.use_sbert_model:
            if isinstance(self.model._first_module(), Transformer):
                logger.info(
                    f"Queries will be truncated to {self.model.get_max_seq_length()} tokens."
                )
            elif isinstance(self.model._first_module(), WordEmbeddings):
                logger.warning(
                    "Queries will not be truncated. This could lead to memory issues. In that case please lower the batch_size."
                )
        if "instructions" in kwargs:
            if kwargs["instructions"] is not None:
                queries = [
                    (query + " " + instruction).strip()
                    for query, instruction in zip(queries, kwargs["instructions"])
                ]
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            # can't just delete, cuz assign by reference on kwargs
            new_kwargs = kwargs

        return self.model.encode(queries, batch_size=batch_size, **new_kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if (
            "qid" in kwargs
            and self.save_corpus_embeddings
            and len(self.corpus_embeddings) > 0
        ):
            return self.corpus_embeddings[kwargs["qid"]]

        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + " " + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + " " + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]

        if "instructions" in kwargs:  # not used on the doc side
            new_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["instructions", "qid"]
            }
        else:
            # can't just delete, cuz assign by reference on kwargs
            new_kwargs = kwargs

        corpus_embeddings = self.model.encode(
            sentences, batch_size=batch_size, **new_kwargs
        )
        if self.save_corpus_embeddings and "qid" in kwargs:
            if isinstance(corpus_embeddings, torch.tensor):
                corpus_embeddings = corpus_embeddings.cpu().detach()
            self.corpus_embeddings[kwargs["qid"]] = corpus_embeddings
        return corpus_embeddings


def is_dres_compatible(model):
    for method in ["encode_queries", "encode_corpus"]:
        op = getattr(model, method, None)
        if not (callable(op)):
            return False
    return True


def is_cross_encoder_compatible(model):
    op = getattr(model, "predict", None)
    if not (callable(op)):
        return False
    return True


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L9
class RetrievalEvaluator(Evaluator):
    def __init__(
        self,
        retriever=None,
        k_values: List[int] = [1, 3, 5, 10, 20, 100, 1000],
        score_function: str = "cos_sim",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_cross_encoder = False
        if is_cross_encoder_compatible(retriever):
            logger.info(
                "The custom predict function of the model will be used if not a SentenceTransformer CrossEncoder"
            )
            self.retriever = DenseRetrievalExactSearch(retriever, **kwargs)
            self.is_cross_encoder = True
        elif is_dres_compatible(retriever):
            logger.info(
                "The custom encode_queries and encode_corpus functions of the model will be used"
            )
            self.retriever = DenseRetrievalExactSearch(retriever, **kwargs)
        else:
            self.retriever = DenseRetrievalExactSearch(DRESModel(retriever), **kwargs)
        self.k_values = k_values
        self.top_k = (
            max(k_values) if "top_k" not in kwargs else kwargs["top_k"]
        )  # can lower it if reranking
        self.score_function = score_function

    def __call__(
        self, corpus: dict[str, dict[str, str]], queries: dict[str, str]
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        if self.is_cross_encoder:
            return self.retriever.search_cross_encoder(corpus, queries, self.top_k)
        else:
            return self.retriever.search(
                corpus, queries, self.top_k, self.score_function
            )

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
    ) -> Tuple[Dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int],
        metric: str,
    ) -> Tuple[Dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)
        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)
        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)
        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            return top_k_accuracy(qrels, results, k_values)
