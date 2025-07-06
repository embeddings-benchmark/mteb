from __future__ import annotations

import heapq
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytrec_eval
import torch
import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer

from mteb.encoder_interface import AudioBatch, Encoder, PromptType
from mteb.model_meta import ModelMeta

from ..Evaluator import Evaluator
from ..utils import (
    confidence_scores,
    convert_conv_history_to_query,
    cos_sim,
    download,
    hole,
    mrr,
    nAUC,
    recall_cap,
    top_k_accuracy,
)

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search.py#L12
class DenseRetrievalExactSearch:
    def __init__(
        self,
        model: Encoder,
        encode_kwargs: dict[str, Any] = {},
        corpus_chunk_size: int = 50000,
        previous_results: str | Path | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 128
        if "show_progress_bar" not in encode_kwargs:
            encode_kwargs["show_progress_bar"] = True
        if "convert_to_tensor" not in encode_kwargs:
            encode_kwargs["convert_to_tensor"] = True

        self.corpus_chunk_size = corpus_chunk_size
        if isinstance(previous_results, Path):
            self.previous_results = str(previous_results)
        else:
            self.previous_results = previous_results
        self.batch_size = encode_kwargs.get("batch_size")
        self.show_progress_bar = encode_kwargs.get("show_progress_bar")
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
        queries: dict[str, np.array | list[np.array]],
        top_k: int,
        task_name: str,
        instructions: dict[str, str] | None = None,
        request_qid: str | None = None,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        logger.info("Encoding Queries.")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}

        query_data = [queries[qid] for qid in query_ids]

        # detect query modality from first query
        if isinstance(query_data[0], str):
            query_modality = "text"
        elif isinstance(query_data[0], (dict, np.ndarray)):
            query_modality = "audio"
        else:
            query_modality = "text"  # fallback

        logger.info(f"Detected query modality: {query_modality}")

        # encode queries based on modality
        if query_modality == "text":
            if instructions:
                query_data = [
                    f"{query} {instructions[query]}".strip() for query in query_data
                ]
            query_embeddings = self.model.encode(
                query_data,
                task_name=task_name,
                prompt_type=PromptType.query,
                **self.encode_kwargs,
            )
        else:  # audio
            query_embeddings = self.model.encode(
                query_data,
                task_name=task_name,
                prompt_type=PromptType.query,
                **self.encode_kwargs,
            )

        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, reverse=True)
        corpus_data = [corpus[cid] for cid in corpus_ids]

        # detect corpus modality from first corpus item
        if isinstance(corpus_data[0], str):
            corpus_modality = "text"
        elif isinstance(corpus_data[0], (dict, np.ndarray)):
            corpus_modality = "audio"
        else:
            corpus_modality = "text"  # fallback

        logger.info(f"Detected corpus modality: {corpus_modality}")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")

        itr = range(0, len(corpus_data), self.corpus_chunk_size)

        result_heaps = {
            qid: [] for qid in query_ids
        }  # keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size, len(corpus_data)
            )

            # encode chunk of corpus
            if (
                self.save_corpus_embeddings
                and request_qid
                and len(self.corpus_embeddings[request_qid])
            ):
                sub_corpus_embeddings = torch.tensor(
                    self.corpus_embeddings[request_qid][batch_num]
                )
            else:
                # encode chunk of corpus based on modality
                corpus_chunk = corpus_data[corpus_start_idx:corpus_end_idx]
                sub_corpus_embeddings = self.model.encode(
                    corpus_chunk,
                    task_name=task_name,
                    prompt_type=PromptType.passage,
                    request_qid=request_qid,
                    **self.encode_kwargs,
                )
                if self.save_corpus_embeddings and request_qid:
                    self.corpus_embeddings[request_qid].append(sub_corpus_embeddings)

            # compute similarites using self defined similarity otherwise default to cosine-similarity
            if hasattr(self.model, "similarity"):
                similarity_scores = self.model.similarity(
                    query_embeddings, sub_corpus_embeddings
                )
            else:
                similarity_scores = cos_sim(query_embeddings, sub_corpus_embeddings)
            is_nan = torch.isnan(similarity_scores)
            if is_nan.sum() > 0:
                logger.warning(
                    f"Found {is_nan.sum()} NaN values in the similarity scores. Replacing NaN values with -1."
                )
            similarity_scores[is_nan] = -1

            # get top-k values
            similarity_scores_top_k_values, similarity_scores_top_k_idx = torch.topk(
                similarity_scores,
                min(
                    top_k + 1,
                    len(similarity_scores[1])
                    if len(similarity_scores) > 1
                    else len(similarity_scores[-1]),
                ),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            similarity_scores_top_k_values = (
                similarity_scores_top_k_values.cpu().tolist()
            )
            similarity_scores_top_k_idx = similarity_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    similarity_scores_top_k_idx[query_itr],
                    similarity_scores_top_k_values[query_itr],
                ):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

    def load_results_file(self):
        # load the first stage results from file in format {qid: {doc_id: score}}
        if "https://" in self.previous_results:
            # download the file
            if not os.path.exists(self.previous_results):
                url_descriptor = self.previous_results.split("https://")[-1].replace(
                    "/", "--"
                )
                dest_file = os.path.join(
                    "results", f"cached_predictions--{url_descriptor}"
                )
                os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
                download(self.previous_results, dest_file)
                logger.info(
                    f"Downloaded the previous results at {self.previous_results} to {dest_file}"
                )
            self.previous_results = dest_file

        with open(self.previous_results) as f:
            previous_results = json.load(f)
        assert isinstance(previous_results, dict)
        assert isinstance(previous_results[list(previous_results.keys())[0]], dict)
        return previous_results

    def search_cross_encoder(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, np.array | list[np.array]],
        top_k: int,
        instructions: dict[str, str] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """This function provides support for reranker (or cross-encoder) models that encoder query and document at the same time (typically with attention).
        Some notable examples include MonoBERT, MonoT5, RankLlama, etc.
        Note: you must provide the path to the results to rerank to the __init__ function as `previous_results` or else rerank all documents in the corpus
        """
        pairs = []  # create the pairs for reranking
        for qid in queries.keys():
            if self.previous_results is None:
                # try to use all of them
                logging.info(
                    f"previous_results is None. Using all the documents to rerank: {len(corpus)}"
                )
                q_results = dict.fromkeys(corpus.keys(), 0.0)
            else:
                q_results = self.previous_results[qid]
            # take the top-k only
            q_results_sorted = dict(
                sorted(q_results.items(), key=lambda item: item[1], reverse=True)
            )
            top_n = [k for k, v in list(q_results_sorted.items())[:top_k]]
            query = queries[qid]
            query = (
                self.convert_conv_history_to_query(self.model, [query])[0]
                if isinstance(query, list)
                else query
            )
            for doc_id in top_n:
                pairs.append(
                    (
                        query,
                        corpus[doc_id],
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
            # handle corpus data directly
            corpus_in_pair = list(corpus_in_pair)

            if hasattr(self.model, "model") and isinstance(
                self.model.model, CrossEncoder
            ):
                # can't take instructions, so add them here
                if instructions_in_pair[0] is not None:
                    queries_in_pair = [
                        f"{q} {i}".strip()
                        for i, q in zip(instructions_in_pair, queries_in_pair)
                    ]
                scores = self.model.predict(list(zip(queries_in_pair, corpus_in_pair)))  # type: ignore
            else:
                # may use the instructions in a unique way, so give them also
                scores = self.model.predict(  # type: ignore
                    list(zip(queries_in_pair, corpus_in_pair, instructions_in_pair))
                )

            for i, score in enumerate(scores):
                results[query_ids[i]][corpus_ids[i]] = float(score)

        return results

    def predict(self, queries, passages, **kwargs):
        raise NotImplementedError(
            "You must implement a predict method for your reranker model"
        )

    def encode_conversations(
        self,
        model: Encoder,
        conversations: list[list[np.array]],
        task_name: str,
        **kwargs,
    ):
        if callable(getattr(self.model, "encode_conversations", None)):
            return model.encode_conversations(  # type: ignore
                conversations, task_name=task_name, **kwargs
            )
        logger.warning(
            "Model doesn't have encode_conversations fallback to default implementation"
        )
        queries = self.convert_conv_history_to_query(model, conversations)  # type: ignore
        return model.encode(
            queries, task_name=task_name, prompt_type=PromptType.query, **kwargs
        )  # type: ignore

    @staticmethod
    def convert_conv_history_to_query(
        model: Encoder, conversations: list[list[np.array]]
    ) -> str:
        if callable(getattr(model, "convert_conv_history_to_query", None)):
            return model.convert_conv_history_to_query(conversations)  # type: ignore
        return convert_conv_history_to_query(conversations)  # type: ignore


class DRESModel:
    """Dense Retrieval Exact Search (DRES).
    This class converts a model with just an .encode method into DRES format.
    """

    mteb_model_meta: ModelMeta | None

    def __init__(self, model, **kwargs):
        self.model: Any = model
        self.use_sbert_model = isinstance(model, SentenceTransformer)
        self.device = model.device if hasattr(model, "device") else None
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = {}

        if hasattr(self.model, "similarity") and callable(self.model.similarity):
            self.similarity = self.model.similarity

    def encode_corpus(
        self,
        corpus: list[dict[str, str]],
        task_name: str,
        batch_size: int,
        prompt_type: PromptType = PromptType.passage,
        request_qid: str | None = None,
        **kwargs,
    ):
        if (
            request_qid
            and self.save_corpus_embeddings
            and len(self.corpus_embeddings) > 0
        ):
            return self.corpus_embeddings[request_qid]

        # handle different corpus data types directly
        corpus_embeddings = self.model.encode(
            corpus,
            task_name=task_name,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )

        if self.save_corpus_embeddings and request_qid:
            self.corpus_embeddings[request_qid] = corpus_embeddings
        return corpus_embeddings

    def encode(
        self,
        sentences: list[str] | AudioBatch,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ):
        if prompt_type and prompt_type == PromptType.passage:
            return self.encode_corpus(
                sentences, task_name, prompt_type=prompt_type, **kwargs
            )
        return self.model.encode(
            sentences, task_name=task_name, prompt_type=prompt_type, **kwargs
        )


def is_cross_encoder_compatible(model) -> bool:
    op = getattr(model, "predict", None)
    return callable(op)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L9
class Any2AnyRetrievalEvaluator(Evaluator):
    def __init__(
        self,
        retriever,
        task_name: str | None = None,
        k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000],
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_cross_encoder = False
        if is_cross_encoder_compatible(retriever):
            logger.info(
                "The custom predict function of the model will be used if not a SentenceTransformer CrossEncoder"
            )
            self.retriever = DenseRetrievalExactSearch(
                retriever, encode_kwargs=encode_kwargs, **kwargs
            )
            self.is_cross_encoder = True
        else:
            self.retriever = DenseRetrievalExactSearch(
                DRESModel(retriever), encode_kwargs=encode_kwargs, **kwargs
            )
        self.k_values = k_values
        self.top_k = (
            max(k_values) if "top_k" not in kwargs else kwargs["top_k"]
        )  # can lower it if reranking
        self.task_name = task_name

    def __call__(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, np.array | list[np.array]],
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        if self.is_cross_encoder:
            return self.retriever.search_cross_encoder(corpus, queries, self.top_k)
        else:
            return self.retriever.search(
                corpus,
                queries,
                self.top_k,
                task_name=self.task_name,  # type: ignore
            )

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        if ignore_identical_ids:
            logger.debug(
                "For evaluation, ``ignore_identical_ids=True`` is set to True, the evaluator will ignore identical query and document ids."
            )
            # remove identical ids from results dict
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
        else:
            logger.debug(
                "For evaluation, we DO NOT ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=True`` to ignore this."
            )

        all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

        for k in k_values:
            all_ndcgs[f"NDCG@{k}"] = []
            all_aps[f"MAP@{k}"] = []
            all_recalls[f"Recall@{k}"] = []
            all_precisions[f"P@{k}"] = []

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
                all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
                all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
                all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
                all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

        ndcg, _map, recall, precision = (
            all_ndcgs.copy(),
            all_aps.copy(),
            all_recalls.copy(),
            all_precisions.copy(),
        )

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
            _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
            recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
            precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)

        naucs = Any2AnyRetrievalEvaluator.evaluate_abstention(
            results, {**all_ndcgs, **all_aps, **all_recalls, **all_precisions}
        )

        return ndcg, _map, recall, precision, naucs

    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
        output_type: str = "all",
    ) -> tuple[dict[str, float], dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            metric_scores = mrr(qrels, results, k_values, output_type)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            metric_scores = recall_cap(qrels, results, k_values, output_type)

        elif metric.lower() in ["hole", "hole@k"]:
            metric_scores = hole(qrels, results, k_values, output_type)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            metric_scores = top_k_accuracy(qrels, results, k_values, output_type)

        naucs = Any2AnyRetrievalEvaluator.evaluate_abstention(results, metric_scores)
        metric_scores_avg = {k: sum(v) / len(v) for k, v in metric_scores.items()}

        return metric_scores_avg, naucs

    @staticmethod
    def evaluate_abstention(
        results: dict[str, dict[str, float]],
        metric_scores: dict[str, list[float]],
    ) -> dict[str, float]:
        """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997"""
        all_sim_scores = [list(results[qid].values()) for qid in list(results.keys())]
        all_conf_scores = [
            confidence_scores(sim_scores) for sim_scores in all_sim_scores
        ]
        conf_fcts = list(all_conf_scores[0].keys())
        all_conf_scores = {
            fct: np.array([x[fct] for x in all_conf_scores]) for fct in conf_fcts
        }
        metric_scores = {k: np.array(v) for k, v in metric_scores.items()}
        naucs = {}

        for metric_name, scores in metric_scores.items():
            for fct, conf_scores in all_conf_scores.items():
                naucs[f"nAUC_{metric_name}_{fct}"] = nAUC(conf_scores, scores)

        return naucs
