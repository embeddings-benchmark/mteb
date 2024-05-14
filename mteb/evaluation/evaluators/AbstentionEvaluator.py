from __future__ import annotations

import json
import logging
import os
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pytrec_eval
import torch
import tqdm

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator
from .RerankingEvaluator import RerankingEvaluator

logger = logging.getLogger(__name__)


class AbstentionRetrievingEvaluator(Evaluator):
    def __init__(
        self,
        metadata_dict: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metadata_dict = metadata_dict

    def __call__(self, model):
        """This is called during training to evaluate the model.
        It returns scores.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        raise NotImplementedError(
            "The abstention evaluator must not be called directly."
        )

    @staticmethod
    def compute_abstention_scores_retrieval(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int] = None,
        ignore_identical_ids: bool = True,
    ) -> Dict[str, float]:

        if k_values is None:
            k_values = [1, 3, 5, 10]
        # Choose whether to ignore identical ids
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

        # Compute retrieval metrics for each instance
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        # Compute confidence scores (max, std, P1P2)
        conf_scores = {}
        for qid in results.keys():
            scs = list(results[qid].values())
            scs_mean = sum(scs) / len(scs)
            scs_sort = sorted(scs)[::-1]
            conf_scores[qid] = {
                "max": scs_sort[0],
                "std": (sum((sc - scs_mean) ** 2 for sc in scs) / len(scs)) ** (1 / 2),
                "P1P2": scs_sort[0] - scs_sort[1],
            }

        # Compute nAUCs
        metrics = list(list(scores.values())[0].keys())
        abst_funcs = list(list(conf_scores.values())[0].keys())
        abst_rates = [k / 10 for k in range(10)]
        abst_scores = {}

        # Evaluate for all abstention functions (max, std, P1P2)
        for abst_func in abst_funcs:
            conf_scs = {key: val[abst_func] for key, val in conf_scores.items()}
            conf_scs_sort = dict(
                sorted(conf_scs.items(), key=lambda item: item[1])[::-1]
            )
            evals = {metric: [] for metric in metrics}
            oracles = {metric: [] for metric in metrics}

            # Evaluate for all abstention rates
            for abst_rate in abst_rates:
                num_kept = len(conf_scs) - int(abst_rate * len(conf_scs))
                kept_qids = list(conf_scs_sort.keys())[:num_kept]

                # Evaluate for all metrics (ndcg, map, precision, recall)
                for metric in metrics:
                    evals[metric].append(
                        sum(scores[qid][metric] for qid in kept_qids) / num_kept
                    )
                    scs = [scores[qid][metric] for qid in scores.keys()]
                    scs_sort = sorted(scs)[::-1]
                    oracles[metric].append(sum(scs_sort[:num_kept]) / num_kept)

            # Compute nAUCs
            for metric in metrics:
                auc = sum(evals[metric]) / len(abst_rates) * max(abst_rates)
                auc_oracle = sum(oracles[metric]) / len(abst_rates) * max(abst_rates)
                auc_rand = oracles[metric][0] * max(abst_rates)
                abst_scores[
                    f"nAUC_{metric.replace('P_', 'precision_at_').replace('recall_', 'recall_at_').replace('cut', 'at')}_{abst_func}"
                ] = (auc - auc_rand) / (auc_oracle - auc_rand)

        return abst_scores

    def evaluate_monolingual_retrieval_abstention(
        self, retriever, corpus, queries, relevant_docs, lang=None, **kwargs
    ):
        """Copy of Retrieval Evaluator function with abstention scores added"""
        start_time = time()
        results = retriever(corpus, queries)
        end_time = time()
        logger.info(
            "Time taken to retrieve: {:.2f} seconds".format(end_time - start_time)
        )

        if kwargs.get("save_predictions", False):
            output_folder = kwargs.get("output_folder", "results")
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            top_k = kwargs.get("top_k", None)
            if top_k is not None:
                for qid in list(results.keys()):
                    doc_ids = set(
                        sorted(
                            results[qid], key=lambda x: results[qid][x], reverse=True
                        )[:top_k]
                    )
                    results[qid] = {
                        k: v for k, v in results[qid].items() if k in doc_ids
                    }
            if lang is None:
                qrels_save_path = (
                    f"{output_folder}/{self.metadata_dict['name']}_predictions.json"
                )
            else:
                qrels_save_path = f"{output_folder}/{self.metadata_dict['name']}_{lang}_predictions.json"

            with open(qrels_save_path, "w") as f:
                json.dump(results, f)

        ndcg, _map, recall, precision = retriever.evaluate(
            relevant_docs,
            results,
            retriever.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        )
        mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        abstention = self.compute_abstention_scores_retrieval(relevant_docs, results)
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **abstention,
        }
        return scores


class AbstentionRerankingEvaluator(RerankingEvaluator):
    """This class evaluates a SentenceTransformer model for the task of re-ranking.
    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP is compute to measure the quality of the ranking.
    :param samples: Must be a list and each element is of the form:
        - {'query': '', 'positive': [], 'negative': []}. Query is the search query, positive is a list of positive
        (relevant) documents, negative is a list of negative (irrelevant) documents.
        - {'query': [], 'positive': [], 'negative': []}. Where query is a list of strings, which embeddings we average
        to get the query embedding.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def compute_metrics_individual(self, model):
        """Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        raise NotImplementedError(
            f"The {self.__class__.__name__} does not support individual computation of metrics."
        )

    def compute_metrics_batched(self, model):
        """Computes the metrics in a batched way, by batching all queries and
        all documents together
        """
        all_mrr_scores = []
        all_ap_scores = []

        # using encode_queries and encode_corpus functions if they exists,
        # which can be defined by users to add different instructions for query and passage conveniently
        encode_queries_func = (
            model.encode_queries if hasattr(model, "encode_queries") else model.encode
        )
        encode_corpus_func = (
            model.encode_corpus if hasattr(model, "encode_corpus") else model.encode
        )

        logger.info("Encoding queries...")
        if isinstance(self.samples[0]["query"], str):
            all_query_embs = np.asarray(
                encode_queries_func(
                    [sample["query"] for sample in self.samples],
                    batch_size=self.batch_size,
                )
            )
        elif isinstance(self.samples[0]["query"], list):
            # In case the query is a list of strings, we get the most similar embedding to any of the queries
            all_query_flattened = [
                q for sample in self.samples for q in sample["query"]
            ]
            all_query_embs = np.asarray(
                encode_queries_func(all_query_flattened, batch_size=self.batch_size)
            )
        else:
            raise ValueError(
                f"Query must be a string or a list of strings but is {type(self.samples[0]['query'])}"
            )

        logger.info("Encoding candidates...")
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample["positive"])
            all_docs.extend(sample["negative"])

        all_docs_embs = np.asarray(
            encode_corpus_func(all_docs, batch_size=self.batch_size)
        )

        # Compute mrr, ap scores and confidence scores
        logger.info("Evaluating...")
        query_idx, docs_idx = 0, 0
        conf_scores = {}
        for i, instance in enumerate(self.samples):
            num_subqueries = (
                len(instance["query"]) if isinstance(instance["query"], list) else 1
            )
            query_emb = all_query_embs[query_idx : query_idx + num_subqueries]
            query_idx += num_subqueries

            num_pos = len(instance["positive"])
            num_neg = len(instance["negative"])
            docs_emb = all_docs_embs[docs_idx : docs_idx + num_pos + num_neg]
            docs_idx += num_pos + num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            is_relevant = [True] * num_pos + [False] * num_neg

            scores = self._compute_metrics_instance(query_emb, docs_emb, is_relevant)
            all_mrr_scores.append(scores["mrr"])
            all_ap_scores.append(scores["ap"])

            # Compute confidence scores on instance (max, std, 1-2)
            pred_scores = (
                self.similarity_fct(query_emb, docs_emb).cpu().flatten().tolist()
            )
            pred_scores_sort = sorted(pred_scores)[::-1]
            pred_scores_mean = sum(pred_scores) / len(pred_scores)
            conf_scores[i] = {
                "max": pred_scores_sort[0],
                "std": (
                    sum((sc - pred_scores_mean) ** 2 for sc in pred_scores)
                    / len(pred_scores)
                )
                ** (1 / 2),
                "P1P2": pred_scores_sort[0] - pred_scores_sort[1],
            }

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        # # Compute nAUCs
        scores = {
            qid: {"ap": all_ap_scores[qid], "rr": all_mrr_scores[qid]}
            for qid in range(len(self.samples))
        }
        metrics = list(list(scores.values())[0].keys())
        abst_funcs = list(list(conf_scores.values())[0].keys())
        abst_rates = [k / 10 for k in range(10)]
        abst_scores = {}

        # # Evaluate for all abstention functions (max, std, P1P2)
        for abst_func in abst_funcs:
            conf_scs = {key: val[abst_func] for key, val in conf_scores.items()}
            conf_scs_sort = dict(
                sorted(conf_scs.items(), key=lambda item: item[1])[::-1]
            )
            evals = {metric: [] for metric in metrics}
            oracles = {metric: [] for metric in metrics}

            # Evaluate for all abstention rates
            for abst_rate in abst_rates:
                num_kept = len(conf_scs) - int(abst_rate * len(conf_scs))
                kept_qids = list(conf_scs_sort.keys())[:num_kept]

                # Evaluate for metrics (map and mrr)
                for metric in metrics:
                    evals[metric].append(
                        sum(scores[qid][metric] for qid in kept_qids) / num_kept
                    )
                    scs = [scores[qid][metric] for qid in scores.keys()]
                    scs_sort = sorted(scs)[::-1]
                    oracles[metric].append(sum(scs_sort[:num_kept]) / num_kept)

            # Compute nAUCs
            for metric in metrics:
                auc = sum(evals[metric]) / len(abst_rates) * max(abst_rates)
                auc_oracle = sum(oracles[metric]) / len(abst_rates) * max(abst_rates)
                auc_rand = oracles[metric][0] * max(abst_rates)
                abst_scores[f"nAUC_m{metric}_{abst_func}"] = (auc - auc_rand) / (
                    auc_oracle - auc_rand
                )

        return {**{"map": mean_ap, "mrr": mean_mrr}, **abst_scores}
