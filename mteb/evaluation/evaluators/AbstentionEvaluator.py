from __future__ import annotations

from time import time
import os
import json
from typing import Dict, List, Tuple
import pytrec_eval

import logging

import torch

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
        raise NotImplementedError("The abstention evaluator must not be called directly.")

    @staticmethod
    def compute_abstention_scores_retrieval(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: List[int] = [1, 3, 5, 10],
        ignore_identical_ids: bool = True
    ) -> Dict[str, float]:

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
                'max': scs_sort[0], 
                'std': (sum((sc - scs_mean) ** 2 for sc in scs) / len(scs)) ** (1/2), 
                'P1P2': scs_sort[0] - scs_sort[1]
            }
        
        # Compute nAUCs
        metrics = list(list(scores.values())[0].keys())
        abst_funcs = list(list(conf_scores.values())[0].keys())
        abst_rates = [k/10 for k in range(10)]
        abst_scores = {}

        # Evaluate for all abstention functions (max, std, P1P2)
        for abst_func in abst_funcs:
            conf_scs = {key: val[abst_func] for key, val in conf_scores.items()}
            conf_scs_sort = dict(sorted(conf_scs.items(), key=lambda item: item[1])[::-1])
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

        # ndcg, _map, recall, precision = retriever.evaluate(
        #     relevant_docs,
        #     results,
        #     retriever.k_values,
        #     ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
        # )
        # mrr = retriever.evaluate_custom(
        #     relevant_docs, results, retriever.k_values, "mrr"
        # )
        abstention = self.compute_abstention_scores_retrieval(
            relevant_docs, results
        )
        scores = {
            # **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            # **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            # **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            # **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            # **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
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

    def _compute_metrics_instance(self, query_emb, docs_emb, is_relevant):
        """Computes metrics for a single instance = (query, positives, negatives)

        Args:
            query_emb (`torch.Tensor` of shape `(num_queries, hidden_size)`): Query embedding
                if `num_queries` > 0: we take the closest document to any of the queries
            docs_emb (`torch.Tensor` of shape `(num_pos+num_neg, hidden_size)`): Candidates documents embeddings
            is_relevant (`List[bool]` of length `num_pos+num_neg`): True if the document is relevant

        Returns:
            scores (`Dict[str, float]`):
                - `mrr`: Mean Reciprocal Rank @ `self.mrr_at_k`
                - `ap`: Average Precision
        """
        pred_scores = self.similarity_fct(query_emb, docs_emb)
        if len(pred_scores.shape) > 1:
            pred_scores = torch.amax(pred_scores, dim=0)

        pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order

        mrr = self.mrr_at_k_score(is_relevant, pred_scores_argsort, self.mrr_at_k)
        ap = self.ap_score(is_relevant, pred_scores.cpu().tolist())
        # TODO: Here code the metrics (best would be to use the same fincton as above for the retrieving ?)
        return {"mrr": mrr, "ap": ap, "Abstention_Placeholder": 0}
