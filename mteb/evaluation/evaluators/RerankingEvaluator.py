from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score

from .Evaluator import Evaluator
from .utils import confidence_scores, cos_sim, nAUC

logger = logging.getLogger(__name__)


class RerankingEvaluator(Evaluator):
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
        samples,
        mrr_at_k: int = 10,
        name: str = "",
        similarity_fct=cos_sim,
        batch_size: int = 512,
        use_batched_encoding: bool = True,
        limit: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            samples = samples.train_test_split(limit)["test"]
        self.samples = samples
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.similarity_fct = similarity_fct
        self.batch_size = batch_size
        self.use_batched_encoding = use_batched_encoding

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        ### Remove sample with empty positive / negative set
        self.samples = [
            sample
            for sample in self.samples
            if len(sample["positive"]) > 0 and len(sample["negative"]) > 0
        ]

    def __call__(self, model):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model):
        return (
            self.compute_metrics_batched(model)
            if self.use_batched_encoding
            else self.compute_metrics_individual(model)
        )

    def compute_metrics_batched(self, model):
        """Computes the metrices in a batched way, by batching all queries and
        all documents together
        """
        all_mrr_scores = []
        all_ap_scores = []
        all_conf_scores = []

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
            all_query_embs = self._encode_unique_texts(
                all_query_flattened, encode_corpus_func
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

        all_docs_embs = self._encode_unique_texts(all_docs, encode_corpus_func)

        # Compute scores and confidence scores
        logger.info("Evaluating...")
        query_idx, docs_idx = 0, 0
        for instance in self.samples:
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

            sim_scores = self._compute_sim_scores_instance(query_emb, docs_emb)
            scores = self._compute_metrics_instance(sim_scores, is_relevant)
            conf_scores = self.conf_scores(sim_scores.tolist())

            all_mrr_scores.append(scores["mrr"])
            all_ap_scores.append(scores["ap"])
            all_conf_scores.append(conf_scores)

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        # Compute nAUCs
        naucs_map = self.nAUC_scores(all_conf_scores, all_ap_scores, "map")
        naucs_mrr = self.nAUC_scores(all_conf_scores, all_mrr_scores, "mrr")

        return {**{"map": mean_ap, "mrr": mean_mrr}, **naucs_map, **naucs_mrr}

    def compute_metrics_individual(self, model):
        """Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        all_mrr_scores = []
        all_ap_scores = []
        all_conf_scores = []

        # using encode_queries and encode_corpus functions if they exists,
        # which can be defined by users to add different instructions for query and passage conveniently
        encode_queries_func = (
            model.encode_queries if hasattr(model, "encode_queries") else model.encode
        )
        encode_corpus_func = (
            model.encode_corpus if hasattr(model, "encode_corpus") else model.encode
        )

        for instance in tqdm.tqdm(self.samples, desc="Samples"):
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])

            if len(positive) == 0 or len(negative) == 0:
                continue

            docs = positive + negative
            is_relevant = [True] * len(positive) + [False] * len(negative)

            if isinstance(query, str):
                # .encoding interface requires List[str] as input
                query = [query]
            query_emb = np.asarray(
                encode_queries_func(query, batch_size=self.batch_size)
            )
            docs_emb = np.asarray(encode_corpus_func(docs, batch_size=self.batch_size))

            sim_scores = self._compute_sim_scores_instance(query_emb, docs_emb)
            scores = self._compute_metrics_instance(sim_scores, is_relevant)
            conf_scores = self.conf_scores(sim_scores.tolist())

            all_mrr_scores.append(scores["mrr"])
            all_ap_scores.append(scores["ap"])
            all_conf_scores.append(conf_scores)

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        # Compute nAUCs
        naucs_map = self.nAUC_scores(all_conf_scores, all_ap_scores, "map")
        naucs_mrr = self.nAUC_scores(all_conf_scores, all_mrr_scores, "mrr")

        return {**{"map": mean_ap, "mrr": mean_mrr}, **naucs_map, **naucs_mrr}

    def _encode_unique_texts(self, all_texts, encode_queries_func):
        index_map, all_unique_texts, all_texts_indexes = {}, [], []
        for text in all_texts:
            text_hash = hash(text)
            if text_hash not in index_map:
                index_map[text_hash] = len(all_unique_texts)
                all_unique_texts.append(text)
            all_texts_indexes.append(index_map[text_hash])
        logger.warning(
            f"A total on {len(all_texts) - len(all_unique_texts)} duplicate texts were found during encoding. Only encoding unique text and duplicating embeddings across."
        )
        all_unique_texts_embs = np.asarray(
            encode_queries_func(all_unique_texts, batch_size=self.batch_size)
        )
        return all_unique_texts_embs[all_texts_indexes]

    def _compute_sim_scores_instance(
        self, query_emb: torch.Tensor, docs_emb: torch.Tensor
    ) -> torch.Tensor:
        """Computes similarity scores for a single instance = (query, positives, negatives)

        Args:
            query_emb (`torch.Tensor` of shape `(num_queries, hidden_size)`): Query embedding
                if `num_queries` > 0: we take the closest document to any of the queries
            docs_emb (`torch.Tensor` of shape `(num_pos+num_neg, hidden_size)`): Candidates documents embeddings

        Returns:
            sim_scores (`torch.Tensor of shape `(num_pos+num_neg,)`): Query-documents similarity scores
        """
        sim_scores = self.similarity_fct(query_emb, docs_emb)
        if len(sim_scores.shape) > 1:
            sim_scores = torch.amax(sim_scores, dim=0)

        return sim_scores

    def _compute_metrics_instance(
        self, sim_scores: torch.Tensor, is_relevant: List[bool]
    ) -> Dict[str, float]:
        """Computes metrics for a single instance = (query, positives, negatives)

        Args:
            sim_scores (`torch.Tensor of shape `(num_pos+num_neg,)`): Query-documents similarity scores
            is_relevant (`List[bool]` of length `num_pos+num_neg`): True if the document is relevant

        Returns:
            scores (`Dict[str, float]`):
                - `mrr`: Mean Reciprocal Rank @ `self.mrr_at_k`
                - `ap`: Average Precision
        """
        pred_scores_argsort = torch.argsort(-sim_scores)  # Sort in decreasing order
        mrr = self.mrr_at_k_score(is_relevant, pred_scores_argsort, self.mrr_at_k)
        ap = self.ap_score(is_relevant, sim_scores.cpu().tolist())
        return {"mrr": mrr, "ap": ap}

    @staticmethod
    def conf_scores(sim_scores: torch.Tensor) -> Dict[str, float]:
        """Computes confidence scores for a single instance = (query, positives, negatives)

        Args:
            sim_scores (`List[float]` of shape `(num_pos+num_neg,)`): Query-documents similarity scores

        Returns:
            conf_scores (`Dict[str, float]`):
                - `max`: Maximum similarity score
                - `std`: Standard deviation of similarity scores
                - `diff1`: Difference between highest and second highest similarity scores
        """
        return confidence_scores(sim_scores)

    @staticmethod
    def nAUC_scores(
        all_conf_scores: List[Dict[str, float]],
        metrics: List[float],
        metric_name: str,
    ) -> Dict[str, float]:
        """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997

        Args:
            all_conf_scores (`Dict[str, List[float]]`):
                - `max`: Maximum similarity scores
                - `std`: Standard deviations of similarity scores
                - `diff1`: Differences between highest and second highest similarity scores
            metrics (`List[float]` of length `len(samples)`): True if the document is relevant
            metric_name (`str`): Name of the metric (mrr or ap)

        Returns:
            naucs (`Dict[str, float]`): nAUCs for each confidence function
        """
        conf_fcts = list(all_conf_scores[0].keys())
        all_conf_scores = {
            fct: np.array([x[fct] for x in all_conf_scores]) for fct in conf_fcts
        }
        metrics = np.array(metrics)
        naucs = {
            f"nAUC_{metric_name}_{fct}": nAUC(all_conf_scores[fct], metrics)
            for fct in conf_fcts
        }
        return naucs

    @staticmethod
    def mrr_at_k_score(
        is_relevant: list[bool], pred_ranking: list[int], k: int
    ) -> float:
        """Computes MRR@k score

        Args:
            is_relevant: True if the document is relevant
            pred_ranking: Indices of the documents sorted in decreasing order
                of the similarity score
            k: Top-k documents to consider

        Returns:
            The MRR@k score
        """
        mrr_score = 0
        for rank, index in enumerate(pred_ranking[:k]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break

        return mrr_score

    @staticmethod
    def ap_score(is_relevant, pred_scores):
        """Computes AP score

        Args:
            is_relevant (`List[bool]` of length `num_pos+num_neg`): True if the document is relevant
            pred_scores (`List[float]` of length `num_pos+num_neg`): Predicted similarity scores

        Returns:
            ap_score (`float`): AP score
        """
        # preds = np.array(is_relevant)[pred_scores_argsort]
        # precision_at_k = np.mean(preds[:k])
        # ap = np.mean([np.mean(preds[: k + 1]) for k in range(len(preds)) if preds[k]])
        ap = average_precision_score(is_relevant, pred_scores)
        return ap
