from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score

from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator

from ...encoder_interface import Encoder, PromptType
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
        task_name: str | None = None,
        mrr_at_k: int = 10,
        name: str = "",
        encode_kwargs: dict[str, Any] = {},
        use_batched_encoding: bool = True,
        limit: int | None = None,
        k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000],
        evaluator_type: str = "standard",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            samples = samples.train_test_split(limit)["test"]
        self.samples = samples
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.use_batched_encoding = use_batched_encoding
        self.task_name = task_name
        self.k_values = k_values
        self.evaluator_type = evaluator_type
        self.encode_kwargs = encode_kwargs

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 512

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        ### Remove sample with empty positive / negative set
        self.samples = [
            sample
            for sample in self.samples
            if len(sample["positive"]) > 0 and len(sample["negative"]) > 0
        ]

    def __call__(self, model: Encoder):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model: Encoder):
        return (
            self.compute_metrics_batched(model)
            if self.use_batched_encoding
            else self.compute_metrics_individual(model)
        )

    def compute_metrics_batched(self, model: Encoder):
        """Computes the metrices in a batched way, by batching all queries and
        all documents together
        """
        logger.info("Encoding queries...")
        if isinstance(self.samples[0]["query"], str):
            all_query_embs = np.asarray(
                model.encode(
                    [sample["query"] for sample in self.samples],
                    task_name=self.task_name,
                    prompt_type=PromptType.query,
                    **self.encode_kwargs,
                )
            )
        elif isinstance(self.samples[0]["query"], list):
            # In case the query is a list of strings, we get the most similar embedding to any of the queries
            all_query_flattened = [
                q for sample in self.samples for q in sample["query"]
            ]
            all_query_embs = self._encode_unique_texts(
                all_query_flattened,
                model,
                task_name=self.task_name,
                prompt_type=PromptType.query,
                **self.encode_kwargs,
            )
        else:
            raise ValueError(
                f"Query must be a string or a list of strings but is {type(self.samples[0]['query'])}"
            )

        if self.evaluator_type == "standard":
            results = self._encode_candidates(
                model=model,
                batched=True,
                all_query_embs=all_query_embs,
            )
        elif self.evaluator_type == "miracl":
            results = self._encode_candidates_miracl(
                model=model,
                batched=True,
                all_query_embs=all_query_embs,
            )
        return results

    def compute_metrics_individual(self, model: Encoder):
        """Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        if self.evaluator_type == "standard":
            results = self._encode_candidates(
                model=model,
                batched=False,
            )
        elif self.evaluator_type == "miracl":
            results = self._encode_candidates_miracl(
                model=model,
                batched=False,
            )
        return results

    def _encode_candidates(self, model: Encoder, batched: bool, all_query_embs=None):
        all_mrr_scores = []
        all_ap_scores = []
        all_conf_scores = []
        logger.info("Encoding candidates...")
        if batched:
            self._encode_candidates_batched(
                model=model,
                all_query_embs=all_query_embs,
                all_mrr_scores=all_mrr_scores,
                all_ap_scores=all_ap_scores,
                all_conf_scores=all_conf_scores,
            )
        else:
            self._encode_candidates_individual(
                model=model,
                all_mrr_scores=all_mrr_scores,
                all_ap_scores=all_ap_scores,
                all_conf_scores=all_conf_scores,
            )
        scores = self._collect_results(all_mrr_scores, all_ap_scores, all_conf_scores)
        return scores

    def _encode_candidates_batched(
        self,
        all_query_embs,
        model: Encoder,
        all_mrr_scores,
        all_ap_scores,
        all_conf_scores,
    ):
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample["positive"])
            all_docs.extend(sample["negative"])

        all_docs_embs = self._encode_unique_texts(
            all_docs,
            model,
            task_name=self.task_name,
            prompt_type=PromptType.passage,
            **self.encode_kwargs,
        )

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
            self._apply_sim_scores(
                query_emb,
                docs_emb,
                is_relevant,
                all_mrr_scores,
                all_ap_scores,
                all_conf_scores,
                model,
            )

    def _encode_candidates_individual(
        self,
        model: Encoder,
        all_mrr_scores,
        all_ap_scores,
        all_conf_scores,
    ):
        for instance in tqdm.tqdm(self.samples, desc="Samples"):
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])

            if len(positive) == 0 or len(negative) == 0:
                continue

            docs = positive + negative
            is_relevant = [True] * len(positive) + [False] * len(negative)

            if isinstance(query, str):
                # .encoding interface requires list[str] as input
                query = [query]
            query_emb = np.asarray(
                model.encode(
                    query,
                    task_name=self.task_name,
                    prompt_type=PromptType.query,
                    **self.encode_kwargs,
                )
            )
            docs_emb = np.asarray(
                model.encode(
                    docs,
                    task_name=self.task_name,
                    prompt_type=PromptType.passage,
                    **self.encode_kwargs,
                )
            )
            self._apply_sim_scores(
                query_emb,
                docs_emb,
                is_relevant,
                all_mrr_scores,
                all_ap_scores,
                all_conf_scores,
                model,
            )

    def _collect_results(self, all_mrr_scores, all_ap_scores, all_conf_scores):
        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        # Compute nAUCs
        naucs_map = self.nAUC_scores(all_conf_scores, all_ap_scores, "map")
        naucs_mrr = self.nAUC_scores(all_conf_scores, all_mrr_scores, "mrr")

        return {**{"map": mean_ap, "mrr": mean_mrr}, **naucs_map, **naucs_mrr}

    def _encode_candidates_miracl(
        self,
        model: Encoder,
        batched,
        all_query_embs=None,
    ):
        if batched:
            return self._encode_candidates_miracl_batched(
                model=model, all_query_embs=all_query_embs
            )
        else:
            return self._encode_candidates_miracl_individual(
                model=model,
            )

    def _encode_candidates_miracl_batched(self, all_query_embs, model: Encoder):
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample["candidates"])

        all_docs_embs = np.asarray(
            model.encode(
                all_docs,
                task_name=self.task_name,
                prompt_type=PromptType.passage,
                **self.encode_kwargs,
            )
        )

        # Compute scores
        logger.info("Evaluating...")
        query_idx, docs_idx = 0, 0
        results, qrels = {}, {}
        for instance in self.samples:
            num_subqueries = (
                len(instance["query"]) if isinstance(instance["query"], list) else 1
            )
            query_emb = all_query_embs[query_idx : query_idx + num_subqueries]
            query_idx += num_subqueries

            positive = instance["positive"]
            docs = instance["candidates"]
            num_doc = len(docs)
            docs_emb = all_docs_embs[docs_idx : docs_idx + num_doc]
            docs_idx += num_doc

            fake_qid = str(query_idx)
            results[fake_qid] = self.rerank(query_emb, docs_emb, model)
            qrels[fake_qid] = {
                str(i): 1 if doc in positive else 0 for i, doc in enumerate(docs)
            }

        scores_miracl = self._collect_miracl_results(results, qrels)
        return scores_miracl

    def _encode_candidates_miracl_individual(self, model: Encoder):
        results, qrels = {}, {}
        for i, instance in enumerate(tqdm.tqdm(self.samples, desc="Samples")):
            query = instance["query"]
            positive = set(instance["positive"])
            docs = list(instance["candidates"])

            if isinstance(query, str):
                # .encoding interface requires list[str] as input
                query_emb = np.asarray(
                    model.encode(
                        [query],
                        task_name=self.task_name,
                        prompt_type=PromptType.query,
                        **self.encode_kwargs,
                    )
                )
                docs_emb = np.asarray(
                    model.encode(
                        docs,
                        task_name=self.task_name,
                        prompt_type=PromptType.passage,
                        **self.encode_kwargs,
                    )
                )

            fake_qid = str(i)
            results[fake_qid] = self.rerank(query_emb, docs_emb, model)
            qrels[fake_qid] = {
                str(i): 1 if doc in positive else 0 for i, doc in enumerate(docs)
            }

        scores_miracl = self._collect_miracl_results(results, qrels)
        return scores_miracl

    def _collect_miracl_results(self, results, qrels):
        ndcg, _map, recall, precision, naucs = RetrievalEvaluator.evaluate(
            qrels=qrels,
            results=results,
            k_values=self.k_values,
            ignore_identical_ids=False,
        )
        scores = {**ndcg, **_map, **recall, **precision, **naucs}
        scores_miracl = {f"{k}(MIRACL)": v for k, v in scores.items()}
        return scores_miracl

    def rerank(
        self, query_emb: np.ndarray, docs_emb: np.ndarray, model: Encoder
    ) -> dict[str, float]:
        """Rerank documents (docs_emb) given the query (query_emb)

        Args:
            query_emb: Query embedding of shape `(num_queries, hidden_size)`)
                if `num_queries` > 0: we take the closest document to any of the queries
            docs_emb: Candidates documents embeddings of shape `(num_pos+num_neg, hidden_size)`)
            model: Model to use for computing similarity scores if model.similarity is available

        Returns:
            similarity_scores:
        """
        if not query_emb.shape[0]:
            raise ValueError("Empty query embedding")

        if not docs_emb.shape[0]:
            return {"empty-docid": 0}

        if hasattr(model, "similarity"):
            pred_scores = model.similarity(query_emb, docs_emb)
        else:
            pred_scores = cos_sim(query_emb, docs_emb)
        if len(pred_scores.shape) > 1:
            pred_scores = torch.amax(pred_scores, dim=0)

        return {
            str(i): score.detach().numpy().item() for i, score in enumerate(pred_scores)
        }

    def _apply_sim_scores(
        self,
        query_emb,
        docs_emb,
        is_relevant,
        all_mrr_scores,
        all_ap_scores,
        all_conf_scores,
        model: Encoder,
    ):
        sim_scores = self._compute_sim_scores_instance(query_emb, docs_emb, model)
        scores = self._compute_metrics_instance(sim_scores, is_relevant)
        conf_scores = self.conf_scores(sim_scores.tolist())

        all_mrr_scores.append(scores["mrr"])
        all_ap_scores.append(scores["ap"])
        all_conf_scores.append(conf_scores)

    @staticmethod
    def _encode_unique_texts(
        all_texts: list[str],
        model: Encoder,
        task_name: str | None,
        prompt_type: PromptType | None,
        **encode_kwargs: Any,
    ):
        index_map, all_unique_texts, all_texts_indexes = {}, [], []
        for text in all_texts:
            text_hash = hash(text)
            if text_hash not in index_map:
                index_map[text_hash] = len(all_unique_texts)
                all_unique_texts.append(text)
            all_texts_indexes.append(index_map[text_hash])
        logger.warning(
            f"A total of {len(all_texts) - len(all_unique_texts)}/{len(all_texts)} duplicate texts were found during encoding. Only encoding unique text and duplicating embeddings across."
        )
        all_unique_texts_embs = np.asarray(
            model.encode(
                all_unique_texts,
                task_name=task_name,
                prompt_type=prompt_type,
                **encode_kwargs,
            )
        )
        return all_unique_texts_embs[all_texts_indexes]

    def _compute_sim_scores_instance(
        self, query_emb: np.ndarray, docs_emb: np.ndarray, model: Encoder
    ) -> torch.Tensor:
        """Computes similarity scores for a single instance = (query, positives, negatives)

        Args:
            query_emb: Query embedding, with shape `(num_queries, hidden_size)`
                if `num_queries` > 0: we take the closest document to any of the queries
            docs_emb: Candidates documents embeddings, with shape `(num_pos+num_neg, hidden_size)`
            model: Model to use for computing similarity scores if model.similarity is available

        Returns:
            sim_scores: Query-documents similarity scores, with shape `(num_pos+num_neg,)`
        """
        if hasattr(model, "similarity"):
            sim_scores = model.similarity(query_emb, docs_emb)
        else:
            sim_scores = cos_sim(query_emb, docs_emb)
        if len(sim_scores.shape) > 1:
            sim_scores = torch.amax(sim_scores, dim=0)

        return sim_scores

    def _compute_metrics_instance(
        self, sim_scores: torch.Tensor, is_relevant: list[bool]
    ) -> dict[str, float]:
        """Computes metrics for a single instance = (query, positives, negatives)

        Args:
            sim_scores: Query-documents similarity scores, with shape `(num_pos+num_neg,)`
            is_relevant: True if the document is relevant, with length `num_pos+num_neg`

        Returns:
            scores:
                - `mrr`: Mean Reciprocal Rank @ `self.mrr_at_k`
                - `ap`: Average Precision
        """
        pred_scores_argsort = torch.argsort(-sim_scores)  # Sort in decreasing order
        mrr = self.mrr_at_k_score(is_relevant, pred_scores_argsort, self.mrr_at_k)
        ap = self.ap_score(is_relevant, sim_scores.cpu().tolist())
        return {"mrr": mrr, "ap": ap}

    @staticmethod
    def conf_scores(sim_scores: torch.Tensor) -> dict[str, float]:
        """Computes confidence scores for a single instance = (query, positives, negatives)

        Args:
            sim_scores: Query-documents similarity scores, with shape `(num_pos+num_neg,)`

        Returns:
            conf_scores:
                - `max`: Maximum similarity score
                - `std`: Standard deviation of similarity scores
                - `diff1`: Difference between highest and second highest similarity scores
        """
        return confidence_scores(sim_scores)

    @staticmethod
    def nAUC_scores(
        all_conf_scores: list[dict[str, float]],
        metrics: list[float],
        metric_name: str,
    ) -> dict[str, float]:
        """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997

        Args:
            all_conf_scores: Confidence scores for all instances, with length `len(samples)`
            metrics: Metric scores for all instances, with length `len(samples)`
            metric_name: Name of the metric (mrr or ap)

        Returns:
            naucs: nAUCs for each confidence function
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
            is_relevant (`list[bool]` of length `num_pos+num_neg`): True if the document is relevant
            pred_scores (`list[float]` of length `num_pos+num_neg`): Predicted similarity scores

        Returns:
            ap_score (`float`): AP score
        """
        # preds = np.array(is_relevant)[pred_scores_argsort]
        # precision_at_k = np.mean(preds[:k])
        # ap = np.mean([np.mean(preds[: k + 1]) for k in range(len(preds)) if preds[k]])
        ap = average_precision_score(is_relevant, pred_scores)
        return ap
