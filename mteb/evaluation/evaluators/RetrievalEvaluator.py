from __future__ import annotations

import logging
from typing import Any

from .Evaluator import Evaluator
from .model_classes import (
    DenseRetrievalExactSearch,
    is_cross_encoder_compatible,
)
from .utils import (
    add_task_specific_scores,
    calculate_retrieval_scores,
    evaluate_abstention,
    hole,
    mrr,
    recall_cap,
    top_k_accuracy,
)

logger = logging.getLogger(__name__)


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L9
class RetrievalEvaluator(Evaluator):
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
                retriever, encode_kwargs=encode_kwargs, **kwargs
            )
        self.k_values = k_values
        self.top_k = (
            max(k_values) if "top_k" not in kwargs else kwargs["top_k"]
        )  # can lower it if reranking
        self.task_name = task_name

    def __call__(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        instructions: dict[str, str] | None = None,
        qid: str | None = None,
        top_ranked: dict[str, list[str]] | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")

        # allow kwargs top-k to override the class top-k
        if "top_k" in kwargs:
            self.top_k = kwargs["top_k"]
            del kwargs["top_k"]

        if self.is_cross_encoder:
            return self.retriever.search_cross_encoder(
                corpus, queries, self.top_k, instructions=instructions, **kwargs
            )
        elif (
            hasattr(self.retriever.model.model, "mteb_model_meta")
            and self.retriever.model.model.mteb_model_meta.name == "bm25s"
        ):
            return self.retriever.model.model.search(
                corpus,
                queries,
                self.top_k,
                task_name=self.task_name,  # type: ignore
                instructions=instructions,
                **kwargs,
            )
        else:
            return self.retriever.search(
                corpus,
                queries,
                self.top_k,
                instructions=instructions,
                top_ranked=top_ranked,
                request_qid=qid,
                task_name=self.task_name,
                **kwargs,
            )

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
        task_name: str = None,
    ) -> tuple[
        dict[str, float],
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
            # Remove identical ids from results dict
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
        else:
            logger.debug(
                "For evaluation, we DO NOT ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=True`` to ignore this."
            )

        all_scores, ndcg, _map, recall, precision, naucs = calculate_retrieval_scores(
            results, qrels, k_values
        )
        task_scores = add_task_specific_scores(
            all_scores, qrels, results, task_name, k_values
        )

        return ndcg, _map, recall, precision, naucs, task_scores

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

        naucs = evaluate_abstention(results, metric_scores)
        metric_scores_avg = {k: sum(v) / len(v) for k, v in metric_scores.items()}

        return metric_scores_avg, naucs
