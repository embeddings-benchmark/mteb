from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.encoder_interface import Encoder

from ._evaluator import Evaluator
from .dense_retrieval_exact_search import (
    DenseRetrievalExactSearch,
    is_cross_encoder_compatible,
)
from .retrieval_metrics import (
    calculate_retrieval_scores,
)

logger = logging.getLogger(__name__)


class RetrievalEvaluator(Evaluator):
    def __init__(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        instructions: dict[str, str] | None = None,
        top_ranked: dict[str, list[str]] | None = None,
        qid: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.corpus = corpus
        self.queries = queries
        self.instructions = instructions
        self.top_ranked = top_ranked

        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.qid = qid
        self.top_k = top_k

    def __call__(
        self,
        model: Encoder,
        encode_kwargs: dict[str, Any],
        previous_results: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        self.is_cross_encoder = is_cross_encoder_compatible(model)
        if self.is_cross_encoder:
            logger.info(
                "The custom predict function of the model will be used if not a SentenceTransformer CrossEncoder"
            )
        self.retriever = DenseRetrievalExactSearch(
            model, encode_kwargs=encode_kwargs, previous_results=previous_results
        )

        if self.is_cross_encoder:
            score_function = self.retriever.search_cross_encoder
        elif (
            hasattr(model, "mteb_model_meta")
            and model.mteb_model_meta is not None
            and model.mteb_model_meta.name == "bm25s"
        ):
            score_function = self.retriever.model.search
        else:
            score_function = self.retriever.search

        return score_function(
            self.corpus,
            self.queries,
            self.top_k,
            instructions=self.instructions,
            top_ranked=self.top_ranked,
            request_qid=self.qid,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **kwargs,
        )

    def evaluate(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
    ) -> tuple[
        dict[str, dict[str, float]],
        dict[str, float],
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

        all_scores, ndcg, _map, recall, precision, naucs, mrr, naucs_mrr = (
            calculate_retrieval_scores(results, qrels, k_values)
        )

        return all_scores, ndcg, _map, recall, precision, naucs, mrr, naucs_mrr
