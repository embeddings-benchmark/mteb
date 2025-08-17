from __future__ import annotations

import logging
from typing import Any

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.models_protocols import CrossEncoderProtocol, Encoder, SearchProtocol
from mteb.models.search_wrappers import SearchCrossEncoderWrapper, SearchEncoderWrapper
from mteb.types import (
    CorpusDatasetType,
    QueryDatasetType,
    RelevantDocumentsType,
    RetrievalEvaluationResult,
    RetrievalOutputType,
    TopRankedDocumentsType,
)

from .evaluator import Evaluator
from .retrieval_metrics import (
    calculate_retrieval_scores,
)

logger = logging.getLogger(__name__)


class RetrievalEvaluator(Evaluator):
    def __init__(
        self,
        corpus: CorpusDatasetType,
        queries: QueryDatasetType,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        top_ranked: TopRankedDocumentsType | None = None,
        qid: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.corpus = corpus
        self.queries = queries
        self.top_ranked = top_ranked

        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.qid = qid
        self.top_k = top_k

    def __call__(
        self,
        model: SearchProtocol | Encoder | CrossEncoderProtocol,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> RetrievalOutputType:
        if isinstance(model, Encoder) and not isinstance(model, SearchProtocol):
            search_model = SearchEncoderWrapper(model)
        elif isinstance(model, CrossEncoderProtocol):
            search_model = SearchCrossEncoderWrapper(model)
        elif isinstance(model, SearchProtocol):
            search_model = model
        else:
            raise TypeError(
                f"RetrievalEvaluator expects a SearchInterface, Encoder, or CrossEncoder, got {type(model)}"
            )

        search_model.index(
            corpus=self.corpus,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            encode_kwargs=encode_kwargs,
        )
        return search_model.search(
            queries=self.queries,
            top_k=self.top_k,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            encode_kwargs=encode_kwargs,
            top_ranked=self.top_ranked,
        )

    def evaluate(
        self,
        qrels: RelevantDocumentsType,
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
    ) -> RetrievalEvaluationResult:
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
        return calculate_retrieval_scores(results, qrels, k_values)
