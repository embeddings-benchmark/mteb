from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm.auto import tqdm

from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import (
    CrossEncoderProtocol,
    EncoderProtocol,
    SearchProtocol,
)
from mteb.models.search_wrappers import (
    SearchCrossEncoderWrapper,
    SearchEncoderWrapper,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import MTEBModels
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search wrapper that combines multiple models using a specified fusion strategy."""

    def __init__(
        self,
        models: Sequence[MTEBModels],
        weights: Sequence[float] | None = None,
        sub_model_top_k: int | None = None,
        fusion_strategy: Literal["rrf", "dbsf", "relative-score-fusion"]
        | Callable[
            [Sequence[Mapping[str, float]], Sequence[float]], Mapping[str, float]
        ] = "rrf",
        rrf_k: int = 60,
    ) -> None:
        """Initialize the HybridSearch wrapper.

        Args:
            models: Sequence of sub-models to combine. Must contain at least two models.
            weights: Optional Sequence of weights for each sub-model. If None, equal weights are assigned.
            sub_model_top_k: Optional top-k documents to retrieve from individual retriever sub-models.
            fusion_strategy: Fusion strategy to combine sub-model scores.
                Options: "rrf" (Reciprocal Rank Fusion), "dbsf" (Distribution-Based Score Fusion),
                "relative-score-fusion", or a custom Callable.
            rrf_k: The rank constant used for Reciprocal Rank Fusion (default: 60).
        """
        if len(models) < 2:
            raise ValueError("At least two models must be provided for hybrid search.")

        self.weights: Sequence[float] = (
            [1.0 / len(models)] * len(models) if weights is None else weights
        )
        if len(self.weights) != len(models):
            raise ValueError("Length of weights must match the number of models.")

        if sub_model_top_k is not None and sub_model_top_k <= 0:
            raise ValueError("sub_model_top_k must be greater than 0")
        self.sub_model_top_k = sub_model_top_k

        if not isinstance(rrf_k, int):
            raise TypeError("rrf_k must be an integer")
        if rrf_k < 0:
            raise ValueError("rrf_k must be greater than or equal to 0")
        self.rrf_k = rrf_k

        self.wrapped_models: list[SearchProtocol] = []
        names = []

        for model in models:
            wrapped: SearchProtocol
            if isinstance(model, EncoderProtocol) and not isinstance(
                model, SearchProtocol
            ):
                wrapped = SearchEncoderWrapper(model)
            elif isinstance(model, CrossEncoderProtocol):
                wrapped = SearchCrossEncoderWrapper(model)
            elif isinstance(model, SearchProtocol):
                wrapped = model
            else:
                raise TypeError(
                    f"Expected a SearchProtocol, EncoderProtocol, or CrossEncoderProtocol, got {type(model)}"
                )

            self.wrapped_models.append(wrapped)

            meta = wrapped.mteb_model_meta
            if meta and meta.name:
                names.append(meta.name.rsplit("/", 1)[-1])
            else:
                names.append("unknown")

        self.fusion_name: str
        self._fuse_fn: Callable[
            [Sequence[Mapping[str, float]], Sequence[float]], Mapping[str, float]
        ]

        if isinstance(fusion_strategy, str):
            self.fusion_name = fusion_strategy
            if fusion_strategy == "rrf":
                self._fuse_fn = lambda q_scores, w: fuse_rrf(
                    q_scores, w, rrf_k=self.rrf_k
                )
            elif fusion_strategy == "dbsf":
                self._fuse_fn = fuse_dbsf
            elif fusion_strategy in {"relative-score-fusion", "relative_score_fusion"}:
                self.fusion_name = "relative-score-fusion"
                self._fuse_fn = fuse_relative_score_fusion
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        elif callable(fusion_strategy):
            self.fusion_name = getattr(fusion_strategy, "__name__", "custom")
            self._fuse_fn = fusion_strategy
        else:
            raise TypeError(
                "fusion_strategy must be one of 'rrf', 'dbsf', 'relative-score-fusion', or a callable"
            )

        combined_name = f"mteb/hybrid-{self.fusion_name}-{'-'.join(names)}"
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": combined_name,
                "model_type": ["hybrid"],
            }
        )

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        """Index the corpus dataset using all sub-models."""
        logger.info("Indexing corpus using sub-models...")
        for model in tqdm(self.wrapped_models, desc="Indexing sub-models"):
            model.index(
                corpus,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                num_proc=num_proc,
            )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        """Search the queries using all sub-models and fuse the results."""
        if self.sub_model_top_k is None:
            sub_top_k = max(top_k, 100)
        else:
            sub_top_k = max(top_k, self.sub_model_top_k)

        logger.info("Running all sub-models...")

        all_results = []
        for model in tqdm(self.wrapped_models, desc="Sub-models"):
            res = model.search(
                queries=queries,
                top_k=sub_top_k,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                top_ranked=top_ranked,
                num_proc=num_proc,
            )
            all_results.append(res)

        fused_results: RetrievalOutputType = {}
        for row in queries:
            qid = row["id"]

            query_scores_list = []
            for res in all_results:
                query_scores_list.append(res.get(qid, {}))

            fused_query_scores = self.fuse(query_scores_list)

            sorted_fused = sorted(
                fused_query_scores.items(), key=lambda x: x[1], reverse=True
            )
            fused_results[qid] = dict(sorted_fused[:top_k])

        return fused_results

    def fuse(
        self, query_scores_list: Sequence[Mapping[str, float]]
    ) -> Mapping[str, float]:
        """Fuse the query scores from multiple sub-models."""
        return self._fuse_fn(query_scores_list, self.weights)


def fuse_dbsf(
    query_scores_list: Sequence[Mapping[str, float]], weights: Sequence[float]
) -> Mapping[str, float]:
    """Fuse the query scores using Distribution-Based Score Fusion. (https://arxiv.org/html/2410.20878v1)"""
    fused: dict[str, float] = {}
    for scores, weight in zip(query_scores_list, weights, strict=True):
        if not scores:
            continue

        doc_ids = list(scores.keys())
        raw_scores = np.array(list(scores.values()), dtype=float)

        mu = np.mean(raw_scores)
        sigma = np.std(raw_scores)

        if sigma < 1e-9:
            normalized = np.full_like(raw_scores, 0.5)
        else:
            lower_limit = mu - 3 * sigma
            upper_limit = mu + 3 * sigma
            denom = upper_limit - lower_limit
            normalized = (raw_scores - lower_limit) / denom
            normalized = np.clip(normalized, 0.0, 1.0)

        for doc_id, norm_score in zip(doc_ids, normalized):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * float(norm_score)

    return fused


def fuse_rrf(
    query_scores_list: Sequence[Mapping[str, float]],
    weights: Sequence[float],
    rrf_k: int = 60,
) -> Mapping[str, float]:
    """Fuse the query scores using Reciprocal Rank Fusion."""
    fused: dict[str, float] = {}
    for scores, weight in zip(query_scores_list, weights, strict=True):
        sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        for rank_idx, doc_id in enumerate(sorted_docs):
            rank = rank_idx + 1
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * (1.0 / (rrf_k + rank))
    return fused


def fuse_relative_score_fusion(
    query_scores_list: Sequence[Mapping[str, float]], weights: Sequence[float]
) -> Mapping[str, float]:
    """Fuse the query scores using Relative Score MinMax normalisation."""
    fused: dict[str, float] = {}
    for scores, weight in zip(query_scores_list, weights, strict=True):
        if not scores:
            continue

        doc_ids = list(scores.keys())
        raw_scores = np.array(list(scores.values()), dtype=float)

        min_s = np.min(raw_scores)
        max_s = np.max(raw_scores)

        if max_s == min_s:
            normalized = np.full_like(raw_scores, 0.5)
        else:
            normalized = (raw_scores - min_s) / (max_s - min_s)

        for doc_id, norm_score in zip(doc_ids, normalized):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * float(norm_score)

    return fused
