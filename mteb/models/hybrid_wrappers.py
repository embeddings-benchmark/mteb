from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import MTEBModels
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )


class HybridSearch:
    """Hybrid search wrapper that combines multiple models using a specified fusion strategy."""

    def __init__(
        self,
        models: Sequence[MTEBModels],
        weights: list[float] | None = None,
        sub_model_top_k: int | None = None,
        fusion_strategy: Literal["rrf", "dbsf", "relative-score-fusion"]
        | Callable[[list[dict[str, float]], list[float]], dict[str, float]] = "rrf",
        rrf_k: int = 60,
    ) -> None:
        models = list(models)
        if len(models) < 2:
            raise ValueError("At least two models must be provided for hybrid search.")
        self.models = models

        self.wrapped_models: list[SearchProtocol] = []
        for model in models:
            if isinstance(model, EncoderProtocol) and not isinstance(
                model, SearchProtocol
            ):
                self.wrapped_models.append(SearchEncoderWrapper(model))
            elif isinstance(model, CrossEncoderProtocol):
                self.wrapped_models.append(SearchCrossEncoderWrapper(model))
            elif isinstance(model, SearchProtocol):
                self.wrapped_models.append(model)
            else:
                raise TypeError(
                    f"Expected a SearchProtocol, EncoderProtocol, or CrossEncoderProtocol, got {type(model)}"
                )

        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Length of weights must match the number of models.")
            self.weights = list(weights)

        if sub_model_top_k is not None:
            if sub_model_top_k <= 0:
                raise ValueError("sub_model_top_k must be greater than 0")
        self.sub_model_top_k = sub_model_top_k

        if not isinstance(rrf_k, int):
            raise TypeError("rrf_k must be an integer")
        if rrf_k < 0:
            raise ValueError("rrf_k must be greater than or equal to 0")
        self.rrf_k = rrf_k

        self.fusion_name: str
        self._fuse_fn: Callable[[list[dict[str, float]], list[float]], dict[str, float]]

        if isinstance(fusion_strategy, str):
            self.fusion_name = fusion_strategy
            if fusion_strategy == "rrf":
                self._fuse_fn = self._fuse_rrf
            elif fusion_strategy == "dbsf":
                self._fuse_fn = self._fuse_dbsf
            elif fusion_strategy in {"relative-score-fusion", "relative_score_fusion"}:
                self.fusion_name = "relative-score-fusion"
                self._fuse_fn = self._fuse_relative_score_fusion
            else:
                raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        elif callable(fusion_strategy):
            self.fusion_name = getattr(fusion_strategy, "__name__", "custom")
            self._fuse_fn = fusion_strategy
        else:
            raise TypeError("fusion_strategy must be a string or a callable")

        self.cross_encoder_models: list[SearchCrossEncoderWrapper] = []
        self.retriever_models: list[SearchProtocol] = []
        self.model_is_cross_encoder: list[bool] = []
        for model in self.wrapped_models:
            if isinstance(model, SearchCrossEncoderWrapper):
                self.cross_encoder_models.append(model)
                self.model_is_cross_encoder.append(True)
            else:
                self.retriever_models.append(model)
                self.model_is_cross_encoder.append(False)

        names = []
        for model in self.wrapped_models:
            meta = model.mteb_model_meta
            if meta and meta.name:
                names.append(meta.name.rsplit("/", 1)[-1])
            else:
                names.append("unknown")

        combined_name = f"hybrid-{self.fusion_name}/{'-'.join(names)}"
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
        for model in self.wrapped_models:
            model.index(
                corpus,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                num_proc=num_proc,
            )

    def search(  # noqa: PLR0914
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

        cross_encoder_models = self.cross_encoder_models
        retriever_models = self.retriever_models
        model_is_cross_encoder = self.model_is_cross_encoder

        effective_top_ranked = top_ranked

        if cross_encoder_models and effective_top_ranked is None:
            if not retriever_models:
                raise ValueError(
                    "CrossEncoder sub-models require top_ranked documents for reranking, "
                    "or at least one retriever sub-model in the hybrid wrapper to generate candidates."
                )

            logger.info("Running retriever sub-models...")
            retriever_results = []
            for model in tqdm(retriever_models, desc="Retriever sub-models"):
                res = model.search(
                    queries=queries,
                    top_k=sub_top_k,
                    task_metadata=task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    encode_kwargs=encode_kwargs,
                    top_ranked=None,
                    num_proc=num_proc,
                )
                retriever_results.append(res)

            generated_top_ranked: dict[str, list[str]] = {}
            for row in queries:
                qid = row["id"]
                candidate_rrf: dict[str, float] = {}
                for res in retriever_results:
                    if qid in res:
                        scores = res[qid]
                        sorted_docs = sorted(
                            scores.keys(), key=lambda d: scores[d], reverse=True
                        )
                        for rank_idx, doc_id in enumerate(sorted_docs):
                            rank = rank_idx + 1
                            candidate_rrf[doc_id] = candidate_rrf.get(
                                doc_id, 0.0
                            ) + 1.0 / (60 + rank)
                sorted_candidates = sorted(
                    candidate_rrf.keys(), key=lambda d: candidate_rrf[d], reverse=True
                )
                generated_top_ranked[qid] = sorted_candidates[:sub_top_k]

            effective_top_ranked = generated_top_ranked

            logger.info("Running cross-encoder sub-models...")
            cross_encoder_results = []
            for model in tqdm(cross_encoder_models, desc="Cross-encoder sub-models"):
                res = model.search(
                    queries=queries,
                    top_k=sub_top_k,
                    task_metadata=task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    encode_kwargs=encode_kwargs,
                    top_ranked=effective_top_ranked,
                    num_proc=num_proc,
                )
                cross_encoder_results.append(res)

            ret_iter = iter(retriever_results)
            ce_iter = iter(cross_encoder_results)
            all_results = [
                next(ce_iter) if is_ce else next(ret_iter)
                for is_ce in model_is_cross_encoder
            ]
        else:
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
                    top_ranked=effective_top_ranked,
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

    def fuse(self, query_scores_list: list[dict[str, float]]) -> dict[str, float]:
        """Fuse the query scores from multiple sub-models."""
        return self._fuse_fn(query_scores_list, self.weights)

    @staticmethod
    def _fuse_dbsf(
        query_scores_list: list[dict[str, float]], weights: list[float]
    ) -> dict[str, float]:
        """Fuse the query scores using Distribution-Based Score Fusion."""
        fused: dict[str, float] = {}
        for scores, weight in zip(query_scores_list, weights):
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

    def _fuse_rrf(
        self, query_scores_list: list[dict[str, float]], weights: list[float]
    ) -> dict[str, float]:
        """Fuse the query scores using Reciprocal Rank Fusion."""
        fused: dict[str, float] = {}
        for scores, weight in zip(query_scores_list, weights):
            sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
            for rank_idx, doc_id in enumerate(sorted_docs):
                rank = rank_idx + 1
                fused[doc_id] = fused.get(doc_id, 0.0) + weight * (
                    1.0 / (self.rrf_k + rank)
                )
        return fused

    @staticmethod
    def _fuse_relative_score_fusion(
        query_scores_list: list[dict[str, float]], weights: list[float]
    ) -> dict[str, float]:
        """Fuse the query scores using Relative Score MinMax normalisation."""
        fused: dict[str, float] = {}
        for scores, weight in zip(query_scores_list, weights):
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
