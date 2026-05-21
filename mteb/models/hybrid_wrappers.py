from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

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
    from collections.abc import Sequence

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import MTEBModels
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )


class BaseHybridSearch(ABC):
    """Base class for hybrid search wrappers implementing the SearchProtocol."""

    fusion_name: str | None = None

    def __init__(
        self,
        models: Sequence[MTEBModels],
        weights: list[float] | None = None,
        sub_model_top_k: int | None = None,
    ) -> None:
        self.models = list(models)
        if len(self.models) < 2:
            raise ValueError("At least two models must be provided for hybrid search.")

        self.wrapped_models: list[SearchProtocol] = []
        for model in self.models:
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
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            if len(weights) != len(self.models):
                raise ValueError("Length of weights must match the number of models.")
            self.weights = list(weights)

        if sub_model_top_k is not None:
            if sub_model_top_k <= 0:
                raise ValueError("sub_model_top_k must be greater than 0")
        self.sub_model_top_k = sub_model_top_k

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

        cross_encoder_models = []
        retriever_models = []
        model_is_cross_encoder = []
        for model in self.wrapped_models:
            if isinstance(model, SearchCrossEncoderWrapper):
                cross_encoder_models.append(model)
                model_is_cross_encoder.append(True)
            else:
                retriever_models.append(model)
                model_is_cross_encoder.append(False)

        effective_top_ranked = top_ranked

        if cross_encoder_models and effective_top_ranked is None:
            if not retriever_models:
                raise ValueError(
                    "CrossEncoder sub-models require top_ranked documents for reranking, "
                    "or at least one retriever sub-model in the hybrid wrapper to generate candidates."
                )

            retriever_results = []
            for model in retriever_models:
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
                candidates = set()
                for res in retriever_results:
                    if qid in res:
                        candidates.update(res[qid].keys())
                generated_top_ranked[qid] = list(candidates)

            effective_top_ranked = generated_top_ranked

            cross_encoder_results = []
            for model in cross_encoder_models:
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
            all_results = []
            for model in self.wrapped_models:
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

    @abstractmethod
    def fuse(self, query_scores_list: list[dict[str, float]]) -> dict[str, float]:
        """Fuse the query scores from multiple sub-models."""
        raise NotImplementedError("Subclasses must implement the fuse method.")

    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Generate combined ModelMeta for the hybrid model."""
        names = []
        model_types = set()
        for model in self.wrapped_models:
            meta = getattr(model, "mteb_model_meta", None)
            if meta and meta.name:
                names.append(meta.name.split("/")[-1])
            else:
                names.append("unknown")

            if meta and getattr(meta, "model_type", None):
                for m_type in meta.model_type:
                    model_types.add(m_type)

        if self.fusion_name:
            fusion_name = self.fusion_name
        else:
            class_name = self.__class__.__name__
            fusion_name = class_name
            for suffix in ["HybridSearch", "Search"]:
                if fusion_name.endswith(suffix):
                    fusion_name = fusion_name[: -len(suffix)]
            fusion_name = fusion_name.lower()

        combined_name = f"hybrid-{fusion_name}/{'-'.join(names)}"
        final_model_types = sorted(model_types) if model_types else ["dense"]

        return ModelMeta.create_empty(
            overwrites={
                "name": combined_name,
                "model_type": final_model_types,
            }
        )


class DBSFHybridSearch(BaseHybridSearch):
    """Distribution-Based Score Fusion (DBSF) hybrid search wrapper."""

    fusion_name = "dbsf"

    def fuse(self, query_scores_list: list[dict[str, float]]) -> dict[str, float]:
        """Fuse the query scores using Distribution-Based Score Fusion. (https://arxiv.org/html/2410.20878v1)"""
        fused: dict[str, float] = {}
        for scores, weight in zip(query_scores_list, self.weights):
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


class RRFHybridSearch(BaseHybridSearch):
    """Reciprocal Rank Fusion (RRF) hybrid search wrapper."""

    fusion_name = "rrf"

    def __init__(
        self,
        models: Sequence[MTEBModels],
        weights: list[float] | None = None,
        sub_model_top_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(models, weights=weights, sub_model_top_k=sub_model_top_k)
        if not isinstance(rrf_k, int):
            raise TypeError("rrf_k must be an integer")
        if rrf_k < 0:
            raise ValueError("rrf_k must be greater than or equal to 0")
        self.rrf_k = rrf_k

    def fuse(self, query_scores_list: list[dict[str, float]]) -> dict[str, float]:
        """Fuse the query scores using Reciprocal Rank Fusion."""
        fused: dict[str, float] = {}
        for scores, weight in zip(query_scores_list, self.weights):
            sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
            for rank_idx, doc_id in enumerate(sorted_docs):
                rank = rank_idx + 1
                fused[doc_id] = fused.get(doc_id, 0.0) + weight * (
                    1.0 / (self.rrf_k + rank)
                )
        return fused


class RelativeScoreFusionHybridSearch(BaseHybridSearch):
    """Relative Score Fusion (MinMax) hybrid search wrapper."""

    fusion_name = "relative-score-fusion"

    def fuse(self, query_scores_list: list[dict[str, float]]) -> dict[str, float]:
        """Fuse the query scores using Relative Score MinMax normalisation."""
        fused: dict[str, float] = {}
        for scores, weight in zip(query_scores_list, self.weights):
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
