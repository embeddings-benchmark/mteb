"""ViDoRe v3 reranking views using pinned first-stage predictions from the Hub."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.tasks.retrieval.multilingual.vidore3_bench_retrieval import (
    Vidore3ComputerScienceRetrievalv2,
    Vidore3EnergyRetrievalv2,
    Vidore3FinanceEnRetrievalv2,
    Vidore3FinanceFrRetrievalv2,
    Vidore3HrRetrievalv2,
    Vidore3IndustrialRetrievalv2,
    Vidore3PharmaceuticalsRetrievalv2,
    Vidore3PhysicsRetrievalv2,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import HFSubset, ScoresDict


def _reranking_metadata(parent: TaskMetadata) -> TaskMetadata:
    return parent.model_copy(
        deep=True,
        update={
            "name": parent.name.replace("Retrieval.v2", "Reranking"),
            "description": parent.description
            + " Rerank precomputed first-stage candidates.",
            "type": "Reranking",
            "prompt": {
                "query": "Retrieve images or text relevant to the user's query."
            },
            "adapted_from": [parent.name],
            "superseded_by": None,
            "is_beta": True,  # The reranking result format is an upstream proposal.
        },
    )


def _prediction_sources(retrieval_name: str) -> dict[str, FirstStagePredictionSource]:
    sources: tuple[tuple[str, str, Literal["text", "image", "text-image"]], ...] = (
        ("bm25-text", "mteb__baseline-bm25s", "text"),
        ("bge-text", "BAAI__bge-m3", "text"),
        ("qwen-text-image", "Qwen__Qwen3-VL-Embedding-2B", "text-image"),
    )
    return {
        name: FirstStagePredictionSource(
            repo_id="mteb/Vidore3RetrievalPredictions",
            filename=f"{model}/{representation}/{retrieval_name}_predictions.json",
            revision="3d6834bc0d3aded9de65eb2e431d875f654c96e8",
            document_representation=representation,
        )
        for name, model, representation in sources
    }


class _PreparedReranking(AbsTaskRetrieval):
    def evaluate(self, *args: Any, **kwargs: Any) -> Mapping[HFSubset, ScoresDict]:
        if self.reranking_configuration is None:
            raise ValueError(
                "Select candidates with convert_to_reranking(first_stage=..., top_k=50) "
                "before evaluating this reranking task."
            )
        return super().evaluate(*args, **kwargs)


class Vidore3ComputerScienceReranking(
    _PreparedReranking, Vidore3ComputerScienceRetrievalv2
):
    metadata = _reranking_metadata(Vidore3ComputerScienceRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(
        Vidore3ComputerScienceRetrievalv2.metadata.name
    )


class Vidore3EnergyReranking(_PreparedReranking, Vidore3EnergyRetrievalv2):
    metadata = _reranking_metadata(Vidore3EnergyRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(
        Vidore3EnergyRetrievalv2.metadata.name
    )


class Vidore3FinanceEnReranking(_PreparedReranking, Vidore3FinanceEnRetrievalv2):
    metadata = _reranking_metadata(Vidore3FinanceEnRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(
        Vidore3FinanceEnRetrievalv2.metadata.name
    )


class Vidore3FinanceFrReranking(_PreparedReranking, Vidore3FinanceFrRetrievalv2):
    metadata = _reranking_metadata(Vidore3FinanceFrRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(
        Vidore3FinanceFrRetrievalv2.metadata.name
    )


class Vidore3HrReranking(_PreparedReranking, Vidore3HrRetrievalv2):
    metadata = _reranking_metadata(Vidore3HrRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(Vidore3HrRetrievalv2.metadata.name)


class Vidore3IndustrialReranking(_PreparedReranking, Vidore3IndustrialRetrievalv2):
    metadata = _reranking_metadata(Vidore3IndustrialRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(
        Vidore3IndustrialRetrievalv2.metadata.name
    )


class Vidore3PharmaceuticalsReranking(
    _PreparedReranking, Vidore3PharmaceuticalsRetrievalv2
):
    metadata = _reranking_metadata(Vidore3PharmaceuticalsRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(
        Vidore3PharmaceuticalsRetrievalv2.metadata.name
    )


class Vidore3PhysicsReranking(_PreparedReranking, Vidore3PhysicsRetrievalv2):
    metadata = _reranking_metadata(Vidore3PhysicsRetrievalv2.metadata)
    first_stage_predictions = _prediction_sources(
        Vidore3PhysicsRetrievalv2.metadata.name
    )
