"""Fixed top-50 first stages on shared ViDoRe corpora, exposed as task subsets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.abstasks.first_stage_predictions import FirstStagePredictionSource
from mteb.abstasks.reranking_from_predictions import AbsTaskRerankingFromPredictions
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
    from mteb.abstasks.retrieval import AbsTaskRetrieval
    from mteb.abstasks.task_metadata import TaskMetadata

_FIRST_STAGES = {
    "bm25-text": "mteb__baseline-bm25s/text",
    "bge-text": "BAAI__bge-m3/text",
    "qwen-text-image": "Qwen__Qwen3-VL-Embedding-2B/text-image",
}


def _sources(task: type[AbsTaskRetrieval]) -> dict[str, FirstStagePredictionSource]:
    return {
        name: FirstStagePredictionSource(
            repo_id="mteb/Vidore3RetrievalPredictions",
            filename=f"{directory}/{task.metadata.name}_predictions.json",
            revision="3d6834bc0d3aded9de65eb2e431d875f654c96e8",
        )
        for name, directory in _FIRST_STAGES.items()
    }


def _metadata(task: type[AbsTaskRetrieval], top_k: int) -> TaskMetadata:
    return task.metadata.model_copy(
        update={
            "name": task.metadata.name.replace("Retrieval.v2", "Reranking"),
            "type": "Reranking",
            "description": task.metadata.description
            + f" Rerank the top-{top_k} candidates from BM25, BGE-M3 and Qwen3-VL-Embedding-2B,"
            " evaluated separately for each retriever and language.",
            "eval_langs": {
                f"{source}__top{top_k}__{subset}": languages
                for source in _FIRST_STAGES
                for subset, languages in task.metadata.hf_subsets_to_langscripts.items()
            },
            "prompt": {
                "query": "Retrieve images or text relevant to the user's query."
            },
            "adapted_from": [task.metadata.name],
            "is_beta": True,
        }
    )


class Vidore3ComputerScienceReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3ComputerScienceRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)


class Vidore3EnergyReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3EnergyRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)


class Vidore3FinanceEnReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3FinanceEnRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)


class Vidore3FinanceFrReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3FinanceFrRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)


class Vidore3HrReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3HrRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)


class Vidore3IndustrialReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3IndustrialRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)


class Vidore3PharmaceuticalsReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3PharmaceuticalsRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)


class Vidore3PhysicsReranking(AbsTaskRerankingFromPredictions):
    retrieval_task = Vidore3PhysicsRetrievalv2
    first_stage_predictions = _sources(retrieval_task)
    top_k = 50
    metadata = _metadata(retrieval_task, top_k)
