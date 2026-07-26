from __future__ import annotations

from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_BIBTEX_CITATION = r"""
@article{Zhao2026RethinkingRR,
  author = {Yilun Zhao and Jinbiao Wei and Tingyu Song and Siyue Zhang and Chen Zhao and Arman Cohan},
  journal = {arXiv preprint arXiv:2605.04018},
  title = {Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems},
  year = {2026},
}
"""


_REFERENCE = "https://huggingface.co/datasets/yale-nlp/Bright-Pro"
_DATE = ("2025-09-01", "2026-04-30")


class BrightProBiologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProBiologyRetrieval",
        dataset={
            "path": "mteb/BrightProBiologyRetrieval",
            "revision": "8d356ed8a3b65123b5ec78793fbb7e80010345db",
        },
        reference=_REFERENCE,
        description=(
            "Reasoning-intensive retrieval over Biology StackExchange posts. Each query is "
            "paired with multi-aspect gold evidence: a long-form reference answer "
            "whose cited passages collectively cover several reasoning aspects, so "
            "retrievers are scored on surfacing that aspect-diverse evidence set "
            "rather than a single passage. Was developed as part of the BRIGHT-Pro "
            "benchmark for agentic search settings."
        ),
        type="Retrieval",
        prompt={
            "query": "Given a Biology post, retrieve relevant passages that help answer the post"
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )


class BrightProEarthScienceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProEarthScienceRetrieval",
        dataset={
            "path": "mteb/BrightProEarthScienceRetrieval",
            "revision": "66a901dddee938a175f87d3dededeae506c13af8",
        },
        reference=_REFERENCE,
        description=(
            "Reasoning-intensive retrieval over Earth Science StackExchange posts. Each query is "
            "paired with multi-aspect gold evidence: a long-form reference answer "
            "whose cited passages collectively cover several reasoning aspects, so "
            "retrievers are scored on surfacing that aspect-diverse evidence set "
            "rather than a single passage. Was developed as part of the BRIGHT-Pro "
            "benchmark for agentic search settings."
        ),
        type="Retrieval",
        prompt={
            "query": "Given an Earth Science post, retrieve relevant passages that help answer the post"
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )


class BrightProEconomicsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProEconomicsRetrieval",
        dataset={
            "path": "mteb/BrightProEconomicsRetrieval",
            "revision": "e7ce2bbbfc5fab8c0048dadc83abf2b5dd98bc05",
        },
        reference=_REFERENCE,
        description=(
            "Reasoning-intensive retrieval over Economics StackExchange posts. Each query is "
            "paired with multi-aspect gold evidence: a long-form reference answer "
            "whose cited passages collectively cover several reasoning aspects, so "
            "retrievers are scored on surfacing that aspect-diverse evidence set "
            "rather than a single passage. Was developed as part of the BRIGHT-Pro "
            "benchmark for agentic search settings."
        ),
        type="Retrieval",
        prompt={
            "query": "Given an Economics post, retrieve relevant passages that help answer the post"
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )


class BrightProPsychologyRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProPsychologyRetrieval",
        dataset={
            "path": "mteb/BrightProPsychologyRetrieval",
            "revision": "18b7681648b7064e0a0f46843f3ab677d7e6a2c4",
        },
        reference=_REFERENCE,
        description=(
            "Reasoning-intensive retrieval over Psychology StackExchange posts. Each query is "
            "paired with multi-aspect gold evidence: a long-form reference answer "
            "whose cited passages collectively cover several reasoning aspects, so "
            "retrievers are scored on surfacing that aspect-diverse evidence set "
            "rather than a single passage. Was developed as part of the BRIGHT-Pro "
            "benchmark for agentic search settings."
        ),
        type="Retrieval",
        prompt={
            "query": "Given a Psychology post, retrieve relevant passages that help answer the post"
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )


class BrightProRoboticsRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProRoboticsRetrieval",
        dataset={
            "path": "mteb/BrightProRoboticsRetrieval",
            "revision": "25648862e812eef5fbeb51de859ce3127dbc055e",
        },
        reference=_REFERENCE,
        description=(
            "Reasoning-intensive retrieval over Robotics StackExchange posts. Each query is "
            "paired with multi-aspect gold evidence: a long-form reference answer "
            "whose cited passages collectively cover several reasoning aspects, so "
            "retrievers are scored on surfacing that aspect-diverse evidence set "
            "rather than a single passage. Was developed as part of the BRIGHT-Pro "
            "benchmark for agentic search settings."
        ),
        type="Retrieval",
        prompt={
            "query": "Given a Robotics post, retrieve relevant passages that help answer the post"
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )


class BrightProStackoverflowRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProStackoverflowRetrieval",
        dataset={
            "path": "mteb/BrightProStackoverflowRetrieval",
            "revision": "cfadbcdc4d1ee17e91751218ae0ebfa001c8feea",
        },
        reference=_REFERENCE,
        description=(
            "Reasoning-intensive retrieval over Stack Overflow posts. Each query is "
            "paired with multi-aspect gold evidence: a long-form reference answer "
            "whose cited passages collectively cover several reasoning aspects, so "
            "retrievers are scored on surfacing that aspect-diverse evidence set "
            "rather than a single passage. Was developed as part of the BRIGHT-Pro "
            "benchmark for agentic search settings."
        ),
        type="Retrieval",
        prompt={
            "query": "Given a Stack Overflow post, retrieve relevant passages that help answer the post"
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )


class BrightProSustainableLivingRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="BrightProSustainableLivingRetrieval",
        dataset={
            "path": "mteb/BrightProSustainableLivingRetrieval",
            "revision": "3f10f4998714e03b9fda8064360a768b0a3fbb4f",
        },
        reference=_REFERENCE,
        description=(
            "Reasoning-intensive retrieval over Sustainable Living StackExchange posts. Each query is "
            "paired with multi-aspect gold evidence: a long-form reference answer "
            "whose cited passages collectively cover several reasoning aspects, so "
            "retrievers are scored on surfacing that aspect-diverse evidence set "
            "rather than a single passage. Was developed as part of the BRIGHT-Pro "
            "benchmark for agentic search settings."
        ),
        type="Retrieval",
        prompt={
            "query": "Given a Sustainable Living post, retrieve relevant passages that help answer the post"
        },
        category="t2t",
        eval_splits=["standard"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=_DATE,
        domains=["Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        modalities=["text"],
        bibtex_citation=_BIBTEX_CITATION,
    )
