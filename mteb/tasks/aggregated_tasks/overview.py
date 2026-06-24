from __future__ import annotations

from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate

__all__ = [
    "FollowIRAggregate",
    "LongEmbedAggregate",
    "MIEBEngAggregate",
    "MTEBCodeV1Aggregate",
    "MTEBEngV2Aggregate",
    "MTEBEuropeV1Aggregate",
    "MTEBIndicV1Aggregate",
    "MTEBLawV1Aggregate",
    "MTEBMedicalV1Aggregate",
    "MTEBMultilingualV1Aggregate",
    "MTEBMultilingualV2Aggregate",
]


class MTEBEngV2Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(eng, v2)",
        description="English benchmark tasks for evaluating text embeddings.",
        reference=None,
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MTEBMultilingualV1Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(Multilingual, v1)",
        description="MTEB(Multilingual, v1) Aggregate Task",
        reference="https://arxiv.org/abs/2502.13595",
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MTEBMultilingualV2Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(Multilingual, v2)",
        description="MTEB(Multilingual, v2) Aggregate Task",
        reference="https://arxiv.org/abs/2502.13595",
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MTEBEuropeV1Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(Europe, v1)",
        description="MTEB(Europe, v1) Aggregate Task",
        reference=None,
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MTEBIndicV1Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(Indic, v1)",
        description="MTEB(Indic, v1) Aggregate Task",
        reference=None,
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MTEBCodeV1Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(Code, v1)",
        description="MTEB(Code, v1) Aggregate Task",
        reference=None,
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MTEBLawV1Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(Law, v1)",
        description="MTEB(Law, v1) Aggregate Task",
        reference=None,
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MTEBMedicalV1Aggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MTEB(Medical, v1)",
        description="MTEB(Medical, v1) Aggregate Task",
        reference=None,
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class FollowIRAggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="FollowIR",
        description="FollowIR Aggregate Task",
        reference="https://arxiv.org/abs/2403.15246",
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class LongEmbedAggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="LongEmbed",
        description="LongEmbed Aggregate Task",
        reference="https://arxiv.org/abs/2404.12096v2",
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )


class MIEBEngAggregate(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MIEB(eng)",
        description="MIEB(eng) Aggregate Task",
        reference="https://arxiv.org/abs/2504.10471",
        tasks=[],
        main_score="main_score",
        type="Retrieval",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        domains=["Academic"],
    )
