from __future__ import annotations

from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.retrieval import (
    MomentSeekerT2VEventLevelRetrieval,
    MomentSeekerT2VGlobalLevelRetrieval,
    MomentSeekerT2VObjectLevelRetrieval,
    MomentSeekerTI2VEventLevelRetrieval,
    MomentSeekerTI2VGlobalLevelRetrieval,
    MomentSeekerTI2VObjectLevelRetrieval,
    MomentSeekerTV2VEventLevelRetrieval,
    MomentSeekerTV2VGlobalLevelRetrieval,
    MomentSeekerTV2VObjectLevelRetrieval,
)

_BIBTEX = r"""
@misc{yuan2025momentseeker,
  archiveprefix = {arXiv},
  author = {Huaying Yuan and Jian Ni and Zheng Liu and Yueze Wang and Junjie Zhou and Zhengyang Liang and Bo Zhao and Zhao Cao and Zhicheng Dou and Ji-Rong Wen},
  eprint = {2502.12558},
  primaryclass = {cs.CV},
  title = {MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval},
  url = {https://arxiv.org/abs/2502.12558},
  year = {2025},
}
"""
_REFERENCE = "https://arxiv.org/abs/2502.12558"

_GLOBAL = [
    MomentSeekerTI2VGlobalLevelRetrieval(),
    MomentSeekerTV2VGlobalLevelRetrieval(),
    MomentSeekerT2VGlobalLevelRetrieval(),
]
_EVENT = [
    MomentSeekerTI2VEventLevelRetrieval(),
    MomentSeekerTV2VEventLevelRetrieval(),
    MomentSeekerT2VEventLevelRetrieval(),
]
_OBJECT = [
    MomentSeekerTI2VObjectLevelRetrieval(),
    MomentSeekerTV2VObjectLevelRetrieval(),
    MomentSeekerT2VObjectLevelRetrieval(),
]


class MomentSeekerGlobalLevelRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerGlobalLevelRetrieval",
        description="MomentSeeker full-video retrieval, global-level moments "
        "(Causal Reasoning, Spatial Relation), averaged over the image+text and "
        "video+text query directions.",
        reference=_REFERENCE,
        tasks=_GLOBAL,
        main_score="map_at_5",
        type="Retrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )


class MomentSeekerEventLevelRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerEventLevelRetrieval",
        description="MomentSeeker full-video retrieval, event-level moments "
        "(Description Location, Action Recognition, Anomaly Detection), averaged "
        "over the image+text and video+text query directions.",
        reference=_REFERENCE,
        tasks=_EVENT,
        main_score="map_at_5",
        type="Retrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )


class MomentSeekerObjectLevelRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerObjectLevelRetrieval",
        description="MomentSeeker full-video retrieval, object-level moments "
        "(Object Recognition, Object Location, Attribute Recognition, OCR), "
        "averaged over the image+text, video+text, and text-only query directions.",
        reference=_REFERENCE,
        tasks=_OBJECT,
        main_score="map_at_5",
        type="Retrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )


class MomentSeekerRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerRetrieval",
        description="MomentSeeker full-video moment retrieval: retrieve the "
        "complete source video that contains a query's answer moment. Averages "
        "the nine subtasks spanning three query directions (image+text, "
        "video+text, text-only) and three moment levels (global, event, object).",
        reference=_REFERENCE,
        tasks=_GLOBAL + _EVENT + _OBJECT,
        main_score="map_at_5",
        type="Retrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )
