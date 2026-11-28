from __future__ import annotations

from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.retrieval import (
    MomentSeekerT2VEventLevelFullVideoRetrieval,
    MomentSeekerT2VGlobalLevelFullVideoRetrieval,
    MomentSeekerT2VObjectLevelFullVideoRetrieval,
    MomentSeekerTI2VEventLevelFullVideoRetrieval,
    MomentSeekerTI2VGlobalLevelFullVideoRetrieval,
    MomentSeekerTI2VObjectLevelFullVideoRetrieval,
    MomentSeekerTV2VEventLevelFullVideoRetrieval,
    MomentSeekerTV2VGlobalLevelFullVideoRetrieval,
    MomentSeekerTV2VObjectLevelFullVideoRetrieval,
)

_REFERENCE = "https://arxiv.org/abs/2502.12558"
_BIBTEX = r"""
@inproceedings{yuan2025momentseeker,
  author = {Yuan, Huaying and Ni, Jian and Liu, Zheng and Wang, Yueze and Zhou, Junjie and Liang, Zhengyang and Zhao, Bo and Cao, Zhao and Wen, Ji-Rong and Dou, Zhicheng},
  booktitle = {Advances in Neural Information Processing Systems 38},
  collection = {NeurIPS 2025},
  doi = {10.52202/085713-0934},
  pages = {31491–31517},
  publisher = {Neural Information Processing Systems Foundation, Inc. (NeurIPS)},
  series = {NeurIPS 2025},
  title = {MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval},
  url = {https://doi.org/10.52202/085713-0934},
  year = {2025},
}
"""

_FULL_GLOBAL = [
    MomentSeekerT2VGlobalLevelFullVideoRetrieval(),
    MomentSeekerTI2VGlobalLevelFullVideoRetrieval(),
    MomentSeekerTV2VGlobalLevelFullVideoRetrieval(),
]

_FULL_EVENT = [
    MomentSeekerT2VEventLevelFullVideoRetrieval(),
    MomentSeekerTI2VEventLevelFullVideoRetrieval(),
    MomentSeekerTV2VEventLevelFullVideoRetrieval(),
]

_FULL_OBJECT = [
    MomentSeekerT2VObjectLevelFullVideoRetrieval(),
    MomentSeekerTI2VObjectLevelFullVideoRetrieval(),
    MomentSeekerTV2VObjectLevelFullVideoRetrieval(),
]


class MomentSeekerGlobalLevelFullVideoRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerGlobalLevelFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval, global-level moments (Causal Reasoning, Spatial Relation), averaged over the text, image+text and video+text query types."
        ),
        reference=_REFERENCE,
        tasks=_FULL_GLOBAL,
        main_score="map_at_5",
        # subtasks mix t2v, it2v and vt2v; every query has a text question and a video target
        category="t2v",
        type="Any2AnyRetrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )


class MomentSeekerEventLevelFullVideoRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerEventLevelFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval, event-level moments (Description Location, Action Recognition, Anomaly Detection), averaged over the text, image+text and video+text query types."
        ),
        reference=_REFERENCE,
        tasks=_FULL_EVENT,
        main_score="map_at_5",
        # subtasks mix t2v, it2v and vt2v; every query has a text question and a video target
        category="t2v",
        type="Any2AnyRetrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )


class MomentSeekerObjectLevelFullVideoRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerObjectLevelFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval, object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR), averaged over the text, image+text and video+text query types."
        ),
        reference=_REFERENCE,
        tasks=_FULL_OBJECT,
        main_score="map_at_5",
        # subtasks mix t2v, it2v and vt2v; every query has a text question and a video target
        category="t2v",
        type="Any2AnyRetrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )


class MomentSeekerFullVideoRetrieval(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval: the average of the nine subtasks spanning three query types "
            "(text, image+text, video+text) and three moment levels (global, event, "
            "object)."
        ),
        reference=_REFERENCE,
        tasks=_FULL_GLOBAL + _FULL_EVENT + _FULL_OBJECT,
        main_score="map_at_5",
        # subtasks mix t2v, it2v and vt2v; every query has a text question and a video target
        category="t2v",
        type="Any2AnyRetrieval",
        eval_splits=["test"],
        bibtex_citation=_BIBTEX,
    )
