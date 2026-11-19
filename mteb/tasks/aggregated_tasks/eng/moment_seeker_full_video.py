from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mteb.results import TaskResult
    from mteb.types import HFSubset, ScoresDict

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

# Queries per subtask (unanswerable queries included). Weighting by these pools the
# score over queries, which is how the paper computes its per-level and overall columns.
_N_QUERIES = {
    "MomentSeekerT2VGlobalLevelFullVideoRetrieval": 74,
    "MomentSeekerTI2VGlobalLevelFullVideoRetrieval": 156,
    "MomentSeekerTV2VGlobalLevelFullVideoRetrieval": 156,
    "MomentSeekerT2VEventLevelFullVideoRetrieval": 699,
    "MomentSeekerTI2VEventLevelFullVideoRetrieval": 88,
    "MomentSeekerTV2VEventLevelFullVideoRetrieval": 88,
    "MomentSeekerT2VObjectLevelFullVideoRetrieval": 227,
    "MomentSeekerTI2VObjectLevelFullVideoRetrieval": 156,
    "MomentSeekerTV2VObjectLevelFullVideoRetrieval": 156,
}


class _QueryWeightedAggregate(AbsTaskAggregate):
    """Average subtask scores weighted by query count instead of uniformly."""

    def task_results_to_scores(
        self, task_results: list[TaskResult]
    ) -> dict[str, Mapping[HFSubset, ScoresDict]]:
        scores = super().task_results_to_scores(task_results)
        by_name = {
            tr.task_name: tr
            for tr in task_results
            if tr.task_name in self.taskname_to_task
        }
        if set(by_name) != set(self.taskname_to_task):
            return scores  # missing subtasks: the parent already set the score to None
        names = list(self.taskname_to_task)
        for split in self.metadata.eval_splits:
            values = [
                by_name[n]._get_score_fast(languages=["eng"], splits=[split])
                for n in names
            ]
            weighted = float(np.average(values, weights=[_N_QUERIES[n] for n in names]))
            scores[split] = {
                "default": {
                    self.metadata.main_score: weighted,
                    "main_score": weighted,
                }
            }
        return scores


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


class MomentSeekerGlobalLevelFullVideoRetrieval(_QueryWeightedAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerGlobalLevelFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval, global-level moments (Causal Reasoning, Spatial Relation), averaged over the text, image+text and video+text "
            "query types, weighted by query count as in the paper."
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


class MomentSeekerEventLevelFullVideoRetrieval(_QueryWeightedAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerEventLevelFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval, event-level moments (Description Location, Action Recognition, Anomaly Detection), averaged over the text, image+text and video+text "
            "query types, weighted by query count as in the paper."
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


class MomentSeekerObjectLevelFullVideoRetrieval(_QueryWeightedAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerObjectLevelFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval, object-level moments (Object Recognition, Object Location, Attribute Recognition, OCR), averaged over the text, image+text and video+text "
            "query types, weighted by query count as in the paper."
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


class MomentSeekerFullVideoRetrieval(_QueryWeightedAggregate):
    metadata = AggregateTaskMetadata(
        name="MomentSeekerFullVideoRetrieval",
        description=(
            "MomentSeeker full-video retrieval: the average of the nine subtasks spanning three query types "
            "(text, image+text, video+text) and three moment levels (global, event, "
            "object), weighted by query count as in the paper's overall score."
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
