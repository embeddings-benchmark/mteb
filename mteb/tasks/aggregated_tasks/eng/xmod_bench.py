from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.tasks.retrieval import (
    XModBenchAT2IRetrieval,
    XModBenchAT2TRetrieval,
    XModBenchAT2VRetrieval,
    XModBenchIT2ARetrieval,
    XModBenchIT2TRetrieval,
    XModBenchT2ARetrieval,
    XModBenchT2IRetrieval,
    XModBenchT2VRetrieval,
    XModBenchVT2ARetrieval,
    XModBenchVT2TRetrieval,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mteb.results.task_result import TaskResult
    from mteb.types import HFSubset, ScoresDict

logger = logging.getLogger(__name__)

_TASKS = [
    XModBenchAT2TRetrieval(),
    XModBenchAT2IRetrieval(),
    XModBenchAT2VRetrieval(),
    XModBenchT2ARetrieval(),
    XModBenchT2IRetrieval(),
    XModBenchT2VRetrieval(),
    XModBenchIT2ARetrieval(),
    XModBenchVT2ARetrieval(),
    XModBenchIT2TRetrieval(),
    XModBenchVT2TRetrieval(),
]

# Image and video are split internally because MTEB modality metadata is
# task-level. Weighting by retained query count reports micro accuracy over the
# 5,981 decodable questions in this MTEB adaptation.
_TASK_WEIGHTS = {
    "XModBenchAT2TRetrieval": 1_000,
    "XModBenchAT2IRetrieval": 617,
    "XModBenchAT2VRetrieval": 373,
    "XModBenchT2ARetrieval": 1_000,
    "XModBenchT2IRetrieval": 617,
    "XModBenchT2VRetrieval": 379,
    "XModBenchIT2ARetrieval": 617,
    "XModBenchVT2ARetrieval": 382,
    "XModBenchIT2TRetrieval": 617,
    "XModBenchVT2TRetrieval": 379,
}


class XModBench(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="XModBench",
        description=(
            "XModBench-Lite is a unified cross-modal multiple-choice question "
            "answering benchmark spanning text, audio, images, and video. It "
            "retains 5,981 of 6,000 questions across six official "
            "Audio/Vision/Text directions, excluding 19 questions that "
            "reference unusable source MP4s. Each question ranks its four "
            "original answer candidates; the aggregate reports query-weighted "
            "accuracy across ten concrete MTEB modality tasks."
        ),
        reference="https://arxiv.org/abs/2510.15148",
        tasks=_TASKS,
        main_score="accuracy",
        type="Any2AnyRetrieval",
        modalities=["audio", "image", "text", "video"],
        eval_splits=["test"],
        is_public=True,
        bibtex_citation=r"""
@inproceedings{wang2026xmodbench,
  author = {Wang, Xingrui and Liu, Jiang and Huang, Chao and Yu, Xiaodong and Wang, Ze and Sun, Ximeng and Wu, Jialian and Yuille, Alan and Barsoum, Emad and Liu, Zicheng},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  title = {XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models},
  url = {https://arxiv.org/abs/2510.15148},
  year = {2026},
}
""",
    )

    def task_results_to_scores(
        self, task_results: list[TaskResult]
    ) -> dict[str, Mapping[HFSubset, ScoresDict]]:
        results_by_name = {
            result.task_name: result
            for result in task_results
            if result.task_name in self.taskname_to_task
        }
        missing = set(self.taskname_to_task) - set(results_by_name)

        scores: dict[str, Mapping[HFSubset, ScoresDict]] = {}
        for split in self.metadata.eval_splits:
            if missing:
                logger.info(
                    "Missing task results for required XModBench tasks: %s. "
                    "Setting aggregate accuracy to None.",
                    sorted(missing),
                )
                accuracy = None
            else:
                weighted_sum = 0.0
                total_weight = 0
                for task_name, task in self.taskname_to_task.items():
                    weight = _TASK_WEIGHTS[task_name]
                    direction = task.metadata.hf_subsets[0]
                    task_score = results_by_name[task_name]._get_score_fast(
                        splits=[split], subsets=[direction]
                    )
                    weighted_sum += weight * task_score
                    total_weight += weight
                accuracy = weighted_sum / total_weight

            scores[split] = {
                "default": {
                    "accuracy": accuracy,
                    "main_score": accuracy,
                }
            }
        return scores


__all__ = ["XModBench"]
