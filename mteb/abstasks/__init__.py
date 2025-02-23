from __future__ import annotations

from .abs_task import AbsTask
from .any_modality import (
    AbsTaskAny2AnyMultiChoice,
    AbsTaskAny2AnyRetrieval,
)
from .image import (
    AbsTaskAny2TextMultipleChoice,
    AbsTaskImageClassification,
    AbsTaskImageClustering,
    AbsTaskImageMultilabelClassification,
    AbsTaskImageTextPairClassification,
    AbsTaskVisualSTS,
    AbsTaskZeroshotClassification,
)
from .task_metadata import TaskMetadata
from .text import (
    AbsTextBitextMining,
    AbsTextClassification,
    AbsTextClustering,
    AbsTextClusteringFast,
    AbsTextMultilabelClassification,
    AbsTextPairClassification,
    AbsTextReranking,
    AbsTextRetrieval,
    AbsTextSpeedTask,
    AbsTextSTS,
    AbsTextSummarization,
)

__all__ = [
    "AbsTask",
    "AbsTextBitextMining",
    "AbsTextClassification",
    "AbsTextClustering",
    "AbsTextClusteringFast",
    "AbsTextMultilabelClassification",
    "AbsTextPairClassification",
    "AbsTextReranking",
    "AbsTextRetrieval",
    "AbsTextSpeedTask",
    "AbsTextSTS",
    "AbsTextSummarization",
    "TaskMetadata",
    "AbsTaskAny2AnyMultiChoice",
    "AbsTaskAny2AnyRetrieval",
    "AbsTaskAny2TextMultipleChoice",
    "AbsTaskImageClassification",
    "AbsTaskImageClustering",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskImageTextPairClassification",
    "AbsTaskVisualSTS",
    "AbsTaskZeroshotClassification",
]
