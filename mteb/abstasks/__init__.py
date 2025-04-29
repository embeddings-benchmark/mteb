from __future__ import annotations

from .AbsAnyTaskSTS import AbsAnyTaskSTS
from .AbsTask import AbsTask
from .AbsTaskBitextMining import AbsTaskBitextMining
from .AbsTaskClassification import AbsTaskClassification
from .AbsTaskClustering import AbsTaskClustering
from .AbsTaskClusteringFast import AbsTaskClusteringFast
from .AbsTaskMultilabelClassification import AbsTaskMultilabelClassification
from .AbsTaskPairClassification import AbsTaskPairClassification
from .AbsTaskRetrieval import AbsTaskRetrieval
from .AbsTaskSummarization import AbsTaskSummarization
from .Image import (
    AbsTaskAny2AnyMultiChoice,
    AbsTaskAny2AnyRetrieval,
    AbsTaskImageClassification,
    AbsTaskImageClustering,
    AbsTaskImageMultilabelClassification,
    AbsTaskImageTextPairClassification,
    AbsTaskZeroShotClassification,
)
from .TaskMetadata import TaskMetadata

__all__ = [
    "AbsTask",
    "AbsTaskBitextMining",
    "AbsTaskClassification",
    "AbsTaskClustering",
    "AbsTaskClusteringFast",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskRetrieval",
    "AbsAnyTaskSTS",
    "AbsTaskSummarization",
    "TaskMetadata",
    "AbsTaskAny2AnyMultiChoice",
    "AbsTaskAny2AnyRetrieval",
    "AbsTaskImageClassification",
    "AbsTaskImageClustering",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskImageTextPairClassification",
    "AbsTaskZeroShotClassification",
]
