from __future__ import annotations

from .AbsTask import AbsTask
from .AbsTaskBitextMining import AbsTaskBitextMining
from .AbsTaskClassification import AbsTaskClassification
from .AbsTaskClustering import AbsTaskClustering
from .AbsTaskClusteringFast import AbsTaskClusteringFast
from .AbsTaskMultilabelClassification import AbsTaskMultilabelClassification
from .AbsTaskPairClassification import AbsTaskPairClassification
from .AbsTaskRetrieval import AbsTaskRetrieval
from .AbsTaskSpeedTask import AbsTaskSpeedTask
from .AbsTaskSTS import AbsTaskSTS
from .AbsTaskSummarization import AbsTaskSummarization
from .Image import (
    AbsTaskAny2AnyMultiChoice,
    AbsTaskAny2AnyRetrieval,
    AbsTaskAny2TextMultipleChoice,
    AbsTaskImageClassification,
    AbsTaskImageClustering,
    AbsTaskImageMultilabelClassification,
    AbsTaskImageTextPairClassification,
    AbsTaskVisualSTS,
    AbsTaskZeroshotClassification,
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
    "AbsTaskSpeedTask",
    "AbsTaskSTS",
    "AbsTaskSummarization",
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
