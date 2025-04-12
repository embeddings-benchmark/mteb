from __future__ import annotations

from .AbsTask import AbsTask
from .AbsTaskAnyClassification import AbsTaskAnyClassification
from .AbsTaskBitextMining import AbsTaskBitextMining
from .AbsTaskClustering import AbsTaskClustering
from .AbsTaskClusteringFast import AbsTaskClusteringFast
from .AbsTaskMultilabelClassification import AbsTaskMultilabelClassification
from .AbsTaskPairClassification import AbsTaskPairClassification
from .AbsTaskRetrieval import AbsTaskRetrieval
from .AbsTaskSTS import AbsTaskSTS
from .AbsTaskSummarization import AbsTaskSummarization
from .Image import (
    AbsTaskAny2AnyMultiChoice,
    AbsTaskAny2AnyRetrieval,
    AbsTaskImageClustering,
    AbsTaskImageMultilabelClassification,
    AbsTaskImageTextPairClassification,
    AbsTaskVisualSTS,
    AbsTaskZeroShotClassification,
)
from .TaskMetadata import TaskMetadata

__all__ = [
    "AbsTask",
    "AbsTaskBitextMining",
    "AbsTaskAnyClassification",
    "AbsTaskClustering",
    "AbsTaskClusteringFast",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskRetrieval",
    "AbsTaskSTS",
    "AbsTaskSummarization",
    "TaskMetadata",
    "AbsTaskAny2AnyMultiChoice",
    "AbsTaskAny2AnyRetrieval",
    "AbsTaskImageClustering",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskImageTextPairClassification",
    "AbsTaskVisualSTS",
    "AbsTaskZeroShotClassification",
]
