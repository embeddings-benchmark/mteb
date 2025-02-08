from __future__ import annotations

from .AbsTask import AbsTask
from .AbsTaskBitextMining import AbsTaskBitextMining
from .AbsTaskClassification import AbsTaskClassification
from .AbsTaskClustering import AbsTaskClustering
from .AbsTaskClusteringFast import AbsTaskClusteringFast
from .AbsTaskMultilabelClassification import AbsTaskMultilabelClassification
from .AbsTaskPairClassification import AbsTaskPairClassification
from .AbsTaskReranking import AbsTaskReranking
from .AbsTaskRetrieval import AbsTaskRetrieval
from .AbsTaskSpeedTask import AbsTaskSpeedTask
from .AbsTaskSTS import AbsTaskSTS
from .AbsTaskSummarization import AbsTaskSummarization
from .Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from .Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from .Image.AbsTaskAny2TextMultipleChoice import AbsTaskAny2TextMultipleChoice
from .Image.AbsTaskImageClassification import AbsTaskImageClassification
from .Image.AbsTaskImageClustering import AbsTaskImageClustering
from .Image.AbsTaskImageMultilabelClassification import (
    AbsTaskImageMultilabelClassification,
)
from .Image.AbsTaskImageTextPairClassification import AbsTaskImageTextPairClassification
from .Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from .Image.AbsTaskZeroshotClassification import AbsTaskZeroshotClassification
from .TaskMetadata import TaskMetadata

__all__ = [
    "AbsTask",
    "AbsTaskBitextMining",
    "AbsTaskClassification",
    "AbsTaskClustering",
    "AbsTaskClusteringFast",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskReranking",
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
