from __future__ import annotations

from .AbsTask import AbsTask
from .AbsTaskAnyClassification import AbsTaskAnyClassification
from .AbsTaskAnySTS import AbsTaskAnySTS
from .AbsTaskBitextMining import AbsTaskBitextMining
from .AbsTaskClustering import AbsTaskClustering
from .AbsTaskClusteringFast import AbsTaskClusteringFast
from .AbsTaskMultilabelClassification import AbsTaskMultilabelClassification
from .AbsTaskPairClassification import AbsTaskPairClassification
from .AbsTaskRetrieval import AbsTaskRetrieval
from .AbsTaskSummarization import AbsTaskSummarization
from .Image import (
    AbsTaskAny2AnyMultiChoice,
    AbsTaskAny2AnyRetrieval,
    AbsTaskImageClustering,
    AbsTaskImageMultilabelClassification,
    AbsTaskImageTextPairClassification,
    AbsTaskZeroShotClassification,
)

__all__ = [
    "AbsTask",
    "AbsTaskBitextMining",
    "AbsTaskAnyClassification",
    "AbsTaskClustering",
    "AbsTaskClusteringFast",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskRetrieval",
    "AbsTaskAnySTS",
    "AbsTaskSummarization",
    "AbsTaskAny2AnyMultiChoice",
    "AbsTaskAny2AnyRetrieval",
    "AbsTaskImageClustering",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskImageTextPairClassification",
    "AbsTaskZeroShotClassification",
]
