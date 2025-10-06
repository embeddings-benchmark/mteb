from __future__ import annotations

from .AbsTask import AbsTask
from .AbsTaskAnyClassification import AbsTaskAnyClassification
from .AbsTaskAnyClustering import AbsTaskAnyClustering
from .AbsTaskAnySTS import AbsTaskAnySTS
from .AbsTaskAnyZeroShotClassification import AbsTaskAnyZeroShotClassification
from .AbsTaskBitextMining import AbsTaskBitextMining
from .AbsTaskClusteringFast import AbsTaskClusteringFast
from .AbsTaskMultilabelClassification import AbsTaskMultilabelClassification
from .AbsTaskPairClassification import AbsTaskPairClassification
from .AbsTaskReranking import AbsTaskReranking
from .AbsTaskRetrieval import AbsTaskRetrieval
from .AbsTaskSummarization import AbsTaskSummarization
from .AbsTaskTextRegression import AbsTaskTextRegression
from .Image import (
    AbsTaskImageTextPairClassification,
)

__all__ = [
    "AbsTask",
    "AbsTaskAnyClassification",
    "AbsTaskAnyClustering",
    "AbsTaskAnySTS",
    "AbsTaskAnyZeroShotClassification",
    "AbsTaskBitextMining",
    "AbsTaskClusteringFast",
    "AbsTaskImageTextPairClassification",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskReranking",
    "AbsTaskRetrieval",
    "AbsTaskSummarization",
    "AbsTaskTextRegression",
]
