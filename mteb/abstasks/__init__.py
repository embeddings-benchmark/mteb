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
from .AbsTaskRetrieval import AbsTaskRetrieval
from .AbsTaskSummarization import AbsTaskSummarization
from .AbsTaskTextRegression import AbsTaskTextRegression
from .Image import (
    AbsTaskImageMultilabelClassification,
    AbsTaskImageTextPairClassification,
)

__all__ = [
    "AbsTask",
    "AbsTaskBitextMining",
    "AbsTaskAnyClassification",
    "AbsTaskAnyClustering",
    "AbsTaskClusteringFast",
    "AbsTaskMultilabelClassification",
    "AbsTaskPairClassification",
    "AbsTaskRetrieval",
    "AbsTaskAnySTS",
    "AbsTaskSummarization",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskImageTextPairClassification",
    "AbsTaskAnyZeroShotClassification",
    "AbsTaskTextRegression",
]
