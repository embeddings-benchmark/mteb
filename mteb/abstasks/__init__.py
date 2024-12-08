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
from .MultilingualTask import MultilingualTask

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
    "MultilingualTask",
]
