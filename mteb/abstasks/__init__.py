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

# todo todo todo
from .Image.AbsTaskAny2AnyMultiChoice import *
from .Image.AbsTaskAny2AnyRetrieval import *
from .Image.AbsTaskAny2TextMultipleChoice import *
from .Image.AbsTaskImageClassification import *
from .Image.AbsTaskImageClustering import *
from .Image.AbsTaskImageMultilabelClassification import *
from .Image.AbsTaskImageTextPairClassification import *
from .Image.AbsTaskVisualSTS import *
from .Image.AbsTaskZeroshotClassification import *
from .MultilingualTask import *
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
]
