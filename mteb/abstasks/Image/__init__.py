from __future__ import annotations

from .AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from .AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from .AbsTaskAny2TextMultipleChoice import AbsTaskAny2TextMultipleChoice
from .AbsTaskImageClassification import AbsTaskImageClassification
from .AbsTaskImageClustering import AbsTaskImageClustering
from .AbsTaskImageMultilabelClassification import AbsTaskImageMultilabelClassification
from .AbsTaskImageTextPairClassification import AbsTaskImageTextPairClassification
from .AbsTaskVisualSTS import AbsTaskVisualSTS
from .AbsTaskZeroshotClassification import AbsTaskZeroshotClassification

__all__ = [
    "AbsTaskZeroshotClassification",
    "AbsTaskVisualSTS",
    "AbsTaskImageTextPairClassification",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskImageClustering",
    "AbsTaskImageClassification",
    "AbsTaskAny2TextMultipleChoice",
    "AbsTaskAny2AnyRetrieval",
    "AbsTaskAny2AnyMultiChoice",
]
