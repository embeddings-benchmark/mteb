from __future__ import annotations

from .AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from .AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from .AbsTaskImageClassification import AbsTaskImageClassification
from .AbsTaskImageClustering import AbsTaskImageClustering
from .AbsTaskImageMultilabelClassification import AbsTaskImageMultilabelClassification
from .AbsTaskImageTextPairClassification import AbsTaskImageTextPairClassification
from .AbsTaskZeroShotClassification import AbsTaskZeroShotClassification

__all__ = [
    "AbsTaskZeroShotClassification",
    "AbsTaskImageTextPairClassification",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskImageClustering",
    "AbsTaskImageClassification",
    "AbsTaskAny2AnyRetrieval",
    "AbsTaskAny2AnyMultiChoice",
]
