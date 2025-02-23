from __future__ import annotations

from .AbsTaskAny2TextMultipleChoice import AbsTaskAny2TextMultipleChoice
from .AbsTaskImageClassification import AbsTaskImageClassification
from .AbsTaskImageClustering import AbsTaskImageClustering
from .AbsTaskImageMultilabelClassification import AbsTaskImageMultilabelClassification
from .AbsTaskImageTextPairClassification import AbsTaskImageTextPairClassification
from .AbsTaskVisualSTS import AbsTaskVisualSTS
from .AbsTaskZeroshotClassification import AbsTaskZeroshotClassification

__all__ = [
    "AbsTaskZeroshotClassification",
    "AbsTaskImageClassification",
    "AbsTaskImageClustering",
    "AbsTaskImageTextPairClassification",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskVisualSTS",
    "AbsTaskAny2TextMultipleChoice",
]
