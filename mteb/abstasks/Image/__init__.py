from __future__ import annotations

from .AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from .AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from .AbsTaskImageMultilabelClassification import AbsTaskImageMultilabelClassification
from .AbsTaskImageTextPairClassification import AbsTaskImageTextPairClassification

__all__ = [
    "AbsTaskImageTextPairClassification",
    "AbsTaskImageMultilabelClassification",
    "AbsTaskAny2AnyRetrieval",
    "AbsTaskAny2AnyMultiChoice",
]
