from __future__ import annotations

from .AngryTweetsClassification import (
    AngryTweetsClassification,
    AngryTweetsClassificationV2,
)
from .DanishPoliticalCommentsClassification import (
    DanishPoliticalCommentsClassification,
    DanishPoliticalCommentsClassificationV2,
)
from .DdiscoCohesionClassification import (
    DdiscoCohesionClassification,
    DdiscoCohesionClassificationV2,
)
from .DKHateClassification import DKHateClassification, DKHateClassificationV2
from .LccSentimentClassification import LccSentimentClassification

__all__ = [
    "AngryTweetsClassification",
    "AngryTweetsClassificationV2",
    "DKHateClassification",
    "DKHateClassificationV2",
    "DanishPoliticalCommentsClassification",
    "DanishPoliticalCommentsClassificationV2",
    "DdiscoCohesionClassification",
    "DdiscoCohesionClassificationV2",
    "LccSentimentClassification",
]
