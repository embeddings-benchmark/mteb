from __future__ import annotations

from .kor import KorHateSpeechMLClassification
from .mlt import MalteseNewsClassification
from .multilingual import MultiEURLEXMultilabelClassification
from .por import BrazilianToxicTweetsClassification
from .rus import CEDRClassification, SensitiveTopicsClassification

__all__ = [
    "BrazilianToxicTweetsClassification",
    "MalteseNewsClassification",
    "KorHateSpeechMLClassification",
    "MultiEURLEXMultilabelClassification",
    "SensitiveTopicsClassification",
    "CEDRClassification",
]
