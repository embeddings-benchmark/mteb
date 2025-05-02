from __future__ import annotations

from .GeoreviewClassification import GeoreviewClassification
from .HeadlineClassification import HeadlineClassification
from .InappropriatenessClassification import (
    InappropriatenessClassification,
    InappropriatenessClassificationv2,
)
from .KinopoiskClassification import KinopoiskClassification
from .ru_nlu_intent_classification import RuNLUIntentClassification
from .ru_toixic_classification_okmlcup import RuToxicOKMLCUPClassification
from .RuReviewsClassification import RuReviewsClassification
from .RuSciBenchGRNTIClassification import RuSciBenchGRNTIClassification
from .RuSciBenchOECDClassification import RuSciBenchOECDClassification
from .senti_ru_eval import SentiRuEval2016Classification

__all__ = [
    "RuNLUIntentClassification",
    "RuToxicOKMLCUPClassification",
    "KinopoiskClassification",
    "HeadlineClassification",
    "InappropriatenessClassification",
    "InappropriatenessClassificationv2",
    "SentiRuEval2016Classification",
    "RuSciBenchGRNTIClassification",
    "RuSciBenchOECDClassification",
    "RuReviewsClassification",
    "GeoreviewClassification",
]
