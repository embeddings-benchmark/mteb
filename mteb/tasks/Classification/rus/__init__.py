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
    "GeoreviewClassification",
    "HeadlineClassification",
    "InappropriatenessClassification",
    "InappropriatenessClassificationv2",
    "KinopoiskClassification",
    "RuNLUIntentClassification",
    "RuReviewsClassification",
    "RuSciBenchGRNTIClassification",
    "RuSciBenchOECDClassification",
    "RuToxicOKMLCUPClassification",
    "SentiRuEval2016Classification",
]
