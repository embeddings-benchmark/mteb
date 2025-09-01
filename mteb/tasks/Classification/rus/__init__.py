from __future__ import annotations

from .GeoreviewClassification import GeoreviewClassification, GeoreviewClassificationV2
from .HeadlineClassification import HeadlineClassification, HeadlineClassificationV2
from .InappropriatenessClassification import (
    InappropriatenessClassification,
    InappropriatenessClassificationV2,
    InappropriatenessClassificationv2,
)
from .KinopoiskClassification import KinopoiskClassification
from .ru_toixic_classification_okmlcup import (
    RuToxicOKMLCUPClassification,
    RuToxicOKMLCUPClassificationV2,
)
from .RuReviewsClassification import RuReviewsClassification, RuReviewsClassificationV2
from .RuSciBenchGRNTIClassification import RuSciBenchGRNTIClassification
from .RuSciBenchOECDClassification import RuSciBenchOECDClassification
from .senti_ru_eval import (
    SentiRuEval2016Classification,
    SentiRuEval2016ClassificationV2,
)

__all__ = [
    "GeoreviewClassification",
    "GeoreviewClassificationV2",
    "HeadlineClassification",
    "HeadlineClassificationV2",
    "InappropriatenessClassification",
    "InappropriatenessClassificationV2",
    "InappropriatenessClassificationv2",
    "KinopoiskClassification",
    "RuReviewsClassification",
    "RuReviewsClassificationV2",
    "RuSciBenchGRNTIClassification",
    "RuSciBenchOECDClassification",
    "RuToxicOKMLCUPClassification",
    "RuToxicOKMLCUPClassificationV2",
    "SentiRuEval2016Classification",
    "SentiRuEval2016ClassificationV2",
]
