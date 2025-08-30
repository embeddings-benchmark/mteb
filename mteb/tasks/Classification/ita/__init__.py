from __future__ import annotations

from .DadoEvalCoarseClassification import DadoEvalCoarseClassification
from .ItaCaseholdClassification import ItaCaseholdClassification
from .ItalianLinguistAcceptabilityClassification import (
    ItalianLinguisticAcceptabilityClassification,
    ItalianLinguisticAcceptabilityClassificationV2,
)
from .SardiStanceClassification import SardiStanceClassification

__all__ = [
    "DadoEvalCoarseClassification",
    "ItaCaseholdClassification",
    "ItalianLinguisticAcceptabilityClassification",
    "ItalianLinguisticAcceptabilityClassificationV2",
    "SardiStanceClassification",
]
