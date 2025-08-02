from __future__ import annotations

from .DadoEvalCoarseClassification import DadoEvalCoarseClassification
from .ItaCaseholdClassification import ItaCaseholdClassification
from .ItalianLinguistAcceptabilityClassification import (
    ItalianLinguisticAcceptabilityClassification,
)
from .SardiStanceClassification import SardiStanceClassification

__all__ = [
    "ItaCaseholdClassification",
    "ItalianLinguisticAcceptabilityClassification",
    "DadoEvalCoarseClassification",
    "SardiStanceClassification",
]
