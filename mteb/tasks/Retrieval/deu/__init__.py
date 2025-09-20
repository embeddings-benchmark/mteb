from __future__ import annotations

from .GerDaLIRRetrieval import GerDaLIR
from .GerDaLIRSmallRetrieval import GerDaLIRSmall
from .German1Retrieval import German1Retrieval
from .GermanDPRRetrieval import GermanDPR
from .GermanGovServiceRetrieval import GermanGovServiceRetrieval
from .GermanHealthcare1Retrieval import GermanHealthcare1Retrieval
from .GermanLegal1Retrieval import GermanLegal1Retrieval
from .GermanQuADRetrieval import GermanQuADRetrieval
from .LegalQuADRetrieval import LegalQuAD

__all__ = [
    "GerDaLIR",
    "GerDaLIRSmall",
    "GermanDPR",
    "GermanGovServiceRetrieval",
    "GermanQuADRetrieval",
    "LegalQuAD",
    "GermanLegal1Retrieval",
    "German1Retrieval",
    "GermanHealthcare1Retrieval",
]
