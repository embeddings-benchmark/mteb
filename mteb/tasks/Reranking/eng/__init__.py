from __future__ import annotations

from .AskUbuntuDupQuestions import AskUbuntuDupQuestions
from .BIRCO_Reranking import (
    BIRCOArguAnaReranking,
    BIRCOClinicalTrialReranking,
    BIRCODorisMaeReranking,
    BIRCORelicReranking,
    BIRCOWhatsThatBookReranking,
)
from .MindSmallReranking import MindSmallReranking
from .NevIR import NevIR
from .SciDocsReranking import SciDocsReranking
from .StackOverflowDupQuestions import StackOverflowDupQuestions
from .WebLINXCandidatesReranking import WebLINXCandidatesReranking

__all__ = [
    "AskUbuntuDupQuestions",
    "BIRCODorisMaeReranking",
    "BIRCOArguAnaReranking",
    "BIRCOClinicalTrialReranking",
    "BIRCOWhatsThatBookReranking",
    "BIRCORelicReranking",
    "WebLINXCandidatesReranking",
    "StackOverflowDupQuestions",
    "NevIR",
    "MindSmallReranking",
    "SciDocsReranking",
]
