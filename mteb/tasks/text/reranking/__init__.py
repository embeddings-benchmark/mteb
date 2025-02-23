from __future__ import annotations

from .ara import NamaaMrTydiReranking
from .eng import (
    AskUbuntuDupQuestions,
    BIRCOArguAnaReranking,
    BIRCOClinicalTrialReranking,
    BIRCODorisMaeReranking,
    BIRCORelicReranking,
    BIRCOWhatsThatBookReranking,
    MindSmallReranking,
    NevIR,
    SciDocsReranking,
    StackOverflowDupQuestions,
    WebLINXCandidatesReranking,
)
from .fra import AlloprofReranking, SyntecReranking
from .jpn import VoyageMMarcoReranking
from .multilingual import ESCIReranking, MIRACLReranking, WikipediaRerankingMultilingual
from .rus import RuBQReranking
from .zho import CMedQAv1, CMedQAv2, MMarcoReranking, T2Reranking

__all__ = [
    "AlloprofReranking",
    "AskUbuntuDupQuestions",
    "BIRCOArguAnaReranking",
    "BIRCOClinicalTrialReranking",
    "BIRCODorisMaeReranking",
    "BIRCORelicReranking",
    "BIRCOWhatsThatBookReranking",
    "CMedQAv1",
    "CMedQAv2",
    "ESCIReranking",
    "MIRACLReranking",
    "MMarcoReranking",
    "MindSmallReranking",
    "NamaaMrTydiReranking",
    "NevIR",
    "RuBQReranking",
    "SciDocsReranking",
    "StackOverflowDupQuestions",
    "SyntecReranking",
    "T2Reranking",
    "VoyageMMarcoReranking",
    "WebLINXCandidatesReranking",
    "WikipediaRerankingMultilingual",
]
